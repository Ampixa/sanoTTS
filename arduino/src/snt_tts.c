/* snt_tts.c -- saanotts-mcu core: the full TTS pipeline in platform-free
 * C99. All platform speed lives behind snt_port.h. Extracted from the
 * ESP32-S3 lab harness (esp32c3/fsd/fsd_e2e.c) after the optimization
 * campaign reached 0.22x RT; the math is golden-gated bit-exact.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snt_port.h"
#include "snt_tts.h"

#include "model/fsd_meta.h"
#include "model/fsd_q8_meta.h"
#include "model/front_q8_meta.h"
#include "model/frozen_norm.h"
#include "model/chain_scales.h"
#include "model/rb_scales.h"

#define NOW_US() snt_now_us()

#define HOP FSD_HOP
#define NFFT FSD_N_FFT
#define TILE 8
#define LENGTH_SCALE 1.08f

/* ---- SanoTTS Arduino-library addition: sibilant fricative-noise
 * injection -------------------------------------------------------------
 * NOT part of the upstream mcu/ portable core (see mcu/src/snt_tts.c in
 * the saanoTTS repo) -- ported here from the ESP32-S3 standalone-app
 * reference implementation, mcu/ports/esp32s3/firmware/main/fsd_e2e.c
 * (search "sibilant"), which never fed back into the portable runtime.
 * Fixes the whistly/metallic /s z sh zh/ artifact: the deterministic
 * acoustic student regresses the MEAN 40-dim latent (FSD_CODE_DIM), so
 * broadband-noise sibilants collapse to a tone. At sibilant frames we add
 * per-channel Gaussian noise (scaled by that voice's own teacher latent
 * std) into the latent just before quantization, restoring the variance
 * the decoder needs to render hiss. Host-verified on the r7/Kristin stack:
 * sibilant 2-8kHz flatness 0.597 -> 0.686 at beta 0.9 (teacher 0.689),
 * confined to sibilant frames.
 *
 * Off by default (g_sib_beta == 0): every branch below is skipped, no
 * extra allocation happens, and output is bit-identical to the unmodified
 * upstream core. Configure via snt_sibilant_configure() (declared in
 * snt_tts.h) with a voice-specific calibration -- tea_std and sib_ids are
 * NOT portable across voices (tools/calibrate_sibilant_noise.py, or the
 * `sibilant-injection/calib.npz` shipped alongside a voice release). */
static const float *g_sib_tea_std;  /* FSD_CODE_DIM floats, or NULL = off */
static const int32_t *g_sib_ids;    /* voice's sibilant phoneme ids, or NULL */
static int g_sib_n_ids;
static float g_sib_beta;            /* 0 = disabled (default) */
static uint32_t g_sib_rng = 0x9e3779b9u;

void snt_sibilant_configure(const float *tea_std, const int32_t *sib_ids,
                            int n_sib_ids, float beta) {
    g_sib_tea_std = tea_std;
    g_sib_ids = sib_ids;
    g_sib_n_ids = n_sib_ids;
    g_sib_beta = beta;
}

static int sib_is_sibilant(int32_t id) {
    for (int i = 0; i < g_sib_n_ids; i++) if (g_sib_ids[i] == id) return 1;
    return 0;
}

/* xorshift32 + sum-of-12-uniforms approx-normal; deterministic seed so
 * runs are reproducible (matches the fsd_e2e.c reference generator). */
static inline float sib_randn(void) {
    float acc = 0.0f;
    for (int k = 0; k < 12; k++) {
        g_sib_rng ^= g_sib_rng << 13; g_sib_rng ^= g_sib_rng >> 17; g_sib_rng ^= g_sib_rng << 5;
        acc += (float)(g_sib_rng & 0xFFFFFF) / (float)0x1000000;
    }
    return acc - 6.0f;
}

static const unsigned char *g_front, *g_dec;
static const signed char *FQ(long o) { return (const signed char *)(g_front + o); }
static const float *FF(long o) { return (const float *)(g_front + o); }
static const signed char *DQ(long o) { return (const signed char *)(g_dec + o); }
static const float *DF(long o) { return (const float *)(g_dec + o); }

/* inline round-half-away-from-zero: lroundf is a ~50-cycle libm call on LX7 */
static inline int fast_round(float y) {
    return (int)(y + (y >= 0.0f ? 0.5f : -0.5f));
}
#ifdef FSD_FAST_MATH
/* Newton reciprocal from exponent-flip seed: no FPU divide (~40 cycles saved) */
static inline float fast_recip(float d) {
    union { float fv; int iv; } u;
    u.fv = d;
    u.iv = 0x7EF311C3 - u.iv;
    float r = u.fv;
    r = r * (2.0f - d * r);
    r = r * (2.0f - d * r);
    return r;
}
#else
static inline float fast_recip(float d) { return 1.0f / d; }
#endif

#ifdef FSD_FAST_MATH
/* exp via exponent-bit split + degree-4 poly on 2^f, rel err ~2e-6 */
static float g_exp2_lut[65];
static int g_exp2_init = 0;
static inline float fast_exp(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    if (!g_exp2_init) {
        for (int i = 0; i <= 64; i++) g_exp2_lut[i] = powf(2.0f, i / 64.0f);
        g_exp2_init = 1;
    }
    float y = x * 1.44269504088896f; /* log2(e) */
    float fi = floorf(y);
    float f = y - fi;
    int idx = (int)(f * 64.0f);
    float pf = g_exp2_lut[idx] +
               (g_exp2_lut[idx + 1] - g_exp2_lut[idx]) * (f * 64.0f - idx);
    union { float fv; int iv; } u;
    u.iv = (int)((fi + 127.0f) * 8388608.0f);
    return u.fv * pf;
}
/* tanh via Pade 3/2, clamped: err <2e-3 on the range GELU feeds it */
static inline float fast_tanh(float y) {
    if (y > 4.97f) return 1.0f;
    if (y < -4.97f) return -1.0f;
    float y2 = y * y;
    return y * (27.0f + y2) * fast_recip(27.0f + 9.0f * y2);
}
#ifdef SNT_ACT_LUT
/* Shared activation LUTs: gelu(x)=x for x>8, =0 for x<-8 (same for silu at
 * +-9), so ONE 512-entry table over the saturating range covers every
 * activation point, error <= 0.02 - below the int8 pipeline noise floor.
 * On FPU-less cores this turns ~15 soft-float ops into ~4. */
#define ALUT_N 512
static float g_gelu_lut[ALUT_N + 1], g_silu_lut[ALUT_N + 1];
static int g_alut_init = 0;
static float gelu_exact(float x) {
    float y = 0.7978845608f * (x + 0.044715f * x * x * x);
    float y2 = y * y;
    float th = (y > 4.97f) ? 1.0f : (y < -4.97f) ? -1.0f
             : y * (27.0f + y2) / (27.0f + 9.0f * y2);
    return 0.5f * x * (1.0f + th);
}
static void alut_init(void) {
    for (int i = 0; i <= ALUT_N; i++) {
        float xg = -8.0f + 16.0f * i / ALUT_N;
        g_gelu_lut[i] = gelu_exact(xg);
        float xs = -9.0f + 18.0f * i / ALUT_N;
        g_silu_lut[i] = xs / (1.0f + expf(-xs));
    }
    g_alut_init = 1;
}
static inline float gelu(float x) {
    if (x >= 8.0f) return x;
    if (x <= -8.0f) return 0.0f;
    return g_gelu_lut[(int)((x + 8.0f) * (ALUT_N / 16.0f))];
}
static inline float silu(float x) {
    if (x >= 9.0f) return x;
    if (x <= -9.0f) return 0.0f;
    return g_silu_lut[(int)((x + 9.0f) * (ALUT_N / 18.0f))];
}
#else
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + fast_tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}
static inline float silu(float x) { return x * fast_recip(1.0f + fast_exp(-x)); }
#endif
static inline float fast_rsqrt(float x) {
    union { float fv; int iv; } u;
    u.fv = x;
    u.iv = 0x5f3759df - (u.iv >> 1);
    float r = u.fv;
    r = r * (1.5f - 0.5f * x * r * r); /* one iteration: ~1e-3, unit-phase ok */
    return r;
}
#define EXPF_M(x) fast_exp(x)
#else
static float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }
static float silu(float x) { return x / (1.0f + expf(-x)); }
#define EXPF_M(x) expf(x)
#endif

typedef struct {
    float gather[320];          /* max gathered vector: pw1 input = 304 */
    signed char qbuf[320] __attribute__((aligned(16)));
    int32_t acc32[320];         /* max rows outside the head: pw0 = 304 */
    float col[64];
} SnScratch;
static int32_t g_acc_head[1600]; /* head_out rows (single-core section only) */
static SnScratch g_scr[2];

#define par_run(f, n, ctx) snt_par_run((f), (n), (ctx))
extern int snt_scratch_id(void);   /* 0 on the main core, 1 on the worker */
#define SCR() (&g_scr[snt_scratch_id()])

/* substage profiling (microseconds, accumulated across the run) */
static long long g_prof[8];
static long long g_ola_us, g_emit_us; /* 0 gather+quant, 1 dots, 2 dw, 3 norm/stats, 4 gelu/silu, 5 fft, 6 ola/emit, 7 film */
#define PROF(i, expr) do { long long _p0 = NOW_US(); expr; g_prof[i] += NOW_US() - _p0; } while (0)

/* one contiguous arena, bump-allocated; phases reset to marks (LIFO) */
static unsigned char *g_arena;
static size_t g_arena_cap, g_arena_top;
static void *aa(size_t n) {
    g_arena_top = (g_arena_top + 15) & ~(size_t)15;
    if (g_arena_top + n > g_arena_cap) {
        printf("ARENA OOM %zu (top %zu cap %zu)\n", n, g_arena_top, g_arena_cap);
        abort();
    }
    void *p = g_arena + g_arena_top;
    g_arena_top += n;
    return p;
}

#define dot_s8(a, b, n) snt_dot_s8((a), (b), (n))

/* copy a weight block into a resident buffer (SIMD requires residency).
 * Graceful degradation: when dst is NULL (arena could not afford the
 * buffer), keep the source pointer - kernels fall back to scalar on
 * non-resident operands, trading speed, never correctness. */
static const signed char *res_copy(const signed char *src, size_t bytes, signed char *dst) {
    if (!dst) return src;
    memcpy(dst, src, bytes);
    return dst;
}

/* arena alloc that returns NULL instead of aborting when out of space */
static void *aa_try(size_t n) {
    size_t top = (g_arena_top + 15) & ~(size_t)15;
    if (top + n > g_arena_cap) return NULL;
    g_arena_top = top + n;
    return g_arena + top;
}

/* out[r] = dot(act, w + r*len) into S->acc32; w resident, 16B tail pad */
static void matvec_s8(SnScratch *S, const signed char *act, const signed char *w, int rows, int len) {
    snt_matvec_s8(act, w, S->acc32, rows, len);
}

/* quantize the scratch gather vector into the scratch qbuf */
static float quant_gather(SnScratch *S, int n, int pad) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = fabsf(S->gather[i]);
        if (v > m) m = v;
    }
    float s = (m > 0.0f) ? m / 127.0f : 1.0f;
    float inv = fast_recip(s);
    for (int i = 0; i < n; i++) S->qbuf[i] = (signed char)fast_round(S->gather[i] * inv);
    for (int i = n; i < pad; i++) S->qbuf[i] = 0;
    return s;
}

/* 1x1 conv column: gather from x (stride xT) at col, quantize, dot rows */
static void q1x1_col(const signed char *w8, const float *sc, const float *bi, int pad,
                     const float *x, int xT, int col, float *out, int out_stride,
                     int out_col, int in_ch, int out_ch) {
    SnScratch *S = SCR();
    for (int i = 0; i < in_ch; i++) S->gather[i] = x[(size_t)i * xT + col];
    float s_act = quant_gather(S, in_ch, pad);
    matvec_s8(S, S->qbuf, w8, out_ch, pad);
    for (int o = 0; o < out_ch; o++)
        out[(size_t)o * out_stride + out_col] = s_act * sc[o] * (float)S->acc32[o] + bi[o];
}

/* 1x1 conv on a PRE-QUANTIZED column (int8 + scale), bitwise-identical to
 * quantizing the float column at the call site */
static void q1x1_q8col(const signed char *w8, const float *sc, const float *bi, int pad,
                       const signed char *col8, float s_act,
                       float *out, int out_stride, int out_col, int out_ch) {
    SnScratch *S = SCR();
    matvec_s8(S, col8, w8, out_ch, pad);
    for (int o = 0; o < out_ch; o++)
        out[(size_t)o * out_stride + out_col] = s_act * sc[o] * (float)S->acc32[o] + bi[o];
}

/* k-tap conv column with same padding: gather window ch-major/k-minor */
static void qkconv_col(const signed char *w8, const float *sc, const float *bi, int pad,
                       const float *x, int xT, int col, float *out, int out_stride,
                       int out_col, int in_ch, int out_ch, int K) {
    SnScratch *S = SCR();
    int half = K / 2;
    for (int i = 0; i < in_ch; i++)
        for (int k = 0; k < K; k++) {
            int idx = col + k - half;
            S->gather[i * K + k] = (idx >= 0 && idx < xT) ? x[(size_t)i * xT + idx] : 0.0f;
        }
    float s_act = quant_gather(S, in_ch * K, pad);
    matvec_s8(S, S->qbuf, w8, out_ch, pad);
    for (int o = 0; o < out_ch; o++)
        out[(size_t)o * out_stride + out_col] = s_act * sc[o] * (float)S->acc32[o] + bi[o];
}

/* residual conv block over a whole [ch, T] buffer (conv-silu-conv, x+s*res).
 * Column-parallel across both cores: conv0, silu, conv1 each split by range
 * with a barrier between stages (conv1 needs the full silu'd tmp). */
#ifdef SNT_INT_CHAIN
static int32_t g_rb_m0[RB_INSTANCES][64];   /* Q28 conv0 row scales */
static int32_t g_rb_b0[RB_INSTANCES][64];   /* Q12 bias in silu-in units */
static int16_t g_rb_slut16[RB_INSTANCES][257]; /* silu, int16-in (interp), int16-out */
static float g_rb_kinv[RB_INSTANCES], g_rb_so[RB_INSTANCES];
static unsigned char g_rb_built[RB_INSTANCES];
static void rb_ic_build(int inst, const float *c0s, const float *c0b, int ch) {
    float s_si = rb_si_max[inst] / 32767.0f;   /* int16 grids */
    float s_so = rb_so_max[inst] / 32767.0f;
    g_rb_kinv[inst] = 1.0f / s_si;
    g_rb_so[inst] = s_so;
    for (int o = 0; o < ch; o++) {
        g_rb_m0[inst][o] = (int32_t)(c0s[o] * 268435456.0f);
        g_rb_b0[inst][o] = (int32_t)(c0b[o] / s_si * 4096.0f);
    }
    /* 256 segments over the int16 silu-in range; entries in silu-out grid */
    for (int i = 0; i <= 256; i++) {
        float x = ((float)(i - 128) * 256.0f) * s_si; /* segment start in fp */
        float y = x / (1.0f + expf(-x));
        int v = fast_round(y / s_so);
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        g_rb_slut16[inst][i] = (int16_t)v;
    }
    g_rb_built[inst] = 1;
}
#endif

typedef struct {
    const signed char *c0w, *c1w;
    const float *c0s, *c0b, *c1s, *c1b;
    int c0pad, c1pad, ch, T, K, inst;
    float bscale, *x, *tmp;
} RbCtx;
static void rb_conv0_range(int lo, int hi, void *vc) {
    RbCtx *c = (RbCtx *)vc;
#ifdef SNT_INT_CHAIN
    if (c->inst < 0) {
        for (int t = lo; t < hi; t++)
            qkconv_col(c->c0w, c->c0s, c->c0b, c->c0pad, c->x, c->T, t, c->tmp, c->T, t, c->ch, c->ch, c->K);
        float *row = c->tmp; /* fused silu for the float path too */
        for (int ch = 0; ch < c->ch; ch++)
            for (int t = lo; t < hi; t++)
                row[(size_t)ch * c->T + t] = silu(row[(size_t)ch * c->T + t]);
        return;
    }
    SnScratch *S = SCR();
    int16_t *tmp16 = (int16_t *)c->tmp;
    int half = c->K / 2;
    const int32_t *m0 = g_rb_m0[c->inst];
    const int32_t *b0 = g_rb_b0[c->inst];
    const int16_t *slut = g_rb_slut16[c->inst];
    float kinv = g_rb_kinv[c->inst];
    for (int t = lo; t < hi; t++) {
        for (int i = 0; i < c->ch; i++)
            for (int k = 0; k < c->K; k++) {
                int idx = t + k - half;
                S->gather[i * c->K + k] = (idx >= 0 && idx < c->T) ? c->x[(size_t)i * c->T + idx] : 0.0f;
            }
        float s_col = quant_gather(S, c->ch * c->K, c->c0pad);
        matvec_s8(S, S->qbuf, c->c0w, c->ch, c->c0pad);
        int32_t kcol = (int32_t)(s_col * kinv * 65536.0f);
        for (int o = 0; o < c->ch; o++) {
            int32_t r12 = (int32_t)(((int64_t)S->acc32[o] * m0[o]) >> 16);
            int32_t qq = (int32_t)((((int64_t)r12 * kcol) >> 16) + b0[o] + 2048) >> 12;
            if (qq > 32767) qq = 32767;
            if (qq < -32768) qq = -32768;
            /* silu via 256-segment integer interp on the int16 grid */
            int32_t u = qq + 32768;             /* 0..65535 */
            int hidx = u >> 8;                  /* 0..255   */
            int frac = u & 255;
            int32_t y0 = slut[hidx], y1 = slut[hidx + 1];
            tmp16[(size_t)o * c->T + t] = (int16_t)(y0 + (((y1 - y0) * frac) >> 8));
        }
    }
#else
    for (int t = lo; t < hi; t++)
        qkconv_col(c->c0w, c->c0s, c->c0b, c->c0pad, c->x, c->T, t, c->tmp, c->T, t, c->ch, c->ch, c->K);
#endif
}
static void rb_silu_range(int lo, int hi, void *vc) {
    RbCtx *c = (RbCtx *)vc;
    float *tmp = c->tmp;
    for (int ch = 0; ch < c->ch; ch++) {
        float *row = tmp + (size_t)ch * c->T;
        for (int t = lo; t < hi; t++) row[t] = silu(row[t]);
    }
}
static void rb_conv1_range(int lo, int hi, void *vc) {
    RbCtx *c = (RbCtx *)vc;
    SnScratch *S = SCR();
    int half = c->K / 2;
#ifdef SNT_INT_CHAIN
    if (c->inst < 0) {
        for (int t = lo; t < hi; t++) {
            for (int i = 0; i < c->ch; i++)
                for (int k = 0; k < c->K; k++) {
                    int idx = t + k - half;
                    S->gather[i * c->K + k] = (idx >= 0 && idx < c->T) ? c->tmp[(size_t)i * c->T + idx] : 0.0f;
                }
            float s_act = quant_gather(S, c->ch * c->K, c->c1pad);
            matvec_s8(S, S->qbuf, c->c1w, c->ch, c->c1pad);
            for (int o = 0; o < c->ch; o++)
                c->x[(size_t)o * c->T + t] += c->bscale * (s_act * c->c1s[o] * (float)S->acc32[o] + c->c1b[o]);
        }
        return;
    }
    const int16_t *tmp16 = (const int16_t *)c->tmp;
    float s_so = g_rb_so[c->inst];
    static int16_t a16[2][512] __attribute__((aligned(16)));
    int16_t *aw = a16[snt_scratch_id()];
    for (int t = lo; t < hi; t++) {
        for (int i = 0; i < c->ch; i++)
            for (int k = 0; k < c->K; k++) {
                int idx = t + k - half;
                aw[i * c->K + k] = (idx >= 0 && idx < c->T) ? tmp16[(size_t)i * c->T + idx] : 0;
            }
        for (int i = c->ch * c->K; i < c->c1pad; i++) aw[i] = 0;
        snt_matvec_s16s8(aw, c->c1w, S->acc32, c->ch, c->c1pad);
        for (int o = 0; o < c->ch; o++)
            c->x[(size_t)o * c->T + t] += c->bscale * (s_so * c->c1s[o] * (float)S->acc32[o] + c->c1b[o]);
    }
#else
    for (int t = lo; t < hi; t++) {
        for (int i = 0; i < c->ch; i++)
            for (int k = 0; k < c->K; k++) {
                int idx = t + k - half;
                S->gather[i * c->K + k] = (idx >= 0 && idx < c->T) ? c->tmp[(size_t)i * c->T + idx] : 0.0f;
            }
        float s_act = quant_gather(S, c->ch * c->K, c->c1pad);
        matvec_s8(S, S->qbuf, c->c1w, c->ch, c->c1pad);
        for (int o = 0; o < c->ch; o++)
            c->x[(size_t)o * c->T + t] += c->bscale * (s_act * c->c1s[o] * (float)S->acc32[o] + c->c1b[o]);
    }
#endif
}
#ifdef SNT_INT_CHAIN
static int32_t g_ic_m0[5][FSD_PW_HIDDEN];
static int32_t g_ic_b0[5][FSD_PW_HIDDEN];
static signed char g_ic_glut[5][256];
static float g_ic_sa_inv[5], g_ic_sgo[5];
static int16_t g_dw_w16[5][FSD_DIM * FSD_DW_KERNEL];
static float g_dw_k[5][FSD_DIM];   /* s_wch * ninv * nw[ch] (x s_x at runtime) */
static float g_dw_b[5][FSD_DIM];   /* (dwb - mean) * ninv * nw + nb            */
static unsigned char g_dw_built[5];
static int g_ic_init = 0;
static void ic_init(const long qpw[5][6], const float *pw0in_max,
                    const float *geluin_max, const float *geluout_max) {
    for (int bi = 0; bi < 5; bi++) {
        float s_g = geluin_max[bi] / 127.0f;
        float s_go = geluout_max[bi] / 127.0f;
        g_ic_sa_inv[bi] = 1.0f / s_g; /* repurposed: 1/s_g for kcol */
        g_ic_sgo[bi] = s_go;
        const float *sc = DF(qpw[bi][1]);
        const float *bb = DF(qpw[bi][2]);
        for (int o = 0; o < FSD_PW_HIDDEN; o++) {
            /* per-row static (weight scale only); per-column dynamic factor
             * applied as a second integer multiply at runtime */
            g_ic_m0[bi][o] = (int32_t)(sc[o] * 268435456.0f);   /* Q28 */
            g_ic_b0[bi][o] = (int32_t)(bb[o] / s_g * 4096.0f);  /* bias in gelu-in units, Q12 */
        }
        for (int q = -128; q < 128; q++) {
            float y = gelu_exact((float)q * s_g);
            int v = fast_round(y / s_go);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            g_ic_glut[bi][q + 128] = (signed char)v;
        }
    }
    g_ic_init = 1;
}
#endif

#ifdef SNT_INT_CHAIN
static void dw_ic_build(int bi, const float *dww, const float *dwb,
                        const float *nw, const float *nb, float mean, float ninv) {
    for (int ch = 0; ch < FSD_DIM; ch++) {
        float m = 0.0f;
        for (int k = 0; k < FSD_DW_KERNEL; k++) {
            float v = fabsf(dww[(size_t)ch * FSD_DW_KERNEL + k]);
            if (v > m) m = v;
        }
        float s_w = (m > 0.0f) ? m / 32767.0f : 1.0f;
        for (int k = 0; k < FSD_DW_KERNEL; k++)
            g_dw_w16[bi][ch * FSD_DW_KERNEL + k] =
                (int16_t)fast_round(dww[(size_t)ch * FSD_DW_KERNEL + k] / s_w);
        g_dw_k[bi][ch] = s_w * ninv * nw[ch];
        g_dw_b[bi][ch] = (dwb[ch] - mean) * ninv * nw[ch] + nb[ch];
    }
    g_dw_built[bi] = 1;
}
#endif

typedef struct {
    const signed char *fr, *fo, *pw0, *pw1;
    const float *fr_s, *fr_b, *fo_s, *fo_b, *pw0_s, *pw0_b, *pw1_s, *pw1_b;
    const signed char *c8;
    const float *cscale;
    float *tile_f2, *tile_c, *tile_h2, *x;
    int t0, TL, T, bi;
    float bscale;
    /* depthwise + frozen-norm folded into the parallel column pipeline */
    const float *dww, *dwb, *nw, *nb;
    float mean, ninv;
    const float (*halo)[FSD_DIM];
} TileCtx;
/* stage 1: depthwise + frozen norm for a column range (reads x, writes tile_c).
 * MUST complete for the whole tile before stage 2 writes x (pw1 residual). */
static void tile_dw_range(int lo, int hi, void *vc) {
    TileCtx *c = (TileCtx *)vc;
    int half = FSD_DW_KERNEL / 2;
    for (int t = lo; t < hi; t++) {
        int col = c->t0 + t;
        int interior = (col >= half && col + half < c->T && t >= half);
        if (interior) {
#ifdef SNT_INT_CHAIN
            /* int16 dw with ONCE-PER-RANGE quantization: x columns for this
             * range (+halo) quantized a single time into xq16; the sliding
             * windows read it (v1 re-quantized each value ~7x - measured
             * SLOWER than float). */
            static int16_t xq16[2][FSD_DIM][32] __attribute__((aligned(4)));
            static float xq_sx[2];
            static int xq_from[2], xq_to[2], xq_tile[2], xq_bi[2];
            int sid = snt_scratch_id();
            int rfrom = c->t0 + lo - half, rto = c->t0 + hi + half; /* col span */
            if (xq_tile[sid] != c->t0 + 1 || xq_bi[sid] != c->bi + 1
                || xq_from[sid] != rfrom || xq_to[sid] != rto) {
                float mx = 1e-20f;
                for (int ch = 0; ch < FSD_DIM; ch++) {
                    const float *xr = c->x + (size_t)ch * c->T;
                    for (int cc = rfrom; cc < rto; cc++) {
                        float v = fabsf(xr[cc]);
                        if (v > mx) mx = v;
                    }
                }
                float inv_sx = 32767.0f / mx;
                for (int ch = 0; ch < FSD_DIM; ch++) {
                    const float *xr = c->x + (size_t)ch * c->T;
                    for (int cc = rfrom; cc < rto; cc++)
                        xq16[sid][ch][cc - rfrom] = (int16_t)fast_round(xr[cc] * inv_sx);
                }
                xq_sx[sid] = mx / 32767.0f;
                xq_from[sid] = rfrom; xq_to[sid] = rto;
                xq_tile[sid] = c->t0 + 1; xq_bi[sid] = c->bi + 1;
            }
            float s_x = xq_sx[sid];
            const int16_t *w16 = g_dw_w16[c->bi];
            const float *kk = g_dw_k[c->bi];
            const float *bb2 = g_dw_b[c->bi];
            int woff = col - half - xq_from[sid];
            for (int ch = 0; ch < FSD_DIM; ch++) {
                const int16_t *xr = xq16[sid][ch] + woff;
                const int16_t *wr = w16 + ch * FSD_DW_KERNEL;
                int32_t acc = (int32_t)wr[0] * xr[0] + (int32_t)wr[1] * xr[1]
                            + (int32_t)wr[2] * xr[2] + (int32_t)wr[3] * xr[3]
                            + (int32_t)wr[4] * xr[4] + (int32_t)wr[5] * xr[5]
                            + (int32_t)wr[6] * xr[6];
                c->tile_c[(size_t)ch * c->TL + t] = (float)acc * (s_x * kk[ch]) + bb2[ch];
            }
#else
            for (int ch = 0; ch < FSD_DIM; ch++) {
                const float *wr = c->dww + (size_t)ch * FSD_DW_KERNEL;
                const float *xr = c->x + (size_t)ch * c->T + col - half;
                float a2 = c->dwb[ch] + wr[0] * xr[0] + wr[1] * xr[1] + wr[2] * xr[2]
                         + wr[3] * xr[3] + wr[4] * xr[4] + wr[5] * xr[5] + wr[6] * xr[6];
                c->tile_c[(size_t)ch * c->TL + t] = (a2 - c->mean) * c->ninv * c->nw[ch] + c->nb[ch];
            }
#endif
        } else {
            for (int ch = 0; ch < FSD_DIM; ch++) {
                const float *wr = c->dww + (size_t)ch * FSD_DW_KERNEL;
                const float *xr = c->x + (size_t)ch * c->T;
                float a2 = c->dwb[ch];
                for (int k = 0; k < FSD_DW_KERNEL; k++) {
                    int idx = col + k - half;
                    if (idx < 0 || idx >= c->T) continue;
                    float v = xr[idx];
                    if (idx < c->t0) {
                        int h = idx - (c->t0 - half);
                        if (h >= 0) v = c->halo[h][ch];
                    }
                    a2 += wr[k] * v;
                }
                c->tile_c[(size_t)ch * c->TL + t] = (a2 - c->mean) * c->ninv * c->nw[ch] + c->nb[ch];
            }
        }
    }
}

/* stage 2: film + pw0 + gelu + pw1-residual for a column range */
static void tile_filmpw_range(int lo, int hi, void *vc) {
    TileCtx *c = (TileCtx *)vc;
    SnScratch *S = SCR();
    float *red = c->tile_f2;
    float *fo = c->tile_f2 + (size_t)FSD_FILM_RANK * c->TL;
    for (int t = lo; t < hi; t++) {
        long long _f0 = NOW_US();
        q1x1_q8col(c->fr, c->fr_s, c->fr_b, Q8_B0_FILMR_N16,
                   c->c8 + (size_t)(c->t0 + t) * Q8_PRE_N16, c->cscale[c->t0 + t],
                   red, c->TL, t, FSD_FILM_RANK);
        q1x1_col(c->fo, c->fo_s, c->fo_b, Q8_B0_FILMO_N16,
                 red, c->TL, t, fo, c->TL, t, FSD_FILM_RANK, 2 * FSD_DIM);
        for (int ch = 0; ch < FSD_DIM; ch++) {
            size_t li = (size_t)ch * c->TL + t;
            c->tile_c[li] = c->tile_c[li] * (1.0f + fo[li]) + fo[(size_t)(FSD_DIM + ch) * c->TL + t];
        }
        g_prof[6] += NOW_US() - _f0;
#ifdef SNT_INT_CHAIN
        {
            /* dynamic per-column input quant (as the float path) */
            for (int i = 0; i < FSD_DIM; i++) S->gather[i] = c->tile_c[(size_t)i * c->TL + t];
            float s_col = quant_gather(S, FSD_DIM, Q8_B0_PW0_N16);
            matvec_s8(S, S->qbuf, c->pw0, FSD_PW_HIDDEN, Q8_B0_PW0_N16);
            /* kcol: dynamic column factor in Q16 (one float mul per COLUMN) */
            int32_t kcol = (int32_t)(s_col * g_ic_sa_inv[c->bi] * 65536.0f);
            const int32_t *m0 = g_ic_m0[c->bi];
            const int32_t *b0 = g_ic_b0[c->bi];
            const signed char *glut = g_ic_glut[c->bi];
            for (int o = 0; o < FSD_PW_HIDDEN; o++) {
                /* acc * sc[o] (Q28) -> Q12; * kcol (Q16) -> Q12 gelu-in units */
                int32_t r12 = (int32_t)(((int64_t)S->acc32[o] * m0[o]) >> 16); /* Q12 */
                int32_t qq = (int32_t)((((int64_t)r12 * kcol) >> 16) + b0[o] + 2048) >> 12;
                if (qq > 127) qq = 127;
                if (qq < -128) qq = -128;
                S->qbuf[o] = glut[qq + 128];
            }
            for (int i = FSD_PW_HIDDEN; i < Q8_B0_PW1_N16; i++) S->qbuf[i] = 0;
            matvec_s8(S, S->qbuf, c->pw1, FSD_DIM, Q8_B0_PW1_N16);
            float s_act = g_ic_sgo[c->bi];
            for (int ch = 0; ch < FSD_DIM; ch++)
                c->x[(size_t)ch * c->T + c->t0 + t] += c->bscale * (s_act * c->pw1_s[ch] * (float)S->acc32[ch] + c->pw1_b[ch]);
        }
#else
        q1x1_col(c->pw0, c->pw0_s, c->pw0_b, Q8_B0_PW0_N16,
                 c->tile_c, c->TL, t, c->tile_h2, c->TL, t, FSD_DIM, FSD_PW_HIDDEN);
        for (int i = 0; i < FSD_PW_HIDDEN; i++) {
            size_t li = (size_t)i * c->TL + t;
            c->tile_h2[li] = gelu(c->tile_h2[li]);
        }
        for (int i = 0; i < FSD_PW_HIDDEN; i++) S->gather[i] = c->tile_h2[(size_t)i * c->TL + t];
        float s_act = quant_gather(S, FSD_PW_HIDDEN, Q8_B0_PW1_N16);
        matvec_s8(S, S->qbuf, c->pw1, FSD_DIM, Q8_B0_PW1_N16);
        for (int ch = 0; ch < FSD_DIM; ch++)
            c->x[(size_t)ch * c->T + c->t0 + t] += c->bscale * (s_act * c->pw1_s[ch] * (float)S->acc32[ch] + c->pw1_b[ch]);
#endif
    }
}

static void resblock(int inst,
                     const signed char *c0w, const float *c0s, const float *c0b, int c0pad,
                     const signed char *c1w, const float *c1s, const float *c1b, int c1pad,
                     float bscale, float *x, float *tmp, int ch, int T, int K) {
    RbCtx c = {c0w, c1w, c0s, c0b, c1s, c1b, c0pad, c1pad, ch, T, K, inst, bscale, x, tmp};
#ifdef SNT_INT_CHAIN
    /* int chain only where accumulation tolerates it: the 5 frame blocks
     * (80% of rbc cost). dur/token blocks stay float: 11 stacked silu
     * grids measurably cratered corr (0.933 short golden). */
    /* rbc int chain DISABLED pending precision redesign (int16 silu
     * grid): 11-instance int8 grids cratered corr, and partial gating
     * produced inconsistent movement = fallback bug risk. Stage-1
     * (pw chain) remains the proven int path. */
    c.inst = inst; /* int16 grids: all instances */
    if (!g_rb_built[inst]) rb_ic_build(inst, c0s, c0b, ch);
#endif
    long long _r0 = NOW_US();
    par_run(rb_conv0_range, T, &c);
    long long _r1 = NOW_US();
    g_prof[2] += _r1 - _r0;
#ifndef SNT_INT_CHAIN
    par_run(rb_silu_range, T, &c);
#endif
    long long _r2 = NOW_US();
    g_prof[4] += _r2 - _r1;
    par_run(rb_conv1_range, T, &c);
    g_prof[7] += NOW_US() - _r2;
}

typedef struct {
    const float *hp1;           /* magnitude logits (513, float)          */
    const int32_t *acc;         /* raw head accumulators (all 1539 rows)  */
    const int32_t *pm;          /* per-row Q20 phase multipliers (static) */
    const int32_t *pb;          /* per-row Q12 phase biases / s2 folding  */
    int32_t k2;                 /* per-frame s2 factor, Q16               */
    float *fre, *fim;
} SpecCtx;
#ifdef SNT_INT_CHAIN
/* integer exp2: 2^(q/256) for q in [0,256) as Q30, plus shift by int part */
static float g_pow2_tab[64];
static int32_t g_e2lut[257];
static int g_e2_init = 0;
static void e2_init(void) {
    for (int i = 0; i <= 256; i++)
        g_e2lut[i] = (int32_t)(pow(2.0, i / 256.0) * (double)(1 << 22)); /* Q22 */
    for (int i = 0; i < 64; i++)
        g_pow2_tab[i] = powf(2.0f, (float)(i - 32 - 22));
    g_e2_init = 1;
}
#endif
static void spec_range(int lo, int hi, void *vc) {
    SpecCtx *c = (SpecCtx *)vc;
    for (int k = lo; k < hi; k++) {
#ifdef SNT_INT_CHAIN
        /* logmag (float, 1 mul from head dequant) -> Q8 log2 domain int,
         * exp via LUT + shift: ~6 int ops replace the fast_exp chain */
        float lm = c->hp1[k];
        if (lm < -12.0f) lm = -12.0f;
        if (lm > 8.0f) lm = 8.0f;
        int32_t q8 = (int32_t)(lm * 369.3299304675746f) + 8192; /* log2e*256, bias 32*256 */
        int e2i = q8 >> 8;              /* 0..43   (lm in [-12,8] -> log2 in [-17.3,11.6], biased +32) */
        int e2f = q8 & 255;
        /* m = lut[frac] * 2^(e2i-32-22) : keep as float via ldexpf-free scale */
        float m = (float)g_e2lut[e2f] * g_pow2_tab[e2i];
        if (m < 1e-7f) m = 1e-7f;
#else
        float m = c->hp1[k];
        if (m < -12.0f) m = -12.0f;
        if (m > 8.0f) m = 8.0f;
        m = EXPF_M(m);
        if (m < 1e-7f) m = 1e-7f;
#endif
#ifdef SNT_INT_CHAIN
        /* phase rows never dequantize: unit vector is scale-free from the
         * integer pair (per-row scale via Q20 mult, bias via Q12, s2 via
         * the frame factor k2 - all integer) */
        int pr_k = FSD_BINS + k, pi_k = 2 * FSD_BINS + k;
        int32_t pri = (int32_t)((((int64_t)((((int64_t)c->acc[pr_k] * c->pm[pr_k]) >> 16)) * c->k2) >> 16) + c->pb[pr_k]);
        int32_t pii = (int32_t)((((int64_t)((((int64_t)c->acc[pi_k] * c->pm[pi_k]) >> 16)) * c->k2) >> 16) + c->pb[pi_k]);
        float pr = (float)pri, pi = (float)pii;
#else
        float pr = c->hp1[FSD_BINS + k], pi = c->hp1[2 * FSD_BINS + k];
#endif
#ifdef FSD_FAST_MATH
        float n2 = pr * pr + pi * pi;
        float r = (n2 > 1e-12f) ? fast_rsqrt(n2) : 0.0f;
        c->fre[k] = m * pr * r;
        c->fim[k] = m * pi * r;
#else
        float nrm = sqrtf(pr * pr + pi * pi);
        if (nrm < 1e-6f) nrm = 1e-6f;
        c->fre[k] = m * pr / nrm;
        c->fim[k] = m * pi / nrm;
#endif
    }
}

#ifdef SNT_INT_FFT
/* int32 inverse FFT, Q(30-log2 n) block-scaled: RV32 MUL ~3 cycles vs
 * ~200-cycle soft-float. Twiddles Q30; butterflies keep Q via 64-bit
 * products (MULH on RV32M). Input/output through per-frame block scale. */
#define FFTN2 (NFFT / 2)
static int32_t g_twr[FFTN2 / 2], g_twi[FFTN2 / 2];
static int g_tw_init = 0;
static inline int32_t qmul(int32_t a, int32_t b) {
    return (int32_t)(((int64_t)a * b) >> 30);
}
static void fft_inv_i32(int32_t *re, int32_t *im, int n) {
    if (!g_tw_init) {
        for (int k = 0; k < FFTN2 / 2; k++) {
            g_twr[k] = (int32_t)(cos(2.0 * M_PI * k / FFTN2) * (double)(1 << 30));
            g_twi[k] = (int32_t)(sin(2.0 * M_PI * k / FFTN2) * (double)(1 << 30));
        }
        g_tw_init = 1;
    }
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            int32_t tr = re[i]; re[i] = re[j]; re[j] = tr;
            int32_t ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        int step = n / len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < len / 2; k++) {
                int tw = k * step;
                int32_t cr = g_twr[tw], ci = g_twi[tw];
                int a2 = i + k, b2 = i + k + len / 2;
                int32_t xr = qmul(re[b2], cr) - qmul(im[b2], ci);
                int32_t xi = qmul(re[b2], ci) + qmul(im[b2], cr);
                /* halve each stage: total 1/n normalization built in */
                int32_t ar = re[a2] >> 1, ai = im[a2] >> 1;
                xr >>= 1; xi >>= 1;
                re[b2] = ar - xr; im[b2] = ai - xi;
                re[a2] = ar + xr; im[a2] = ai + xi;
            }
        }
    }
}
#endif

static void fft_inv(float *re, float *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / len;
        float wr = (float)cos(ang), wi = (float)sin(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                int a = i + k, b = i + k + len / 2;
                float xr = re[b] * cr - im[b] * ci;
                float xi = re[b] * ci + im[b] * cr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr;        im[a] += xi;
                float ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
    float s = 1.0f / n;
    for (int i = 0; i < n; i++) { re[i] *= s; im[i] *= s; }
}



int snt_synthesize(const snt_config *cfg,
                   const int32_t *phoneme_ids, int n_ids,
                   snt_pcm_cb cb, void *user, snt_stats *stats_out) {
    SnScratch *S = SCR();
    g_front = (const unsigned char *)cfg->front_blob;
    g_dec = (const unsigned char *)cfg->dec_blob;
    const int32_t *ids = phoneme_ids;
    int n_tokens = n_ids;
    g_arena = (unsigned char *)(((uintptr_t)cfg->arena + 15) & ~(uintptr_t)15);
    g_arena_cap = cfg->arena_size - (size_t)(g_arena - (unsigned char *)cfg->arena);
    g_arena_top = 0;
#ifdef SNT_ACT_LUT
    if (!g_alut_init) alut_init();
#endif
#ifdef SNT_INT_CHAIN
    /* dw constants fold per-utterance norm stats: rebuild every call */
    for (int i = 0; i < 5; i++) g_dw_built[i] = 0;
    if (!g_e2_init) e2_init();
#endif
    long long t_start = NOW_US();

    /* ---------- duration student ---------- */
    int H = FRONT_DUR_HIDDEN;
    const float *demb = FF(FOFF_DUR_EMB_F32);
    size_t mark_dur = g_arena_top;
    float *dh = (float *)aa((size_t)H * n_tokens * 4);
    float *dtmp = (float *)aa((size_t)H * n_tokens * 4);
    /* SRAM-resident duration weights (were scalar-from-flash: 247 ms!) */
    const signed char *r_dproj = res_copy(FQ(FOFF_DUR_PROJ_W8),
        (size_t)H * FRONT_DUR_PROJ_N16, (signed char *)aa_try((size_t)H * FRONT_DUR_PROJ_N16 + 16));
    signed char *r_dc0 = (signed char *)aa_try((size_t)H * FRONT_DUR_B0_C0_N16 + 16);
    signed char *r_dc1 = (signed char *)aa_try((size_t)H * FRONT_DUR_B0_C1_N16 + 16);
    const signed char *r_dout = res_copy(FQ(FOFF_DUR_OUT_W8),
        (size_t)FRONT_DUR_OUT_N16, (signed char *)aa_try((size_t)FRONT_DUR_OUT_N16 + 16));
    for (int t = 0; t < n_tokens; t++) {
        for (int h = 0; h < H; h++) S->gather[h] = demb[(size_t)ids[t] * H + h];
        S->gather[H] = (n_tokens > 1) ? (float)t / (float)(n_tokens - 1) : 0.0f;
        S->gather[H + 1] = log1pf((float)n_tokens) / log1pf((float)FRONT_DUR_MAX_TOKENS);
        S->gather[H + 2] = 1.0f;
        float s_act = quant_gather(S, H + 3, FRONT_DUR_PROJ_N16);
        const signed char *w8 = r_dproj;
        const float *sc = FF(FOFF_DUR_PROJ_SCALE);
        const float *bi = FF(FOFF_DUR_PROJ_BIAS);
        matvec_s8(S, S->qbuf, w8, H, FRONT_DUR_PROJ_N16);
        for (int o = 0; o < H; o++)
            dh[(size_t)o * n_tokens + t] = s_act * sc[o] * (float)S->acc32[o] + bi[o];
    }
    long dur_off[3][7] = {
        {FOFF_DUR_B0_C0_W8, FOFF_DUR_B0_C0_SCALE, FOFF_DUR_B0_C0_BIAS, FOFF_DUR_B0_C1_W8, FOFF_DUR_B0_C1_SCALE, FOFF_DUR_B0_C1_BIAS, FOFF_DUR_B0_SCALE_F32},
        {FOFF_DUR_B1_C0_W8, FOFF_DUR_B1_C0_SCALE, FOFF_DUR_B1_C0_BIAS, FOFF_DUR_B1_C1_W8, FOFF_DUR_B1_C1_SCALE, FOFF_DUR_B1_C1_BIAS, FOFF_DUR_B1_SCALE_F32},
        {FOFF_DUR_B2_C0_W8, FOFF_DUR_B2_C0_SCALE, FOFF_DUR_B2_C0_BIAS, FOFF_DUR_B2_C1_W8, FOFF_DUR_B2_C1_SCALE, FOFF_DUR_B2_C1_BIAS, FOFF_DUR_B2_SCALE_F32},
    };
    for (int b = 0; b < FRONT_DUR_DEPTH; b++) {
        const signed char *dc0 = res_copy(FQ(dur_off[b][0]), (size_t)H * FRONT_DUR_B0_C0_N16, r_dc0);
        const signed char *dc1 = res_copy(FQ(dur_off[b][3]), (size_t)H * FRONT_DUR_B0_C1_N16, r_dc1);
        resblock(b, dc0, FF(dur_off[b][1]), FF(dur_off[b][2]), FRONT_DUR_B0_C0_N16,
                 dc1, FF(dur_off[b][4]), FF(dur_off[b][5]), FRONT_DUR_B0_C1_N16,
                 *FF(dur_off[b][6]), dh, dtmp, H, n_tokens, FRONT_DUR_KERNEL);
    }
    static int durs[1024];
    if (n_tokens > 1024) { printf("too many tokens\n"); return 1; }
    {

        const float *os = FF(FOFF_DUR_OUT_SCALE);
        const float *ob = FF(FOFF_DUR_OUT_BIAS);
        for (int t = 0; t < n_tokens; t++) {
            for (int i = 0; i < H; i++) S->gather[i] = dh[(size_t)i * n_tokens + t];
            float s_act = quant_gather(S, H, FRONT_DUR_OUT_N16);
            matvec_s8(S, S->qbuf, r_dout, 1, FRONT_DUR_OUT_N16);
            float logd = s_act * os[0] * (float)S->acc32[0] + ob[0];
            float d = roundf(fmaxf(expf(logd), 1.0f) * LENGTH_SCALE);
            if (d < 1.0f) d = 1.0f;
            if (d > (float)FRONT_DUR_MAX_DURATION) d = (float)FRONT_DUR_MAX_DURATION;
            durs[t] = (int)d;
        }
    }
    g_arena_top = mark_dur;
    if (cfg->dur_override)   /* test hook: frame-exact golden alignment */
        for (int t = 0; t < n_tokens; t++) durs[t] = cfg->dur_override[t];
    int T = 0;
    for (int t = 0; t < n_tokens; t++) T += durs[t];

    /* ---------- acoustic ---------- */
    int AH = FRONT_AC_HIDDEN;
    const float *aemb = FF(FOFF_AC_EMB_F32);
    float maxdur = 1.0f;
    for (int t = 0; t < n_tokens; t++)
        if ((float)durs[t] > maxdur) maxdur = (float)durs[t];
    /* persistent across both phases: trunk buffer + pre-quantized code */
    float *x = (float *)aa((size_t)FSD_DIM * T * 4);
    signed char *c8 = (signed char *)aa((size_t)T * Q8_PRE_N16); /* frame-major, padded */
    float *cscale = (float *)aa((size_t)T * 4);
    size_t mark_front = g_arena_top;
    float *ax = (float *)aa((size_t)AH * T * 4);
    float *th = (float *)aa((size_t)AH * n_tokens * 4);
    signed char *rbuf_kc0 = (signed char *)aa_try((size_t)FRONT_AC_HIDDEN * FRONT_AC_TB0_C0_N16 + 16);
    signed char *rbuf_kc1 = (signed char *)aa_try((size_t)FRONT_AC_HIDDEN * FRONT_AC_TB0_C1_N16 + 16);
    float *ttmp = ax; /* token temp borrows the frame buffer */
    const signed char *tproj_w = res_copy(FQ(FOFF_AC_TPROJ_W8), (size_t)AH * FRONT_AC_TPROJ_N16, rbuf_kc0);
    for (int t = 0; t < n_tokens; t++) {
        for (int h = 0; h < AH; h++) S->gather[h] = aemb[(size_t)ids[t] * AH + h];
        S->gather[AH] = (n_tokens > 1) ? (float)t / (float)(n_tokens - 1) : 0.0f;
        S->gather[AH + 1] = log1pf((float)durs[t]) / log1pf(maxdur);
        float s_act = quant_gather(S, AH + 2, FRONT_AC_TPROJ_N16);
        const signed char *w8 = tproj_w;
        const float *sc = FF(FOFF_AC_TPROJ_SCALE);
        const float *bi = FF(FOFF_AC_TPROJ_BIAS);
        matvec_s8(S, S->qbuf, w8, AH, FRONT_AC_TPROJ_N16);
        for (int o = 0; o < AH; o++)
            th[(size_t)o * n_tokens + t] = s_act * sc[o] * (float)S->acc32[o] + bi[o];
    }
    long tb_off[3][7] = {
        {FOFF_AC_TB0_C0_W8, FOFF_AC_TB0_C0_SCALE, FOFF_AC_TB0_C0_BIAS, FOFF_AC_TB0_C1_W8, FOFF_AC_TB0_C1_SCALE, FOFF_AC_TB0_C1_BIAS, FOFF_AC_TB0_SCALE_F32},
        {FOFF_AC_TB1_C0_W8, FOFF_AC_TB1_C0_SCALE, FOFF_AC_TB1_C0_BIAS, FOFF_AC_TB1_C1_W8, FOFF_AC_TB1_C1_SCALE, FOFF_AC_TB1_C1_BIAS, FOFF_AC_TB1_SCALE_F32},
        {FOFF_AC_TB2_C0_W8, FOFF_AC_TB2_C0_SCALE, FOFF_AC_TB2_C0_BIAS, FOFF_AC_TB2_C1_W8, FOFF_AC_TB2_C1_SCALE, FOFF_AC_TB2_C1_BIAS, FOFF_AC_TB2_SCALE_F32},
    };
    for (int b = 0; b < FRONT_AC_TOKEN_DEPTH; b++) {
        const signed char *c0 = res_copy(FQ(tb_off[b][0]), (size_t)AH * FRONT_AC_TB0_C0_N16, rbuf_kc0);
        const signed char *c1 = res_copy(FQ(tb_off[b][3]), (size_t)AH * FRONT_AC_TB0_C1_N16, rbuf_kc1);
        resblock(3 + b, c0, FF(tb_off[b][1]), FF(tb_off[b][2]), FRONT_AC_TB0_C0_N16,
                 c1, FF(tb_off[b][4]), FF(tb_off[b][5]), FRONT_AC_TB0_C1_N16,
                 *FF(tb_off[b][6]), th, ttmp, AH, n_tokens, FRONT_AC_KERNEL);
    }
    /* expand tokens -> frames on the fly (no [AH+3, T] buffer) */
    float *atmp = x; /* borrow the decoder buffer: AH*T <= FSD_DIM*T */
    /* sibilant injection (see block above): one flag byte per frame, only
     * allocated when the feature is armed; freed with ax/th below. */
    signed char *frame_sib = (g_sib_beta > 0.0f) ? (signed char *)aa_try((size_t)T) : NULL;
    {
        const signed char *w8 = res_copy(FQ(FOFF_AC_FPROJ_W8),
                                         (size_t)AH * FRONT_AC_FPROJ_N16, rbuf_kc0);
        const float *sc = FF(FOFF_AC_FPROJ_SCALE);
        const float *bi = FF(FOFF_AC_FPROJ_BIAS);
        int f = 0;
        for (int tok = 0; tok < n_tokens; tok++) {
            int sib = frame_sib ? sib_is_sibilant(ids[tok]) : 0;
            for (int d = 0; d < durs[tok]; d++, f++) {
                if (frame_sib) frame_sib[f] = (signed char)sib;
                for (int h = 0; h < AH; h++) S->gather[h] = th[(size_t)h * n_tokens + tok];
                S->gather[AH] = (T > 1) ? (float)f / (float)(T - 1) : 0.0f;
                S->gather[AH + 1] = (float)tok / (float)(n_tokens > 1 ? n_tokens - 1 : 1);
                S->gather[AH + 2] = (durs[tok] > 1) ? (float)d / (float)(durs[tok] - 1) : 0.0f;
                float s_act = quant_gather(S, AH + 3, FRONT_AC_FPROJ_N16);
                matvec_s8(S, S->qbuf, w8, AH, FRONT_AC_FPROJ_N16);
                for (int o = 0; o < AH; o++)
                    ax[(size_t)o * T + f] = s_act * sc[o] * (float)S->acc32[o] + bi[o];
            }
        }
    }
    long fb_off[5][7] = {
        {FOFF_AC_FB0_C0_W8, FOFF_AC_FB0_C0_SCALE, FOFF_AC_FB0_C0_BIAS, FOFF_AC_FB0_C1_W8, FOFF_AC_FB0_C1_SCALE, FOFF_AC_FB0_C1_BIAS, FOFF_AC_FB0_SCALE_F32},
        {FOFF_AC_FB1_C0_W8, FOFF_AC_FB1_C0_SCALE, FOFF_AC_FB1_C0_BIAS, FOFF_AC_FB1_C1_W8, FOFF_AC_FB1_C1_SCALE, FOFF_AC_FB1_C1_BIAS, FOFF_AC_FB1_SCALE_F32},
        {FOFF_AC_FB2_C0_W8, FOFF_AC_FB2_C0_SCALE, FOFF_AC_FB2_C0_BIAS, FOFF_AC_FB2_C1_W8, FOFF_AC_FB2_C1_SCALE, FOFF_AC_FB2_C1_BIAS, FOFF_AC_FB2_SCALE_F32},
        {FOFF_AC_FB3_C0_W8, FOFF_AC_FB3_C0_SCALE, FOFF_AC_FB3_C0_BIAS, FOFF_AC_FB3_C1_W8, FOFF_AC_FB3_C1_SCALE, FOFF_AC_FB3_C1_BIAS, FOFF_AC_FB3_SCALE_F32},
        {FOFF_AC_FB4_C0_W8, FOFF_AC_FB4_C0_SCALE, FOFF_AC_FB4_C0_BIAS, FOFF_AC_FB4_C1_W8, FOFF_AC_FB4_C1_SCALE, FOFF_AC_FB4_C1_BIAS, FOFF_AC_FB4_SCALE_F32},
    };
    for (int b = 0; b < FRONT_AC_DEPTH; b++) {
        const signed char *c0 = res_copy(FQ(fb_off[b][0]), (size_t)AH * FRONT_AC_FB0_C0_N16, rbuf_kc0);
        const signed char *c1 = res_copy(FQ(fb_off[b][3]), (size_t)AH * FRONT_AC_FB0_C1_N16, rbuf_kc1);
        resblock(6 + b, c0, FF(fb_off[b][1]), FF(fb_off[b][2]), FRONT_AC_FB0_C0_N16,
                 c1, FF(fb_off[b][4]), FF(fb_off[b][5]), FRONT_AC_FB0_C1_N16,
                 *FF(fb_off[b][6]), ax, atmp, AH, T, FRONT_AC_KERNEL);
    }
    {
        const signed char *ow = res_copy(FQ(FOFF_AC_OUT_W8),
                                         (size_t)FSD_CODE_DIM * FRONT_AC_OUT_N16, rbuf_kc1);
        float ccol[64];
        for (int t = 0; t < T; t++) {
            q1x1_col(ow, FF(FOFF_AC_OUT_SCALE), FF(FOFF_AC_OUT_BIAS),
                     FRONT_AC_OUT_N16, ax, T, t, ccol, 1, 0, AH, FSD_CODE_DIM);
            /* sibilant hiss restoration: inject per-channel noise at
             * this voice's /s z sh zh/ frames, before quantization (same
             * point as the fsd_e2e.c reference). No-op unless armed. */
            if (g_sib_beta > 0.0f && frame_sib && frame_sib[t] && g_sib_tea_std)
                for (int i = 0; i < FSD_CODE_DIM; i++)
                    ccol[i] += sib_randn() * g_sib_tea_std[i] * g_sib_beta;
            float m = 0.0f;
            for (int i = 0; i < FSD_CODE_DIM; i++) {
                float v = fabsf(ccol[i]);
                if (v > m) m = v;
            }
            float s = (m > 0.0f) ? m / 127.0f : 1.0f;
            float inv = fast_recip(s);
            signed char *dst = c8 + (size_t)t * Q8_PRE_N16;
            for (int i = 0; i < FSD_CODE_DIM; i++) dst[i] = (signed char)fast_round(ccol[i] * inv);
            for (int i = FSD_CODE_DIM; i < Q8_PRE_N16; i++) dst[i] = 0;
            cscale[t] = s;
        }
    }
    g_arena_top = mark_front; /* release ax/th/rbuf_kc* for the decoder phase */




    /* ---------- decoder: two-pass norm, tiled, fused head, ring OLA ---------- */
    long long t_front = NOW_US();

    /* decoder weight residency (freed flash traffic is the 4.5x lever) */
    /* decoder phase: ax/th/rbuf_kc* are dead; give the head/pw residency
     * its own arena block (arena still has headroom) */
    /* essentials BEFORE opportunistic residency (priority inversion
     * starved the tile buffers on the ESP32-C3's 131 KB arena) */
    float *dwcol = (float *)aa((size_t)FSD_DIM * 4);
    (void)dwcol;
    float *tile_c = (float *)aa((size_t)FSD_DIM * TILE * 4);
    float *tile_h2 = (float *)aa((size_t)FSD_PW_HIDDEN * TILE * 4);
    float *tile_f2 = (float *)aa((size_t)(FSD_FILM_RANK + 2 * FSD_DIM) * TILE * 4);
    signed char *rbuf_ho = (signed char *)aa_try((size_t)FSD_BINS * 3 * Q8_HEAD_OUT_N16 + 16);
    const signed char *ho_res = res_copy(DQ(Q8OFF_HEAD_OUT_W8), (size_t)FSD_BINS * 3 * Q8_HEAD_OUT_N16, rbuf_ho);
    signed char *rbuf_hi = (signed char *)aa_try((size_t)FSD_HEAD_RANK * Q8_HEAD_IN_N16 + 16);
    const signed char *hi_res = res_copy(DQ(Q8OFF_HEAD_IN_W8), (size_t)FSD_HEAD_RANK * Q8_HEAD_IN_N16, rbuf_hi);
    signed char *rbuf_pre = (signed char *)aa_try((size_t)FSD_DIM * Q8_PRE_N16 + 16);
    const signed char *pre_res = res_copy(DQ(Q8OFF_PRE_W8), (size_t)FSD_DIM * Q8_PRE_N16, rbuf_pre);
    signed char *rbuf_pw0 = (signed char *)aa_try((size_t)FSD_PW_HIDDEN * Q8_B0_PW0_N16 + 16);
    signed char *rbuf_pw1 = (signed char *)aa_try((size_t)FSD_DIM * Q8_B0_PW1_N16 + 16);
    signed char *rbuf_fr = (signed char *)aa_try((size_t)FSD_FILM_RANK * Q8_B0_FILMR_N16 + 16);
    signed char *rbuf_fo2 = (signed char *)aa_try((size_t)2 * FSD_DIM * Q8_B0_FILMO_N16 + 16);
    for (int t = 0; t < T; t++)
        q1x1_q8col(pre_res, DF(Q8OFF_PRE_SCALE), DF(Q8OFF_PRE_BIAS), Q8_PRE_N16,
                   c8 + (size_t)t * Q8_PRE_N16, cscale[t], x, T, t, FSD_DIM);

    long q[5][12] = {
        {Q8OFF_B0_DW_W_F32, Q8OFF_B0_DW_B_F32, Q8OFF_B0_NORM_W_F32, Q8OFF_B0_NORM_B_F32, Q8OFF_B0_FILMR_W8, Q8OFF_B0_FILMR_SCALE, Q8OFF_B0_FILMR_BIAS, Q8OFF_B0_FILMO_W8, Q8OFF_B0_FILMO_SCALE, Q8OFF_B0_FILMO_BIAS, 0, Q8OFF_B0_SCALE_F32},
        {Q8OFF_B1_DW_W_F32, Q8OFF_B1_DW_B_F32, Q8OFF_B1_NORM_W_F32, Q8OFF_B1_NORM_B_F32, Q8OFF_B1_FILMR_W8, Q8OFF_B1_FILMR_SCALE, Q8OFF_B1_FILMR_BIAS, Q8OFF_B1_FILMO_W8, Q8OFF_B1_FILMO_SCALE, Q8OFF_B1_FILMO_BIAS, 0, Q8OFF_B1_SCALE_F32},
        {Q8OFF_B2_DW_W_F32, Q8OFF_B2_DW_B_F32, Q8OFF_B2_NORM_W_F32, Q8OFF_B2_NORM_B_F32, Q8OFF_B2_FILMR_W8, Q8OFF_B2_FILMR_SCALE, Q8OFF_B2_FILMR_BIAS, Q8OFF_B2_FILMO_W8, Q8OFF_B2_FILMO_SCALE, Q8OFF_B2_FILMO_BIAS, 0, Q8OFF_B2_SCALE_F32},
        {Q8OFF_B3_DW_W_F32, Q8OFF_B3_DW_B_F32, Q8OFF_B3_NORM_W_F32, Q8OFF_B3_NORM_B_F32, Q8OFF_B3_FILMR_W8, Q8OFF_B3_FILMR_SCALE, Q8OFF_B3_FILMR_BIAS, Q8OFF_B3_FILMO_W8, Q8OFF_B3_FILMO_SCALE, Q8OFF_B3_FILMO_BIAS, 0, Q8OFF_B3_SCALE_F32},
        {Q8OFF_B4_DW_W_F32, Q8OFF_B4_DW_B_F32, Q8OFF_B4_NORM_W_F32, Q8OFF_B4_NORM_B_F32, Q8OFF_B4_FILMR_W8, Q8OFF_B4_FILMR_SCALE, Q8OFF_B4_FILMR_BIAS, Q8OFF_B4_FILMO_W8, Q8OFF_B4_FILMO_SCALE, Q8OFF_B4_FILMO_BIAS, 0, Q8OFF_B4_SCALE_F32},
    };
    long qpw[5][6] = {
        {Q8OFF_B0_PW0_W8, Q8OFF_B0_PW0_SCALE, Q8OFF_B0_PW0_BIAS, Q8OFF_B0_PW1_W8, Q8OFF_B0_PW1_SCALE, Q8OFF_B0_PW1_BIAS},
        {Q8OFF_B1_PW0_W8, Q8OFF_B1_PW0_SCALE, Q8OFF_B1_PW0_BIAS, Q8OFF_B1_PW1_W8, Q8OFF_B1_PW1_SCALE, Q8OFF_B1_PW1_BIAS},
        {Q8OFF_B2_PW0_W8, Q8OFF_B2_PW0_SCALE, Q8OFF_B2_PW0_BIAS, Q8OFF_B2_PW1_W8, Q8OFF_B2_PW1_SCALE, Q8OFF_B2_PW1_BIAS},
        {Q8OFF_B3_PW0_W8, Q8OFF_B3_PW0_SCALE, Q8OFF_B3_PW0_BIAS, Q8OFF_B3_PW1_W8, Q8OFF_B3_PW1_SCALE, Q8OFF_B3_PW1_BIAS},
        {Q8OFF_B4_PW0_W8, Q8OFF_B4_PW0_SCALE, Q8OFF_B4_PW0_BIAS, Q8OFF_B4_PW1_W8, Q8OFF_B4_PW1_SCALE, Q8OFF_B4_PW1_BIAS},
    };


#ifdef SNT_INT_CHAIN
    {
        static const float pmax[5] = {CHAIN_PW0IN_MAX_B0, CHAIN_PW0IN_MAX_B1, CHAIN_PW0IN_MAX_B2, CHAIN_PW0IN_MAX_B3, CHAIN_PW0IN_MAX_B4};
        static const float gmax[5] = {CHAIN_GELUIN_MAX_B0, CHAIN_GELUIN_MAX_B1, CHAIN_GELUIN_MAX_B2, CHAIN_GELUIN_MAX_B3, CHAIN_GELUIN_MAX_B4};
        static const float omax[5] = {CHAIN_GELUOUT_MAX_B0, CHAIN_GELUOUT_MAX_B1, CHAIN_GELUOUT_MAX_B2, CHAIN_GELUOUT_MAX_B3, CHAIN_GELUOUT_MAX_B4};
        if (!g_ic_init) ic_init(qpw, pmax, gmax, omax);
    }
#endif
    for (int bi = 0; bi < FSD_BLOCKS; bi++) {
        const float *dww = DF(q[bi][0]);
        const float *dwb = DF(q[bi][1]);
        const float *nw = DF(q[bi][2]);
        const float *nb2 = DF(q[bi][3]);
        float bscale = *DF(q[bi][11]);
        int half = FSD_DW_KERNEL / 2;
        const signed char *pw0_res = res_copy(DQ(qpw[bi][0]), (size_t)FSD_PW_HIDDEN * Q8_B0_PW0_N16, rbuf_pw0);
        const signed char *pw1_res = res_copy(DQ(qpw[bi][3]), (size_t)FSD_DIM * Q8_B0_PW1_N16, rbuf_pw1);
        const signed char *fr_res = res_copy(DQ(q[bi][4]), (size_t)FSD_FILM_RANK * Q8_B0_FILMR_N16, rbuf_fr);
        const signed char *fo_res = res_copy(DQ(q[bi][7]), (size_t)2 * FSD_DIM * Q8_B0_FILMO_N16, rbuf_fo2);
#ifdef FSD_FROZEN_NORM
        static const float fz_mean[5] = {FROZEN_MEAN_B0, FROZEN_MEAN_B1, FROZEN_MEAN_B2, FROZEN_MEAN_B3, FROZEN_MEAN_B4};
        static const float fz_var[5] = {FROZEN_VAR_B0, FROZEN_VAR_B1, FROZEN_VAR_B2, FROZEN_VAR_B3, FROZEN_VAR_B4};
        float mean = fz_mean[bi];
        float ninv = 1.0f / sqrtf(fz_var[bi] + 1e-5f);
#ifdef SNT_INT_CHAIN
        if (!g_dw_built[bi]) dw_ic_build(bi, dww, dwb, nw, nb2, mean, ninv);
#endif
#else
        /* pass 1: dw conv stats only */
        long long _dw0 = NOW_US();
        double mean = 0.0, sq = 0.0;
        for (int ch = 0; ch < FSD_DIM; ch++) {
            const float *wr = dww + (size_t)ch * FSD_DW_KERNEL;
            const float *xr = x + (size_t)ch * T;
            for (int t = 0; t < T; t++) {
                float a2 = dwb[ch];
                for (int k = 0; k < FSD_DW_KERNEL; k++) {
                    int idx = t + k - half;
                    if (idx >= 0 && idx < T) a2 += wr[k] * xr[idx];
                }
                mean += a2;
                sq += (double)a2 * a2;
            }
        }
        size_t nel = (size_t)FSD_DIM * T;
        mean /= (double)nel;
        double var = sq / (double)nel - mean * mean;
        float ninv = (float)(1.0 / sqrt(var + 1e-5));
        g_prof[3] += NOW_US() - _dw0;
#ifdef SNT_INT_CHAIN
        if (!g_dw_built[bi]) dw_ic_build(bi, dww, dwb, nw, nb2, (float)mean, ninv);
#endif
#endif
        /* pass 2: per tile recompute dw, norm, film, pw0, gelu, pw1-residual.
         * The left dw halo must see PRE-residual x: carry a half-width
         * snapshot of the previous tile's trailing columns. */
        static float halo[8][FSD_DIM];
        static float halo_next[8][FSD_DIM];
        for (int t0 = 0; t0 < T; t0 += TILE) {
            int TL = (t0 + TILE <= T) ? TILE : (T - t0);
            for (int k = 0; k < half; k++) {
                int col2 = t0 + TL - half + k;
                for (int ch = 0; ch < FSD_DIM; ch++)
                    halo_next[k][ch] = (col2 >= 0 && col2 < T) ? x[(size_t)ch * T + col2] : 0.0f;
            }
            long long _fl0 = NOW_US();
            TileCtx tc = {fr_res, fo_res, pw0_res, pw1_res,
                          DF(q[bi][5]), DF(q[bi][6]), DF(q[bi][8]), DF(q[bi][9]),
                          DF(qpw[bi][1]), DF(qpw[bi][2]), DF(qpw[bi][4]), DF(qpw[bi][5]),
                          c8, cscale, tile_f2, tile_c, tile_h2, x,
                          t0, TL, T, bi, bscale,
                          dww, dwb, nw, nb2, mean, ninv,
                          (const float (*)[FSD_DIM])halo};
            long long _td = NOW_US();
            par_run(tile_dw_range, TL, &tc);      /* barrier: x reads done */
            g_prof[4] += NOW_US() - _td;
            par_run(tile_filmpw_range, TL, &tc);  /* then x writes */
            g_prof[0] += NOW_US() - _fl0;
            memcpy(halo, halo_next, sizeof halo);
        }
    }

    /* fused head + ring OLA + on-the-fly corr */
    static float win[NFFT], fre[NFFT], fim[NFFT], ring[NFFT];
    static float hhcol[64], hp1[FSD_BINS * 3];
    for (int i = 0; i < NFFT; i++) {
        double v = sin(M_PI * i / NFFT);
        win[i] = (float)(v * v);
    }
    memset(ring, 0, sizeof ring);
    static float pcm_buf[HOP + NFFT / 2]; /* per-hop emission + flush;
        static: 3 KB would overflow default MCU task stacks (measured:
        stack protection fault on ESP32-C3's 3.5 KB main task) */
    int pcm_n = 0;
    int emitted = 0;
    int aborted = 0;
#ifdef SNT_INT_CHAIN
    static int32_t ph_m[FSD_BINS * 3], ph_b[FSD_BINS * 3];
    {
        const float *hos0 = DF(Q8OFF_HEAD_OUT_SCALE);
        const float *hob0 = DF(Q8OFF_HEAD_OUT_BIAS);
        for (int o = FSD_BINS; o < FSD_BINS * 3; o++) {
            ph_m[o] = (int32_t)(hos0[o] * 1048576.0f);      /* Q20 */
            ph_b[o] = (int32_t)(hob0[o] * 4096.0f);         /* Q12 units */
        }
    }
#endif
    for (int t = 0; t < T; t++) {
        long long _h0 = NOW_US();
        for (int i = 0; i < FSD_DIM; i++) S->gather[i] = x[(size_t)i * T + t];
        float s_act = quant_gather(S, FSD_DIM, Q8_HEAD_IN_N16);
        const float *his = DF(Q8OFF_HEAD_IN_SCALE);
        const float *hib = DF(Q8OFF_HEAD_IN_BIAS);
        matvec_s8(S, S->qbuf, hi_res, FSD_HEAD_RANK, Q8_HEAD_IN_N16);
        for (int o = 0; o < FSD_HEAD_RANK; o++)
            hhcol[o] = gelu(s_act * his[o] * (float)S->acc32[o] + hib[o]);
        float m2 = 0.0f;
        for (int i = 0; i < FSD_HEAD_RANK; i++) {
            float v = fabsf(hhcol[i]);
            if (v > m2) m2 = v;
        }
        float s2 = (m2 > 0.0f) ? m2 / 127.0f : 1.0f;
        float inv2 = fast_recip(s2);
        for (int i = 0; i < FSD_HEAD_RANK; i++) S->qbuf[i] = (signed char)fast_round(hhcol[i] * inv2);
        for (int i = FSD_HEAD_RANK; i < Q8_HEAD_OUT_N16; i++) S->qbuf[i] = 0;
        const float *hos = DF(Q8OFF_HEAD_OUT_SCALE);
        const float *hob = DF(Q8OFF_HEAD_OUT_BIAS);
        snt_matvec_s8(S->qbuf, ho_res, g_acc_head, FSD_BINS * 3, Q8_HEAD_OUT_N16);
#ifdef SNT_INT_CHAIN
        for (int o = 0; o < FSD_BINS; o++)   /* magnitude rows only */
            hp1[o] = s2 * hos[o] * (float)g_acc_head[o] + hob[o];
#else
        for (int o = 0; o < FSD_BINS * 3; o++)
            hp1[o] = s2 * hos[o] * (float)g_acc_head[o] + hob[o];
#endif
        g_prof[1] += NOW_US() - _h0;
        long long _sp0 = NOW_US();
#ifdef SNT_INT_CHAIN
        /* Q12/Q20/Q16 chain: value_u12 = acc*ph_m>>16 * k2>>16 + ph_b */
        int32_t k2 = (int32_t)(s2 * 268435456.0f); /* s2 in Q28... see below */
        /* dimensional check: acc*Q20(hos) >> 16 = Q4 units; * Q28(s2) >> 16
         * = Q16... align to ph_b's Q12: shift net so both are Q12 */
        k2 = (int32_t)(s2 * 16777216.0f);          /* Q24: (Q4 * Q24) >> 16 = Q12 */
        SpecCtx sctx = {hp1, g_acc_head, ph_m, ph_b, k2, fre, fim};
#else
        SpecCtx sctx = {hp1, NULL, NULL, NULL, 0, fre, fim};
#endif
        par_run(spec_range, FSD_BINS, &sctx);
        g_prof[3] += NOW_US() - _sp0;
#ifdef FSD_FULL_FFT
        for (int k = FSD_BINS; k < NFFT; k++) {
            fre[k] = fre[NFFT - k];
            fim[k] = -fim[NFFT - k];
        }
        PROF(5, fft_inv(fre, fim, NFFT));
#else
        /* real IFFT via N/2 complex IFFT: pack hermitian X[0..N/2] into Z[0..N/2-1] */
        {
            long long _f0 = NOW_US();
            static float cwre[NFFT / 2], cwim[NFFT / 2];
            static int cw_init = 0;
            if (!cw_init) {
                for (int k = 0; k < NFFT / 2; k++) {
                    cwre[k] = (float)cos(2.0 * M_PI * k / NFFT);
                    cwim[k] = (float)sin(2.0 * M_PI * k / NFFT);
                }
                cw_init = 1;
            }
#ifdef SNT_INT_FFT
            {
                /* INTEGER hermitian pack: spectrum -> Q29 ints ONCE, then
                 * the A/B/twiddle complex arithmetic runs in int32 (Q30
                 * twiddles), feeding the int FFT with no float middle. */
                static int32_t qre[NFFT / 2 + 1], qim[NFFT / 2 + 1];
                static int32_t icw_re[NFFT / 2], icw_im[NFFT / 2];
                static int icw_init = 0;
                if (!icw_init) {
                    for (int k = 0; k < NFFT / 2; k++) {
                        icw_re[k] = (int32_t)(cos(2.0 * M_PI * k / NFFT) * (double)(1 << 30));
                        icw_im[k] = (int32_t)(sin(2.0 * M_PI * k / NFFT) * (double)(1 << 30));
                    }
                    icw_init = 1;
                }
                float mx = 1e-20f;
                for (int k = 0; k <= NFFT / 2; k++) {
                    float v = fabsf(fre[k]);
                    if (v > mx) mx = v;
                    v = fabsf(fim[k]);
                    if (v > mx) mx = v;
                }
                float qs = (float)(1 << 28) / mx; /* 2 guard bits for pack adds */
                float dq = mx / (float)(1 << 28) * (float)(NFFT / 2);
                for (int k = 0; k <= NFFT / 2; k++) {
                    qre[k] = (int32_t)(fre[k] * qs);
                    qim[k] = (int32_t)(fim[k] * qs);
                }
                static int32_t ire[NFFT / 2], iim[NFFT / 2];
                for (int k = 0; k < NFFT / 2; k++) {
                    int32_t xr = qre[k], xi = qim[k];
                    int32_t mr = qre[NFFT / 2 - k], mi = qim[NFFT / 2 - k];
                    int32_t ar = (xr + mr) >> 1, ai = (xi - mi) >> 1;
                    int32_t dr = xr - mr, di = xi + mi;
                    int32_t br = (int32_t)((qmul(dr, icw_re[k]) - qmul(di, icw_im[k])) >> 1);
                    int32_t bi2 = (int32_t)((qmul(dr, icw_im[k]) + qmul(di, icw_re[k])) >> 1);
                    ire[k] = ar - bi2;
                    iim[k] = ai + br;
                }
                fft_inv_i32(ire, iim, NFFT / 2);
                for (int n = 0; n < NFFT / 2; n++) {
                    fre[2 * n] = (float)ire[n] * dq;
                    fre[2 * n + 1] = (float)iim[n] * dq;
                }
            }
#else
            static float zre[NFFT / 2], zim[NFFT / 2];
            for (int k = 0; k < NFFT / 2; k++) {
                float xr = fre[k], xi = fim[k];
                float mr = fre[NFFT / 2 - k], mi = fim[NFFT / 2 - k];
                float ar = 0.5f * (xr + mr), ai = 0.5f * (xi + mi * -1.0f);
                float dr = xr - mr, di = xi + mi;
                float br = 0.5f * (dr * cwre[k] - di * cwim[k]);
                float bi2 = 0.5f * (dr * cwim[k] + di * cwre[k]);
                zre[k] = ar - bi2;
                zim[k] = ai + br;
            }
            fft_inv(zre, zim, NFFT / 2);
            for (int n = 0; n < NFFT / 2; n++) {
                fre[2 * n] = zre[n];
                fre[2 * n + 1] = zim[n];
            }
#endif
            g_prof[5] += NOW_US() - _f0;
        }
#endif
        int base = t * HOP;
        long long _o0 = NOW_US();
        for (int i = 0; i < NFFT; i++) ring[(base + i) % NFFT] += fre[i] * win[i];
        g_ola_us += NOW_US() - _o0;
        long long _e0 = NOW_US();
        /* steady-state envelope is periodic: env(s) = sum_j win^2[(s%HOP)+j*HOP] */
        static float env_inv[HOP];
        static int env_init = 0;
        if (!env_init) {
            for (int r = 0; r < HOP; r++) {
                float e = 0.0f;
                for (int j = 0; j < NFFT / HOP; j++) {
                    float w2 = win[r + j * HOP];
                    e += w2 * w2;
                }
                env_inv[r] = (e > 1e-11f) ? 1.0f / e : 0.0f;
            }
            env_init = 1;
        }
        /* samples [base, base+HOP) are now final (no future frame reaches them) */
        int emit_from = base, emit_to = base + HOP;
        if (t == T - 1) emit_to = T * HOP + NFFT / 2; /* flush tail */
        int interior_from = NFFT - HOP;               /* all 4 windows present */
        int interior_to = (T - 1) * HOP + HOP;        /* last full-coverage sample */
        for (int s = emit_from; s < emit_to; s++) {
            int out_idx = s - NFFT / 2;
            if (out_idx >= 0 && out_idx < T * HOP) {
                float sample;
                if (s >= interior_from && s < interior_to) {
                    sample = ring[s & (NFFT - 1)] * env_inv[s & (HOP - 1)];
                } else {
                    float e = 0.0f;
                    int j0 = (s - NFFT + HOP) / HOP;
                    if (j0 < 0) j0 = 0;
                    for (int j = j0; j <= s / HOP && j < T; j++) {
                        float w2 = win[s - j * HOP];
                        e += w2 * w2;
                    }
                    sample = (e > 1e-11f) ? ring[s & (NFFT - 1)] / e : 0.0f;
                }
                pcm_buf[pcm_n++] = sample;
                emitted++;
            }
            ring[s & (NFFT - 1)] = 0.0f; /* slot consumed; next use is s+NFFT */
        }
        if (pcm_n && cb) {
            if (cb(pcm_buf, pcm_n, user)) { aborted = 1; }
        }
        pcm_n = 0;
        g_emit_us += NOW_US() - _e0;
        if (aborted) break;
    }
    long long t_done = NOW_US();
    if (stats_out) {
        stats_out->frames = T;
        stats_out->samples = emitted;
        stats_out->elapsed_us = t_done - t_start;
    }
#ifdef SNT_PROF
    printf("SNTPROF front %lld | tiles %lld | head %lld | rbc0 %lld | spec %lld | tdw %lld | fft %lld | rbc1 %lld | film %lld | emit %lld (us)\n",
           (long long)(t_front - t_start), g_prof[0], g_prof[1], g_prof[2],
           g_prof[3], g_prof[4], g_prof[5], g_prof[7], g_prof[6], g_emit_us);
    for (int i = 0; i < 8; i++) g_prof[i] = 0;
    g_ola_us = g_emit_us = 0;
#endif
    (void)t_front;
    return aborted ? 1 : 0;
}
