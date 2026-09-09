/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Ampixa
 *
 * The sanoTTS inference runtime is MIT; see LICENSE.MIT for the exact file
 * list and why the split is sound. The repository as a whole is GPL-3.0,
 * because the grapheme-to-phoneme layer embeds espeak-ng. This file does not.
 */
/* snt_nano.c -- the E12-nano stack (294,642 parameters) in platform-free C99.
 *
 * E13 NOTE: the same file also runs the E13 substitution arms (DyT for every
 * LayerNorm, ReLU for GELU, widths 48/62). Which operators and widths a build
 * uses comes ENTIRELY from the generated nano_q8_meta.h on the include path
 * (NANO_NORM_TYPE / NANO_ACT_TYPE / NANO_DIM ...); an E12 header reproduces
 * the pre-E13 runtime bit-identically. Sizes quoted in the comments below are
 * the E12 numbers.
 *
 * Companion to snt_tts.c (the 567,008-parameter R7 line). What is REUSED from
 * R7, unchanged in semantics:
 *
 *   - the snt_port.h contract (snt_dot_s8 / snt_matvec_s8 / residency / par_run)
 *   - the dynamic activation quantiser: ONE symmetric scale per output column
 *     over the whole gathered in_ch*K receptive field, rounded half-away-from-
 *     zero (quant_gather / qkconv_col / q1x1_col)
 *   - the bump arena, res_copy residency staging and fast-math primitives
 *   - the ENTIRE iSTFT path: real IFFT via an N/2 complex IFFT with a Hermitian
 *     pack, sin^2 window (== torch.hann_window(1024) periodic), ring
 *     overlap-add and the periodic sum-of-w^2 envelope. n_fft 1024 and hop 256
 *     are identical in both lineages, so this is a copy, not a port.
 *
 * What is NEW here, because R7 has no counterpart:
 *
 *   - dynamic per-frame LayerNorm over NANO_DIM channels with a runtime
 *     reciprocal square root. R7 ships FROZEN calibrated GroupNorm statistics
 *     (frozen_norm.h) and never computes a norm at inference.
 *   - range-reduced fixed-point sincos. The head emits mag*(cos phi + i sin phi)
 *     at 2 * 513 * 93.75 ~= 96,000 evaluations per second of audio, over a
 *     measured |phi| <= 29.03 rad, so neither a small-angle approximation nor a
 *     libm call is viable. R7's head emits an unnormalised (re, im) PAIR and
 *     only ever needs one rsqrt per bin.
 *   - LayerScale (per-channel gamma on the block output).
 *   - a deterministic 4-channel Gaussian noise generator reproducing
 *     torch.randn(generator=Generator().manual_seed(sha256(row_id)[:8])) exactly:
 *     ATen's MT19937, its 24-bit uniform, and normal_fill_16's Box-Muller in
 *     the same lane order. Ported from snt_trellis.c, which is validated
 *     against the PyTorch stream.
 *   - a DC blocker as the first-order IIR it actually is,
 *     H(z) = (1 - z^-1)/(1 - 0.9973 z^-1), 2 MAC/sample. The trainer applies a
 *     4096-tap truncated-IIR FIR by FFT convolution; that exists only because
 *     MPS conv1d rejects spatial lengths > 65536. The truncation drops tail
 *     taps summing to 1.6e-5 of the DC gain, so the IIR is the more faithful
 *     of the two, not an approximation of it.
 *
 * MEMORY MODE: WINDOWED. Nothing in the arena scales with the utterance.
 *
 * It used to be whole-utterance: the [DIM, T] activation plane cost 192 B per
 * mel frame, the noise plane another 16, and the acoustic token plane 176 B
 * per TOKEN, so the arena was a line in the length and the line was a ceiling.
 * On an ESP32-S3 running ESPHome, which leaves 122,880 B of contiguous
 * internal SRAM, three seconds of speech fitted, 4.4 s starved the weight
 * staging (correct audio, three times the compute) and 5.3 s returned ERR_OOM.
 *
 * Every plane is now a fixed-width WINDOW. The pipeline runs in chunks of
 * NANO_FRAME_CHUNK output frames; each stage inside a chunk works on a range
 * shrunken by its own receptive field from the stage before it, so the first
 * stage recomputes a skirt of PIPE_RF columns at each end that the neighbouring
 * chunks also compute, and nothing has to survive a chunk boundary inside a
 * plane. See "streaming window geometry" below for the arithmetic.
 *
 * Because the GLOBAL frame index still decides where the zero padding is and
 * which token owns a frame, every column is computed from exactly the inputs
 * the whole-utterance version gave it: the emitted PCM is BYTE-IDENTICAL, and
 * it is identical for every chunk size too (verified from 2 to 2048 frames,
 * on all three lineages, in all four build configurations).
 *
 * The mel-100 interface is still streamed through an EM_RING ring so it never
 * materialises, and the head/iSTFT/OLA still run inside the per-frame loop --
 * materialising the [513, T] complex spectrum would cost 4,104 B/frame. The
 * measured peak is reported in snt_nano_stats.arena_peak; for en_us_e12nano at
 * NANO_FRAME_CHUNK 128 it is 84,208 B for any utterance longer than the
 * window, HOST-measured and DEVICE-confirmed from 192 to 2,905 frames.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "snt_port.h"
#include "snt_nano.h"

/* Generated by tools/export_e12_nano_q8.py. Unlike the R7 line, which keeps a
 * second copy under src/model/, the nano headers live only in their versioned
 * model directory and the build adds it to the include path. */
#include "nano_q8_meta.h"

/* ---- E13 operator substitutions (experiments/e13-substitution-
 * reallocation-sub300k-20260823.json) ------------------------------------
 * The exporter stamps the trained model's operator choices into the generated
 * header. Headers from E12-era exports predate these defines, so they default
 * to the E12 operators here -- an E12 build of this file is bit-identical to
 * the pre-E13 runtime (proven by the PCM byte-identity harness, not asserted).
 *   NANO_NORM_TYPE  0 = dynamic per-frame LayerNorm (E12)
 *                   1 = DyT, y = gamma * tanh(alpha * x) + beta (Zhu et al.
 *                       2025, arXiv:2503.10622; alpha PER-CHANNEL, a
 *                       registered deviation from the paper's scalar)
 *   NANO_ACT_TYPE   0 = exact erf GELU (E12)
 *                   1 = ReLU (replaces the erff call entirely)             */
#ifndef NANO_NORM_TYPE
#define NANO_NORM_TYPE 0
#endif
#ifndef NANO_ACT_TYPE
#define NANO_ACT_TYPE 0
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define HOP NANO_HOP
#define NFFT NANO_N_FFT
#define BINS NANO_BINS
#define DIM NANO_DIM
#define MELS NANO_MELS
#define EK NANO_EMBED_KERNEL          /* mel embedding kernel, 7 */
#define DWK NANO_DW_KERNEL            /* ConvNeXt depthwise kernel, 7 */
#define LENGTH_SCALE 1.0f             /* the nano renders at length_scale 1.0 */
#define DC_POLE 0.9973f
#define MAG_CLIP 100.0f
#define LN_EPS 1e-6f                  /* nn.LayerNorm(eps=1e-6) */

/* gathered receptive fields: embed is the widest at MELS*EK = 700 */
#define GATHER_MAX (((MELS * EK) + 31) & ~31)
#define ACC_MAX (NANO_PW_HIDDEN + 16)

/* ---- streaming window geometry ------------------------------------------
 * Every activation plane in this file used to be allocated for the WHOLE
 * utterance, so the arena grew linearly with the frame count T and the token
 * count. The planes are now WINDOWS: a chunk of output columns is produced,
 * the window slides, and the arena is a constant.
 *
 * A window is not a ring. Each chunk recomputes its own inputs from the last
 * point that is cheap to reproduce (the token embedding for the token planes,
 * the frame projection for the decoder plane), so nothing has to survive from
 * one chunk to the next inside a plane and no modulo addressing is needed:
 * the plane is addressed as x[ch * xw + (t - x0)], where x0 is the global
 * column of the window's column 0. The cost is that the columns within one
 * receptive field of a chunk boundary are computed twice; the benefit is that
 * every column is computed from exactly the same inputs as the whole-utterance
 * version computed it from, which is what makes the output BYTE-IDENTICAL
 * rather than merely close.
 *
 * The width of the recomputed skirt is the pipeline's receptive-field radius.
 * Per stage, "radius" is how far the stage's INPUT columns reach either side
 * of the output column it produces:
 *
 *   resblock  conv0 at column a reads x[a-K/2 .. a+K/2] and conv1 at column t
 *             reads conv0[t-K/2 .. t+K/2], so one block reaches 2*(K/2) = K-1
 *             columns each way.
 *   embed     mel column a is a 1x1 of ax[:,a] (radius 0), then the k=EK
 *             embedding reads mel[t-EK/2 .. t+EK/2]: EK/2 each way.
 *   trunk     the depthwise reads x[t-DWK/2 .. t+DWK/2] pre-residual and the
 *             pointwise is 1x1 at the same column: DWK/2 each way.
 *   head      per column: radius 0.
 *
 * PIPE_RF is their sum over the decoder plane (the frame-projection output
 * through the last trunk block); TOK_RF and DUR_RF are the same sum over the
 * acoustic token blocks and the duration blocks, which live on token planes.
 * For en_us_e12nano / en_us_e13b that is 3*4 + 3 + 4*3 = 27 frames and 3*4 =
 * 12 tokens; for en_us_r227f32, 4*4 + 3 + 5*3 = 34 frames and 4*4 = 16.  */
#define RF_RESBLOCK(K) ((K) - 1)
#define RF_EMBED (EK / 2)
#define RF_TRUNK (DWK / 2)
#define PIPE_RF (NANO_AC_DEPTH * RF_RESBLOCK(NANO_AC_KERNEL) + RF_EMBED \
                 + NANO_BLOCKS * RF_TRUNK)
#define TOK_RF (NANO_AC_TOKEN_DEPTH * RF_RESBLOCK(NANO_AC_KERNEL))
#define DUR_RF (NANO_DUR_DEPTH * RF_RESBLOCK(NANO_DUR_KERNEL))

/* Chunk sizes. Larger chunks recompute proportionally less and cost
 * proportionally more arena; these are the shipped compromise and a build may
 * override either. The frame chunk MUST be even: the head processes frames in
 * pairs and a chunk boundary at an odd frame would split a pair. */
#ifndef NANO_FRAME_CHUNK
#define NANO_FRAME_CHUNK 128
#endif
#ifndef NANO_TOKEN_CHUNK
#define NANO_TOKEN_CHUNK 128
#endif
#if (NANO_FRAME_CHUNK) % 2 != 0
#error "NANO_FRAME_CHUNK must be even: the head emits frames in pairs"
#endif
#if (NANO_FRAME_CHUNK) < 2 || (NANO_TOKEN_CHUNK) < 1
#error "chunk sizes must be positive"
#endif

/* Window bounds checking, host debug builds only. A stage that read outside
 * its window would read a neighbouring allocation inside the same arena --
 * live memory, so neither the allocator nor a sanitiser would notice, and the
 * damage would show up only as changed audio. This turns that into a message.
 * Off in every shipped build; it costs a compare per plane access. */
#ifdef SNT_NANO_WINDOW_GUARD
static long g_win_bad = 0;
static int win_col(int w, int o, int idx) {
    int c = idx - o;
    if (c < 0 || c >= w) {
        if (g_win_bad < 16)
            fprintf(stderr, "snt_nano: WINDOW VIOLATION idx=%d origin=%d width=%d\n",
                    idx, o, w);
        g_win_bad++;
        return 0;
    }
    return c;
}
long snt_nano_window_violations(void) { return g_win_bad; }
#define WCOL(w, o, idx) win_col((w), (o), (idx))
#else
#define WCOL(w, o, idx) ((idx) - (o))
#endif

static int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* First and last token whose frames intersect [flo, fhi). Linear in the token
 * count and called once per chunk; a prefix-sum table would be faster and
 * would cost MAX_TOKENS_RT ints of .bss on a part where .bss is the scarce
 * resource this whole change is about. */
static void tok_span(const int *durs, int n_tokens, int flo, int fhi,
                     int *t_lo, int *t_hi) {
    int f = 0, lo = 0, hi = 0, seen = 0;
    for (int t = 0; t < n_tokens; t++) {
        int e = f + durs[t];
        if (e > flo && f < fhi) {
            if (!seen) { lo = t; seen = 1; }
            hi = t;
        }
        f = e;
        if (f >= fhi) break;
    }
    *t_lo = lo;
    *t_hi = seen ? hi : lo;
}

/* ---------------------------------------------------------------------- */
/* blob accessors                                                          */
/* ---------------------------------------------------------------------- */

/* ---- weight element type --------------------------------------------------
 * The exporter writes every dense weight as [out_ch, n16] rows of ONE element
 * type and records it as NANO_WEIGHT_FORMAT: 0 = int8 with per-output-channel
 * scales (the device format, every MCU gate), 1 = float32 rows with unit
 * scales (--weights f32, the browser build). Offsets, shapes and every macro
 * are the same in both; only the element type changes, so a build must be
 * told which blob it is compiled against -- SNT_NANO_W_F32 -- and refuses the
 * other. Float weights only make sense with float activations, so
 * SNT_NANO_W_F32 implies SNT_NANO_ACT_F32; staging into the arena is also
 * skipped because there is no SIMD int8 path to feed. */
#ifndef NANO_WEIGHT_FORMAT
#define NANO_WEIGHT_FORMAT 0            /* pre-f32 headers are int8 exports */
#endif
#ifdef SNT_NANO_W_F32
#if NANO_WEIGHT_FORMAT != 1
#error "SNT_NANO_W_F32 build against an int8 blob header (NANO_WEIGHT_FORMAT 0)"
#endif
#ifndef SNT_NANO_ACT_F32
#define SNT_NANO_ACT_F32
#endif
typedef float nano_w_t;
#else
#if NANO_WEIGHT_FORMAT != 0
#error "int8 build against a float32 blob header (NANO_WEIGHT_FORMAT 1): define SNT_NANO_W_F32"
#endif
typedef signed char nano_w_t;
#endif

static const unsigned char *g_front, *g_dec;
static const nano_w_t *FQ(long o) { return (const nano_w_t *)(g_front + o); }
static const float *FF(long o) { return (const float *)(g_front + o); }
static const nano_w_t *DQ(long o) { return (const nano_w_t *)(g_dec + o); }
static const float *DF(long o) { return (const float *)(g_dec + o); }

/* ---------------------------------------------------------------------- */
/* fast math (identical policy to snt_tts.c)                               */
/* ---------------------------------------------------------------------- */

/* round-half-away-from-zero; lroundf is a ~50-cycle libm call on LX7 */
static inline int fast_round(float y) {
    return (int)(y + (y >= 0.0f ? 0.5f : -0.5f));
}

#ifdef SNT_NANO_FAST_MATH
static inline float fast_recip(float d) {
    union { float fv; int iv; } u;
    u.fv = d;
    u.iv = 0x7EF311C3 - u.iv;
    float r = u.fv;
    r = r * (2.0f - d * r);
    r = r * (2.0f - d * r);
    return r;
}
/* Two Newton steps, NOT R7's one. R7 uses rsqrt only to normalise a phase unit
 * vector, where 1e-3 is invisible; here it scales every LayerNorm output, and
 * 1e-3 of multiplicative error is the same order as the int8 error the whole
 * exercise is trying to keep. Two steps land near 1e-6. */
static inline float fast_rsqrt(float x) {
    union { float fv; int iv; } u;
    u.fv = x;
    u.iv = 0x5f3759df - (u.iv >> 1);
    float r = u.fv;
    r = r * (1.5f - 0.5f * x * r * r);
    r = r * (1.5f - 0.5f * x * r * r);
    return r;
}
static float g_exp2_lut[65];
static int g_exp2_init = 0;
/* Built eagerly at entry, not lazily inside fast_exp: the spectrum loop now
 * runs on both cores, and a lazily-built table is a data race whose failure
 * mode is a half-filled LUT read as valid. */
static void exp2_init(void) {
    for (int i = 0; i <= 64; i++) g_exp2_lut[i] = powf(2.0f, i / 64.0f);
    g_exp2_init = 1;
}
static inline float fast_exp(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    if (!g_exp2_init) exp2_init();
    float y = x * 1.44269504088896f;
    float fi = floorf(y);
    float f = y - fi;
    int idx = (int)(f * 64.0f);
    float pf = g_exp2_lut[idx] + (g_exp2_lut[idx + 1] - g_exp2_lut[idx]) * (f * 64.0f - idx);
    union { float fv; int iv; } u;
    u.iv = (int)((fi + 127.0f) * 8388608.0f);
    return u.fv * pf;
}

/* EXACT (erf) GELU, not the tanh approximation R7's fast path uses.
 * nn.GELU() defaults to approximate='none', and the two differ by up to 1e-3
 * absolute -- large next to the interface error budget measured for this stack
 * (docs/e12-nano-int8-quantisation.md section 4). A 512-segment table with
 * LINEAR INTERPOLATION over the saturating range costs 2 KB and lands at ~1e-4;
 * R7's uninterpolated table is 0.02, which is not good enough here. */
#define GLUT_N 512
static float g_gelu_lut[GLUT_N + 1];
static float g_silu_lut[GLUT_N + 1];
static int g_alut_init = 0;
static void alut_init(void) {
    for (int i = 0; i <= GLUT_N; i++) {
        float xg = -8.0f + 16.0f * (float)i / (float)GLUT_N;
        g_gelu_lut[i] = 0.5f * xg * (1.0f + erff(xg * 0.70710678118654752f));
        float xs = -9.0f + 18.0f * (float)i / (float)GLUT_N;
        g_silu_lut[i] = xs / (1.0f + expf(-xs));
    }
    g_alut_init = 1;
}
/* unused (not deleted) in a NANO_ACT_TYPE==1 build: ReLU replaces every call */
static inline float __attribute__((unused)) gelu(float x) {
    if (x >= 8.0f) return x;
    if (x <= -8.0f) return 0.0f;
    float p = (x + 8.0f) * (GLUT_N / 16.0f);
    int i = (int)p;
    float f = p - (float)i;
    return g_gelu_lut[i] + (g_gelu_lut[i + 1] - g_gelu_lut[i]) * f;
}
static inline float silu(float x) {
    if (x >= 9.0f) return x;
    if (x <= -9.0f) return 0.0f;
    float p = (x + 9.0f) * (GLUT_N / 18.0f);
    int i = (int)p;
    float f = p - (float)i;
    return g_silu_lut[i] + (g_silu_lut[i + 1] - g_silu_lut[i]) * f;
}
#define EXPF_M(x) fast_exp(x)
#else
static inline float fast_recip(float d) { return 1.0f / d; }
static inline float fast_rsqrt(float x) { return 1.0f / sqrtf(x); }
/* unused (not deleted) in a NANO_ACT_TYPE==1 build: ReLU replaces every call */
static inline float __attribute__((unused)) gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }
static inline float silu(float x) { return x / (1.0f + expf(-x)); }
#define EXPF_M(x) expf(x)
#endif

/* ---------------------------------------------------------------------- */
/* range-reduced fixed-point sincos (NEW -- no R7 counterpart)             */
/* ---------------------------------------------------------------------- */
/* The head emits a raw phase in radians with a measured |phi| <= 29.03. Both
 * cos and sin are wanted for every one of 513 bins of every frame, i.e. ~96k
 * evaluations per second of audio, so cosf/sinf (each with its own libm range
 * reduction) is not affordable on an MCU.
 *
 * Reduction is a UNSIGNED WRAP, not a modulo: phi is scaled to turns, the
 * integer part is dropped with a plain truncation (no floorf, no fmodf), and
 * the fraction is carried as a Q32 phase word whose top two bits are the
 * quadrant. One 257-entry quarter-wave table with linear interpolation then
 * serves both functions; max error is (h^2/8)*max|sin''| with h = pi/512,
 * i.e. ~4.7e-6 absolute -- three orders below the int8 pipeline's noise floor
 * and below the ~1e-7 the float32 phase itself resolves at |phi| ~ 29.
 */
#define SC_Q 256
static float g_sin_q[SC_Q + 1];
static int g_sc_init = 0;
static void sc_init(void) {
    for (int i = 0; i <= SC_Q; i++)
        g_sin_q[i] = (float)sin(M_PI * 0.5 * (double)i / (double)SC_Q);
    g_sc_init = 1;
}

/* sin of a Q32 turn phase (0 == 0 rad, 2^30 == pi/2). */
static inline float sc_sin_q32(uint32_t ph) {
    int quad = (int)(ph >> 30);
    uint32_t within = ph & 0x3FFFFFFFu;          /* 0 .. 2^30-1 inside quadrant */
    if (quad & 1) within = 0x40000000u - within; /* mirror in quadrants 1 and 3 */
    /* index into the quarter table: within * SC_Q / 2^30, with the remainder
     * kept as the interpolation fraction (pure integer, no float divide) */
    uint32_t scaled = within >> (30 - 8 - 16);   /* Q16 index, 8 integer bits */
    int idx = (int)(scaled >> 16);
    float frac = (float)(scaled & 0xFFFFu) * (1.0f / 65536.0f);
    if (idx >= SC_Q) { idx = SC_Q; frac = 0.0f; }
    float v = g_sin_q[idx] + (g_sin_q[idx + 1] - g_sin_q[idx]) * frac;
    return (quad & 2) ? -v : v;
}

void snt_nano_sincos(float phi, float *cos_out, float *sin_out) {
    if (!g_sc_init) sc_init();
    float turns = phi * 0.15915494309189535f;    /* 1 / (2 pi) */
    int ti = (int)turns;                         /* truncation, |turns| <= ~5 */
    float fr = turns - (float)ti;
    if (fr < 0.0f) fr += 1.0f;
    if (fr >= 1.0f) fr = 0.0f;                   /* guard the rounding edge */
    uint32_t ph = (uint32_t)(fr * 4294967296.0f);
    *sin_out = sc_sin_q32(ph);
    *cos_out = sc_sin_q32(ph + 0x40000000u);     /* cos x == sin(x + pi/2) */
}

/* ---------------------------------------------------------------------- */
/* deterministic Gaussian noise (ported verbatim from snt_trellis.c)       */
/* ---------------------------------------------------------------------- */

static const uint32_t SHA_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

static uint32_t rotr32(uint32_t v, int n) { return (v >> n) | (v << (32 - n)); }

static void sha256_block(uint32_t *h, const unsigned char *block) {
    uint32_t w[64], a, b, c, d, e, f, g, hh, t1, t2;
    int i;
    for (i = 0; i < 16; i++)
        w[i] = ((uint32_t)block[4 * i] << 24) | ((uint32_t)block[4 * i + 1] << 16) |
               ((uint32_t)block[4 * i + 2] << 8) | (uint32_t)block[4 * i + 3];
    for (i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    a = h[0]; b = h[1]; c = h[2]; d = h[3];
    e = h[4]; f = h[5]; g = h[6]; hh = h[7];
    for (i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        t1 = hh + S1 + ch + SHA_K[i] + w[i];
        t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

/* Small-input SHA-256 (row ids are short); one or two blocks, no streaming. */
int snt_nano_sha256_seed(const char *text, uint64_t *seed_out) {
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    unsigned char tail[128];
    size_t len = strlen(text), full = len / 64, rem = len % 64, i, tail_len;
    uint64_t bits = (uint64_t)len * 8u, seed = 0;
    for (i = 0; i < full; i++) sha256_block(h, (const unsigned char *)text + i * 64);
    memset(tail, 0, sizeof tail);
    memcpy(tail, text + full * 64, rem);
    tail[rem] = 0x80;
    tail_len = (rem < 56) ? 64u : 128u;
    for (i = 0; i < 8; i++)
        tail[tail_len - 1 - i] = (unsigned char)((bits >> (8 * i)) & 0xffu);
    sha256_block(h, tail);
    if (tail_len == 128) sha256_block(h, tail + 64);
    for (i = 0; i < 2; i++) seed = (seed << 32) | (uint64_t)h[i];
    *seed_out = seed;   /* == int.from_bytes(sha256(text).digest()[:8], "big") */
    return 0;
}

#define MT_N 624
#define MT_M 397
typedef struct { uint32_t state[MT_N]; int left, next; } mt19937;

static uint32_t mt_twist(uint32_t u, uint32_t v) {
    uint32_t mixed = (u & 0x80000000u) | (v & 0x7fffffffu);
    return (mixed >> 1) ^ ((v & 1u) ? 0x9908b0dfu : 0u);
}
static void mt_seed(mt19937 *g, uint64_t seed) {
    /* ATen mt19937_engine::init_with_uint32 -- only the low 32 bits are used */
    g->state[0] = (uint32_t)(seed & 0xffffffffu);
    for (int i = 1; i < MT_N; i++)
        g->state[i] = (uint32_t)(1812433253u * (g->state[i - 1] ^ (g->state[i - 1] >> 30)) + (uint32_t)i);
    g->left = 1;
    g->next = 0;
}
static void mt_next_state(mt19937 *g) {
    uint32_t *s = g->state;
    int i;
    g->left = MT_N;
    g->next = 0;
    for (i = 0; i < MT_N - MT_M; i++) s[i] = s[i + MT_M] ^ mt_twist(s[i], s[i + 1]);
    for (; i < MT_N - 1; i++) s[i] = s[i + MT_M - MT_N] ^ mt_twist(s[i], s[i + 1]);
    s[MT_N - 1] = s[MT_M - 1] ^ mt_twist(s[MT_N - 1], s[0]);
}
static uint32_t mt_random(mt19937 *g) {
    uint32_t y;
    if (--g->left <= 0) mt_next_state(g);
    y = g->state[g->next++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}
/* at::uniform_real_distribution<float>: (raw & (2^24 - 1)) * 2^-24 */
static float mt_uniform(mt19937 *g) {
    return (float)(mt_random(g) & 0xffffffu) * (1.0f / 16777216.0f);
}
/* ATen normal_fill_16 with mean 0, std 1, in float32 throughout
 * (aten/src/ATen/native/DistributionTemplates.h).
 *
 * HOW EXACT THIS IS, MEASURED, not assumed. The integer half of the generator
 * is bit-exact: the MT19937 state, the tempering and the 24-bit uniform
 * reproduce torch.rand's float32 stream exactly for the shipped seed (the
 * fixture carries e2e_uniform.bin so the harness proves it every run). Bits
 * beyond that depend on libm: logf, cosf and sinf are not bit-identical
 * between the C library this links and the one PyTorch was built against.
 * Measured on the shipped fixture, 26% of draws differ by at most 1.9e-6
 * absolute, i.e. a few ulp on a unit-normal value.
 *
 * Doing the angle in double instead (which older ATen did, via
 * c10::pi<double>) moves the worst case to 6.0e-7 but makes MORE draws differ,
 * so float is both the faithful reading of the current source and the cheaper
 * MCU path. Neither variant is bit-exact and claiming otherwise would be a
 * lie the harness would have to be weakened to support. */
static void normal_fill_16(float *d) {
    for (int j = 0; j < 8; j++) {
        float u1 = 1.0f - d[j];
        float u2 = d[j + 8];
        float radius = sqrtf(-2.0f * logf(u1));
        float theta = (float)(2.0 * M_PI) * u2;
        d[j] = radius * cosf(theta);
        d[j + 8] = radius * sinf(theta);
    }
}

/* The uniform stream on its own, so the harness can hold the integer half of
 * the generator to a bit-exact contract independently of libm. */
int snt_nano_uniform_stream(uint64_t seed, int n, float *out) {
    mt19937 g;
    mt_seed(&g, seed);
    for (int i = 0; i < n; i++) out[i] = mt_uniform(&g);
    return 0;
}

int snt_nano_seeded_noise(uint64_t seed, int channels, int frames, float *out) {
    mt19937 g;
    long size = (long)channels * (long)frames, i;
    if (size < 16) return -1; /* torch dispatches sizes < 16 to a scalar path */
    mt_seed(&g, seed);
    for (i = 0; i < size; i++) out[i] = mt_uniform(&g);
    for (i = 0; i + 16 <= size; i += 16) normal_fill_16(out + i);
    if (size % 16 != 0) {
        float tail[16];
        for (int j = 0; j < 16; j++) tail[j] = mt_uniform(&g);
        normal_fill_16(tail);
        for (int j = 0; j < 16; j++) out[size - 16 + j] = tail[j];
    }
    return 0;
}

/* A WINDOW of the same stream, without materialising it.
 *
 * snt_nano_seeded_noise writes [channels, frames] in that memory order, so
 * channel c's frame t is stream element c*frames + t: the four channels are
 * `frames` apart in a strictly sequential MT19937 stream and there is no way
 * to have all four of one frame's values without having driven the generator
 * past the last of them. Keeping only the wanted columns is therefore the
 * only way to stop the noise buffer growing with the utterance.
 *
 * The DRAW ORDER here is identical to the full generator's, element for
 * element, so the values are identical. What changes is that a 16-block is
 * transformed only when it overlaps one of the four wanted ranges: the
 * Box-Muller transform is a pure function of the block, it consumes no draws,
 * and skipping it for a block nobody reads cannot perturb the stream. Cost is
 * `channels * frames` integer draws per call -- microseconds -- against the
 * transcendental cost of a full transform, which this avoids.
 *
 * Reproduced exactly, including the two edges the full generator has:
 *   - elements in the trailing partial 16-block are never transformed;
 *   - if the total size is not a multiple of 16, sixteen FURTHER draws are
 *     taken, transformed, and overwrite the LAST sixteen elements -- which
 *     reaches back into the last complete block.
 * out[ch * stride + (t - lo)] receives channel ch, frame t, for t in [lo,hi).
 */
static int noise_window(uint64_t seed, int channels, int frames,
                        int lo, int hi, float *out, int stride) {
    mt19937 g;
    long size = (long)channels * (long)frames;
    long i;
    float blk[16];
    if (size < 16) return -1;   /* torch dispatches sizes < 16 to a scalar path */
    if (lo < 0 || hi > frames || lo > hi) return -2;
    mt_seed(&g, seed);
    for (i = 0; i + 16 <= size; i += 16) {
        int j;
        int wanted = 0;
        for (j = 0; j < 16; j++) blk[j] = mt_uniform(&g);
        for (j = 0; j < channels; j++) {
            long clo = (long)j * frames + lo, chi = (long)j * frames + hi;
            if (i < chi && i + 16 > clo) { wanted = 1; break; }
        }
        if (!wanted) continue;
        normal_fill_16(blk);
        for (j = 0; j < 16; j++) {
            long abs = i + j;
            int ch = (int)(abs / frames);
            int t = (int)(abs - (long)ch * frames);
            if (t >= lo && t < hi) out[(size_t)ch * stride + (t - lo)] = blk[j];
        }
    }
    {   /* trailing partial block: raw uniforms, never transformed. Always
         * overwritten by the tail below (its range covers this one whenever
         * it exists), but written here so the two generators stay literally
         * the same procedure rather than the same result by argument. */
        int rem = (int)(size - i), j;
        for (j = 0; j < rem; j++) blk[j] = mt_uniform(&g);
        for (j = 0; j < rem; j++) {
            long abs = i + j;
            int ch = (int)(abs / frames);
            int t = (int)(abs - (long)ch * frames);
            if (t >= lo && t < hi) out[(size_t)ch * stride + (t - lo)] = blk[j];
        }
    }
    if (size % 16 != 0) {
        int j;
        for (j = 0; j < 16; j++) blk[j] = mt_uniform(&g);
        normal_fill_16(blk);
        for (j = 0; j < 16; j++) {
            long abs = size - 16 + j;
            int ch = (int)(abs / frames);
            int t = (int)(abs - (long)ch * frames);
            if (t >= lo && t < hi) out[(size_t)ch * stride + (t - lo)] = blk[j];
        }
    }
    return 0;
}

/* ---------------------------------------------------------------------- */
/* arena + int8 column kernels (semantics identical to snt_tts.c)          */
/* ---------------------------------------------------------------------- */

/* ---- activation precision -----------------------------------------------
 * The device build quantises every gathered activation vector to int8 (one
 * dynamic scale per output column) and runs the int8 x int8 kernels behind
 * snt_port.h; that is the path the MCU golden gates hold to 0.98.
 *
 * SNT_NANO_ACT_F32 keeps the int8 WEIGHT blobs and every offset in the
 * generated header exactly as exported, but skips the activation quantiser
 * and accumulates in float: y = sc[o] * sum_i x[i] * w8[o][i] + b[o]. The
 * per-output-channel weight scale is applied after the dot exactly as in the
 * int8 path, so a site's arithmetic is unchanged apart from s_act == 1 and a
 * float accumulator. It exists for hosts with no int8 advantage (the
 * WebAssembly build), where the 2,272,145-parameter release stack measures
 * 0.951 minimum correlation under activation int8 but stays above the gate
 * with float activations. Not a device path: it is never used on the S3/C3.
 */
#ifdef SNT_NANO_ACT_F32
typedef float nano_acc_t;
typedef const float *nano_act_ptr;
#define NANO_ACTBUF(S) ((S)->gather)
static void nano_matvec(const float *act, const nano_w_t *w, float *out,
                        int rows, int len) {
    for (int r = 0; r < rows; r++) {
        const nano_w_t *wr = w + (size_t)r * len;
        float acc = 0.0f;
        for (int i = 0; i < len; i++) acc += act[i] * (float)wr[i];
        out[r] = acc;
    }
}
#else
typedef int32_t nano_acc_t;
typedef const signed char *nano_act_ptr;
#define NANO_ACTBUF(S) ((S)->qbuf)
#define nano_matvec snt_matvec_s8
#endif

typedef struct {
    float gather[GATHER_MAX];
    signed char qbuf[GATHER_MAX] __attribute__((aligned(16)));
    nano_acc_t acc32[ACC_MAX];
} NanoScratch;
static NanoScratch g_scr[2];
static nano_acc_t g_acc_head[NANO_HEAD_OUT];

extern int snt_scratch_id(void);
#define SCR() (&g_scr[snt_scratch_id()])

static unsigned char *g_arena;
static size_t g_arena_cap, g_arena_top, g_arena_peak;
static int g_arena_oom;

static void *aa(size_t n) {
    g_arena_top = (g_arena_top + 15) & ~(size_t)15;
    if (g_arena_top + n > g_arena_cap) { g_arena_oom = 1; return NULL; }
    void *p = g_arena + g_arena_top;
    g_arena_top += n;
    if (g_arena_top > g_arena_peak) g_arena_peak = g_arena_top;
    return p;
}
static void *aa_try(size_t n) {
    size_t top = (g_arena_top + 15) & ~(size_t)15;
    if (top + n > g_arena_cap) return NULL;
    g_arena_top = top + n;
    if (g_arena_top > g_arena_peak) g_arena_peak = g_arena_top;
    return g_arena + top;
}
/* `n` is an ELEMENT count (== bytes for int8 rows). */
static const nano_w_t *res_copy(const nano_w_t *src, size_t n, nano_w_t *dst) {
    if (!dst) return src;
    memcpy(dst, src, n * sizeof(nano_w_t));
    return dst;
}

/* ---- weight residency staging ----------------------------------------
 * snt_weights_resident() is the port's answer to "may the SIMD kernels read
 * this address?". On the ESP32-S3 it is false for flash XIP, and the port
 * then silently takes the scalar C loop -- measured at ~26x the cost per MAC
 * of the PIE path. Until this commit only the duration/acoustic RESBLOCK
 * weights were staged (via res_copy into aa_try buffers); embed, the noise
 * adapter, the mel projection, both trunk pointwise matrices and the head
 * were handed to snt_matvec_s8 straight from flash, so 67.0% of all int8 MACs
 * -- 82.8% of measured runtime -- executed scalar.
 *
 * The staged stages run SEQUENTIALLY, so they do not need to be resident
 * simultaneously; the arena marks below hand the same bytes to embed, then to
 * the trunk pointwise pair, then to the head.
 *
 * Two invariants the kernel contract demands and every caller here honours:
 *   - 16-byte alignment: aa/aa_try align the bump top before returning.
 *   - >= 16 readable bytes past the last row: every request adds RES_PAD,
 *     because the fused MAC+load prefetches one chunk past the end.
 * If the arena cannot spare the bytes, aa_try returns NULL, res_copy hands
 * back the flash pointer and the stage degrades to scalar. Slower, never
 * wrong. */
#define RES_PAD 16
static int g_stage_ok;   /* is the arena itself SIMD-readable? */

static const nano_w_t *stage_w(const nano_w_t *src, size_t n) {
    if (!g_stage_ok) return src;
    return res_copy(src, n, (nano_w_t *)aa_try(n * sizeof(nano_w_t) + RES_PAD));
}
static nano_w_t *stage_buf(size_t n) {
    return g_stage_ok ? (nano_w_t *)aa_try(n * sizeof(nano_w_t) + RES_PAD) : NULL;
}

/* one symmetric scale over the WHOLE gathered vector, round-half-away-from-zero */
static float quant_gather(NanoScratch *S, int n, int pad) {
#ifdef SNT_NANO_ACT_F32
    /* no quantisation: the float gather is the activation; zero the padding
     * columns so the dot over `pad` matches the zero-padded weight rows */
    for (int i = n; i < pad; i++) S->gather[i] = 0.0f;
    return 1.0f;
#endif
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

/* `col` is a LOCAL column of x, i.e. already offset by the window origin. */
static void q1x1_col(const nano_w_t *w8, const float *sc, const float *bi, int pad,
                     const float *x, int xw, int col, float *out, int out_stride,
                     int out_col, int in_ch, int out_ch) {
    NanoScratch *S = SCR();
    for (int i = 0; i < in_ch; i++) S->gather[i] = x[(size_t)i * xw + col];
    float s_act = quant_gather(S, in_ch, pad);
    nano_matvec(NANO_ACTBUF(S), w8, S->acc32, out_ch, pad);
    for (int o = 0; o < out_ch; o++)
        out[(size_t)o * out_stride + out_col] = s_act * sc[o] * (float)S->acc32[o] + bi[o];
}

/* residual conv block: x += scale * conv1(silu(conv0(x))).
 *
 * MEMORY: the intermediate conv0/silu plane used to be a full [ch, T] scratch
 * -- 110,704 B on r06, the single largest allocation after the activations
 * themselves. It is now a RING of RB_TILE + K - 1 columns, because conv1 at
 * column m only ever looks at conv0 columns [m-half, m+half].
 *
 * The trick that makes the ring free of recomputation AND free of the halo
 * bookkeeping R7's tiled trunk needs: conv0 RUNS AHEAD of conv1 by exactly
 * `half` columns. Per tile conv0 produces columns [t0+half, t0+TL+half) and
 * therefore reads x over [t0, t0+TL+2*half) -- entirely columns whose residual
 * has not been added yet, since conv1 has only ever written [0, t0). So no
 * pre-residual snapshot is required at all: the run-ahead IS the halo.
 *
 * Layout note: the ring is COLUMN-major (tmp[a % W][ch]) where the old plane
 * was channel-major. Only the addressing changes; the gathered vectors, their
 * quantisation scale and hence every downstream bit are identical. */
#define RB_TILE 32
#define RB_RING(K) (RB_TILE + (K) - 1)

typedef struct {
    const nano_w_t *c0w, *c1w;
    const float *c0s, *c0b, *c1s, *c1b;
    int c0pad, c1pad, ch, T, K, W;
    int xw, x0;             /* x window: column count, and its global column 0 */
    int a0;                 /* absolute column of unit 0 of the current pass */
    float bscale, *x, *tmp;
} RbCtx;

static void rb_conv0_range(int lo, int hi, void *vc) {
    RbCtx *c = (RbCtx *)vc;
    NanoScratch *S = SCR();
    int half = c->K / 2;
    for (int u = lo; u < hi; u++) {
        int a = c->a0 + u;
        for (int i = 0; i < c->ch; i++)
            for (int k = 0; k < c->K; k++) {
                int idx = a + k - half;
                S->gather[i * c->K + k] =
                    (idx >= 0 && idx < c->T)
                        ? c->x[(size_t)i * c->xw + WCOL(c->xw, c->x0, idx)] : 0.0f;
            }
        float s_act = quant_gather(S, c->ch * c->K, c->c0pad);
        nano_matvec(NANO_ACTBUF(S), c->c0w, S->acc32, c->ch, c->c0pad);
        float *dst = c->tmp + (size_t)(a % c->W) * c->ch;
        /* store, then activate in place: identical rounding to the old
         * write-plane-then-silu-pass, which is what the golden hashes check */
        for (int o = 0; o < c->ch; o++)
            dst[o] = s_act * c->c0s[o] * (float)S->acc32[o] + c->c0b[o];
        for (int o = 0; o < c->ch; o++) dst[o] = silu(dst[o]);
    }
}

static void rb_conv1_range(int lo, int hi, void *vc) {
    RbCtx *c = (RbCtx *)vc;
    NanoScratch *S = SCR();
    int half = c->K / 2;
    for (int u = lo; u < hi; u++) {
        int t = c->a0 + u;
        for (int i = 0; i < c->ch; i++)
            for (int k = 0; k < c->K; k++) {
                int idx = t + k - half;
                S->gather[i * c->K + k] = (idx >= 0 && idx < c->T)
                    ? c->tmp[(size_t)(idx % c->W) * c->ch + i] : 0.0f;
            }
        float s_act = quant_gather(S, c->ch * c->K, c->c1pad);
        nano_matvec(NANO_ACTBUF(S), c->c1w, S->acc32, c->ch, c->c1pad);
        for (int o = 0; o < c->ch; o++)
            c->x[(size_t)o * c->xw + WCOL(c->xw, c->x0, t)] +=
                c->bscale * (s_act * c->c1s[o] * (float)S->acc32[o] + c->c1b[o]);
    }
}

/* Writes output columns [lo, hi) of x, reading x over [lo-(K-1), hi+(K-1))
 * -- which is why the caller's window must be that much wider than the range
 * it asks for. `T` is the GLOBAL column count and still decides where the
 * zero padding starts; `xw`/`x0` describe the window x actually occupies.
 *
 * conv0 runs ahead of conv1 by `half` columns exactly as before; the only
 * change is that the run-ahead starts at lo-half instead of at 0, so the
 * pre-residual columns a chunk boundary needs are produced rather than
 * assumed. The ring is RB_TILE + K - 1 wide, which is still exactly the span
 * conv1 reaches over one tile. */
static void resblock(const nano_w_t *c0w, const float *c0s, const float *c0b, int c0pad,
                     const nano_w_t *c1w, const float *c1s, const float *c1b, int c1pad,
                     float bscale, float *x, float *tmp, int ch, int T, int K,
                     int xw, int x0, int lo, int hi) {
    int half = K / 2;
    RbCtx c = {c0w, c1w, c0s, c0b, c1s, c1b, c0pad, c1pad, ch, T, K,
               RB_RING(K), xw, x0, 0, bscale, x, tmp};
    int produced = lo - half;         /* conv0 columns [lo-half, produced) live */
    if (produced < 0) produced = 0;   /* columns < 0 are zero by the gather */
    for (int t0 = lo; t0 < hi; t0 += RB_TILE) {
        int TL = (t0 + RB_TILE <= hi) ? RB_TILE : (hi - t0);
        int need = t0 + TL + half;
        if (need > T) need = T;       /* columns >= T are zero by the gather */
        if (need > produced) {
            c.a0 = produced;
            snt_par_run(rb_conv0_range, need - produced, &c);
            produced = need;
        }
        c.a0 = t0;
        snt_par_run(rb_conv1_range, TL, &c);   /* barrier: conv0 reads are done */
    }
}

/* ---------------------------------------------------------------------- */
#ifdef SNT_NANO_PROF
enum { GL_LN = 0, GL_N };
extern int g_nano_prof_on;
extern int64_t g_glue[GL_N][2];
extern int64_t g_glue_n[GL_N][2];
#endif
/* LayerNorm over DIM channels of one frame (NEW)                          */
/* ---------------------------------------------------------------------- */
/* nn.LayerNorm(dim, eps=1e-6) applied to x.transpose(1,2), i.e. per (batch,
 * frame) over channels, with the BIASED variance torch uses. R7 has nothing
 * like this: its GroupNorm statistics are calibrated once and baked into
 * frozen_norm.h, so the runtime never divides by a live standard deviation. */
static void __attribute__((unused)) layernorm_col(float *dst, const float *src,
                                                  const float *gamma,
                                                  const float *beta, int C) {
#ifdef SNT_NANO_PROF
    int64_t _lt0 = g_nano_prof_on ? snt_now_us() : 0;
#endif
    float sum = 0.0f;
    for (int c = 0; c < C; c++) sum += src[c];
    float mean = sum * (1.0f / (float)C);
    float var = 0.0f;
    for (int c = 0; c < C; c++) {
        float d = src[c] - mean;
        var += d * d;
    }
    var *= (1.0f / (float)C);
    float inv = fast_rsqrt(var + LN_EPS);
    for (int c = 0; c < C; c++) dst[c] = (src[c] - mean) * inv * gamma[c] + beta[c];
#ifdef SNT_NANO_PROF
    if (g_nano_prof_on) {
        int _sid = snt_scratch_id();   /* per core: the trunk tile runs on both */
        g_glue[GL_LN][_sid] += snt_now_us() - _lt0;
        g_glue_n[GL_LN][_sid]++;
    }
#endif
}

#if NANO_NORM_TYPE == 1
/* DyT (E13): y = gamma * tanh(alpha * x) + beta, per channel. No mean, no
 * variance, no rsqrt -- the whole point of the substitution is that the only
 * transcendental left is tanh, and it is per-element rather than per-column.
 *
 * tanhf here is the libm call: correct on the host and affordable on the
 * ESP32-S3 (hardware FPU; one tanhf replaces the LayerNorm's two reduction
 * passes plus fast_rsqrt). The ESP32-C3 has NO FPU, so every tanhf is a
 * soft-float libm call there -- the C3 path wants a range-reduced LUT like
 * the gelu/silu tables above. NOT built this task: the C3 port ships when a
 * final E13 model does, and the LUT must be validated against the golden
 * fixture then, not guessed at now. */
static void dyt_col(float *dst, const float *src, const float *alpha,
                    const float *gamma, const float *beta, int C) {
#ifdef SNT_NANO_PROF
    int64_t _lt0 = g_nano_prof_on ? snt_now_us() : 0;
#endif
    for (int c = 0; c < C; c++)
        dst[c] = gamma[c] * tanhf(alpha[c] * src[c]) + beta[c];
#ifdef SNT_NANO_PROF
    if (g_nano_prof_on) {
        int _sid = snt_scratch_id();
        g_glue[GL_LN][_sid] += snt_now_us() - _lt0;
        g_glue_n[GL_LN][_sid]++;
    }
#endif
}
/* Every norm call site routes through NORM_COL so the two operators cannot
 * drift apart in call structure. In the LayerNorm build the alpha argument is
 * dropped by the preprocessor -- no expression is evaluated for it. */
#define NORM_COL(dst, src, alpha, g, b, C) dyt_col(dst, src, alpha, g, b, C)
#else
#define NORM_COL(dst, src, alpha, g, b, C) layernorm_col(dst, src, g, b, C)
#endif

#if NANO_ACT_TYPE == 1
/* ReLU (E13): replaces the erff GELU call entirely -- removing erf from the
 * inner pointwise loop was SlimTTS's whole argument for the substitution. */
static inline float act_fn(float x) { return x > 0.0f ? x : 0.0f; }
#define ACT(x) act_fn(x)
#else
/* Expands to the gelu() of the selected math config (exact erff, or the
 * interpolated LUT under SNT_NANO_FAST_MATH); the SNT_NANO_ABLATE==4 stub
 * still applies because expansion happens at the use site. */
#define ACT(x) gelu(x)
#endif

/* ---------------------------------------------------------------------- */
/* mel projection + embed/noise/stem-LN, tiled across cores                 */
/* ---------------------------------------------------------------------- */
/* Same shape as the residual block: the mel projection RUNS AHEAD of embed by
 * `half` columns, so per tile each pass is a flat, column-independent range.
 * The mel ring is column-major and RB-style: EM_TILE + EK - 1 columns wide,
 * which is all embed's k=7 window can reach back over a tile.
 *
 * The stem LayerNorm is folded into the embed pass. It was a separate serial
 * sweep over x, and it depends on nothing but the column it normalises. */
#define EM_TILE 32
#define EM_RING (EM_TILE + EK - 1)

typedef struct {
    const nano_w_t *ow, *ew, *nw;
    const float *os, *ob, *es, *eb, *ns, *nb, *gw, *gb;
    const float *ga;                  /* stem DyT alpha; NULL under LayerNorm */
    float *ax, *x, *noise, *mel;
    int T, a0;
    int xw, x0;               /* ax/x window: column count and global column 0 */
    int noisew, noise0;       /* noise window: the same, over its own range    */
} EmCtx;

/* pass 1: mel column a = ac_out(ax[:, a]), a 1x1 conv. Reads ax, writes mel. */
static void em_mel_range(int lo, int hi, void *vc) {
    EmCtx *c = (EmCtx *)vc;
    for (int u = lo; u < hi; u++) {
        int a = c->a0 + u;
        q1x1_col(c->ow, c->os, c->ob, NANO_AC_OUT_N16, c->ax, c->xw,
                 WCOL(c->xw, c->x0, a),
                 c->mel + (size_t)(a % EM_RING) * MELS, 1, 0, NANO_AC_HIDDEN, MELS);
    }
}

/* pass 2: embed(mel) + noise_adapter(noise) + stem LayerNorm. Writes x.
 * NOTE the ordering contract with the shared ax/x plane: this pass writes
 * column t while pass 1 has already read ax as far as t+half, and pass 1 of
 * the NEXT tile starts at t0+TL+half -- strictly ahead of anything written
 * here. That is what makes the aliasing safe under tiling too. */
static void em_col_range(int lo, int hi, void *vc) {
    EmCtx *c = (EmCtx *)vc;
    NanoScratch *S = SCR();
    const int half = EK / 2;
    for (int u = lo; u < hi; u++) {
        int t = c->a0 + u;
        for (int k = 0; k < EK; k++) {
            int idx = t + k - half;
            if (idx >= 0 && idx < c->T) {
                const float *slot = c->mel + (size_t)(idx % EM_RING) * MELS;
                for (int i = 0; i < MELS; i++) S->gather[i * EK + k] = slot[i];
            } else {
                for (int i = 0; i < MELS; i++) S->gather[i * EK + k] = 0.0f;
            }
        }
        float s_act = quant_gather(S, MELS * EK, NANO_EMBED_N16);
        nano_matvec(NANO_ACTBUF(S), c->ew, S->acc32, DIM, NANO_EMBED_N16);
        int xc = WCOL(c->xw, c->x0, t);
        for (int o = 0; o < DIM; o++)
            c->x[(size_t)o * c->xw + xc] = s_act * c->es[o] * (float)S->acc32[o] + c->eb[o];

        for (int i = 0; i < NANO_NOISE_CH; i++)
            for (int k = 0; k < EK; k++) {
                int idx = t + k - half;
                S->gather[i * EK + k] = (idx >= 0 && idx < c->T)
                    ? c->noise[(size_t)i * c->noisew
                               + WCOL(c->noisew, c->noise0, idx)] : 0.0f;
            }
        float s_n = quant_gather(S, NANO_NOISE_CH * EK, NANO_NOISE_N16);
        nano_matvec(NANO_ACTBUF(S), c->nw, S->acc32, DIM, NANO_NOISE_N16);
        for (int o = 0; o < DIM; o++)
            c->x[(size_t)o * c->xw + xc] += s_n * c->ns[o] * (float)S->acc32[o] + c->nb[o];

        {   /* stem norm (LayerNorm or DyT), folded in */
            float col[DIM], outc[DIM];
            for (int ch = 0; ch < DIM; ch++) col[ch] = c->x[(size_t)ch * c->xw + xc];
            NORM_COL(outc, col, c->ga, c->gw, c->gb, DIM);
            for (int ch = 0; ch < DIM; ch++) c->x[(size_t)ch * c->xw + xc] = outc[ch];
        }
    }
}

/* ---------------------------------------------------------------------- */
/* ConvNeXt trunk: tiled so the two cores can share a block                 */
/* ---------------------------------------------------------------------- */
/* The trunk was the one stage that looked inherently sequential: the block
 * writes its residual back into x, and the depthwise window of column t reads
 * columns t-3..t+3, so once column t-1 has been updated the window would see
 * post-residual data. The old code solved that per column with a rolling
 * `halo` of pre-residual columns, which forced a strictly serial t loop.
 *
 * R7 already solved the same problem at TILE granularity (snt_tts.c,
 * tile_dw_range): snapshot the tile's trailing `half` PRE-residual columns
 * before the residual pass writes, run the depthwise for the whole tile in
 * parallel, barrier, then run the pointwise/residual pass in parallel. Inside
 * a tile nothing has been written yet, so every column sees pre-residual x
 * exactly as the serial version did -- the tiled result is bit-identical, not
 * merely equivalent.
 *
 * Two par_run calls per tile, with the barrier between them being the whole
 * correctness argument: pass 1 only reads x, pass 2 only writes it. */
#define TR_TILE 32

typedef struct {
    const float *dww, *dwb, *nwt, *nbt;
    const float *nat;                 /* block DyT alpha; NULL under LayerNorm */
    const nano_w_t *pw0, *pw1;
    const float *pw0s, *pw0b, *pw1s, *pw1b, *gamma;
    float *x, *tc;                    /* tc: [TR_TILE][DIM], dw+LN of the tile */
    const float (*halo)[DIM];         /* pre-residual columns t0-1 .. t0-half */
    int t0, TL, T;
    int xw, x0;                       /* x window: columns, and global column 0 */
} TrCtx;

/* pass 1: depthwise + LayerNorm for a column range. Reads x only. */
static void tr_dw_range(int lo, int hi, void *vc) {
    TrCtx *c = (TrCtx *)vc;
    const int half = DWK / 2;
    for (int u = lo; u < hi; u++) {
        int t = c->t0 + u;
        float acol[DIM];
        for (int ch = 0; ch < DIM; ch++) {
            const float *wr = c->dww + (size_t)ch * DWK;
            const float *xr = c->x + (size_t)ch * c->xw;
            float a = c->dwb[ch];
            for (int k = 0; k < DWK; k++) {
                int idx = t + k - half;
                if (idx < 0 || idx >= c->T) continue;
                /* halo[j] is the PRE-residual column t0-1-j, j < half */
                float v = (idx < c->t0) ? c->halo[c->t0 - 1 - idx][ch]
                                        : xr[WCOL(c->xw, c->x0, idx)];
                a += wr[k] * v;
            }
            acol[ch] = a;
        }
        NORM_COL(c->tc + (size_t)u * DIM, acol, c->nat, c->nwt, c->nbt, DIM);
    }
}

/* pass 2: pw0 -> GELU -> pw1 -> LayerScale -> residual. Writes x only. */
static void tr_pw_range(int lo, int hi, void *vc) {
    TrCtx *c = (TrCtx *)vc;
    NanoScratch *S = SCR();
    for (int u = lo; u < hi; u++) {
        int t = c->t0 + u;
        const float *ncol = c->tc + (size_t)u * DIM;
        float hid[NANO_PW_HIDDEN];
        for (int i = 0; i < DIM; i++) S->gather[i] = ncol[i];
        float s0 = quant_gather(S, DIM, NANO_B0_PW0_N16);
        nano_matvec(NANO_ACTBUF(S), c->pw0, S->acc32, NANO_PW_HIDDEN, NANO_B0_PW0_N16);
        for (int o = 0; o < NANO_PW_HIDDEN; o++)
            hid[o] = ACT(s0 * c->pw0s[o] * (float)S->acc32[o] + c->pw0b[o]);
        for (int i = 0; i < NANO_PW_HIDDEN; i++) S->gather[i] = hid[i];
        float s1 = quant_gather(S, NANO_PW_HIDDEN, NANO_B0_PW1_N16);
        nano_matvec(NANO_ACTBUF(S), c->pw1, S->acc32, DIM, NANO_B0_PW1_N16);
        int xc = WCOL(c->xw, c->x0, t);
        for (int ch = 0; ch < DIM; ch++)
            c->x[(size_t)ch * c->xw + xc] += c->gamma[ch] *
                (s1 * c->pw1s[ch] * (float)S->acc32[ch] + c->pw1b[ch]);
    }
}

/* ---------------------------------------------------------------------- */
/* head int8 + spectrum, fused and split across cores                       */
/* ---------------------------------------------------------------------- */
/* The head emits 1026 rows per frame: 513 log-magnitudes then 513 phases.
 * Both the matvec and the exp/sincos that consume it are per-BIN independent,
 * so one par_run over the 513 bins covers what the profile called "head
 * LN+int8" (5.2%) and "spec exp+sincos" (25.1%) at once.
 *
 * Each half-range issues TWO row-streaming matvec calls (its magnitude rows
 * and its phase rows) rather than 2*n one-row dots, so the PIE kernel keeps
 * its continuous weight stream. Row offsets are multiples of NANO_HEAD_N16
 * (ceil16(NANO_DIM): 48 at width 48, 64 at width 62), itself a multiple of
 * 16, so every sub-block is still 16-byte aligned as the kernel contract
 * demands.
 *
 * The activation qbuf belongs to whichever core quantised it; both cores only
 * READ it, so it is passed in the context instead of via SCR(). */
typedef struct {
    nano_act_ptr qbuf;                /* int8 qbuf, or the float gather under SNT_NANO_ACT_F32 */
    const nano_w_t *hw;
    const float *hs, *hb;
    nano_acc_t *acc;
    float s_act, *fre, *fim;
} HeadCtx;

static void head_spec_range(int lo, int hi, void *vc) {
    HeadCtx *c = (HeadCtx *)vc;
    int n = hi - lo;
    nano_matvec(c->qbuf, c->hw + (size_t)lo * NANO_HEAD_N16,
                  c->acc + lo, n, NANO_HEAD_N16);
    nano_matvec(c->qbuf, c->hw + (size_t)(BINS + lo) * NANO_HEAD_N16,
                  c->acc + BINS + lo, n, NANO_HEAD_N16);
    for (int k = lo; k < hi; k++) {
        float m;
        if (k == 0 || k == BINS - 1) {
            m = 0.0f;                      /* DC and Nyquist stay zeroed */
        } else {
            float lm = c->s_act * c->hs[k] * (float)c->acc[k] + c->hb[k];
            if (lm > 60.0f) lm = 60.0f;    /* overflow guard only */
            m = EXPF_M(lm);
            if (m > MAG_CLIP) m = MAG_CLIP;
        }
        float phi = c->s_act * c->hs[BINS + k] * (float)c->acc[BINS + k]
                  + c->hb[BINS + k];
        float cs, sn;
        snt_nano_sincos(phi, &cs, &sn);
        c->fre[k] = m * cs;
        c->fim[k] = m * sn;
    }
}

/* ---------------------------------------------------------------------- */
/* iSTFT (reused unchanged from snt_tts.c: same n_fft, same hop, same win)  */
/* ---------------------------------------------------------------------- */

/* Per-stage twiddle bases e^{i*2pi/len}. These were recomputed on every call
 * -- 2 * log2(512) = 18 DOUBLE-precision trig calls per frame on a part with
 * no hardware double, i.e. ~7,500 soft-double calls per utterance for 9
 * distinct constants. Same float values, computed once. */
static float g_fftw_re[16], g_fftw_im[16];
static int g_fftw_init = 0;
static void fft_tw_init(void) {
    for (int s = 1; s < 16; s++) {
        double ang = 2.0 * M_PI / (double)(1 << s);
        g_fftw_re[s] = (float)cos(ang);
        g_fftw_im[s] = (float)sin(ang);
    }
    g_fftw_init = 1;
}

/* Two frames' inverse FFTs, one per core. The iSTFT is the last stage with no
 * cross-bin parallelism left -- its butterflies are sequential in `len` -- but
 * consecutive FRAMES are independent up to the overlap-add, which is applied
 * afterwards in order. So the unit of parallelism here is the frame, not the
 * bin: pack two frames, run both transforms under one par_run, then overlap-add
 * frame t and then frame t+1 exactly as the serial loop did. */
typedef struct { float *zre[2], *zim[2]; int n; } FftCtx;
static void fft_inv(float *re, float *im, int n);
static void fft_pair_range(int lo, int hi, void *vc) {
    FftCtx *c = (FftCtx *)vc;
    for (int i = lo; i < hi; i++) fft_inv(c->zre[i], c->zim[i], c->n);
}

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
    for (int len = 2, s = 1; len <= n; len <<= 1, s++) {
        float wr = g_fftw_re[s], wi = g_fftw_im[s];
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

/* ---------------------------------------------------------------------- */
/* main entry point                                                        */
/* ---------------------------------------------------------------------- */

#define ERR_OOM (-2)
#define ERR_TOKENS (-3)
#define ERR_ABORT (1)

/* ---- float-glue ablation (host analysis only) -------------------------
 * SNT_NANO_ABLATE replaces one glue primitive with a cheap, input-dependent
 * stub and leaves everything else -- memory layout, loop structure, branch
 * behaviour -- alone. The difference in total runtime attributes that
 * primitive IN SITU, which an isolated microbenchmark cannot do on an
 * out-of-order host: measured serially, exp and sincos each look ~8 ns, but
 * the real spectrum loop is 513 independent bins the CPU overlaps.
 *
 * The stubs depend on their argument so nothing folds; the AUDIO IS WRONG and
 * these builds must never be gated or shipped.
 *   1 = exp, 2 = sincos, 3 = LayerNorm, 4 = GELU, 5 = SiLU
 *
 * CAVEAT that cost me a wrong number once: an ablation is only clean if the
 * primitive is DOWNSTREAM of every data-dependent-cost primitive. Ablation 3
 * is not -- LayerNorm feeds GELU, whose erff is data-dependent, so stubbing it
 * changes GELU's cost as well as its own and over-attributes. LayerNorm is
 * therefore measured by a dedicated PROF timer (PR_LN) instead, and 3 is kept
 * only to show the discrepancy. 1, 2 and 4 sit downstream of all of it.
 */
#ifdef SNT_NANO_ABLATE
#if SNT_NANO_ABLATE == 1
#undef EXPF_M
#define EXPF_M(x) (1.0f + 0.001f * (x))
#elif SNT_NANO_ABLATE == 2
#define snt_nano_sincos(phi, c, s) do { *(c) = 0.5f * (phi); *(s) = -0.25f * (phi); } while (0)
#elif SNT_NANO_ABLATE == 4
#define gelu(x) (0.5f * (x) + 0.05f)
#elif SNT_NANO_ABLATE == 5
#define silu(x) (0.5f * (x) + 0.05f)
#elif SNT_NANO_ABLATE == 3
#define layernorm_col(dst, src, g, b, C) \
    do { for (int _c = 0; _c < (C); _c++) (dst)[_c] = (src)[_c] * (g)[_c] + (b)[_c]; } while (0)
#else
#error "SNT_NANO_ABLATE must be 1, 2, 3, 4 or 5"
#endif
#endif

/* ---- stage profiling (host analysis only) -----------------------------
 * Mirrors R7's SNT_PROF. Off by default; the timers themselves cost ~2
 * snt_now_us() calls per stage per frame, so the PROF build's total is a few
 * percent above the shipped build's and only the SHARE of each stage should be
 * read from it. Absolute RTF must come from a non-PROF build. */
#ifdef SNT_NANO_PROF
/* RUNTIME switch (device build). Compile-time gating would mean flashing two
 * different binaries for the RTF and the breakdown; a global test costs one
 * load+branch per stage boundary (~13k branches per utterance, unmeasurable)
 * and guarantees both numbers come from identical machine code. */
enum { PR_DUR = 0, PR_AC, PR_EMBED, PR_TRUNK_DW, PR_TRUNK_PW, PR_HEAD,
       PR_SPEC, PR_FFT, PR_OLA, PR_N };
/* Float-glue counters. These are NOT stages: they are "of which" slices that
 * already live inside the stages above, so they must never be summed with
 * them. GL_LN sits inside EMBED/TRUNK_DW/HEAD.
 *
 * The exp/sincos/complex-mul split that used to live here is gone: those now
 * execute inside a par_run over the 513 bins, and a wall-clock timer around a
 * two-core range does not measure one core's work. The last serial numbers
 * for them are in the pre-parallel log (exp 557 ns/call, sincos 549 ns/call,
 * 11.9% of runtime each). */
int g_nano_prof_on = 0;
static int64_t g_prof[PR_N];
int64_t g_glue[GL_N][2];
int64_t g_glue_n[GL_N][2];
static const char *g_prof_name[PR_N] = {
    "duration", "acoustic", "embed+stemLN", "trunk dw+LN", "trunk pw int8",
    "head LN+quant", "head int8+spec par", "ifft", "ola+emit+dc"};
static const char *g_glue_name[GL_N] = {
    NANO_NORM_TYPE == 1 ? "DyT calls" : "LayerNorm calls"};
#define PROF_T0() int64_t _p0 = g_nano_prof_on ? snt_now_us() : 0
#define PROF_ADD(i) do { if (g_nano_prof_on) g_prof[i] += snt_now_us() - _p0; } while (0)
#define PROF_T0N(n) int64_t _p##n = g_nano_prof_on ? snt_now_us() : 0
#define PROF_ADDN(i, n) do { if (g_nano_prof_on) g_prof[i] += snt_now_us() - _p##n; } while (0)
void snt_nano_prof_set(int on);
void snt_nano_prof_report(int frames, double audio_seconds);
void snt_nano_prof_set(int on) { g_nano_prof_on = on; }
void snt_nano_prof_reset(void) {
    for (int i = 0; i < PR_N; i++) g_prof[i] = 0;
    for (int i = 0; i < GL_N; i++)
        for (int c = 0; c < 2; c++) { g_glue[i][c] = 0; g_glue_n[i][c] = 0; }
}
void snt_nano_prof_report(int frames, double audio_seconds) {
    int64_t total = 0;
    for (int i = 0; i < PR_N; i++) total += g_prof[i];
    printf("  %-18s %10s %8s %12s\n", "stage", "us", "share", "us/s-audio");
    for (int i = 0; i < PR_N; i++)
        printf("  %-18s %10lld %7.1f%% %12.0f\n", g_prof_name[i],
               (long long)g_prof[i], total ? 100.0 * g_prof[i] / total : 0.0,
               audio_seconds > 0 ? g_prof[i] / audio_seconds : 0.0);
    printf("  %-18s %10lld %7.1f%% %12.0f   (frames %d)\n", "TOTAL(profiled)",
           (long long)total, 100.0, audio_seconds > 0 ? total / audio_seconds : 0.0,
           frames);
    int64_t glue = g_glue[GL_LN][0] + g_glue[GL_LN][1];
    printf("  -- FLOAT GLUE (already counted inside the stages above) --\n");
    printf("  %-18s %10s %8s %12s %12s %10s\n",
           "primitive", "us", "share", "us/s-audio", "calls", "ns/call");
    for (int i = 0; i < GL_N; i++) {
        int64_t gu = g_glue[i][0] + g_glue[i][1];
        int64_t gn = g_glue_n[i][0] + g_glue_n[i][1];
        printf("  %-18s %10lld %7.1f%% %12.0f %12lld %10.1f  [CPU-us summed over cores]\n",
               g_glue_name[i], (long long)gu, total ? 100.0 * gu / total : 0.0,
               audio_seconds > 0 ? gu / audio_seconds : 0.0, (long long)gn,
               gn ? 1000.0 * (double)gu / (double)gn : 0.0);
    }
    printf("  %-18s %10lld %7.1f%% %12.0f   (LayerNorm only; exp/sincos are\n"
           "                                                  inside the parallel bin range)\n",
           "GLUE TOTAL", (long long)glue, total ? 100.0 * glue / total : 0.0,
           audio_seconds > 0 ? glue / audio_seconds : 0.0);
}
#else
int g_nano_prof_on = 0;
#define PROF_T0() do {} while (0)
#define PROF_ADD(i) do {} while (0)
#define PROF_T0N(n) do {} while (0)
#define PROF_ADDN(i, n) do {} while (0)
#endif

#define MAX_TOKENS_RT 1024

/* Per-block region tables. The exporter names every block's regions
 * <PREFIX>B<i>_..., and the block count comes from the generated header
 * (NANO_DUR_DEPTH / NANO_AC_TOKEN_DEPTH / NANO_AC_DEPTH / NANO_BLOCKS). The
 * rows below are produced by token pasting and guarded on that count, so one
 * source serves every lineage: the nano stacks (depth 3, 4 decoder blocks)
 * and the 2,272,145-parameter release stack (acoustic depth 4, 5 decoder
 * blocks) alike. An E12 header expands to exactly the hand-written tables
 * this replaced, so the bit-identity proof still holds. */
#define NANO_MAX_DEPTH 8
#if NANO_DUR_DEPTH > NANO_MAX_DEPTH || NANO_AC_TOKEN_DEPTH > NANO_MAX_DEPTH || \
    NANO_AC_DEPTH > NANO_MAX_DEPTH || NANO_BLOCKS > NANO_MAX_DEPTH
#error "snt_nano.c: extend the per-block tables (NANO_MAX_DEPTH) for this lineage"
#endif
#define NANO_RB_ROW(P, i) \
    {P##i##_C0_W8, P##i##_C0_SCALE, P##i##_C0_BIAS, \
     P##i##_C1_W8, P##i##_C1_SCALE, P##i##_C1_BIAS, P##i##_SCALE_F32}
#define NANO_DEC_ROW(i) \
    {DOFF_B##i##_DW_W_F32, DOFF_B##i##_DW_B_F32, DOFF_B##i##_NORM_W_F32, DOFF_B##i##_NORM_B_F32, \
     DOFF_B##i##_PW0_W8, DOFF_B##i##_PW0_SCALE, DOFF_B##i##_PW0_BIAS, \
     DOFF_B##i##_PW1_W8, DOFF_B##i##_PW1_SCALE, DOFF_B##i##_PW1_BIAS, DOFF_B##i##_GAMMA_F32}

int snt_nano_synthesize(const snt_nano_config *cfg,
                        const int32_t *phoneme_ids, int n_ids,
                        snt_nano_pcm_cb cb, void *user,
                        snt_nano_stats *stats_out) {
    NanoScratch *S = SCR();
    if (!cfg || !cfg->front_blob || !cfg->dec_blob || !cfg->arena) return -1;
    if (n_ids <= 0 || n_ids > MAX_TOKENS_RT) return ERR_TOKENS;
    g_front = (const unsigned char *)cfg->front_blob;
    g_dec = (const unsigned char *)cfg->dec_blob;
    g_arena = (unsigned char *)(((uintptr_t)cfg->arena + 15) & ~(uintptr_t)15);
    g_arena_cap = cfg->arena_size - (size_t)(g_arena - (unsigned char *)cfg->arena);
    g_arena_top = 0;
    g_arena_peak = 0;
    g_arena_oom = 0;
    /* Staging only buys anything if the destination is itself SIMD-readable.
     * With a PSRAM arena on the S3 it is not, and copying there would cost a
     * memcpy to still land on the scalar path -- so skip it and say so. */
#ifdef SNT_NANO_W_F32
    g_stage_ok = 0;   /* float rows: nothing to stage for an int8 SIMD path */
#else
    g_stage_ok = snt_weights_resident(g_arena);
#endif
#ifdef SNT_NANO_FAST_MATH
    if (!g_alut_init) alut_init();
    if (!g_exp2_init) exp2_init();
#endif
    if (!g_sc_init) sc_init();
    int64_t t_start = snt_now_us();

    const int32_t *ids = phoneme_ids;
    int n_tokens = n_ids;

    /* ---------------- duration student ---------------- */
    /* The duration trunk runs on a token WINDOW: NANO_TOKEN_CHUNK output
     * tokens plus DUR_RF tokens of recomputed context on each side, rather
     * than one [H, n_tokens] plane for the whole utterance. */
    PROF_T0N(dur);
    const int H = NANO_DUR_HIDDEN;
    const float *demb = FF(NOFF_DUR_EMB_F32);
    size_t mark_dur = g_arena_top;
    const int dchunk = (n_tokens < NANO_TOKEN_CHUNK) ? n_tokens : NANO_TOKEN_CHUNK;
    int dwin = dchunk + 2 * DUR_RF;
    if (dwin > n_tokens) dwin = n_tokens;
    float *dh = (float *)aa((size_t)H * dwin * 4);
    float *dtmp = (float *)aa((size_t)H * RB_RING(NANO_DUR_KERNEL) * 4);
    if (g_arena_oom) return ERR_OOM;
    nano_w_t *r_dc0 = stage_buf((size_t)H * NANO_DUR_B0_C0_N16);
    nano_w_t *r_dc1 = stage_buf((size_t)H * NANO_DUR_B0_C1_N16);
    size_t mark_dproj = g_arena_top;
    static int durs[MAX_TOKENS_RT];
    {
        const long off[NANO_DUR_DEPTH][7] = {
            NANO_RB_ROW(NOFF_DUR_B, 0),
            NANO_RB_ROW(NOFF_DUR_B, 1),
            NANO_RB_ROW(NOFF_DUR_B, 2),
#if NANO_DUR_DEPTH > 3
            NANO_RB_ROW(NOFF_DUR_B, 3),
#endif
#if NANO_DUR_DEPTH > 4
            NANO_RB_ROW(NOFF_DUR_B, 4),
#endif
#if NANO_DUR_DEPTH > 5
            NANO_RB_ROW(NOFF_DUR_B, 5),
#endif
#if NANO_DUR_DEPTH > 6
            NANO_RB_ROW(NOFF_DUR_B, 6),
#endif
#if NANO_DUR_DEPTH > 7
            NANO_RB_ROW(NOFF_DUR_B, 7),
#endif
        };
        const float len_hint =
            log1pf((float)n_tokens) / log1pf((float)NANO_DUR_MAX_TOKENS);
        for (int ta = 0; ta < n_tokens; ta += dchunk) {
            int tb = ta + dchunk;
            if (tb > n_tokens) tb = n_tokens;
            /* ulo/uhi are the UNCLAMPED range of the stage about to run; each
             * block shrinks them by its own receptive field, and after the
             * last one they are exactly [ta, tb). Clamping is applied at the
             * point of use only -- clamping them in place would stop the
             * shrink from landing on [ta, tb) at the utterance edges. */
            int ulo = ta - DUR_RF, uhi = tb + DUR_RF;
            const int dorg = clampi(ulo, 0, n_tokens);
            g_arena_top = mark_dproj;
            {
                const nano_w_t *w8 = stage_w(FQ(NOFF_DUR_PROJ_W8),
                                                (size_t)H * NANO_DUR_PROJ_N16);
                const float *sc = FF(NOFF_DUR_PROJ_SCALE);
                const float *bi = FF(NOFF_DUR_PROJ_BIAS);
                int hi = clampi(uhi, 0, n_tokens);
                for (int t = dorg; t < hi; t++) {
                    int id = ids[t];
                    if (id < 0 || id >= NANO_VOCAB) return -4;
                    for (int h = 0; h < H; h++) S->gather[h] = demb[(size_t)id * H + h];
                    S->gather[H] = (n_tokens > 1) ? (float)t / (float)(n_tokens - 1) : 0.0f;
                    S->gather[H + 1] = len_hint;
                    S->gather[H + 2] = 1.0f;   /* valid_hint: one unpadded sequence */
                    float s_act = quant_gather(S, H + 3, NANO_DUR_PROJ_N16);
                    nano_matvec(NANO_ACTBUF(S), w8, S->acc32, H, NANO_DUR_PROJ_N16);
                    for (int o = 0; o < H; o++)
                        dh[(size_t)o * dwin + (t - dorg)] =
                            s_act * sc[o] * (float)S->acc32[o] + bi[o];
                }
            }
            g_arena_top = mark_dproj;   /* the projection matrix is spent */
            for (int b = 0; b < NANO_DUR_DEPTH; b++) {
                ulo += RF_RESBLOCK(NANO_DUR_KERNEL);
                uhi -= RF_RESBLOCK(NANO_DUR_KERNEL);
                const nano_w_t *c0 = res_copy(FQ(off[b][0]), (size_t)H * NANO_DUR_B0_C0_N16, r_dc0);
                const nano_w_t *c1 = res_copy(FQ(off[b][3]), (size_t)H * NANO_DUR_B0_C1_N16, r_dc1);
                resblock(c0, FF(off[b][1]), FF(off[b][2]), NANO_DUR_B0_C0_N16,
                         c1, FF(off[b][4]), FF(off[b][5]), NANO_DUR_B0_C1_N16,
                         *FF(off[b][6]), dh, dtmp, H, n_tokens, NANO_DUR_KERNEL,
                         dwin, dorg, clampi(ulo, 0, n_tokens), clampi(uhi, 0, n_tokens));
            }
            {
                const float *os = FF(NOFF_DUR_OUT_SCALE);
                const float *ob = FF(NOFF_DUR_OUT_BIAS);
                const nano_w_t *w8 = stage_w(FQ(NOFF_DUR_OUT_W8), NANO_DUR_OUT_N16);
                for (int t = ta; t < tb; t++) {
                    for (int i = 0; i < H; i++)
                        S->gather[i] = dh[(size_t)i * dwin + (t - dorg)];
                    float s_act = quant_gather(S, H, NANO_DUR_OUT_N16);
                    nano_matvec(NANO_ACTBUF(S), w8, S->acc32, 1, NANO_DUR_OUT_N16);
                    float logd = s_act * os[0] * (float)S->acc32[0] + ob[0];
                    float d = roundf(fmaxf(expf(logd), 1.0f) * LENGTH_SCALE);
                    if (d < 1.0f) d = 1.0f;
                    if (d > (float)NANO_DUR_MAX_DURATION) d = (float)NANO_DUR_MAX_DURATION;
                    durs[t] = (int)d;
                }
            }
        }
    }
    g_arena_top = mark_dur;
    if (cfg->dur_override)
        for (int t = 0; t < n_tokens; t++) durs[t] = cfg->dur_override[t];
    int T = 0;
    for (int t = 0; t < n_tokens; t++) T += durs[t];
    if (T < 2) return -5;
    PROF_ADDN(PR_DUR, dur);

    /* ---------------- acoustic student, decoder and head ----------------
     * Everything from here to the PCM callback runs on FRAME CHUNKS. One
     * chunk of `fchunk` output frames is carried the whole way -- acoustic
     * frame blocks, embed, trunk, head, iSTFT, overlap-add -- and then the
     * window slides. Nothing but the iSTFT ring, the DC blocker state and the
     * sample cursor crosses a chunk boundary, so the planes are windows of a
     * FIXED width and the arena no longer grows with the utterance.
     *
     * Each stage inside a chunk works on a range shrunken by its own
     * receptive field from the stage before it, so the first stage recomputes
     * PIPE_RF columns at each end that the neighbouring chunks also compute.
     * Those columns are computed from exactly the inputs the whole-utterance
     * version gave them -- the GLOBAL frame index still decides where the
     * zero padding is and which token owns a frame -- which is what makes the
     * emitted PCM byte-identical rather than merely equivalent. */
    const int AH = NANO_AC_HIDDEN;
    const float *aemb = FF(NOFF_AC_EMB_F32);
    float maxdur = 1.0f;
    for (int t = 0; t < n_tokens; t++)
        if ((float)durs[t] > maxdur) maxdur = (float)durs[t];
    const float lmax = log1pf(maxdur);

    /* Chunk geometry. The chunk is shrunk if the arena the caller gave us
     * cannot hold the windows it implies: recomputing more is strictly better
     * than refusing to speak, and the output does not depend on the chunk
     * size. If even the smallest chunk does not fit, the aa() calls below
     * report it through g_arena_oom exactly as they always have. */
    int fchunk = (T < NANO_FRAME_CHUNK) ? T : NANO_FRAME_CHUNK;
    int pwin = 0, twin = 0;
    {
        size_t avail = (g_arena_cap > g_arena_top) ? g_arena_cap - g_arena_top : 0;
        for (;;) {
            pwin = fchunk + 2 * PIPE_RF;
            if (pwin > T) pwin = T;
            twin = 0;
            for (int fa = 0; fa < T; fa += fchunk) {
                int fb = fa + fchunk;
                int tlo, thi, w;
                if (fb > T) fb = T;
                tok_span(durs, n_tokens, clampi(fa - PIPE_RF, 0, T),
                         clampi(fb + PIPE_RF, 0, T), &tlo, &thi);
                w = clampi(thi + 1 + TOK_RF, 0, n_tokens)
                  - clampi(tlo - TOK_RF, 0, n_tokens);
                if (w > twin) twin = w;
            }
            {   /* every allocation the chunk MUST have, worst phase */
                size_t need = (size_t)DIM * pwin * 4 + 16
                            + (size_t)AH * RB_RING(NANO_AC_KERNEL) * 4 + 16
                            + (size_t)AH * twin * 4 + 16
                            + (size_t)NANO_NOISE_CH * pwin * 4 + 16;
                if (need <= avail || fchunk <= 2) break;
            }
            fchunk = (fchunk / 2) & ~1;   /* stays even: the head pairs frames */
            if (fchunk < 2) fchunk = 2;
        }
    }

    /* ONE [DIM, pwin] plane serves the acoustic frame output ax (which needs
     * only the first AH of the DIM rows) and then the decoder trunk x.
     *
     * They can share it for the reason they always have: the embed loop
     * consumes ax exactly THREE columns AHEAD of where it writes x -- at step
     * t it reads ax[:][t+3] to make the mel column entering the k=7 window,
     * then writes x[:][t]. Row strides are equal, so x[c][t] aliases ax[c][t]
     * for c < AH -- a column last read at step t-3 and never read again --
     * and rows AH..DIM-1 land above ax entirely. Nothing is copied and
     * nothing is delayed; the write simply trails the read. The only thing
     * this change touches is the stride: pwin instead of T. */
    float *plane = (float *)aa((size_t)DIM * pwin * 4);
    if (g_arena_oom) return ERR_OOM;
    float *ax = plane;
    float *x = plane;                  /* written in place over ax, see above */
    const size_t mark_ac = g_arena_top;

    /* Weight-offset tables. Compile-time constants, hoisted out of the chunk
     * loop so crossing a boundary costs no table rebuilds. */
    const long tb_off[NANO_AC_TOKEN_DEPTH][7] = {
        NANO_RB_ROW(NOFF_AC_TB, 0),
        NANO_RB_ROW(NOFF_AC_TB, 1),
        NANO_RB_ROW(NOFF_AC_TB, 2),
#if NANO_AC_TOKEN_DEPTH > 3
        NANO_RB_ROW(NOFF_AC_TB, 3),
#endif
#if NANO_AC_TOKEN_DEPTH > 4
        NANO_RB_ROW(NOFF_AC_TB, 4),
#endif
#if NANO_AC_TOKEN_DEPTH > 5
        NANO_RB_ROW(NOFF_AC_TB, 5),
#endif
#if NANO_AC_TOKEN_DEPTH > 6
        NANO_RB_ROW(NOFF_AC_TB, 6),
#endif
#if NANO_AC_TOKEN_DEPTH > 7
        NANO_RB_ROW(NOFF_AC_TB, 7),
#endif
    };
    const long fb_off[NANO_AC_DEPTH][7] = {
        NANO_RB_ROW(NOFF_AC_FB, 0),
        NANO_RB_ROW(NOFF_AC_FB, 1),
        NANO_RB_ROW(NOFF_AC_FB, 2),
#if NANO_AC_DEPTH > 3
        NANO_RB_ROW(NOFF_AC_FB, 3),
#endif
#if NANO_AC_DEPTH > 4
        NANO_RB_ROW(NOFF_AC_FB, 4),
#endif
#if NANO_AC_DEPTH > 5
        NANO_RB_ROW(NOFF_AC_FB, 5),
#endif
#if NANO_AC_DEPTH > 6
        NANO_RB_ROW(NOFF_AC_FB, 6),
#endif
#if NANO_AC_DEPTH > 7
        NANO_RB_ROW(NOFF_AC_FB, 7),
#endif
    };
    const long qb[NANO_BLOCKS][11] = {
        NANO_DEC_ROW(0),
        NANO_DEC_ROW(1),
        NANO_DEC_ROW(2),
        NANO_DEC_ROW(3),
#if NANO_BLOCKS > 4
        NANO_DEC_ROW(4),
#endif
#if NANO_BLOCKS > 5
        NANO_DEC_ROW(5),
#endif
#if NANO_BLOCKS > 6
        NANO_DEC_ROW(6),
#endif
#if NANO_BLOCKS > 7
        NANO_DEC_ROW(7),
#endif
    };
#if NANO_NORM_TYPE == 1
    /* per-block DyT alpha vectors; region names from the exporter */
    const long qa[NANO_BLOCKS] = {DOFF_B0_NORM_A_F32, DOFF_B1_NORM_A_F32,
                                  DOFF_B2_NORM_A_F32, DOFF_B3_NORM_A_F32,
#if NANO_BLOCKS > 4
                                  DOFF_B4_NORM_A_F32,
#endif
#if NANO_BLOCKS > 5
                                  DOFF_B5_NORM_A_F32,
#endif
#if NANO_BLOCKS > 6
                                  DOFF_B6_NORM_A_F32,
#endif
#if NANO_BLOCKS > 7
                                  DOFF_B7_NORM_A_F32,
#endif
    };
#endif

    /* ---------------- head + iSTFT + OLA + DC block: carried state ------- */
    static float win[NFFT], ring[NFFT];
    static float fre[2][NFFT], fim[2][NFFT];
    static float cwre[NFFT / 2], cwim[NFFT / 2];
    static float zre[2][NFFT / 2], zim[2][NFFT / 2];
    static float env_inv[HOP];
    static float pcm_buf[HOP + NFFT / 2];
    /* One-time: these tables are utterance-independent and the only double
     * arithmetic in the file. Recomputing them per call would pay 1536 soft-
     * float `sin`/`cos` calls on a part without hardware double -- the exact
     * cost R7's portability rule 3 warns about. R7 guards its own with
     * `cw_init` / `env_init`; this does the same. */
    static int tables_init = 0;
    if (!tables_init) {
        fft_tw_init();
        for (int i = 0; i < NFFT; i++) {
            double v = sin(M_PI * i / NFFT);  /* == torch.hann_window(NFFT) */
            win[i] = (float)(v * v);
        }
        for (int k = 0; k < NFFT / 2; k++) {
            cwre[k] = (float)cos(2.0 * M_PI * k / NFFT);
            cwim[k] = (float)sin(2.0 * M_PI * k / NFFT);
        }
        for (int r = 0; r < HOP; r++) {
            float e = 0.0f;
            for (int j = 0; j < NFFT / HOP; j++) {
                float w2 = win[r + j * HOP];
                e += w2 * w2;
            }
            env_inv[r] = (e > 1e-11f) ? 1.0f / e : 0.0f;
        }
        tables_init = 1;
    }
    memset(ring, 0, sizeof ring);

    const float *fgw = DF(DOFF_FNORM_W_F32);
    const float *fgb = DF(DOFF_FNORM_B_F32);
#if NANO_NORM_TYPE == 1
    const float *fga = DF(DOFF_FNORM_A_F32);
#else
    const float *fga = NULL;
    (void)fga;   /* the LayerNorm NORM_COL drops its alpha argument */
#endif
    const int out_total = HOP * (T - 1);   /* torch.istft(center=True) length */
    int emitted = 0, aborted = 0;
    float dc_x1 = 0.0f, dc_y1 = 0.0f;      /* first-order DC blocker state */
    int pcm_n = 0;

    /* ======================= the chunk loop ============================= */
    for (int fa = 0; fa < T && !aborted; fa += fchunk) {
        int fb = fa + fchunk;
        if (fb > T) fb = T;
        /* ulo/uhi are the UNCLAMPED range of the stage about to run. Each
         * stage shrinks them by its own receptive field; after the last trunk
         * block they are exactly [fa, fb). They are clamped at the point of
         * use only -- clamping in place would stop the shrink from landing on
         * [fa, fb) at the utterance edges. */
        int ulo = fa - PIPE_RF, uhi = fb + PIPE_RF;
        const int porg = clampi(ulo, 0, T);    /* plane window origin */
        const int pend = clampi(uhi, 0, T);

        g_arena_top = mark_ac;

        /* ---- acoustic student, on this chunk's token window ---- */
        PROF_T0N(ac);
        int tlo, thi;
        tok_span(durs, n_tokens, porg, pend, &tlo, &thi);
        {
            int tul = tlo - TOK_RF, tuh = thi + 1 + TOK_RF;
            const int torg = clampi(tul, 0, n_tokens);
            /* one conv0 ring, reused by the token blocks and then the frame
             * blocks; th dies with the frame projection that consumes it */
            float *atmp = (float *)aa((size_t)AH * RB_RING(NANO_AC_KERNEL) * 4);
            float *th = (float *)aa((size_t)AH * twin * 4);
            nano_w_t *r_ac0, *r_ac1;
            size_t mark_acproj;
            if (g_arena_oom) return ERR_OOM;
            r_ac0 = stage_buf((size_t)AH * NANO_AC_TB0_C0_N16);
            r_ac1 = stage_buf((size_t)AH * NANO_AC_TB0_C1_N16);
            /* tproj and fproj are one-shot matrices used at either end of the
             * token blocks; they share one mark so neither outlives its loop */
            mark_acproj = g_arena_top;

            {
                const nano_w_t *w8 = stage_w(FQ(NOFF_AC_TPROJ_W8),
                                                (size_t)AH * NANO_AC_TPROJ_N16);
                const float *sc = FF(NOFF_AC_TPROJ_SCALE);
                const float *bi = FF(NOFF_AC_TPROJ_BIAS);
                int hi = clampi(tuh, 0, n_tokens);
                for (int t = torg; t < hi; t++) {
                    int id = ids[t];
                    for (int h = 0; h < AH; h++) S->gather[h] = aemb[(size_t)id * AH + h];
                    S->gather[AH] = (n_tokens > 1) ? (float)t / (float)(n_tokens - 1) : 0.0f;
                    S->gather[AH + 1] = log1pf((float)durs[t]) / lmax;
                    float s_act = quant_gather(S, AH + 2, NANO_AC_TPROJ_N16);
                    nano_matvec(NANO_ACTBUF(S), w8, S->acc32, AH, NANO_AC_TPROJ_N16);
                    for (int o = 0; o < AH; o++)
                        th[(size_t)o * twin + (t - torg)] =
                            s_act * sc[o] * (float)S->acc32[o] + bi[o];
                }
            }
            g_arena_top = mark_acproj;
            for (int b = 0; b < NANO_AC_TOKEN_DEPTH; b++) {
                const nano_w_t *c0 = res_copy(FQ(tb_off[b][0]),
                                              (size_t)AH * NANO_AC_TB0_C0_N16, r_ac0);
                const nano_w_t *c1 = res_copy(FQ(tb_off[b][3]),
                                              (size_t)AH * NANO_AC_TB0_C1_N16, r_ac1);
                tul += RF_RESBLOCK(NANO_AC_KERNEL);
                tuh -= RF_RESBLOCK(NANO_AC_KERNEL);
                resblock(c0, FF(tb_off[b][1]), FF(tb_off[b][2]), NANO_AC_TB0_C0_N16,
                         c1, FF(tb_off[b][4]), FF(tb_off[b][5]), NANO_AC_TB0_C1_N16,
                         *FF(tb_off[b][6]), th, atmp, AH, n_tokens, NANO_AC_KERNEL,
                         twin, torg, clampi(tul, 0, n_tokens), clampi(tuh, 0, n_tokens));
            }
            /* expand tokens -> frames, appending the three positional feats */
            {
                const nano_w_t *w8 = stage_w(FQ(NOFF_AC_FPROJ_W8),
                                                (size_t)AH * NANO_AC_FPROJ_N16);
                const float *sc = FF(NOFF_AC_FPROJ_SCALE);
                const float *bi = FF(NOFF_AC_FPROJ_BIAS);
                int f = 0;
                for (int tok = 0; tok < tlo; tok++) f += durs[tok];
                for (int tok = tlo; tok <= thi; tok++) {
                    for (int d = 0; d < durs[tok]; d++, f++) {
                        if (f < porg || f >= pend) continue;
                        for (int h = 0; h < AH; h++)
                            S->gather[h] = th[(size_t)h * twin + (tok - torg)];
                        S->gather[AH] = (T > 1) ? (float)f / (float)(T - 1) : 0.0f;
                        S->gather[AH + 1] =
                            (float)tok / (float)(n_tokens > 1 ? n_tokens - 1 : 1);
                        S->gather[AH + 2] =
                            (durs[tok] > 1) ? (float)d / (float)(durs[tok] - 1) : 0.0f;
                        float s_act = quant_gather(S, AH + 3, NANO_AC_FPROJ_N16);
                        nano_matvec(NANO_ACTBUF(S), w8, S->acc32, AH, NANO_AC_FPROJ_N16);
                        for (int o = 0; o < AH; o++)
                            ax[(size_t)o * pwin + (f - porg)] =
                                s_act * sc[o] * (float)S->acc32[o] + bi[o];
                    }
                }
            }
            g_arena_top = mark_acproj;
            for (int b = 0; b < NANO_AC_DEPTH; b++) {
                const nano_w_t *c0 = res_copy(FQ(fb_off[b][0]),
                                              (size_t)AH * NANO_AC_FB0_C0_N16, r_ac0);
                const nano_w_t *c1 = res_copy(FQ(fb_off[b][3]),
                                              (size_t)AH * NANO_AC_FB0_C1_N16, r_ac1);
                ulo += RF_RESBLOCK(NANO_AC_KERNEL);
                uhi -= RF_RESBLOCK(NANO_AC_KERNEL);
                resblock(c0, FF(fb_off[b][1]), FF(fb_off[b][2]), NANO_AC_FB0_C0_N16,
                         c1, FF(fb_off[b][4]), FF(fb_off[b][5]), NANO_AC_FB0_C1_N16,
                         *FF(fb_off[b][6]), ax, atmp, AH, T, NANO_AC_KERNEL,
                         pwin, porg, clampi(ulo, 0, T), clampi(uhi, 0, T));
            }
        }
        g_arena_top = mark_ac;   /* atmp / th / residency buffers are dead */
        PROF_ADDN(PR_AC, ac);

        /* ---------------- decoder: embed ---------------- */
        PROF_T0N(emb);
        {
            /* The noise the embed window reads is exactly the range the embed
             * stage's own receptive field reaches, i.e. [ulo, uhi) before the
             * shrink below. It is generated straight into that window: the
             * MT19937 stream writes [channels, T] in channel-major order, so
             * a whole-utterance buffer was the only other way to have all
             * four channels of one frame -- see noise_window(). */
            const int nlo = clampi(ulo, 0, T), nhi = clampi(uhi, 0, T);
            const int nw = nhi - nlo;
            float *noise = (float *)aa((size_t)NANO_NOISE_CH * nw * 4);
            if (g_arena_oom) return ERR_OOM;
            if (noise_window(cfg->noise_seed, NANO_NOISE_CH, T, nlo, nhi,
                             noise, nw) != 0)
                return -6;
            ulo += RF_EMBED;
            uhi -= RF_EMBED;
            {
                const int elo = clampi(ulo, 0, T), ehi = clampi(uhi, 0, T);
                /* embed(mel) + noise_adapter(noise) + stem LayerNorm, tiled.
                 * The mel-100 interface is still streamed through a ring
                 * rather than materialised; the ring is EM_RING columns wide
                 * so a whole tile's worth of embed columns can run off it. */
                static float melring[EM_RING][MELS];
                /* The three matrices of this stage are interleaved inside the
                 * tile, so unlike the phases either side of it they must be
                 * resident SIMULTANEOUSLY: 4,800 + 33,792 + 1,536 = 40,128 B. */
                EmCtx ec;
                const int half = EK / 2;
                int produced;
                ec.ow = stage_w(FQ(NOFF_AC_OUT_W8), (size_t)MELS * NANO_AC_OUT_N16);
                ec.os = FF(NOFF_AC_OUT_SCALE);
                ec.ob = FF(NOFF_AC_OUT_BIAS);
                ec.ew = stage_w(DQ(DOFF_EMBED_W8), (size_t)DIM * NANO_EMBED_N16);
                ec.es = DF(DOFF_EMBED_SCALE);
                ec.eb = DF(DOFF_EMBED_BIAS);
                ec.nw = stage_w(DQ(DOFF_NOISE_W8), (size_t)DIM * NANO_NOISE_N16);
                ec.ns = DF(DOFF_NOISE_SCALE);
                ec.nb = DF(DOFF_NOISE_BIAS);
                ec.gw = DF(DOFF_NORM_W_F32);
                ec.gb = DF(DOFF_NORM_B_F32);
#if NANO_NORM_TYPE == 1
                ec.ga = DF(DOFF_NORM_A_F32);
#else
                ec.ga = NULL;
#endif
                ec.ax = ax;
                ec.x = x;
                ec.noise = noise;
                ec.mel = &melring[0][0];
                ec.T = T;
                ec.xw = pwin;
                ec.x0 = porg;
                ec.noisew = nw;
                ec.noise0 = nlo;
                produced = elo - half;   /* mel columns [., produced) are live */
                if (produced < 0) produced = 0;
                for (int t0 = elo; t0 < ehi; t0 += EM_TILE) {
                    int TL = (t0 + EM_TILE <= ehi) ? EM_TILE : (ehi - t0);
                    int need = t0 + TL + half;
                    if (need > T) need = T;  /* columns >= T are zero by the gather */
                    if (need > produced) {
                        ec.a0 = produced;
                        snt_par_run(em_mel_range, need - produced, &ec);
                        produced = need;
                    }
                    ec.a0 = t0;
                    snt_par_run(em_col_range, TL, &ec);
                }
            }
        }
        /* noise and the embed staging are dead from here; x already occupies
         * the plane, so rewinding the top is the whole of the deallocation. */
        g_arena_top = mark_ac;
        PROF_ADDN(PR_EMBED, emb);

        /* ---------------- decoder: ConvNeXt1D blocks ---------------- */
        {
            const int half = DWK / 2;
            /* pre-residual snapshot of the tile boundary, and the tile's
             * dw+LN output. Both are TILE-shaped, not T-shaped. */
            static float halo[DWK / 2][DIM];
            static float halo_next[DWK / 2][DIM];
            static float tile_c[TR_TILE][DIM];
            /* The blocks run one after another, so ONE pair of buffers sized
             * to a single block's pointwise matrices serves all of them. */
            nano_w_t *r_pw0 = stage_buf((size_t)NANO_PW_HIDDEN * NANO_B0_PW0_N16);
            nano_w_t *r_pw1 = stage_buf((size_t)DIM * NANO_B0_PW1_N16);
            for (int b = 0; b < NANO_BLOCKS; b++) {
                TrCtx tc;
                int blo, bhi;
                ulo += RF_TRUNK;
                uhi -= RF_TRUNK;
                blo = clampi(ulo, 0, T);
                bhi = clampi(uhi, 0, T);
                tc.dww = DF(qb[b][0]);
                tc.dwb = DF(qb[b][1]);
                tc.nwt = DF(qb[b][2]);
                tc.nbt = DF(qb[b][3]);
#if NANO_NORM_TYPE == 1
                tc.nat = DF(qa[b]);
#else
                tc.nat = NULL;
#endif
                tc.pw0 = res_copy(DQ(qb[b][4]),
                                  (size_t)NANO_PW_HIDDEN * NANO_B0_PW0_N16, r_pw0);
                tc.pw0s = DF(qb[b][5]);
                tc.pw0b = DF(qb[b][6]);
                tc.pw1 = res_copy(DQ(qb[b][7]),
                                  (size_t)DIM * NANO_B0_PW1_N16, r_pw1);
                tc.pw1s = DF(qb[b][8]);
                tc.pw1b = DF(qb[b][9]);
                tc.gamma = DF(qb[b][10]);
                tc.x = x;
                tc.tc = &tile_c[0][0];
                tc.halo = (const float (*)[DIM])halo;
                tc.T = T;
                tc.xw = pwin;
                tc.x0 = porg;
                /* The first tile of a chunk needs the block's PRE-residual
                 * columns below blo, which are in the window and untouched by
                 * this block. At blo == 0 every column is out of range and
                 * this fills zeros -- what the old memset did, and the only
                 * case the whole-utterance version ever had. */
                for (int j = 0; j < half; j++) {
                    int col = blo - 1 - j;
                    for (int ch = 0; ch < DIM; ch++)
                        halo[j][ch] = (col >= 0 && col < T)
                            ? x[(size_t)ch * pwin + WCOL(pwin, porg, col)] : 0.0f;
                }
                for (int t0 = blo; t0 < bhi; t0 += TR_TILE) {
                    int TL = (t0 + TR_TILE <= bhi) ? TR_TILE : (bhi - t0);
                    /* snapshot BEFORE pass 2 writes: halo_next[j] is the
                     * pre-residual column (t0+TL-1-j), i.e. the next tile's
                     * t0'-1-j. */
                    for (int j = 0; j < half; j++) {
                        int col = t0 + TL - 1 - j;
                        for (int ch = 0; ch < DIM; ch++)
                            halo_next[j][ch] = (col >= 0 && col < T)
                                ? x[(size_t)ch * pwin + WCOL(pwin, porg, col)] : 0.0f;
                    }
                    tc.t0 = t0;
                    tc.TL = TL;
                    PROF_T0N(dw);
                    snt_par_run(tr_dw_range, TL, &tc);   /* barrier: x reads done */
                    PROF_ADDN(PR_TRUNK_DW, dw);
                    PROF_T0N(pw);
                    snt_par_run(tr_pw_range, TL, &tc);   /* then x writes */
                    PROF_ADDN(PR_TRUNK_PW, pw);
                    memcpy(halo, halo_next, sizeof halo);
                }
            }
        }
        /* the trunk's pointwise staging is spent; the head reuses those bytes */
        g_arena_top = mark_ac;

        /* ---------------- head + iSTFT + OLA + DC block ---------------- */
        {
            const nano_w_t *hw = stage_w(DQ(DOFF_HEAD_W8),
                                            (size_t)NANO_HEAD_OUT * NANO_HEAD_N16);
            const float *hs = DF(DOFF_HEAD_SCALE);
            const float *hb = DF(DOFF_HEAD_BIAS);

            for (int t0 = fa; t0 < fb && !aborted; t0 += 2) {
                int np = (t0 + 1 < fb) ? 2 : 1;   /* frames processed in this pair */

                /* --- per frame: head LayerNorm + quantise, then the parallel
                 *     head-int8 + spectrum range, then the Hermitian pack --- */
                for (int p = 0; p < np; p++) {
                    int t = t0 + p;
                    PROF_T0N(hd);
                    float col[DIM], ncol[DIM];
                    for (int c = 0; c < DIM; c++)
                        col[c] = x[(size_t)c * pwin + WCOL(pwin, porg, t)];
                    NORM_COL(ncol, col, fga, fgw, fgb, DIM);
                    for (int i = 0; i < DIM; i++) S->gather[i] = ncol[i];
                    float s_act = quant_gather(S, DIM, NANO_HEAD_N16);
                    PROF_ADDN(PR_HEAD, hd);

                    PROF_T0N(sp);
                    {
                        HeadCtx hc = {NANO_ACTBUF(S), hw, hs, hb, g_acc_head, s_act,
                                      fre[p], fim[p]};
                        snt_par_run(head_spec_range, BINS, &hc);
                    }
                    PROF_ADDN(PR_SPEC, sp);

                    PROF_T0N(pk);
                    /* real IFFT via an N/2 complex IFFT (Hermitian pack) -- R7's path */
                    for (int k = 0; k < NFFT / 2; k++) {
                        float xr = fre[p][k], xi = fim[p][k];
                        float mr = fre[p][NFFT / 2 - k], mi = fim[p][NFFT / 2 - k];
                        float ar = 0.5f * (xr + mr), ai = 0.5f * (xi - mi);
                        float dr = xr - mr, di = xi + mi;
                        float br = 0.5f * (dr * cwre[k] - di * cwim[k]);
                        float bi2 = 0.5f * (dr * cwim[k] + di * cwre[k]);
                        zre[p][k] = ar - bi2;
                        zim[p][k] = ai + br;
                    }
                    PROF_ADDN(PR_FFT, pk);
                }

                PROF_T0N(ff);
                {
                    FftCtx fc;
                    fc.zre[0] = zre[0]; fc.zre[1] = zre[1];
                    fc.zim[0] = zim[0]; fc.zim[1] = zim[1];
                    fc.n = NFFT / 2;
                    snt_par_run(fft_pair_range, np, &fc);
                }
                for (int p = 0; p < np; p++)
                    for (int n = 0; n < NFFT / 2; n++) {
                        fre[p][2 * n] = zre[p][n];
                        fre[p][2 * n + 1] = zim[p][n];
                    }
                PROF_ADDN(PR_FFT, ff);

                /* --- overlap-add strictly in frame order: the ring, the DC
                 *     blocker state and the PCM callback are all sequential --- */
                for (int p = 0; p < np; p++) {
                    int t = t0 + p;
                    PROF_T0N(ol);
                    const float *fr = fre[p];
                    int base = t * HOP;
                    for (int i = 0; i < NFFT; i++) ring[(base + i) % NFFT] += fr[i] * win[i];

                    int emit_from = base, emit_to = base + HOP;
                    if (t == T - 1) emit_to = out_total + NFFT / 2;
                    int interior_from = NFFT - HOP;
                    int interior_to = (T - 1) * HOP + HOP;
                    for (int s2 = emit_from; s2 < emit_to; s2++) {
                        int out_idx = s2 - NFFT / 2;
                        if (out_idx >= 0 && out_idx < out_total) {
                            float sample;
                            if (s2 >= interior_from && s2 < interior_to) {
                                sample = ring[s2 & (NFFT - 1)] * env_inv[s2 & (HOP - 1)];
                            } else {
                                float e = 0.0f;
                                int j0 = (s2 - NFFT + HOP) / HOP;
                                if (j0 < 0) j0 = 0;
                                for (int j = j0; j <= s2 / HOP && j < T; j++) {
                                    float w2 = win[s2 - j * HOP];
                                    e += w2 * w2;
                                }
                                sample = (e > 1e-11f) ? ring[s2 & (NFFT - 1)] / e : 0.0f;
                            }
                            /* H(z) = (1 - z^-1) / (1 - R z^-1), 2 MAC/sample */
                            float y = sample - dc_x1 + DC_POLE * dc_y1;
                            dc_x1 = sample;
                            dc_y1 = y;
                            pcm_buf[pcm_n++] = y;
                            emitted++;
                        }
                        ring[s2 & (NFFT - 1)] = 0.0f;
                    }
                    if (pcm_n && cb) {
                        if (cb(pcm_buf, pcm_n, user)) aborted = 1;
                    }
                    pcm_n = 0;
                    PROF_ADDN(PR_OLA, ol);
                    if (aborted) break;
                }
            }
        }
    }

    if (stats_out) {
        stats_out->frames = T;
        stats_out->samples = emitted;
        stats_out->elapsed_us = snt_now_us() - t_start;
        stats_out->arena_peak = g_arena_peak;
    }
    return aborted ? ERR_ABORT : 0;
}
