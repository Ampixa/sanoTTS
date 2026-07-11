/* fsd_e2e.c -- host validator for the FULL on-device TTS chain, int8, using
 * the exact memory-lean algorithm the ESP32-S3 firmware runs:
 *   ids -> duration student -> acoustic (token_context) -> c[40,T]
 *       -> lrc decoder with two-pass GroupNorm (no full-T temp buffer),
 *          fused per-frame head, ring-buffer OLA, corr-on-the-fly.
 *
 * Build: cc -O2 -std=c99 fsd_e2e.c -lm -o fsd_e2e_test && ./fsd_e2e_test golden
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "golden/fsd_meta.h"
#include "golden/fsd_q8_meta.h"
#include "golden/front_q8_meta.h"
#include "golden/frozen_norm.h"

#ifdef ESP_PLATFORM
#include <stdint.h>
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
extern const unsigned char front_start[] asm("_binary_front_q8_bin_start");
extern const unsigned char dec_start[] asm("_binary_model_q8_bin_start");
extern const unsigned char ids_start[] asm("_binary_e2e_ids_bin_start");
extern const unsigned char ids_end[] asm("_binary_e2e_ids_bin_end");
extern const unsigned char durs_start[] asm("_binary_e2e_durs_bin_start");
extern const unsigned char gold_start[] asm("_binary_e2e_audio_bin_start");
extern const unsigned char gold_end[] asm("_binary_e2e_audio_bin_end");
extern int32_t esp_nn_dot_s8_aligned_esp32s3(const int8_t *a, const int8_t *b, int32_t len);
/* SIMD only when the weight pointer is in internal SRAM (flash XIP reads garbage) */
static int is_sram(const void *p) {
    uint32_t a = (uint32_t)p;
    return a >= 0x3FC80000u && a < 0x3FD00000u;
}
/* PSRAM is data-cache mapped (unlike flash XIP), so PIE vector loads are safe
 * there too -- verified by the golden-pass corr check at boot */
static int is_psram(const void *p) {
    uint32_t a = (uint32_t)p;
    return a >= 0x3C000000u && a < 0x3E000000u;
}
#define NOW_US() esp_timer_get_time()
/* engine memory MUST be internal SRAM: the SIMD dot only runs on SRAM pointers
 * (is_sram gate), and with PSRAM enabled plain MALLOC_CAP_8BIT could land there */
static void *xmalloc(size_t n) {
    void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!p) { printf("OOM %zu\n", n); abort(); }
    return p;
}
static void *xmalloc16(size_t n) {
    void *p = heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!p) { printf("OOM16 %zu\n", n); abort(); }
    return p;
}

/* ---- LEDC-PWM audio out: GPIO17 -> RC low-pass -> LM386 -> speaker.
 * 156.25 kHz carrier (the RC filter rejects it hard; PDM's shaped noise sat
 * inside the passband and buzzed). A 22 kHz GPTimer ISR drains a ring buffer
 * the emit loop fills; the engine is ~4.5x faster than playback, so pushes
 * throttle on ring-full and the stream plays gaplessly in realtime. */
#include "esp_attr.h"
#include "driver/ledc.h"
#include "driver/gptimer.h"
#define AOUT_GPIO    17
#define AOUT_CARRIER 156250
#define AOUT_SR      22050
/* Whole-utterance PCM buffer in PSRAM. The producer synthesizes UNTHROTTLED
 * (0.53x RT) while the ISR plays ~370ms behind it -- the producer is ~2x
 * faster than playback, so after the initial lead it can never be caught and
 * underruns are structurally impossible. (Scalar ISR reads from cached PSRAM
 * are fine; only PIE vector loads from PSRAM garbage.) */
#define APCM_MAX     (12 * 22050)          /* 12 s ceiling per utterance */
#define APCM_LEAD    44100                 /* 2 s head start. The scalar-PSRAM pw banks
                                            * put synthesis at ~1.05x realtime, so playback
                                            * would slowly outrun it; a 2 s pre-buffer covers
                                            * the whole-utterance deficit (<=12 s * 0.1) with
                                            * margin. Short clips still start at aout_end. */
static int16_t *g_pcm;                     /* PSRAM, allocated once in aout_init */
static volatile int g_pcm_w, g_pcm_r;
static volatile int g_pcm_playing;
static float g_again = 0.0f;              /* 0 = silent measuring pass */
static float g_apeak = 0.0f;
static gptimer_handle_t g_agt;

static bool IRAM_ATTR aout_tick(gptimer_handle_t t,
                                const gptimer_alarm_event_data_t *e, void *u) {
    int16_t s = 0;
    if (g_pcm_playing) {
        int r = g_pcm_r;
        if (r < g_pcm_w) { s = g_pcm[r]; g_pcm_r = r + 1; }
    }
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, (uint32_t)((s >> 8) + 128));
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
    return false;
}

/* arm a fresh utterance (stops any previous playback) */
static void aout_begin(void) {
    g_pcm_playing = 0;
    g_pcm_r = 0;
    g_pcm_w = 0;
}
/* flush: play whatever was produced, however short */
static void aout_end(void) { g_pcm_playing = 1; }

/* short boot chirp: audibly distinct from ANY utterance so a reboot can never
 * be mistaken for a spoken response (bypasses gain; writes pcm directly) */
static void aout_beep(void) {
    aout_begin();
    for (int i = 0; i < AOUT_SR / 5 && i < APCM_MAX; i++)
        g_pcm[i] = (int16_t)(8000.0f * sinf(2.0f * 3.14159265f * 880.0f * i / AOUT_SR));
    g_pcm_w = AOUT_SR / 5;
    aout_end();
}

static void aout_init(void) {
    ledc_timer_config_t tc = {
        .speed_mode = LEDC_LOW_SPEED_MODE, .duty_resolution = LEDC_TIMER_8_BIT,
        .timer_num = LEDC_TIMER_0, .freq_hz = AOUT_CARRIER, .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&tc));
    ledc_channel_config_t cc = {
        .gpio_num = AOUT_GPIO, .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = LEDC_CHANNEL_0, .timer_sel = LEDC_TIMER_0, .duty = 128, .hpoint = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&cc));
    gptimer_config_t gc = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT, .direction = GPTIMER_COUNT_UP,
        .resolution_hz = 1000000,
    };
    ESP_ERROR_CHECK(gptimer_new_timer(&gc, &g_agt));
    gptimer_event_callbacks_t cbs = { .on_alarm = aout_tick };
    ESP_ERROR_CHECK(gptimer_register_event_callbacks(g_agt, &cbs, NULL));
    gptimer_alarm_config_t ac = {
        .reload_count = 0, .alarm_count = 45, .flags = { .auto_reload_on_alarm = true },
    };
    ESP_ERROR_CHECK(gptimer_set_alarm_action(g_agt, &ac));
    ESP_ERROR_CHECK(gptimer_enable(g_agt));
    ESP_ERROR_CHECK(gptimer_start(g_agt));
    g_pcm = (int16_t *)heap_caps_malloc((size_t)APCM_MAX * 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!g_pcm) { printf("FATAL: no PSRAM for PCM buffer\n"); abort(); }
}

static void aout_push(float sample) {
    if (g_again == 0.0f) {                /* pass 0: find peak, stay silent */
        static int wdt_ctr;
        float v = fabsf(sample);
        if (v > g_apeak) g_apeak = v;
        if (++wdt_ctr >= 4096) { wdt_ctr = 0; vTaskDelay(1); }  /* feed IDLE0/WDT */
        return;
    }
    float v = sample * g_again;
    if (v > 32000.0f) v = 32000.0f;
    if (v < -32000.0f) v = -32000.0f;
    if (g_pcm_w < APCM_MAX) { g_pcm[g_pcm_w] = (int16_t)v; g_pcm_w = g_pcm_w + 1; }
    if (!g_pcm_playing && g_pcm_w >= APCM_LEAD) g_pcm_playing = 1;
    {   /* unthrottled multi-second compute: feed IDLE0 so the task WDT stays quiet */
        static int wdt_ctr2;
        if (++wdt_ctr2 >= 4096) { wdt_ctr2 = 0; vTaskDelay(1); }
    }
}

/* ---- arbitrary-utterance mode: when g_ids_ovr is set, run_e2e synthesizes
 * these ids with PREDICTED durations and no golden comparison (dashboard). */
static const int *g_ids_ovr;
static int g_n_ovr;
static volatile int g_synth_busy;
static int g_last_T, g_last_ms;          /* filled by run_e2e for the HTTP summary */
#else
#include <time.h>
#define NOW_US() ((long long)clock() * 1000000LL / CLOCKS_PER_SEC)
static void *xmalloc(size_t n) { void *p = malloc(n); if (!p) abort(); return p; }
static void *xmalloc16(size_t n) { void *p = NULL; if (posix_memalign(&p, 16, n)) abort(); return p; }
static void aout_push(float sample) { (void)sample; }
#endif

#define HOP FSD_HOP
#define NFFT FSD_N_FFT
#define TILE 8
#define LENGTH_SCALE 1.08f

/* ---- Sibilant fricative-noise injection ---------------------------------
 * The deterministic acoustic student regresses the MEAN 40-dim latent, so
 * /s z ʃ ʒ/ (broadband noise) collapse to a whistly tone. At sibilant frames
 * we add per-channel Gaussian noise (scaled by the student latent's own
 * per-channel dynamic range) back into the latent before quantization, so the
 * decoder renders proper hiss. Host-verified on the r7 stack: sibilant 2-8kHz
 * flatness 0.597 -> 0.686 at beta 0.9 (teacher 0.689), confined to sibilants.
 * calib_fsd40.npz / tools/calibrate_sibilant_noise_fsd.py. */
#define SIB_BETA 0.9f
/* per-channel std (40), fsd 40-dim code space; from calib_fsd40.npz */
static const float SIB_TEA_STD[FSD_CODE_DIM] = {
  1.97866f, 1.22888f, 1.12157f, 1.84006f, 2.19765f, 3.22486f, 2.34727f, 1.96785f,
  1.43621f, 1.29783f, 2.81680f, 1.88507f, 1.99120f, 1.93567f, 1.68095f, 2.60358f,
  3.08976f, 2.98307f, 2.71773f, 1.68559f, 1.69745f, 0.96275f, 5.05729f, 3.31400f,
  1.58970f, 1.90072f, 1.74750f, 1.90494f, 3.50463f, 1.78908f, 2.42983f, 4.13734f,
  2.71335f, 2.13357f, 2.36755f, 2.63225f, 2.00407f, 1.78098f, 3.15883f, 2.05270f,
};
static inline int sib_is_sibilant(int id) {   /* s=31 z=38 ʃ=96 ʒ=108 */
    return id == 31 || id == 38 || id == 96 || id == 108;
}
/* xorshift32 + approx-normal (sum of 12 uniforms - 6 ~ N(0,1)); deterministic
 * seed so runs are reproducible (no Date/rand on this build). */
static uint32_t g_sib_rng = 0x9e3779b9u;
static inline float sib_randn(void) {
    float acc = 0.0f;
    for (int k = 0; k < 12; k++) {
        g_sib_rng ^= g_sib_rng << 13; g_sib_rng ^= g_sib_rng >> 17; g_sib_rng ^= g_sib_rng << 5;
        acc += (float)(g_sib_rng & 0xFFFFFF) / (float)0x1000000;   /* U[0,1) */
    }
    return acc - 6.0f;
}
#define MAX_FRAMES (APCM_MAX / HOP + 16)
static signed char g_frame_sib[MAX_FRAMES];   /* 1 at sibilant frames, filled at token expansion */

static unsigned char *g_front, *g_dec;
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
static inline float fast_exp(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    float y = x * 1.44269504088896f; /* log2(e) */
    float fi = floorf(y);
    float f = y - fi;
    /* 2^f on [0,1), minimax-ish degree 4 */
    float pf = 1.0f + f * (0.69314718f + f * (0.24022651f + f * (0.05550411f + f * 0.00961813f)));
    union { float fv; int iv; } u;
    u.iv = (int)((fi + 127.0f) * 8388608.0f); /* exponent bits */
    return u.fv * pf;
}
/* tanh via Pade 3/2, clamped: err <2e-3 on the range GELU feeds it */
static inline float fast_tanh(float y) {
    if (y > 4.97f) return 1.0f;
    if (y < -4.97f) return -1.0f;
    float y2 = y * y;
    return y * (27.0f + y2) * fast_recip(27.0f + 9.0f * y2);
}
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + fast_tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}
static inline float silu(float x) { return x * fast_recip(1.0f + fast_exp(-x)); }
static inline float fast_rsqrt(float x) {
    union { float fv; int iv; } u;
    u.fv = x;
    u.iv = 0x5f3759df - (u.iv >> 1);
    float r = u.fv;
    r = r * (1.5f - 0.5f * x * r * r);
    r = r * (1.5f - 0.5f * x * r * r);
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

typedef void (*par_fn)(int lo, int hi, void *ctx);
#ifdef ESP_PLATFORM
static volatile par_fn g_par_fn;
static void *volatile g_par_ctx;
static volatile int g_par_lo, g_par_hi;
static volatile int g_par_done;
static TaskHandle_t g_worker_handle;
static void sn_worker(void *arg) {
    (void)arg;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);  /* blocked: zero bus traffic */
        __sync_synchronize();
        g_par_fn(g_par_lo, g_par_hi, (void *)g_par_ctx);
        __sync_synchronize();
        g_par_done = 1;
    }
}
static int g_worker_up = 0;
static void par_run(par_fn f, int n, void *ctx) {
    if (!g_worker_up || n < 8) { f(0, n, ctx); return; }
    int mid = n / 2;
    g_par_fn = f; g_par_ctx = ctx; g_par_lo = mid; g_par_hi = n; g_par_done = 0;
    __sync_synchronize();
    xTaskNotifyGive(g_worker_handle);
    f(0, mid, ctx);
    while (!g_par_done) { }
    __sync_synchronize();
}
#else
static void par_run(par_fn f, int n, void *ctx) { f(0, n, ctx); }
#endif
#ifdef ESP_PLATFORM
#define SCR() (&g_scr[xPortGetCoreID()])
#else
#define SCR() (&g_scr[0])
#endif

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

/* second arena in PSRAM for FLOAT activation matrices only. These are consumed
 * exclusively by scalar gathers (q1x1_col/qkconv_col copy columns into internal
 * scratch before any SIMD), so cached-PSRAM residency is safe. Weights and the
 * pre-quantized c8 (fed directly to the PIE asm) must stay internal SRAM. */
#ifdef ESP_PLATFORM
static unsigned char *g_aps;
static size_t g_aps_cap, g_aps_top;
static void *aps(size_t n) {
    g_aps_top = (g_aps_top + 15) & ~(size_t)15;
    if (g_aps_top + n > g_aps_cap) {
        printf("PSRAM ARENA OOM %zu (top %zu cap %zu)\n", n, g_aps_top, g_aps_cap);
        abort();
    }
    void *p = g_aps + g_aps_top;
    g_aps_top += n;
    return p;
}
#else
static void *aps(size_t n) { return aa(n); }
#endif

/* allocate both arenas once. On ESP this MUST run BEFORE wifi_init: the engine
 * takes the largest internal block minus a fixed wifi reservation, so wifi can
 * still bring up its (internal-RAM) buffers afterwards. */
static void arena_init(void) {
#ifdef ESP_PLATFORM
    /* reservation covers what allocates internal AFTER this: httpd task stack
     * (28K, for espeak's clause translator) + wifi runtime (~15K; most wifi/lwip
     * buffers go to PSRAM). Trimmed from 60K once espeak forced the budget tight. */
    g_arena_cap = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) - 42 * 1024;
#else
    g_arena_cap = 320 * 1024;
#endif
    printf("arena: %zu bytes\n", g_arena_cap);
    unsigned char *arena_raw = (unsigned char *)xmalloc(g_arena_cap);
    /* heap_caps_malloc only guarantees 8-byte alignment; the SIMD dot needs 16 */
    g_arena = (unsigned char *)(((uintptr_t)arena_raw + 15) & ~(uintptr_t)15);
    g_arena_cap -= (size_t)(g_arena - arena_raw);
#ifdef ESP_PLATFORM
    g_aps_cap = 4 * 1024 * 1024;
    g_aps = (unsigned char *)heap_caps_malloc(g_aps_cap, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!g_aps) { printf("FATAL: no PSRAM for activation arena\n"); abort(); }
    printf("psram arena: %zu bytes\n", g_aps_cap);
#endif
}

static int32_t dot_s8(const signed char *a, const signed char *b, int n) {
#ifdef ESP_PLATFORM
    if (is_sram(b)) return esp_nn_dot_s8_aligned_esp32s3(a, b, n);
#endif
    int32_t acc = 0;
    int32_t r = 0;
    for (int i = 0; i < n; i++) r += (int32_t)a[i] * (int32_t)b[i];
    acc = r;
    return acc;
}

/* copy a weight block into aligned SRAM (or plain RAM on host) */
static const signed char *res_copy(const signed char *src, size_t bytes, signed char *dst) {
    memcpy(dst, src, bytes);
    return dst;
}

#ifdef ESP_PLATFORM
extern void sn_matvec_s8_c3(const int8_t *act, const int8_t *w, int32_t *out, int rows);
extern void sn_matvec_s8_c5(const int8_t *act, const int8_t *w, int32_t *out, int rows);
extern void sn_matvec_s8_g(const int8_t *act, const int8_t *w, int32_t *out, int rows, int chunks);
#endif
/* out[r] = dot(act, w + r*len) into S->acc32; w SRAM-resident, 16B tail pad */
static void matvec_s8(SnScratch *S, const signed char *act, const signed char *w, int rows, int len) {
#ifdef ESP_PLATFORM
    if (is_sram(w)) {
        int chunks = len >> 4;
        if (chunks == 3) { sn_matvec_s8_c3(act, w, S->acc32, rows); return; }
        if (chunks == 5) { sn_matvec_s8_c5(act, w, S->acc32, rows); return; }
        sn_matvec_s8_g(act, w, S->acc32, rows, chunks);
        return;
    }
#endif
    for (int r = 0; r < rows; r++) S->acc32[r] = dot_s8(act, w + (size_t)r * len, len);
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
typedef struct {
    const signed char *c0w, *c1w;
    const float *c0s, *c0b, *c1s, *c1b;
    int c0pad, c1pad, ch, T, K;
    float bscale, *x, *tmp;
} RbCtx;
static void rb_conv0_range(int lo, int hi, void *vc) {
    RbCtx *c = (RbCtx *)vc;
    for (int t = lo; t < hi; t++)
        qkconv_col(c->c0w, c->c0s, c->c0b, c->c0pad, c->x, c->T, t, c->tmp, c->T, t, c->ch, c->ch, c->K);
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
}
typedef struct {
    const signed char *fr, *fo, *pw0, *pw1;
    const float *fr_s, *fr_b, *fo_s, *fo_b, *pw0_s, *pw0_b, *pw1_s, *pw1_b;
    const signed char *c8;
    const float *cscale;
    float *tile_f2, *tile_c, *tile_h2, *x;
    int t0, TL, T;
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
            for (int ch = 0; ch < FSD_DIM; ch++) {
                const float *wr = c->dww + (size_t)ch * FSD_DW_KERNEL;
                const float *xr = c->x + (size_t)ch * c->T + col - half;
                float a2 = c->dwb[ch] + wr[0] * xr[0] + wr[1] * xr[1] + wr[2] * xr[2]
                         + wr[3] * xr[3] + wr[4] * xr[4] + wr[5] * xr[5] + wr[6] * xr[6];
                c->tile_c[(size_t)ch * c->TL + t] = (a2 - c->mean) * c->ninv * c->nw[ch] + c->nb[ch];
            }
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
        q1x1_q8col(c->fr, c->fr_s, c->fr_b, Q8_B0_FILMR_N16,
                   c->c8 + (size_t)(c->t0 + t) * Q8_PRE_N16, c->cscale[c->t0 + t],
                   red, c->TL, t, FSD_FILM_RANK);
        q1x1_col(c->fo, c->fo_s, c->fo_b, Q8_B0_FILMO_N16,
                 red, c->TL, t, fo, c->TL, t, FSD_FILM_RANK, 2 * FSD_DIM);
        for (int ch = 0; ch < FSD_DIM; ch++) {
            size_t li = (size_t)ch * c->TL + t;
            c->tile_c[li] = c->tile_c[li] * (1.0f + fo[li]) + fo[(size_t)(FSD_DIM + ch) * c->TL + t];
        }
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
    }
}

static void resblock(const signed char *c0w, const float *c0s, const float *c0b, int c0pad,
                     const signed char *c1w, const float *c1s, const float *c1b, int c1pad,
                     float bscale, float *x, float *tmp, int ch, int T, int K) {
    RbCtx c = {c0w, c1w, c0s, c0b, c1s, c1b, c0pad, c1pad, ch, T, K, bscale, x, tmp};
    long long _r0 = NOW_US();
    par_run(rb_conv0_range, T, &c);
    long long _r1 = NOW_US();
    g_prof[2] += _r1 - _r0;
    par_run(rb_silu_range, T, &c);
    long long _r2 = NOW_US();
    g_prof[4] += _r2 - _r1;
    par_run(rb_conv1_range, T, &c);
    g_prof[7] += NOW_US() - _r2;
}

typedef struct { const float *hp1; float *fre, *fim; } SpecCtx;
static void spec_range(int lo, int hi, void *vc) {
    SpecCtx *c = (SpecCtx *)vc;
    for (int k = lo; k < hi; k++) {
        float m = c->hp1[k];
        if (m < -12.0f) m = -12.0f;
        if (m > 8.0f) m = 8.0f;
        m = EXPF_M(m);
        if (m < 1e-7f) m = 1e-7f;
        float pr = c->hp1[FSD_BINS + k], pi = c->hp1[2 * FSD_BINS + k];
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

static void *xload(const char *dir, const char *name, size_t *bytes) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *fh = fopen(path, "rb");
    if (!fh) { fprintf(stderr, "missing %s\n", path); exit(1); }
    fseek(fh, 0, SEEK_END);
    long sz = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    void *buf = malloc((size_t)sz);
    if (!buf || fread(buf, 1, (size_t)sz, fh) != (size_t)sz) exit(1);
    fclose(fh);
    if (bytes) *bytes = (size_t)sz;
    return buf;
}

typedef struct { const signed char *qbuf; const signed char *w; } HeadCtx;
static void head_range(int lo, int hi, void *ctx) {
    HeadCtx *c = (HeadCtx *)ctx;
    for (int r = lo; r < hi; r++)
        g_acc_head[r] = dot_s8(c->qbuf, c->w + (size_t)r * Q8_HEAD_OUT_N16, Q8_HEAD_OUT_N16);
}

static int run_e2e(int use_golden_c, const char *dir) {
    SnScratch *S = SCR();
    size_t nb;
#ifdef ESP_PLATFORM
    (void)dir;
    g_front = (unsigned char *)front_start;
    g_dec = (unsigned char *)dec_start;
    const int *ids = (const int *)ids_start;
    int n_tokens = (int)((size_t)(ids_end - ids_start) / 4);
    const int *g_durs = (const int *)durs_start;
    const float *g_c = NULL;
    const float *g_audio = (const float *)gold_start;
    size_t n_gold = (size_t)(gold_end - gold_start) / 4;
    if (g_ids_ovr) {            /* dashboard utterance: predicted durs, no golden */
        ids = g_ids_ovr;
        n_tokens = g_n_ovr;
        g_durs = NULL;
        g_audio = NULL;
        n_gold = (size_t)-1;
    }
    if (use_golden_c) return 1; /* no c golden embedded */
#else
    g_front = (unsigned char *)xload(dir, "front_q8.bin", NULL);
    g_dec = (unsigned char *)xload(dir, "model_q8.bin", NULL);
    const int *ids = (const int *)xload(dir, "e2e_ids.bin", &nb);
    int n_tokens = (int)(nb / 4);
    const int *g_durs = (const int *)xload(dir, "e2e_durs.bin", NULL);
    const float *g_c = (const float *)xload(dir, "e2e_c.bin", NULL);
    float *g_audio = (float *)xload(dir, "e2e_audio.bin", &nb);
    size_t n_gold = nb / 4;
#endif
    printf("tokens=%d\n", n_tokens);
    if (!g_arena) arena_init();   /* host path; on ESP app_main calls it pre-wifi */
    g_arena_top = 0;
#ifdef ESP_PLATFORM
    g_aps_top = 0;
#endif
    long long t_start = NOW_US();

    /* ---------- duration student ---------- */
    int H = FRONT_DUR_HIDDEN;
    const float *demb = FF(FOFF_DUR_EMB_F32);
    size_t mark_dur = g_arena_top;
    float *dh = (float *)aps((size_t)H * n_tokens * 4);
    float *dtmp = (float *)aps((size_t)H * n_tokens * 4);
    /* SRAM-resident duration weights (were scalar-from-flash: 247 ms!) */
    signed char *r_dproj = (signed char *)aa((size_t)H * FRONT_DUR_PROJ_N16 + 16);
    memcpy(r_dproj, FQ(FOFF_DUR_PROJ_W8), (size_t)H * FRONT_DUR_PROJ_N16);
    signed char *r_dc0 = (signed char *)aa((size_t)H * FRONT_DUR_B0_C0_N16 + 16);
    signed char *r_dc1 = (signed char *)aa((size_t)H * FRONT_DUR_B0_C1_N16 + 16);
    signed char *r_dout = (signed char *)aa((size_t)FRONT_DUR_OUT_N16 + 16);
    memcpy(r_dout, FQ(FOFF_DUR_OUT_W8), (size_t)FRONT_DUR_OUT_N16);
    for (int t = 0; t < n_tokens; t++) {
        /* The duration student's embedding has only FRONT_DUR_VOCAB(127) rows, but
         * the acoustic student + Kristin map span 157 (e.g. 'ᵻ'=128, the common
         * reduced vowel). Ids >=127 are valid for SOUND (acoustic keeps them) but
         * OOB for TIMING -- fall back to schwa(59), whose duration matches a
         * reduced vowel. Only the duration lookup is remapped; acoustic is exact. */
        int did = ids[t] < FRONT_DUR_VOCAB ? ids[t] : 59;
        for (int h = 0; h < H; h++) S->gather[h] = demb[(size_t)did * H + h];
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
        memcpy(r_dc0, FQ(dur_off[b][0]), (size_t)H * FRONT_DUR_B0_C0_N16);
        memcpy(r_dc1, FQ(dur_off[b][3]), (size_t)H * FRONT_DUR_B0_C1_N16);
        resblock(r_dc0, FF(dur_off[b][1]), FF(dur_off[b][2]), FRONT_DUR_B0_C0_N16,
                 r_dc1, FF(dur_off[b][4]), FF(dur_off[b][5]), FRONT_DUR_B0_C1_N16,
                 *FF(dur_off[b][6]), dh, dtmp, H, n_tokens, FRONT_DUR_KERNEL);
    }
    static int durs[1024];
    if (n_tokens > 1024) { printf("too many tokens\n"); return 1; }
    int dur_exact = 0;
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
            if (g_durs && durs[t] == g_durs[t]) dur_exact++;
        }
    }
    printf("durations: %d/%d exact vs float golden\n", dur_exact, n_tokens);

    g_arena_top = mark_dur;
    long long t_durdone = NOW_US();
    int pred_T = 0;
    for (int t = 0; t < n_tokens; t++) pred_T += durs[t];
    /* golden runs use GOLDEN durations so frames align exactly for corr;
     * dashboard runs keep the predicted durations (that's the product path). */
    if (g_durs) {
        printf("frames: predicted %d, golden %zu\n", pred_T, n_gold / HOP);
        for (int t = 0; t < n_tokens; t++) durs[t] = g_durs[t];
    }
    int T = 0;
    for (int t = 0; t < n_tokens; t++) T += durs[t];
#ifdef ESP_PLATFORM
    /* will it fit? Float activations live in the PSRAM arena; internal SRAM holds
     * c8/cscale (~52B/frame) + fixed weight residency (~140K). Reject, don't abort. */
    {
        /* fixed internal residency dropped ~47K by moving pw banks to PSRAM */
        /* fixed internal = the acoustic peak (rbuf_kc0/kc1 ~23K) + margin; measured
         * from an ARENA OOM: 391 frames used 43408 => fixed ~23K, not 45K. The 45K
         * over-estimate was capping utterances at ~3s when ~7s actually fits. */
        size_t need_int = ((size_t)Q8_PRE_N16 + 4) * (size_t)T + 26000;
        size_t need_ps = ((size_t)FSD_DIM * 4 + (size_t)FRONT_AC_HIDDEN * 4) * (size_t)T
                         + (size_t)(FRONT_AC_HIDDEN + 2 * FRONT_DUR_HIDDEN) * 4 * n_tokens + 8192;
        if ((long long)T * HOP + NFFT > APCM_MAX) {
            printf("REJECT: T=%d exceeds %ds PCM buffer\n", T, APCM_MAX / 22050);
            g_last_T = -T;
            return 2;
        }
        if (need_int > g_arena_cap || need_ps > g_aps_cap) {
            printf("REJECT: T=%d needs int %zu/%zu psram %zu/%zu\n",
                   T, need_int, g_arena_cap, need_ps, g_aps_cap);
            g_last_T = -T;   /* signal "too long" with the frame count */
            return 2;
        }
    }
    g_last_T = T;
    g_last_ms = (int)((long long)T * HOP * 1000 / 22050);
#endif

    /* ---------- acoustic ---------- */
    int AH = FRONT_AC_HIDDEN;
    const float *aemb = FF(FOFF_AC_EMB_F32);
    float maxdur = 1.0f;
    for (int t = 0; t < n_tokens; t++)
        if ((float)durs[t] > maxdur) maxdur = (float)durs[t];
    /* persistent across both phases: trunk buffer + pre-quantized code */
    float *x = (float *)aps((size_t)FSD_DIM * T * 4);
    signed char *c8 = (signed char *)aa((size_t)T * Q8_PRE_N16); /* frame-major, padded */
    float *cscale = (float *)aa((size_t)T * 4);
    size_t mark_front = g_arena_top;
    float *ax = (float *)aps((size_t)AH * T * 4);
    float *th = (float *)aps((size_t)AH * n_tokens * 4);
    signed char *rbuf_kc0 = (signed char *)aa((size_t)FRONT_AC_HIDDEN * FRONT_AC_TB0_C0_N16 + 16);
    signed char *rbuf_kc1 = (signed char *)aa((size_t)FRONT_AC_HIDDEN * FRONT_AC_TB0_C1_N16 + 16);
    float *ttmp = ax; /* token temp borrows the frame buffer */
    res_copy(FQ(FOFF_AC_TPROJ_W8), (size_t)AH * FRONT_AC_TPROJ_N16, rbuf_kc0);
    for (int t = 0; t < n_tokens; t++) {
        for (int h = 0; h < AH; h++) S->gather[h] = aemb[(size_t)ids[t] * AH + h];
        S->gather[AH] = (n_tokens > 1) ? (float)t / (float)(n_tokens - 1) : 0.0f;
        S->gather[AH + 1] = log1pf((float)durs[t]) / log1pf(maxdur);
        float s_act = quant_gather(S, AH + 2, FRONT_AC_TPROJ_N16);
        const signed char *w8 = rbuf_kc0; /* res_copy'd below before loop */
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
        resblock(c0, FF(tb_off[b][1]), FF(tb_off[b][2]), FRONT_AC_TB0_C0_N16,
                 c1, FF(tb_off[b][4]), FF(tb_off[b][5]), FRONT_AC_TB0_C1_N16,
                 *FF(tb_off[b][6]), th, ttmp, AH, n_tokens, FRONT_AC_KERNEL);
    }

    long long t_tokdone = NOW_US();
    /* expand tokens -> frames on the fly (no [AH+3, T] buffer) */
    float *atmp = x; /* borrow the decoder buffer: AH*T <= FSD_DIM*T */
    {
        const signed char *w8 = res_copy(FQ(FOFF_AC_FPROJ_W8),
                                         (size_t)AH * FRONT_AC_FPROJ_N16, rbuf_kc0);
        const float *sc = FF(FOFF_AC_FPROJ_SCALE);
        const float *bi = FF(FOFF_AC_FPROJ_BIAS);
        int f = 0;
        for (int tok = 0; tok < n_tokens; tok++) {
            int sib = sib_is_sibilant(ids[tok]);
            for (int d = 0; d < durs[tok]; d++, f++) {
                if (f < MAX_FRAMES) g_frame_sib[f] = (signed char)sib;
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

    long long t_fprojdone = NOW_US();
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
        resblock(c0, FF(fb_off[b][1]), FF(fb_off[b][2]), FRONT_AC_FB0_C0_N16,
                 c1, FF(fb_off[b][4]), FF(fb_off[b][5]), FRONT_AC_FB0_C1_N16,
                 *FF(fb_off[b][6]), ax, atmp, AH, T, FRONT_AC_KERNEL);
    }
    long long t_fbdone = NOW_US();
    {
        const signed char *ow = res_copy(FQ(FOFF_AC_OUT_W8),
                                         (size_t)FSD_CODE_DIM * FRONT_AC_OUT_N16, rbuf_kc1);
        float ccol[64];
        for (int t = 0; t < T; t++) {
            q1x1_col(ow, FF(FOFF_AC_OUT_SCALE), FF(FOFF_AC_OUT_BIAS),
                     FRONT_AC_OUT_N16, ax, T, t, ccol, 1, 0, AH, FSD_CODE_DIM);
            /* sibilant hiss restoration: inject per-channel noise at /s z ʃ ʒ/
             * frames (see SIB_TEA_STD block). Before quant, same point as host. */
            if (SIB_BETA > 0.0f && t < MAX_FRAMES && g_frame_sib[t])
                for (int i = 0; i < FSD_CODE_DIM; i++)
                    ccol[i] += sib_randn() * SIB_TEA_STD[i] * SIB_BETA;
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


    if (g_c) {
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        size_t n = (size_t)FSD_CODE_DIM * T;
        for (size_t i = 0; i < n; i++) {
            size_t ch = i / T, tt = i % T;
            float a2 = (float)c8[(size_t)tt * Q8_PRE_N16 + ch] * cscale[tt];
            float b2 = g_c[i];
            sa += a2; sb += b2; saa += (double)a2 * a2; sbb += (double)b2 * b2;
            sab += (double)a2 * b2;
        }
        double cov = sab - sa * sb / n;
        printf("c: corr %.6f\n", cov / sqrt((saa - sa * sa / n) * (sbb - sb * sb / n) + 1e-30));
    }

    /* ---------- decoder: two-pass norm, tiled, fused head, ring OLA ---------- */
    long long t_front = NOW_US();
    if (use_golden_c && g_c) {
        /* re-quantize the golden float c into c8 (host-only debug path) */
        for (int tt = 0; tt < T; tt++) {
            float m = 0.0f;
            for (int i = 0; i < FSD_CODE_DIM; i++) {
                float v = fabsf(g_c[(size_t)i * T + tt]);
                if (v > m) m = v;
            }
            float s = (m > 0.0f) ? m / 127.0f : 1.0f;
            float inv = fast_recip(s);
            signed char *dst = c8 + (size_t)tt * Q8_PRE_N16;
            for (int i = 0; i < FSD_CODE_DIM; i++)
                dst[i] = (signed char)fast_round(g_c[(size_t)i * T + tt] * inv);
            cscale[tt] = s;
        }
        printf("(decoder running on golden c)\n");
    }
    /* decoder weight residency (freed flash traffic is the 4.5x lever) */
    /* decoder phase: ax/th/rbuf_kc* are dead; give the head/pw residency
     * its own arena block (arena still has headroom) */
    /* head-out (74K) doesn't fit internal SRAM alongside wifi, and PIE from
     * PSRAM reads garbage (measured corr 0.007). Compromise: PSRAM residency with
     * SCALAR reads (correct, and ~4x flash bandwidth), split across both cores. */
    signed char *rbuf_ho = (signed char *)aps((size_t)FSD_BINS * 3 * Q8_HEAD_OUT_N16 + 16);
    const signed char *ho_res = res_copy(DQ(Q8OFF_HEAD_OUT_W8), (size_t)FSD_BINS * 3 * Q8_HEAD_OUT_N16, rbuf_ho);
    signed char *rbuf_hi = (signed char *)aa((size_t)FSD_HEAD_RANK * Q8_HEAD_IN_N16 + 16);
    const signed char *hi_res = res_copy(DQ(Q8OFF_HEAD_IN_W8), (size_t)FSD_HEAD_RANK * Q8_HEAD_IN_N16, rbuf_hi);
    signed char *rbuf_pre = (signed char *)aa((size_t)FSD_DIM * Q8_PRE_N16 + 16);
    const signed char *pre_res = res_copy(DQ(Q8OFF_PRE_W8), (size_t)FSD_DIM * Q8_PRE_N16, rbuf_pre);
    /* pw banks (~47K) -> PSRAM so the internal arena is freed for a longer c8
     * (c8 grows 48 B/frame; this is what caps utterance length). matvec_s8 sees
     * a non-SRAM pointer and auto-uses the scalar dot (both cores via par_run),
     * so it's correct -- costs some decode speed, which we have margin for. */
    signed char *rbuf_pw0 = (signed char *)aps((size_t)FSD_PW_HIDDEN * Q8_B0_PW0_N16 + 16);
    signed char *rbuf_pw1 = (signed char *)aps((size_t)FSD_DIM * Q8_B0_PW1_N16 + 16);
    signed char *rbuf_fr = (signed char *)aa((size_t)FSD_FILM_RANK * Q8_B0_FILMR_N16 + 16);
    signed char *rbuf_fo2 = (signed char *)aa((size_t)2 * FSD_DIM * Q8_B0_FILMO_N16 + 16);
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

    /* Float tile scratch (~17.7K), scalar-accessed only. In the PSRAM arena (aps):
     * frees that internal RAM for the decoder arena (espeak squeezed internal), and
     * the per-run arena reset means no leak. Was static-xmalloc16-internal, which
     * OOM16'd on the first real speak once the golden boot run stopped pre-allocating
     * it (golden now rejects for lack of arena). */
    float *dwcol = (float *)aps((size_t)FSD_DIM * 4);
    float *tile_c = (float *)aps((size_t)FSD_DIM * TILE * 4);
    float *tile_h2 = (float *)aps((size_t)FSD_PW_HIDDEN * TILE * 4);
    float *tile_f2 = (float *)aps((size_t)(FSD_FILM_RANK + 2 * FSD_DIM) * TILE * 4);

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
                          t0, TL, T, bscale,
                          dww, dwb, nw, nb2, mean, ninv,
                          (const float (*)[FSD_DIM])halo};
            par_run(tile_dw_range, TL, &tc);      /* barrier: x reads done */
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
    /* float accumulators: doubles are software-emulated on LX7 (~300 ms
     * of pure instrumentation cost, measured). Deployment emits PCM here. */
    float sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    size_t n_out = 0;
    int emitted = 0;
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
#ifdef ESP_PLATFORM
        if (is_sram(ho_res)) {
            sn_matvec_s8_c3(S->qbuf, ho_res, g_acc_head, FSD_BINS * 3);
        } else {
            HeadCtx hctx = {S->qbuf, ho_res};
            par_run(head_range, FSD_BINS * 3, &hctx);   /* both cores, scalar */
        }
#else
        for (int r = 0; r < FSD_BINS * 3; r++)
            g_acc_head[r] = dot_s8(S->qbuf, ho_res + (size_t)r * Q8_HEAD_OUT_N16, Q8_HEAD_OUT_N16);
#endif
        for (int o = 0; o < FSD_BINS * 3; o++)
            hp1[o] = s2 * hos[o] * (float)g_acc_head[o] + hob[o];
        g_prof[1] += NOW_US() - _h0;
        long long _sp0 = NOW_US();
        SpecCtx sctx = {hp1, fre, fim};
        par_run(spec_range, FSD_BINS, &sctx);
        if (0)
        for (int k = 0; k < FSD_BINS; k++) {
            float m = hp1[k];
            if (m < -12.0f) m = -12.0f;
            if (m > 8.0f) m = 8.0f;
            m = EXPF_M(m);
            if (m < 1e-7f) m = 1e-7f;
            float pr = hp1[FSD_BINS + k], pi = hp1[2 * FSD_BINS + k];
#ifdef FSD_FAST_MATH
            float n2 = pr * pr + pi * pi;
            float r = (n2 > 1e-12f) ? fast_rsqrt(n2) : 0.0f;
            fre[k] = m * pr * r;
            fim[k] = m * pi * r;
#else
            float nrm = sqrtf(pr * pr + pi * pi);
            if (nrm < 1e-6f) nrm = 1e-6f;
            fre[k] = m * pr / nrm;
            fim[k] = m * pi / nrm;
#endif
        }
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
            if (out_idx >= 0 && out_idx < T * HOP && (size_t)out_idx < n_gold) {
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
                aout_push(sample);   /* stream to speaker; paces loop to realtime */
                if (g_audio) {
                    float gg = g_audio[out_idx];
                    sa += sample; sb += gg;
                    saa += sample * sample; sbb += gg * gg;
                    sab += sample * gg;
                }
                n_out++;
                emitted++;
            }
            ring[s & (NFFT - 1)] = 0.0f; /* slot consumed; next use is s+NFFT */
        }
        g_emit_us += NOW_US() - _e0;
    }
    long long t_done = NOW_US();
    double audio_ms = (double)T * HOP * 1000.0 / 22050.0;
    if (!g_audio) {   /* dashboard utterance: no golden to compare against */
        printf("RESULT audio %.1f ms | synth+play wall %.1f ms | %d frames\n",
               audio_ms, (t_done - t_start) / 1000.0, T);
        return 0;
    }
    double cov = (double)sab - (double)sa * sb / (double)n_out;
    double cr = cov / sqrt(((double)saa - (double)sa * sa / (double)n_out) * ((double)sbb - (double)sb * sb / (double)n_out) + 1e-30);
    printf("FRONTPROF dur %.1f | tok %.1f | fproj %.1f | fblocks %.1f | acout %.1f (ms)\n",
           (t_durdone - t_start) / 1000.0, (t_tokdone - t_durdone) / 1000.0,
           (t_fprojdone - t_tokdone) / 1000.0, (t_fbdone - t_fprojdone) / 1000.0,
           (t_front - t_fbdone) / 1000.0);
    printf("RESULT front %.1f ms | decoder %.1f ms | total %.1f ms\n",
           (t_front - t_start) / 1000.0, (t_done - t_front) / 1000.0,
           (t_done - t_start) / 1000.0);
    printf("RESULT audio %.1f ms -> %.2fx realtime, corr %.6f (E2E)\n",
           audio_ms, (t_done - t_start) / 1000.0 / audio_ms, cr);
    printf("AMP out/gold %.4f\n", sb != 0.0 ? sqrt(saa / sbb) : 0.0);
    printf("PROF fft %lld | film %lld | head %lld | tiles %lld | specprep %lld | ola %lld | emit %lld (us)\n",
           g_prof[5], g_prof[6], g_prof[1], g_prof[0], g_prof[3], g_ola_us, g_emit_us);
    printf(cr > 0.97 ? "PASS\n" : "FAIL\n");
    return cr > 0.97 ? 0 : 1;
}

#ifdef ESP_PLATFORM
/* ================= WiFi dashboard (frontend lifted from the u600 dashboard;
 * same Kristin Piper phoneme-ID map -- FRONT_DUR_VOCAB is 127 here too) ===== */
#include <ctype.h>
#include <stdbool.h>
#include "freertos/event_groups.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "esp_http_server.h"
#include "esp_system.h"
#include "esp_g2p.h"   /* on-chip espeak-ng G2P (text -> Kristin ids) */
/* Set your WiFi at build time: idf.py -DWIFI_SSID=... -DWIFI_PASS=... , or edit
 * here. Kept out of source so credentials never land in the repo. */
#ifndef WIFI_SSID
#define WIFI_SSID      "YOUR_WIFI_SSID"
#endif
#ifndef WIFI_PASS
#define WIFI_PASS      "YOUR_WIFI_PASSWORD"
#endif
#define WIFI_CONNECTED BIT0
#define MAX_IDS        600   /* espeak sentences run long; ~1 min of speech */
#define MAX_TEXT       512
#define MAX_BODY       6000  /* holds ~600 urlencoded ids */

#define ID_PAD 0
#define ID_BOS 1
#define ID_EOS 2
#define ID_SPACE 3
#define ID_DOT 10
#define ID_A 14
#define ID_B 15
#define ID_D 17
#define ID_E 18
#define ID_F 19
#define ID_H 20
#define ID_I 21
#define ID_J 22
#define ID_K 23
#define ID_L 24
#define ID_M 25
#define ID_N 26
#define ID_O 27
#define ID_P 28
#define ID_R 30
#define ID_S 31
#define ID_T 32
#define ID_U 33
#define ID_V 34
#define ID_W 35
#define ID_Z 38
#define ID_AE 39
#define ID_DH 41
#define ID_NG 44
#define ID_AH0 50
#define ID_AA 51
#define ID_AO 54
#define ID_ER 60
#define ID_EH 61
#define ID_G 66
#define ID_IH 74
#define ID_RR 88
#define ID_SH 96
#define ID_UH 100
#define ID_AH 102
#define ID_ZH 108
#define ID_STRESS 120
#define ID_LEN 122
#define ID_TH 126

static EventGroupHandle_t g_wifi_events;

static const int DEMO_IDS[] = {
  1,0,41,0,74,0,31,0,3,0,88,0,120,0,102,0,26,0,38,0,3,0,121,0,
  54,0,26,0,3,0,50,0,3,0,126,0,88,0,120,0,21,0,122,0,3,0,17,0,
  120,0,51,0,122,0,24,0,60,0,3,0,32,0,96,0,120,0,74,0,28,0,10,0,2
};

static int add_id(int *out, int *n, int max, int id) {
  if (*n + 2 > max) return -1;
  out[(*n)++] = id;
  out[(*n)++] = ID_PAD;
  return 0;
}
static int add_phone(int *out, int *n, int max, int id) { return add_id(out, n, max, id); }
static int add_space(int *out, int *n, int max) {
  if (*n > 2 && out[*n - 2] != ID_SPACE) return add_id(out, n, max, ID_SPACE);
  return 0;
}

static void normalize_text(const char *in, char *out, int max) {
  int n = 0;
  bool last_space = false;
  for (int i = 0; in[i] && n + 1 < max; i++) {
    unsigned char c = (unsigned char)in[i];
    if (isalnum(c)) {
      out[n++] = (char)tolower(c);
      last_space = false;
    } else if (isspace(c) || c == '-' || c == '_') {
      if (!last_space && n > 0) out[n++] = ' ';
      last_space = true;
    } else if (c == '.' || c == '!' || c == '?') {
      if (!last_space && n > 0) out[n++] = ' ';
      last_space = true;
    }
  }
  while (n > 0 && out[n - 1] == ' ') n--;
  out[n] = 0;
}

static int emit_word_approx(const char *w, int len, int *out, int *n, int max) {
  int i = 0;
  while (i < len) {
    char c = w[i], c2 = (i + 1 < len) ? w[i + 1] : 0;
    char c3 = (i + 2 < len) ? w[i + 2] : 0;
    if (c == 't' && c2 == 'h') {
      add_phone(out, n, max, (i == 0 && len <= 5) ? ID_DH : ID_TH); i += 2; continue;
    }
    if (c == 's' && c2 == 'h') { add_phone(out, n, max, ID_SH); i += 2; continue; }
    if (c == 'c' && c2 == 'h') { add_phone(out, n, max, ID_T); add_phone(out, n, max, ID_SH); i += 2; continue; }
    if (c == 'n' && c2 == 'g') { add_phone(out, n, max, ID_NG); i += 2; continue; }
    if (c == 'p' && c2 == 'h') { add_phone(out, n, max, ID_F); i += 2; continue; }
    if (c == 'q' && c2 == 'u') { add_phone(out, n, max, ID_K); add_phone(out, n, max, ID_W); i += 2; continue; }
    if (c == 'c' && c2 == 'k') { add_phone(out, n, max, ID_K); i += 2; continue; }
    if ((c == 'e' && (c2 == 'e' || c2 == 'a')) || (c == 'i' && c2 == 'e')) {
      add_phone(out, n, max, ID_I); add_phone(out, n, max, ID_LEN); i += 2; continue;
    }
    if (c == 'o' && c2 == 'o') { add_phone(out, n, max, ID_UH); i += 2; continue; }
    if ((c == 'o' && c2 == 'u') || (c == 'o' && c2 == 'w')) {
      add_phone(out, n, max, ID_A); add_phone(out, n, max, ID_UH); i += 2; continue;
    }
    if ((c == 'a' && (c2 == 'i' || c2 == 'y')) || (c == 'e' && c2 == 'y')) {
      add_phone(out, n, max, ID_E); i += 2; continue;
    }
    if (c == 'i' && c2 == 'g' && c3 == 'h') {
      add_phone(out, n, max, ID_A); add_phone(out, n, max, ID_IH); i += 3; continue;
    }
    if ((c == 'e' || c == 'i' || c == 'u') && c2 == 'r') { add_phone(out, n, max, ID_ER); i += 2; continue; }
    if (c == 'a' && c2 == 'r') { add_phone(out, n, max, ID_AA); add_phone(out, n, max, ID_RR); i += 2; continue; }
    if (c == 'o' && c2 == 'r') { add_phone(out, n, max, ID_AO); add_phone(out, n, max, ID_RR); i += 2; continue; }
    switch (c) {
      case 'a': add_phone(out, n, max, ID_AE); break;
      case 'b': add_phone(out, n, max, ID_B); break;
      case 'c': add_phone(out, n, max, (c2 == 'e' || c2 == 'i' || c2 == 'y') ? ID_S : ID_K); break;
      case 'd': add_phone(out, n, max, ID_D); break;
      case 'e': if (i != len - 1) add_phone(out, n, max, ID_EH); break;
      case 'f': add_phone(out, n, max, ID_F); break;
      case 'g': add_phone(out, n, max, ID_G); break;
      case 'h': add_phone(out, n, max, ID_H); break;
      case 'i': add_phone(out, n, max, ID_IH); break;
      case 'j': add_phone(out, n, max, ID_J); break;
      case 'k': add_phone(out, n, max, ID_K); break;
      case 'l': add_phone(out, n, max, ID_L); break;
      case 'm': add_phone(out, n, max, ID_M); break;
      case 'n': add_phone(out, n, max, ID_N); break;
      case 'o': add_phone(out, n, max, ID_AO); break;
      case 'p': add_phone(out, n, max, ID_P); break;
      case 'r': add_phone(out, n, max, ID_RR); break;
      case 's': add_phone(out, n, max, ID_S); break;
      case 't': add_phone(out, n, max, ID_T); break;
      case 'u': add_phone(out, n, max, ID_AH); break;
      case 'v': add_phone(out, n, max, ID_V); break;
      case 'w': add_phone(out, n, max, ID_W); break;
      case 'x': add_phone(out, n, max, ID_K); add_phone(out, n, max, ID_S); break;
      case 'y': add_phone(out, n, max, ID_J); break;
      case 'z': add_phone(out, n, max, ID_Z); break;
      default: break;
    }
    i++;
  }
  return 0;
}

static int text_to_ids(const char *text, int *out, int max) {
  char norm[MAX_TEXT];
  normalize_text(text, norm, sizeof(norm));
  if (strcmp(norm, "this runs on a three dollar chip") == 0) {
    int n = (int)(sizeof(DEMO_IDS) / sizeof(DEMO_IDS[0]));
    if (n > max) return -1;
    memcpy(out, DEMO_IDS, sizeof(DEMO_IDS));
    return n;
  }
  int n = 0;
  if (add_id(out, &n, max, ID_BOS) < 0) return -1;
  const char *p = norm;
  while (*p) {
    while (*p == ' ') p++;
    const char *start = p;
    while (*p && *p != ' ') p++;
    int len = (int)(p - start);
    if (len > 0) {
      if (n > 2 && add_space(out, &n, max) < 0) return -1;
      if (emit_word_approx(start, len, out, &n, max) < 0) return -1;
    }
  }
  if (add_id(out, &n, max, ID_DOT) < 0) return -1;
  if (n + 1 > max) return -1;
  out[n++] = ID_EOS;
  return n;
}

static int parse_ids(const char *s, int *out, int max) {
  int n = 0;
  const char *p = s;
  while (*p) {
    while (*p == ' ' || *p == ',' || *p == ';' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    if (!*p) break;
    char *end = NULL;
    long v = strtol(p, &end, 10);
    /* accept the full acoustic vocab (157); duration remaps its own OOB ids */
    if (end == p || v < 0 || v >= FRONT_AC_VOCAB || n >= max) return -1;
    out[n++] = (int)v;
    p = end;
  }
  return n;
}

static void url_decode(char *s) {
  char *d = s;
  for (char *p = s; *p; p++) {
    if (*p == '+') {
      *d++ = ' ';
    } else if (*p == '%' && isxdigit((unsigned char)p[1]) && isxdigit((unsigned char)p[2])) {
      char h[3] = {p[1], p[2], 0};
      *d++ = (char)strtol(h, NULL, 16);
      p += 2;
    } else {
      *d++ = *p;
    }
  }
  *d = 0;
}

static void get_form_field(char *body, const char *name, char *out, int max) {
  out[0] = 0;
  int name_len = (int)strlen(name);
  char *p = body;
  while (p && *p) {
    char *next = strchr(p, '&');
    if (next) *next = 0;
    if (strncmp(p, name, name_len) == 0 && p[name_len] == '=') {
      snprintf(out, max, "%s", p + name_len + 1);
      url_decode(out);
      if (next) *next = '&';
      return;
    }
    if (next) {
      *next = '&';
      p = next + 1;
    } else {
      break;
    }
  }
}

/* synthesize + stream to speaker; blocks ~utterance-length (realtime) */
static volatile int g_speak_count;

static esp_err_t status_get(httpd_req_t *req) {
  char out[160];
  snprintf(out, sizeof(out), "uptime=%llds speaks=%d busy=%d reset_reason=%d gain=%.1f",
           (long long)(esp_timer_get_time() / 1000000LL), g_speak_count,
           (int)g_synth_busy, (int)esp_reset_reason(), g_again);
  httpd_resp_set_type(req, "text/plain");
  httpd_resp_set_hdr(req, "Connection", "close");
  return httpd_resp_send(req, out, HTTPD_RESP_USE_STRLEN);
}

static int synth_stream_ids(const int *ids, int n_ids, char *summary, int summary_len) {
  if (g_synth_busy) {
    snprintf(summary, summary_len, "busy: synthesis already running");
    return -1;
  }
  g_synth_busy = 1;
  long long t0 = NOW_US();
  aout_begin();
  g_ids_ovr = ids;
  g_n_ovr = n_ids;
  int rc = run_e2e(0, NULL);
  g_ids_ovr = NULL;
  aout_end();
  long long t1 = NOW_US();
  g_synth_busy = 0;
  if (rc == 2) {
    snprintf(summary, summary_len, "error: too long (%d frames won't fit in SRAM) - shorten the text", -g_last_T);
    return -1;
  }
  if (rc != 0) {
    snprintf(summary, summary_len, "error: synthesis failed rc=%d", rc);
    return -1;
  }
  g_speak_count++;
  snprintf(summary, summary_len,
           "ok: ids=%d frames=%d audio=%.2fs synth=%.2fs (playing now) n=%d",
           n_ids, g_last_T, g_last_ms / 1000.0f, (t1 - t0) / 1000000.0f, (int)g_speak_count);
  printf("%s\n", summary);
  return 0;
}

static esp_err_t index_get(httpd_req_t *req) {
  static const char page[] =
    "<!doctype html><html><head><meta name=viewport content='width=device-width,initial-scale=1'>"
    "<title>saanoTTS S3 realtime</title><style>"
    "body{font-family:system-ui;margin:24px;max-width:760px}textarea,input{width:100%;font:16px system-ui;padding:10px;margin:8px 0}"
    "button{font:16px system-ui;padding:10px 14px}.row{display:flex;gap:8px}.row button{flex:1}"
    "#status{white-space:pre-wrap;background:#f4f4f4;padding:12px;min-height:48px}"
    "</style></head><body><h1>saanoTTS ESP32-S3 &mdash; realtime</h1>"
    "<p>fsd r7 engine, 0.27x RT: speech starts as soon as you submit.</p>"
    "<form id=f><label>Text</label><input name=text value='This runs on a three dollar chip'>"
    "<label>Advanced: Piper phoneme IDs (optional)</label><textarea name=ids rows=4 placeholder='1,0,41,0,...,2'></textarea>"
    "<div class=row><button>Synthesize and play</button><button type=button id=demo>Demo phrase</button></div></form>"
    "<p>The text path uses a small approximate English frontend. For exact output, submit Piper IDs.</p>"
    "<pre id=status>ready</pre><script>"
    "async function speak(fd){status.textContent='speaking...';let r=await fetch('/api/speak',{method:'POST',body:new URLSearchParams(fd)});status.textContent=await r.text();}"
    "f.onsubmit=e=>{e.preventDefault();speak(new FormData(f));};"
    "demo.onclick=()=>{f.text.value='This runs on a three dollar chip';f.ids.value='';speak(new FormData(f));};"
    "</script></body></html>";
  httpd_resp_set_type(req, "text/html");
  httpd_resp_set_hdr(req, "Connection", "close");
  return httpd_resp_send(req, page, HTTPD_RESP_USE_STRLEN);
}

/* Worker task: does espeak G2P + synth on a PSRAM stack so the httpd task stays
 * shallow (internal RAM freed for wifi TX). speak_post fills g_req_*, signals
 * g_speak_go, waits g_speak_done. */
static char g_req_text[MAX_TEXT], g_req_idsfield[MAX_BODY];
static char g_worker_summary[220];
static int g_worker_rc;
static SemaphoreHandle_t g_speak_go, g_speak_done;
static TaskHandle_t g_worker_handle2;

static void speak_worker(void *arg) {
    (void)arg;
    static int ids[MAX_IDS];
    for (;;) {
        xSemaphoreTake(g_speak_go, portMAX_DELAY);
        int n_ids; const char *idsrc;
        if (g_req_idsfield[0]) { n_ids = parse_ids(g_req_idsfield, ids, MAX_IDS); idsrc = "raw"; }
        else {
            n_ids = esp_g2p_text_to_ids(g_req_text, ids, MAX_IDS);
            if (n_ids > 0) idsrc = "espeak";
            else { n_ids = text_to_ids(g_req_text, ids, MAX_IDS); idsrc = "crude(fallback)"; }
        }
        printf("[worker] frontend=%s n_ids=%d\n", idsrc, n_ids);
        if (n_ids <= 0) { g_worker_rc = -2; snprintf(g_worker_summary, sizeof(g_worker_summary), "bad input"); }
        else g_worker_rc = synth_stream_ids(ids, n_ids, g_worker_summary, sizeof(g_worker_summary));
        xSemaphoreGive(g_speak_done);
    }
}

static esp_err_t speak_post(httpd_req_t *req) {
  printf("[http] speak_post: content_len=%d\n", (int)req->content_len);
  if (req->content_len >= MAX_BODY) {
    httpd_resp_set_status(req, "413 Payload Too Large");
    return httpd_resp_sendstr(req, "request too large");
  }
  /* static (not stack): MAX_BODY is 6000 now, and the httpd task also runs the
   * synthesizer -- two 6K stack arrays would risk overflow. Safe because
   * g_synth_busy + core-0 pin serialize requests. */
  static char body[MAX_BODY];
  int got = 0;
  while (got < req->content_len) {
    int r = httpd_req_recv(req, body + got, req->content_len - got);
    if (r <= 0) { printf("[http] recv failed r=%d got=%d\n", r, got); return ESP_FAIL; }
    got += r;
  }
  body[got] = 0;

  /* Hand the request to the worker task (PSRAM stack) and wait. espeak's clause
   * translator + the synth are deep-stack; running them on the httpd task forced
   * a 28K internal stack that starved wifi TX -> lwip OOM crash after send. The
   * worker's stack lives in PSRAM so the httpd task stays shallow (internal). */
  get_form_field(body, "text", g_req_text, sizeof(g_req_text));
  get_form_field(body, "ids", g_req_idsfield, sizeof(g_req_idsfield));
  xSemaphoreGive(g_speak_go);
  xSemaphoreTake(g_speak_done, portMAX_DELAY);
  int rc = g_worker_rc;
  if (rc == -2) { httpd_resp_set_status(req, "400 Bad Request");
                  return httpd_resp_sendstr(req, "could not convert input to phoneme IDs"); }
  char *summary = g_worker_summary;
  if (rc < 0) httpd_resp_set_status(req, "500 Internal Server Error");
  httpd_resp_set_type(req, "text/plain");
  /* the server drops sockets after responses anyway; say so honestly, so
   * clients never race a stale keep-alive connection (was: every 2nd request
   * from the same client failed with "connection closed") */
  httpd_resp_set_hdr(req, "Connection", "close");
  return httpd_resp_send(req, summary, HTTPD_RESP_USE_STRLEN);
}

static void start_http(void) {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.stack_size = 12288;   /* shallow: deep work (espeak+synth) is on the PSRAM-stack worker */
  /* pin to core 0: synthesis runs on the httpd task, and if it lands on core 1
   * it shares g_scr[1] with the par_run worker -> scratch corruption mid-frame */
  config.core_id = 0;
  config.recv_wait_timeout = 30;
  config.send_wait_timeout = 30;
  httpd_handle_t server = NULL;
  ESP_ERROR_CHECK(httpd_start(&server, &config));
  httpd_uri_t index_uri = { .uri = "/", .method = HTTP_GET, .handler = index_get };
  httpd_uri_t speak_uri = { .uri = "/api/speak", .method = HTTP_POST, .handler = speak_post };
  httpd_uri_t status_uri = { .uri = "/api/status", .method = HTTP_GET, .handler = status_get };
  ESP_ERROR_CHECK(httpd_register_uri_handler(server, &index_uri));
  ESP_ERROR_CHECK(httpd_register_uri_handler(server, &speak_uri));
  ESP_ERROR_CHECK(httpd_register_uri_handler(server, &status_uri));
}

static void wifi_event(void *arg, esp_event_base_t base, int32_t id, void *data) {
  if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
    esp_wifi_connect();
  } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
    printf("wifi disconnected; reconnecting\n");
    esp_wifi_connect();
  } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
    ip_event_got_ip_t *event = (ip_event_got_ip_t *)data;
    printf("wifi connected: http://" IPSTR "/\n", IP2STR(&event->ip_info.ip));
    xEventGroupSetBits(g_wifi_events, WIFI_CONNECTED);
  }
}

static void wifi_init(void) {
  ESP_ERROR_CHECK(nvs_flash_init());
  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  esp_netif_create_default_wifi_sta();
  g_wifi_events = xEventGroupCreate();
  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&cfg));
  ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event, NULL, NULL));
  ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event, NULL, NULL));
  wifi_config_t wc = { 0 };
  snprintf((char *)wc.sta.ssid, sizeof(wc.sta.ssid), "%s", WIFI_SSID);
  snprintf((char *)wc.sta.password, sizeof(wc.sta.password), "%s", WIFI_PASS);
  wc.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
  ESP_ERROR_CHECK(esp_wifi_start());
  /* modem power-save stalls requests for seconds at a time (observed as
   * intermittent HTTP timeouts); the board is USB-powered, keep the radio on */
  ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
}

void app_main(void) {
    printf("saanotts fsd r7 realtime dashboard: LEDC-PWM GPIO%d, WiFi %s\n", AOUT_GPIO, WIFI_SSID);
    printf("reset reason: %d (1=poweron 3=sw/panic 4=panic 7=brownout... see esp_reset_reason_t)\n",
           (int)esp_reset_reason());
    xTaskCreatePinnedToCore(sn_worker, "snwork", 4096, NULL, configMAX_PRIORITIES - 2, &g_worker_handle, 1);
    g_worker_up = 1;
    aout_init();
    /* arena BEFORE wifi: the engine takes the largest internal block minus a
     * 90K reservation that wifi/lwip/httpd then live inside */
    arena_init();
    /* on-chip espeak G2P: mount spiffs + init espeak (dict loads to PSRAM). If it
     * fails the dashboard still runs (crude fallback in the worker). */
    if (esp_g2p_init() != 0) printf("WARNING: espeak G2P init failed; using crude frontend\n");
    /* worker task with a 48K PSRAM stack for espeak+synth (keeps httpd shallow so
     * wifi has internal RAM). Pinned core 0 to match synth's g_scr[0] usage. */
    g_speak_go = xSemaphoreCreateBinary();
    g_speak_done = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCoreWithCaps(speak_worker, "speak", 48 * 1024, NULL, 5, &g_worker_handle2, 0, MALLOC_CAP_SPIRAM);
    wifi_init();
    xEventGroupWaitBits(g_wifi_events, WIFI_CONNECTED, false, true, portMAX_DELAY);
    start_http();
    printf("free heap after wifi+http: %u\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    /* Fixed output gain: the model's peak is stable (~0.5155 -> 30000/0.5155).
     * We no longer run the golden utterance at boot to measure it — with espeak
     * sharing internal RAM the arena can't fit the 391-frame golden, so that run
     * REJECTED, left g_apeak=0, and every utterance played SILENT. */
    g_again = 58191.7f;
    printf("fixed gain %.1f\n", g_again);
    aout_beep();        /* boot chirp only -- no speech unless requested */
    printf("dashboard ready\n");
}
#else
int main(int argc, char **argv) {
    return run_e2e(argc > 2 && strcmp(argv[2], "--golden-c") == 0,
                   argc > 1 ? argv[1] : "golden");
}
#endif
