/* snt_nano_wasm.c -- WebAssembly (Emscripten) shim around snt_nano.c, the
 * mel-100 / TinyVocos lineage (the 294k nano stacks and the 2,272,145-
 * parameter release stack).
 *
 * One module per lineage: snt_nano.c takes every shape and byte offset from
 * the generated nano_q8_meta.h at COMPILE time, so build_nano.sh compiles a
 * separate module against mcu/models/<lineage>/ and names its export after
 * the voice. The blobs themselves are NOT baked in: the page fetches
 * web/voices/<voice>/{front,model}_{q8,f32}.bin and passes the raw bytes.
 *
 * Two builds of this file exist, chosen by build_nano.sh:
 *   - int8 (default): the device math path, bit-for-bit the C the ESP32
 *     runs, gated by mcu/test/nano_golden_main.c at 0.98;
 *   - SNT_NANO_W_F32: float32 weight rows + float activations, for a stack
 *     that does not survive int8 (the release stack measures 0.951 minimum
 *     correlation under int8 and 1.000000 here). The header's
 *     NANO_WEIGHT_FORMAT makes an int8/f32 mismatch a compile error.
 *
 * Port shims live here rather than in snt_port_wasm.c because that file also
 * defines the R7 entry (snt_web_synthesize) and links snt_tts.c.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "snt_port.h"
#include "snt_nano.h"
#include "nano_q8_meta.h"

/* Headers from int8 exports that predate --weights f32 carry no
 * NANO_WEIGHT_FORMAT; snt_nano.c defaults them to int8 the same way. */
#ifndef NANO_WEIGHT_FORMAT
#define NANO_WEIGHT_FORMAT 0
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define SNT_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define SNT_EXPORT
#endif

/* ---- port shims (identical in spirit to snt_port_wasm.c) -------------- */

/* No second core in a single WASM instance: run the range inline. Every call
 * site is column-disjoint with a barrier, so serial is always correct. */
void snt_par_run(snt_par_fn f, int n, void *ctx) { f(0, n, ctx); }

/* One core => one scratch bank. */
int snt_scratch_id(void) { return 0; }

int64_t snt_now_us(void) {
#ifdef __EMSCRIPTEN__
    return (int64_t)(emscripten_get_now() * 1000.0);
#else
    return 0;
#endif
}

/* ---- browser entry ----------------------------------------------------- */

/* Every stack in this lineage renders at 24 kHz: the frozen-Vocos teacher
 * (charactr/vocos-mel-24khz) and the Kokoro af_heart corpus are 24 kHz, and
 * the iSTFT head (n_fft 1024, hop 256) is defined against that rate. The
 * exporter does not carry the rate in the header, so it is stated here and
 * mirrored in each voice's meta.json. */
#define SNT_NANO_SAMPLE_RATE 24000

/* Error codes returned by snt_nano_wasm_synthesize (all negative). */
#define SNT_NANO_WASM_EARGS   (-1)   /* null pointer / non-positive size      */
#define SNT_NANO_WASM_EOOM    (-2)   /* arena too small (snt_nano ERR_OOM)     */
#define SNT_NANO_WASM_ETOKENS (-3)   /* n_ids outside 1..1024                  */
#define SNT_NANO_WASM_EID     (-4)   /* a phoneme id outside the vocabulary    */
#define SNT_NANO_WASM_EOUT    (-5)   /* out_cap too small for the utterance    */
#define SNT_NANO_WASM_ECORE   (-6)   /* any other snt_nano_synthesize failure  */

typedef struct {
    float *out;
    int cap;
    int pos;
    int overflow;
} NanoSink;

static int nano_sink(const float *pcm, int n, void *user) {
    NanoSink *s = (NanoSink *)user;
    for (int i = 0; i < n; i++) {
        if (s->pos >= s->cap) { s->overflow = 1; return 1; } /* abort cleanly */
        s->out[s->pos++] = pcm[i];
    }
    return 0;
}

static snt_nano_stats g_last_stats;
static int g_last_rc;

/* Speaking-rate divisor for later synthesize calls (0 = the model's own pace),
 * kept out of snt_nano_wasm_synthesize's signature so the existing export
 * stays call-compatible with callers built against an older module. */
SNT_EXPORT
void snt_nano_wasm_set_length_scale(float length_scale) { snt_nano_set_length_scale(length_scale); }

SNT_EXPORT
int snt_nano_wasm_sample_rate(void) { return SNT_NANO_SAMPLE_RATE; }

/* 1 when this module was compiled against float32 weight rows, 0 for int8. */
SNT_EXPORT
int snt_nano_wasm_weight_format(void) { return NANO_WEIGHT_FORMAT; }

/* Derive the decoder-noise seed the Python renderer would use for `text`:
 * sha256(text)[:8] big-endian. Written as two uint32 halves (lo, hi) into
 * `out2` because WebAssembly has no cheap 64-bit crossing. Returns 0, or
 * -1 on bad arguments. */
SNT_EXPORT
int snt_nano_wasm_seed_from_text(const char *text, uint32_t *out2) {
    uint64_t seed = 0;
    if (!text || !out2) return SNT_NANO_WASM_EARGS;
    if (snt_nano_sha256_seed(text, &seed) != 0) return SNT_NANO_WASM_ECORE;
    out2[0] = (uint32_t)(seed & 0xffffffffu);
    out2[1] = (uint32_t)(seed >> 32);
    return 0;
}

/* Synthesize one utterance entirely inside the WASM instance.
 *
 * All pointers are byte offsets into linear memory. `durs` may be 0 (NULL) to
 * run the duration student; the golden gate passes the fixture's frozen
 * frame counts for a frame-exact match. `seed_lo`/`seed_hi` are the two
 * halves of the 64-bit decoder-noise seed (see snt_nano_wasm_seed_from_text).
 * Returns the number of float samples written to `out`, or one of the
 * negative SNT_NANO_WASM_E* codes. */
SNT_EXPORT
int snt_nano_wasm_synthesize(const uint8_t *front, const uint8_t *dec,
                             const int32_t *ids, int n_ids,
                             const int32_t *durs,
                             uint32_t seed_lo, uint32_t seed_hi,
                             uint8_t *arena, int arena_size,
                             float *out, int out_cap) {
    if (!front || !dec || !ids || !arena || !out ||
        n_ids <= 0 || arena_size <= 0 || out_cap <= 0)
        return SNT_NANO_WASM_EARGS;
    NanoSink sink = {out, out_cap, 0, 0};
    snt_nano_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.front_blob = front;
    cfg.dec_blob = dec;
    cfg.arena = arena;
    cfg.arena_size = (size_t)arena_size;
    cfg.dur_override = durs;
    cfg.noise_seed = ((uint64_t)seed_hi << 32) | (uint64_t)seed_lo;
    memset(&g_last_stats, 0, sizeof g_last_stats);
    int rc = snt_nano_synthesize(&cfg, ids, n_ids, nano_sink, &sink, &g_last_stats);
    g_last_rc = rc;
    if (sink.overflow) return SNT_NANO_WASM_EOUT;
    switch (rc) {
        case 0: return sink.pos;
        case -2: return SNT_NANO_WASM_EOOM;
        case -3: return SNT_NANO_WASM_ETOKENS;
        case -4: return SNT_NANO_WASM_EID;
        default: return SNT_NANO_WASM_ECORE;
    }
}

/* Statistics of the most recent call, for the page's readout. */
SNT_EXPORT
int snt_nano_wasm_last_frames(void) { return g_last_stats.frames; }

SNT_EXPORT
int snt_nano_wasm_last_arena_peak(void) { return (int)g_last_stats.arena_peak; }

SNT_EXPORT
int snt_nano_wasm_last_rc(void) { return g_last_rc; }
