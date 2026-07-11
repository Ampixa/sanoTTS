/* snt_port_wasm.c -- WebAssembly (Emscripten) port + browser entry point.
 *
 * The core is platform-free C99; the browser is just another Tier-S target
 * with a flat address space and no second core. This file provides the
 * three port shims (serial par_run, single scratch bank, wall clock) and a
 * single exported entry the JavaScript side calls. The int8/int16 kernels
 * and snt_weights_resident come from src/snt_kernels_ref.c unchanged --
 * everything the WASM linear memory holds is readable, so residency is
 * trivially true and no staging is needed.
 *
 * Correctness is identical to the host port by construction: same core,
 * same scalar reference kernels. The host golden gate is therefore the
 * WASM gate too.
 */
#include <stdint.h>
#include "snt_port.h"
#include "snt_tts.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define SNT_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define SNT_EXPORT
#endif

/* ---- port shims ------------------------------------------------------ */

/* No second core in a single WASM instance: run the range inline. Every
 * call site is column-disjoint with a barrier, so serial is always correct
 * -- it just forgoes the (absent) parallelism. */
void snt_par_run(snt_par_fn f, int n, void *ctx) { f(0, n, ctx); }

/* One core => one scratch bank. */
int snt_scratch_id(void) { return 0; }

/* Monotonic microseconds for the core's own profiling. emscripten_get_now
 * returns wall-clock milliseconds (performance.now under the hood); the
 * JavaScript caller does its own wall-clock timing for the RTF readout, so
 * this is only for snt_stats.elapsed_us. */
int64_t snt_now_us(void) {
#ifdef __EMSCRIPTEN__
    return (int64_t)(emscripten_get_now() * 1000.0);
#else
    return 0;
#endif
}

/* ---- browser entry --------------------------------------------------- */

/* The core streams PCM through a callback; the browser wants one contiguous
 * float buffer it can drop into a WebAudio AudioBuffer. This sink appends
 * completed samples into a caller-owned buffer and stops accepting once it
 * is full (returning nonzero aborts synthesis cleanly rather than writing
 * out of bounds). */
typedef struct {
    float *out;
    int cap;
    int pos;
} WebSink;

static int web_sink(const float *pcm, int n, void *user) {
    WebSink *s = (WebSink *)user;
    for (int i = 0; i < n; i++) {
        if (s->pos >= s->cap) return 1; /* full: abort, don't overrun */
        s->out[s->pos++] = pcm[i];
    }
    return 0;
}

/* Synthesize one utterance entirely inside the WASM instance.
 *
 * All pointers are byte offsets into the module's linear memory; JavaScript
 * malloc()s each region, copies the model blobs / phoneme IDs in, and reads
 * the PCM back out of `out`. `durs` may be 0 (NULL) to run the model's own
 * duration head; pass the reference durations for a frame-exact match to the
 * PyTorch golden audio. Returns the number of float samples written to
 * `out`, or a negative value on failure. */
SNT_EXPORT
int snt_web_synthesize(const uint8_t *front, const uint8_t *dec,
                       const int32_t *ids, int n_ids,
                       const int32_t *durs,
                       uint8_t *arena, int arena_size,
                       float *out, int out_cap) {
    if (!front || !dec || !ids || n_ids <= 0 || !arena || !out || out_cap <= 0)
        return -1;
    WebSink sink = {out, out_cap, 0};
    snt_config cfg;
    cfg.front_blob = front;
    cfg.dec_blob = dec;
    cfg.arena = arena;
    cfg.arena_size = (size_t)arena_size;
    cfg.dur_override = durs; /* NULL => predicted durations */
    snt_stats st;
    int rc = snt_synthesize(&cfg, ids, n_ids, web_sink, &sink, &st);
    if (rc != 0) return -2;
    return sink.pos;
}
