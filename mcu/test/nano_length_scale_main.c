/* nano_length_scale_main.c -- the runtime speaking-rate gate for snt_nano.
 *
 * snt_nano_set_length_scale() divides every duration the duration student
 * predicts. This runs one fixture row through the student (no frozen
 * durations) at several rates and checks what the rate must and must not
 * change:
 *
 *  - unset (0) and an explicit 1.0 are the model's own pace: identical frames
 *    and bit-identical PCM. This is the guard that the option costs the
 *    default path nothing;
 *  - non-positive and NaN values fall back to the model's own pace;
 *  - frame counts follow the rate. Each duration is rounded and held to at
 *    least one frame, so the ratio is approximate, not exact;
 *  - every sample is finite;
 *  - the rate applies to later calls only as set: after a scaled call, 0
 *    restores the default output;
 *  - a frozen-duration call (cfg.dur_override) is unaffected by the rate.
 *
 * Usage: snt_nano_length_scale_test [fixture-dir]   (default: the E12-nano one)
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snt_nano.h"

/* How far a frame-count ratio may sit from the rate, as a fraction of the rate.
 * Each duration is rounded, so the ratio is off by a few percent; measured on
 * the fixture it is 0.46 for a rate of 0.5 and 1.99 for 2.0. */
#define RATIO_SLACK 0.15

static void *xload(const char *dir, const char *name, size_t *bytes) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *fh = fopen(path, "rb");
    if (!fh) { fprintf(stderr, "missing %s\n", path); exit(1); }
    fseek(fh, 0, SEEK_END);
    long sz = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    void *buf = malloc((size_t)sz);
    if (!buf || fread(buf, 1, (size_t)sz, fh) != (size_t)sz) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    fclose(fh);
    if (bytes) *bytes = (size_t)sz;
    return buf;
}

typedef struct {
    uint64_t hash;       /* FNV-1a over the PCM bytes */
    long samples;
    int non_finite;
} Sink;

static int sink_cb(const float *pcm, int n, void *user) {
    Sink *s = (Sink *)user;
    const unsigned char *p = (const unsigned char *)pcm;
    for (size_t i = 0; i < (size_t)n * sizeof(float); i++) {
        s->hash ^= p[i];
        s->hash *= 1099511628211ULL;
    }
    for (int i = 0; i < n; i++) if (!isfinite(pcm[i])) s->non_finite++;
    s->samples += n;
    return 0;
}

typedef struct {
    const void *front, *dec;
    const int32_t *ids;
    int n_ids;
    uint64_t seed;
} Row;

typedef struct {
    int rc, frames;
    Sink sink;
} Run;

static Run render(const Row *row, float scale, const int32_t *durs) {
    static unsigned char arena[768 * 1024] __attribute__((aligned(16)));
    Run out;
    memset(&out, 0, sizeof out);
    out.sink.hash = 1469598103934665603ULL;
    snt_nano_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.front_blob = row->front;
    cfg.dec_blob = row->dec;
    cfg.arena = arena;
    cfg.arena_size = sizeof arena;
    cfg.dur_override = durs;
    cfg.noise_seed = row->seed;
    snt_nano_stats st;
    memset(&st, 0, sizeof st);
    snt_nano_set_length_scale(scale);
    out.rc = snt_nano_synthesize(&cfg, row->ids, row->n_ids, sink_cb, &out.sink, &st);
    out.frames = st.frames;
    return out;
}

static int failures;

static void check(int ok, const char *what) {
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) failures++;
}

static int same(const Run *a, const Run *b) {
    return a->frames == b->frames && a->sink.hash == b->sink.hash && a->sink.samples == b->sink.samples;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "test/fixtures/en_us_e12nano";
    char path[512];
#ifdef SNT_NANO_W_F32
    void *front = xload(dir, "front_f32.bin", NULL);
    void *dec = xload(dir, "model_f32.bin", NULL);
#else
    void *front = xload(dir, "front_q8.bin", NULL);
    void *dec = xload(dir, "model_q8.bin", NULL);
#endif
    snprintf(path, sizeof path, "%s/rows.txt", dir);
    FILE *mf = fopen(path, "r");
    if (!mf) { fprintf(stderr, "missing %s\n", path); return 1; }
    char row_id[64];
    int tokens, frames, samples;
    unsigned long long seed;
    if (fscanf(mf, "%63s %d %d %d %llu", row_id, &tokens, &frames, &samples, &seed) != 5) {
        fprintf(stderr, "%s has no rows\n", path);
        return 1;
    }
    fclose(mf);
    size_t nb;
    int32_t *ids = (int32_t *)xload(dir, "r00_ids.bin", &nb);
    int32_t *durs = (int32_t *)xload(dir, "r00_durs.bin", NULL);
    Row row = { front, dec, ids, (int)(nb / 4), (uint64_t)seed };

    Run base = render(&row, 0.0f, NULL);
    Run one = render(&row, 1.0f, NULL);
    Run slow = render(&row, 2.0f, NULL);
    Run fast = render(&row, 0.5f, NULL);
    Run after = render(&row, 0.0f, NULL);
    Run neg = render(&row, -1.0f, NULL);
    Run nan_ = render(&row, NAN, NULL);
    Run frozen = render(&row, 0.0f, durs);
    Run frozen_scaled = render(&row, 2.0f, durs);
    snt_nano_set_length_scale(0.0f);

    printf("row %s: %d ids\n", row_id, row.n_ids);
    printf("  %-14s %8s %10s\n", "length_scale", "frames", "ratio");
    printf("  %-14s %8d %10.3f\n", "0 (default)", base.frames, 1.0);
    printf("  %-14s %8d %10.3f\n", "1.0", one.frames, (double)one.frames / base.frames);
    printf("  %-14s %8d %10.3f\n", "0.5", fast.frames, (double)fast.frames / base.frames);
    printf("  %-14s %8d %10.3f\n", "2.0", slow.frames, (double)slow.frames / base.frames);

    check(base.rc == 0 && one.rc == 0 && slow.rc == 0 && fast.rc == 0, "every rate synthesizes");
    check(base.frames > 0, "the default renders frames");
    check(same(&base, &one), "an explicit 1.0 is bit-identical to the default");
    check(same(&base, &neg), "a negative rate falls back to the default");
    check(same(&base, &nan_), "a NaN rate falls back to the default");
    check(fast.frames < base.frames && base.frames < slow.frames, "frames grow with the length scale");
    check(fabs((double)fast.frames / base.frames - 0.5) < 0.5 * RATIO_SLACK, "0.5 gives about half the frames");
    check(fabs((double)slow.frames / base.frames - 2.0) < 2.0 * RATIO_SLACK, "2.0 gives about twice the frames");
    check(base.sink.non_finite == 0 && slow.sink.non_finite == 0 && fast.sink.non_finite == 0, "every sample is finite");
    check(slow.sink.hash != base.sink.hash && fast.sink.hash != base.sink.hash, "a different rate changes the audio");
    check(same(&base, &after), "0 restores the default output after a scaled call");
    check(frozen.rc == 0 && same(&frozen, &frozen_scaled), "frozen durations ignore the rate");

    printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
