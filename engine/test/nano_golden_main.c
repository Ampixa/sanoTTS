/* nano_golden_main.c -- the E12-nano gate.
 *
 * Same contract as test/golden_main.c: waveform Pearson correlation against the
 * PyTorch float reference, threshold 0.98. The threshold is deliberately the
 * SAME NUMBER the R7 line is held to. Nothing here is tuned to make the nano
 * pass; a failure printed honestly is the point of the harness.
 *
 * TWO DIFFERENCES FROM THE R7 GATE, both forced by measurement rather than
 * convenience:
 *
 *  1. The gate is the MINIMUM correlation over 8 rows, not one row's.
 *     docs/e12-nano-int8-quantisation.md section 1 measured the minimum moving
 *     by 0.07 across 75k decoder steps with the acoustic held fixed, so a
 *     one-row verdict on this stack is noise at the +-0.03 level.
 *  2. The two new numeric primitives are checked directly, because an
 *     end-to-end correlation would not say which of them broke:
 *       - the MT19937 uniform stream must be BIT-EXACT against torch.rand;
 *       - the Box-Muller output is held to a numeric bound, not bit-exactness,
 *         because logf/cosf/sinf are not bit-identical across libm builds
 *         (measured: 26% of draws differ, by at most 1.9e-6);
 *       - the fixed-point sincos is compared against libm over the phase range
 *         the head actually emits.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snt_nano.h"

#define MAX_ROWS 32
#define NOISE_TOL 1e-5     /* ~80 ulp on a unit-normal draw */
#define SINCOS_TOL 1e-4
#define GATE 0.98

static void *xload(const char *dir, const char *name, size_t *bytes, int required) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *fh = fopen(path, "rb");
    if (!fh) {
        if (!required) return NULL;
        fprintf(stderr, "missing %s\n", path);
        exit(1);
    }
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
    const float *gold;
    size_t n_gold, pos;
    double sa, sb, saa, sbb, sab;
} CorrSink;

static int corr_cb(const float *pcm, int n, void *user) {
    CorrSink *c = (CorrSink *)user;
    for (int i = 0; i < n && c->pos < c->n_gold; i++, c->pos++) {
        double a = pcm[i], b = c->gold[c->pos];
        c->sa += a; c->sb += b;
        c->saa += a * a; c->sbb += b * b; c->sab += a * b;
    }
    return 0;
}

/* ---- component checks ------------------------------------------------- */

static int check_uniform(const char *dir, uint64_t seed) {
    size_t nb;
    float *ref = (float *)xload(dir, "e2e_uniform.bin", &nb, 1);
    int n = (int)(nb / sizeof(float));
    float *got = (float *)malloc(nb);
    if (!got) return 1;
    snt_nano_uniform_stream(seed, n, got);
    int bad = 0;
    for (int i = 0; i < n; i++) if (got[i] != ref[i]) bad++;
    printf("uniform : %d values, %d differ %s\n", n, bad,
           bad == 0 ? "-- BIT-EXACT vs torch.rand" : "-- MT19937 PORT IS WRONG");
    free(ref);
    free(got);
    return bad != 0;
}

static int check_noise(const char *dir, uint64_t seed, int channels) {
    size_t nb;
    float *ref = (float *)xload(dir, "e2e_noise.bin", &nb, 1);
    int n = (int)(nb / sizeof(float));
    float *got = (float *)malloc(nb);
    if (!got) return 1;
    if (snt_nano_seeded_noise(seed, channels, n / channels, got) != 0) {
        printf("noise   : generator refused %d values\n", n);
        return 1;
    }
    int bad = 0;
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)got[i] - (double)ref[i]);
        if (d > worst) worst = d;
        if (got[i] != ref[i]) bad++;
    }
    printf("noise   : %d values, %d differ, max |delta| %.3g (tol %.0e) %s\n",
           n, bad, worst, NOISE_TOL, worst <= NOISE_TOL ? "OK" : "OUT OF BOUND");
    free(ref);
    free(got);
    return worst > NOISE_TOL;
}

static int check_sincos(float phi_max) {
    double worst = 0.0;
    float worst_phi = 0.0f;
    for (int i = 0; i <= 400000; i++) {
        float phi = -phi_max + 2.0f * phi_max * (float)i / 400000.0f;
        float c, s;
        snt_nano_sincos(phi, &c, &s);
        double dc = fabs((double)c - cos((double)phi));
        double ds = fabs((double)s - sin((double)phi));
        double d = dc > ds ? dc : ds;
        if (d > worst) { worst = d; worst_phi = phi; }
    }
    printf("sincos  : max |error| %.3g over |phi| <= %.2f (worst at %.4f, tol %.0e) %s\n",
           worst, phi_max, worst_phi, SINCOS_TOL, worst <= SINCOS_TOL ? "OK" : "OUT OF BOUND");
    return worst > SINCOS_TOL;
}

/* ---- main ------------------------------------------------------------- */

typedef struct {
    char row_id[64];
    int tokens, frames, samples;
    unsigned long long seed;
} RowMeta;

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "test/fixtures/en_us_e12nano";
    char path[512];
    /* int8 fixtures ship front_q8/model_q8; --weights f32 exports ship
     * front_f32/model_f32. The build must match (snt_nano.c refuses the other
     * element type at compile time via NANO_WEIGHT_FORMAT). */
#ifdef SNT_NANO_W_F32
    void *front = xload(dir, "front_f32.bin", NULL, 1);
    void *dec = xload(dir, "model_f32.bin", NULL, 1);
#else
    void *front = xload(dir, "front_q8.bin", NULL, 1);
    void *dec = xload(dir, "model_q8.bin", NULL, 1);
#endif

    snprintf(path, sizeof path, "%s/rows.txt", dir);
    FILE *mf = fopen(path, "r");
    if (!mf) { fprintf(stderr, "missing %s\n", path); return 1; }
    RowMeta rows[MAX_ROWS];
    int n_rows = 0;
    while (n_rows < MAX_ROWS &&
           fscanf(mf, "%63s %d %d %d %llu", rows[n_rows].row_id,
                  &rows[n_rows].tokens, &rows[n_rows].frames,
                  &rows[n_rows].samples, &rows[n_rows].seed) == 5) {
        n_rows++;
    }
    fclose(mf);
    if (n_rows == 0) { fprintf(stderr, "%s has no rows\n", path); return 1; }

    /* the seed the fixture's component references were generated with */
    uint64_t derived = 0;
    snt_nano_sha256_seed(rows[0].row_id, &derived);
    printf("seed    : sha256(\"%s\")[:8] = %llu, fixture says %llu %s\n",
           rows[0].row_id, (unsigned long long)derived, rows[0].seed,
           derived == rows[0].seed ? "OK" : "-- SHA-256 PORT IS WRONG");
    int bad_components = (derived != rows[0].seed);
    bad_components |= check_uniform(dir, rows[0].seed);
    bad_components |= check_noise(dir, rows[0].seed, 4);
    bad_components |= check_sincos(32.0f);  /* measured max 30.33 over the 8 rows */

    static unsigned char arena[768 * 1024] __attribute__((aligned(16)));
    double min_corr = 2.0, sum_corr = 0.0;
    int worst_row = -1, hard_fail = 0;
    size_t peak_bytes = 0;
    double peak_per_frame = 0.0;
    long long total_us = 0;
    int total_frames = 0;

    printf("\n%-14s %6s %8s %10s %10s %9s\n",
           "row", "frames", "samples", "corr", "rms_ratio", "us", "arena");
    for (int r = 0; r < n_rows; r++) {
        char name[64];
        size_t nb;
        snprintf(name, sizeof name, "r%02d_ids.bin", r);
        int32_t *ids = (int32_t *)xload(dir, name, &nb, 1);
        int n_ids = (int)(nb / 4);
        snprintf(name, sizeof name, "r%02d_durs.bin", r);
        int32_t *durs = (int32_t *)xload(dir, name, NULL, 1);
        snprintf(name, sizeof name, "r%02d_audio.bin", r);
        CorrSink sink;
        memset(&sink, 0, sizeof sink);
        sink.gold = (const float *)xload(dir, name, &nb, 1);
        sink.n_gold = nb / 4;

        snt_nano_config cfg;
        cfg.front_blob = front;
        cfg.dec_blob = dec;
        cfg.arena = arena;
        cfg.arena_size = sizeof arena;
        cfg.dur_override = durs;   /* frame-exact alignment, as R7's gate does */
        cfg.noise_seed = (uint64_t)rows[r].seed;

        snt_nano_stats st;
        memset(&st, 0, sizeof st);
        int rc = snt_nano_synthesize(&cfg, ids, n_ids, corr_cb, &sink, &st);
        if (rc != 0) {
            printf("%-14s synthesize returned %d\n", rows[r].row_id, rc);
            hard_fail = 1;
            continue;
        }
        if (st.frames != rows[r].frames || st.samples != rows[r].samples) {
            printf("%-14s length mismatch: got %d frames / %d samples, "
                   "fixture says %d / %d\n", rows[r].row_id, st.frames,
                   st.samples, rows[r].frames, rows[r].samples);
            hard_fail = 1;
        }
        double n = (double)sink.pos;
        double cov = sink.sab - sink.sa * sink.sb / n;
        double cr = cov / sqrt((sink.saa - sink.sa * sink.sa / n) *
                               (sink.sbb - sink.sb * sink.sb / n) + 1e-30);
        double rms_c = sqrt(sink.saa / n), rms_g = sqrt(sink.sbb / n);
        /* Per-row arena, not just the max: the arena is fixed overhead plus a
         * per-frame term, and only the per-row numbers separate the two. That
         * split is what decides which chips can run a given utterance length. */
        printf("%-14s %6d %8d %10.6f %10.6f %9lld %10zu\n", rows[r].row_id, st.frames,
               st.samples, cr, rms_g > 0 ? rms_c / rms_g : 0.0,
               (long long)st.elapsed_us, st.arena_peak);
        /* A non-finite correlation must FAIL, and it will not do so on its
         * own: min_corr starts above the gate and every comparison against
         * NaN is false, so a NaN row would leave min_corr untouched and the
         * gate would print PASS on a model that emitted no valid audio. That
         * is the single most dangerous outcome this harness can have, so NaN
         * is caught explicitly rather than by arithmetic. */
        if (!isfinite(cr)) {
            printf("  ^ row %s produced a non-finite correlation\n", rows[r].row_id);
            hard_fail = 1;
        } else if (cr < min_corr) {
            min_corr = cr; worst_row = r;
        }
        sum_corr += cr;
        if (st.arena_peak > peak_bytes) peak_bytes = st.arena_peak;
        if (st.frames) {
            double per = (double)st.arena_peak / st.frames;
            if (per > peak_per_frame) peak_per_frame = per;
        }
        total_us += st.elapsed_us;
        total_frames += st.frames;
        free(ids);
        free(durs);
        free((void *)sink.gold);
    }

    printf("\nrows %d   mean corr %.6f   MIN corr %.6f (%s)\n", n_rows,
           sum_corr / n_rows, min_corr,
           worst_row >= 0 ? rows[worst_row].row_id : "-");
    printf("arena peak %zu B, %.1f B/frame; %.2f ms per second of audio "
           "(host, scalar kernels)\n", peak_bytes, peak_per_frame,
           total_frames ? (double)total_us / 1000.0 / (total_frames / 93.75) : 0.0);

    if (hard_fail) { printf("FAIL: a row did not synthesize correctly\n"); return 1; }
    if (bad_components) { printf("FAIL: a component check is out of bound\n"); return 1; }
    if (!isfinite(min_corr) || !isfinite(sum_corr) || worst_row < 0) {
        printf("FAIL: correlation is not a finite number over %d rows\n", n_rows);
        return 1;
    }
    if (min_corr > GATE) {
        printf("PASS\n");
        return 0;
    }
    printf("FAIL: corr %.4f < %.2f\n", min_corr, GATE);
    return 1;
}
