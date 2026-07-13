/* front_golden_test.c -- parity gate for the front-half C port (duration +
 * token_context acoustic students). For each golden dir (from
 * tools/export_front_golden.py): load meta.bin, front_weights_f32.bin,
 * ids.bin, run the C duration + latent path and gate against the PyTorch
 * goldens:
 *   - durations: EXACT integer match at length_scale 1.0 (durations.bin)
 *     and 1.25 (durations_ls125.bin), total frame count included;
 *   - latent: Pearson corr > 0.999 vs latent.bin ([C, T], channel-major).
 * Per-stage goldens (dur_log.bin, tok_ctx.bin, ...) are diffed with -s.
 *
 *   ./front_golden_test golden_front/amy golden_front/vi golden_front/hindi
 * Exit 0 iff every dir passes both gates.
 * (The test harness may malloc; the runtime's hot path does not.)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snt_front_f32.h"

#define GATE_CORR 0.999

static void *xload(const char *dir, const char *name, size_t *bytes,
                   int required) {
    char path[512];
    FILE *fh;
    long sz;
    void *buf;
    snprintf(path, sizeof path, "%s/%s", dir, name);
    fh = fopen(path, "rb");
    if (!fh) {
        if (required) { fprintf(stderr, "missing %s\n", path); exit(1); }
        if (bytes) *bytes = 0;
        return NULL;
    }
    fseek(fh, 0, SEEK_END);
    sz = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    buf = malloc((size_t)sz ? (size_t)sz : 1);
    if (!buf || fread(buf, 1, (size_t)sz, fh) != (size_t)sz) {
        fprintf(stderr, "short read %s\n", path);
        exit(1);
    }
    fclose(fh);
    if (bytes) *bytes = (size_t)sz;
    return buf;
}

static void diff_stats(const float *a, const float *b, long n,
                       double *corr, double *maxabs) {
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, mad = 0;
    long i;
    for (i = 0; i < n; i++) {
        double x = a[i], y = b[i], d = fabs(x - y);
        sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
        if (d > mad) mad = d;
    }
    *corr = (sab - sa * sb / (double)n) /
            sqrt((saa - sa * sa / (double)n) * (sbb - sb * sb / (double)n) +
                 1e-30);
    *maxabs = mad;
}

/* per-stage golden comparison via the runtime's stage tap */
typedef struct {
    const char *dir;
} StageCtx;

static void stage_tap(const char *name, const float *data, int ch, int len,
                      void *user) {
    StageCtx *ctx = (StageCtx *)user;
    char fname[128];
    size_t nb = 0;
    float *gold;
    long n = (long)ch * len;
    double corr, mad;
    snprintf(fname, sizeof fname, "%s.bin", name);
    gold = (float *)xload(ctx->dir, fname, &nb, 0);
    if (!gold) return;
    if ((long)(nb / 4) != n) {
        printf("  stage %-14s SIZE MISMATCH: C %ld vs golden %zu floats\n",
               name, n, nb / 4);
        free(gold);
        return;
    }
    diff_stats(data, gold, n, &corr, &mad);
    printf("  stage %-14s corr %.9f  max|diff| %.3e  (%d x %d)\n",
           name, corr, mad, ch, len);
    free(gold);
}

/* exact integer duration comparison; returns mismatch count */
static long check_durations(const char *label, const int32_t *pred,
                            const int32_t *gold, int n) {
    long mism = 0, i, first = -1;
    for (i = 0; i < n; i++)
        if (pred[i] != gold[i]) {
            if (first < 0) first = i;
            mism++;
        }
    if (mism)
        printf("durations %-8s %ld/%d tokens mismatch (first at %ld: C %d vs golden %d)  FAIL\n",
               label, mism, n, first, pred[first], gold[first]);
    else
        printf("durations %-8s %d/%d tokens exact  PASS\n", label, n, n);
    return mism;
}

static int run_dir(const char *dir, int verbose_stages) {
    size_t meta_nb, w_nb, ids_nb, dur_nb, dur125_nb, lat_nb;
    void *meta = xload(dir, "meta.bin", &meta_nb, 1);
    float *weights = (float *)xload(dir, "front_weights_f32.bin", &w_nb, 1);
    int32_t *ids = (int32_t *)xload(dir, "ids.bin", &ids_nb, 1);
    int32_t *gold_dur = (int32_t *)xload(dir, "durations.bin", &dur_nb, 1);
    int32_t *gold_dur125 =
        (int32_t *)xload(dir, "durations_ls125.bin", &dur125_nb, 0);
    float *gold_lat = (float *)xload(dir, "latent.bin", &lat_nb, 1);
    snt_front_model m;
    StageCtx ctx;
    int32_t *dur, *dur125;
    float *latent, *arena;
    size_t arena_n, arena_dur_n;
    int n_tokens, rc, failed = 0;
    long frames, gold_frames;
    double corr, mad;

    printf("== %s\n", dir);
    rc = snt_front_init(&m, meta, meta_nb, weights, w_nb / 4);
    if (rc != 0) {
        fprintf(stderr, "snt_front_init failed: %d\n", rc);
        return 1;
    }
    n_tokens = (int)(ids_nb / 4);
    if (n_tokens <= 0 || dur_nb != ids_nb) {
        fprintf(stderr, "ids.bin/durations.bin token count mismatch\n");
        return 1;
    }
    if ((lat_nb / 4) % (size_t)m.a_out != 0) {
        fprintf(stderr, "latent.bin size %zu not divisible by C %d\n",
                lat_nb / 4, m.a_out);
        return 1;
    }
    gold_frames = (long)(lat_nb / 4 / (size_t)m.a_out);
    if (verbose_stages) {
        ctx.dir = dir;
        m.stage_cb = stage_tap;
        m.stage_user = &ctx;
    }

    dur = (int32_t *)malloc((size_t)n_tokens * sizeof(int32_t));
    dur125 = (int32_t *)malloc((size_t)n_tokens * sizeof(int32_t));
    arena_dur_n = snt_front_duration_arena_floats(&m, n_tokens);
    arena = (float *)malloc(arena_dur_n * sizeof(float));
    if (!dur || !dur125 || !arena) { fprintf(stderr, "oom\n"); exit(1); }

    frames = snt_front_durations(&m, ids, n_tokens, 1.0f, dur,
                                 arena, arena_dur_n);
    if (frames <= 0) {
        fprintf(stderr, "snt_front_durations failed: %ld\n", frames);
        return 1;
    }
    failed += check_durations("ls=1.0", dur, gold_dur, n_tokens) != 0;
    if (frames != gold_frames) {
        printf("frame total %ld != golden latent frames %ld  FAIL\n",
               frames, gold_frames);
        failed++;
    }
    if (gold_dur125) {
        long fr125 = snt_front_durations(&m, ids, n_tokens, 1.25f, dur125,
                                         arena, arena_dur_n);
        if (fr125 <= 0 || dur125_nb != ids_nb) {
            fprintf(stderr, "length_scale 1.25 run failed: %ld\n", fr125);
            return 1;
        }
        failed += check_durations("ls=1.25", dur125, gold_dur125,
                                  n_tokens) != 0;
    }
    free(arena);

    arena_n = snt_front_latent_arena_floats(&m, n_tokens, frames);
    arena = (float *)malloc(arena_n * sizeof(float));
    latent = (float *)malloc((size_t)m.a_out * (size_t)frames * sizeof(float));
    if (!arena || !latent) { fprintf(stderr, "oom\n"); exit(1); }
    rc = snt_front_latent(&m, ids, dur, n_tokens, frames, latent,
                          arena, arena_n);
    if (rc != 0) {
        fprintf(stderr, "snt_front_latent failed: %d\n", rc);
        return 1;
    }
    diff_stats(latent, gold_lat, (long)m.a_out * frames, &corr, &mad);
    printf("latent corr %.9f  max|diff| %.3e  (%d ch x %ld frames)  %s\n",
           corr, mad, m.a_out, frames, corr > GATE_CORR ? "PASS" : "FAIL");
    if (!(corr > GATE_CORR)) failed++;

    free(meta); free(weights); free(ids); free(gold_dur); free(gold_dur125);
    free(gold_lat); free(dur); free(dur125); free(arena); free(latent);
    return failed ? 1 : 0;
}

int main(int argc, char **argv) {
    int i, failures = 0, ran = 0;
    int verbose = 0, argstart = 1;
    if (argc > 1 && strcmp(argv[1], "-s") == 0) { verbose = 1; argstart = 2; }
    if (argc <= argstart) {
        fprintf(stderr, "usage: %s [-s] <golden-dir> [more dirs...]\n"
                        "  -s  also diff per-stage goldens\n", argv[0]);
        return 2;
    }
    for (i = argstart; i < argc; i++) {
        failures += run_dir(argv[i], verbose);
        ran++;
    }
    printf("%d/%d dirs passed (gates: durations exact, latent corr > %.3f)\n",
           ran - failures, ran, GATE_CORR);
    return failures ? 1 : 0;
}
