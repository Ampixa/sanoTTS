/* piperlite_golden_test.c -- fp32 parity gate for the piperlite C port.
 * For each golden dir (from tools/export_piperlite_golden.py): load meta.bin,
 * weights_f32.bin, z.bin, audio.bin, run snt_piperlite_synthesize, and print
 * Pearson correlation + max-abs-diff vs the PyTorch output. Per-stage goldens
 * (pre.bin, up0.bin, ...) are diffed too when present, for bring-up.
 *
 *   ./piperlite_golden_test golden_piperlite/amy golden_piperlite/vi ...
 * GATE: audio corr > 0.999 for every dir. Exit 0 iff all pass.
 * (The test harness may malloc; the runtime's hot path does not.)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snt_piperlite.h"

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
        printf("  stage %-16s SIZE MISMATCH: C %ld vs golden %zu floats\n",
               name, n, nb / 4);
        free(gold);
        return;
    }
    diff_stats(data, gold, n, &corr, &mad);
    printf("  stage %-16s corr %.9f  max|diff| %.3e  (%d x %d)\n",
           name, corr, mad, ch, len);
    free(gold);
}

static int run_dir(const char *dir, int verbose_stages) {
    size_t meta_nb, w_nb, z_nb, a_nb, arena_n;
    void *meta = xload(dir, "meta.bin", &meta_nb, 1);
    float *weights = (float *)xload(dir, "weights_f32.bin", &w_nb, 1);
    float *z = (float *)xload(dir, "z.bin", &z_nb, 1);
    float *gold = (float *)xload(dir, "audio.bin", &a_nb, 1);
    snt_piperlite_model m;
    StageCtx ctx;
    float *audio, *arena;
    long n_gold = (long)(a_nb / 4);
    int frames, rc, pass;
    double corr, mad;

    printf("== %s\n", dir);
    rc = snt_piperlite_init(&m, meta, meta_nb, weights, w_nb / 4);
    if (rc != 0) {
        fprintf(stderr, "snt_piperlite_init failed: %d\n", rc);
        return 1;
    }
    if ((z_nb / 4) % (size_t)m.in_ch != 0) {
        fprintf(stderr, "z.bin size %zu not divisible by in_ch %d\n",
                z_nb / 4, m.in_ch);
        return 1;
    }
    frames = (int)(z_nb / 4 / (size_t)m.in_ch);
    if (n_gold != (long)frames * SNT_PIPERLITE_HOP) {
        fprintf(stderr, "audio.bin has %ld samples, expected %ld\n",
                n_gold, (long)frames * SNT_PIPERLITE_HOP);
        return 1;
    }
    if (verbose_stages) {
        ctx.dir = dir;
        m.stage_cb = stage_tap;
        m.stage_user = &ctx;
    }
    arena_n = snt_piperlite_arena_floats(&m, frames);
    arena = (float *)malloc(arena_n * sizeof(float));
    audio = (float *)malloc((size_t)n_gold * sizeof(float));
    if (!arena || !audio) { fprintf(stderr, "oom\n"); exit(1); }

    rc = snt_piperlite_synthesize(&m, z, frames, audio, arena, arena_n);
    if (rc != 0) {
        fprintf(stderr, "snt_piperlite_synthesize failed: %d\n", rc);
        return 1;
    }
    diff_stats(audio, gold, n_gold, &corr, &mad);
    pass = corr > GATE_CORR;
    printf("audio corr %.9f  max|diff| %.3e  (%ld samples, %d frames)  %s\n",
           corr, mad, n_gold, frames, pass ? "PASS" : "FAIL");
    free(meta); free(weights); free(z); free(gold); free(arena); free(audio);
    return pass ? 0 : 1;
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
    printf("%d/%d dirs passed (gate: corr > %.3f)\n", ran - failures, ran,
           GATE_CORR);
    return failures ? 1 : 0;
}
