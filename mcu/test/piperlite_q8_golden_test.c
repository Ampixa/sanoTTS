/* piperlite_q8_golden_test.c -- int8 gate for the piperlite decoder.
 * For each golden dir: load meta_q8.bin + weights_q8.bin (from
 * tools/export_piperlite_q8.py) and z.bin + audio.bin (fp32 PyTorch goldens
 * from tools/export_piperlite_golden.py), run the int8 forward, and print
 * Pearson correlation + max-abs-diff vs the fp32 golden audio, plus blob
 * sizes. With -s, per-stage int8 tensors (dequantized) are diffed against
 * the fp32 stage goldens to show where quantization error concentrates.
 *
 *   ./piperlite_q8_golden_test [-s] golden_piperlite/amy ...
 * GATE: audio corr > 0.99 for every dir. Exit 0 iff all pass.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snt_piperlite_q8.h"

#define GATE_CORR 0.99

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
    printf("  stage %-16s corr %.6f  max|diff| %.3e  (%d x %d)\n",
           name, corr, mad, ch, len);
    free(gold);
}

static int run_dir(const char *dir, int verbose_stages) {
    size_t meta_nb, w_nb, z_nb, a_nb, arena_nb;
    void *meta = xload(dir, "meta_q8.bin", &meta_nb, 1);
    int8_t *weights = (int8_t *)xload(dir, "weights_q8.bin", &w_nb, 1);
    float *z = (float *)xload(dir, "z.bin", &z_nb, 1);
    float *gold = (float *)xload(dir, "audio.bin", &a_nb, 1);
    snt_piperlite_q8_model m;
    StageCtx ctx;
    float *audio;
    void *arena;
    long n_gold = (long)(a_nb / 4);
    int frames, rc, pass;
    double corr, mad;

    printf("== %s\n", dir);
    rc = snt_piperlite_q8_init(&m, meta, meta_nb, weights, w_nb);
    if (rc != 0) {
        fprintf(stderr, "snt_piperlite_q8_init failed: %d\n", rc);
        return 1;
    }
    if ((z_nb / 4) % (size_t)m.in_ch != 0) {
        fprintf(stderr, "z.bin size %zu not divisible by in_ch %d\n",
                z_nb / 4, m.in_ch);
        return 1;
    }
    frames = (int)(z_nb / 4 / (size_t)m.in_ch);
    if (n_gold != (long)frames * SNT_PIPERLITE_Q8_HOP) {
        fprintf(stderr, "audio.bin has %ld samples, expected %ld\n",
                n_gold, (long)frames * SNT_PIPERLITE_Q8_HOP);
        return 1;
    }
    if (verbose_stages) {
        ctx.dir = dir;
        m.stage_cb = stage_tap;
        m.stage_user = &ctx;
    }
    arena_nb = snt_piperlite_q8_arena_bytes(&m, frames);
    arena = malloc(arena_nb);
    audio = (float *)malloc((size_t)n_gold * sizeof(float));
    if (!arena || !audio) { fprintf(stderr, "oom\n"); exit(1); }

    rc = snt_piperlite_q8_synthesize(&m, z, frames, audio, arena, arena_nb);
    if (rc != 0) {
        fprintf(stderr, "snt_piperlite_q8_synthesize failed: %d\n", rc);
        return 1;
    }
    diff_stats(audio, gold, n_gold, &corr, &mad);
    pass = corr > GATE_CORR;
    printf("audio corr %.6f  max|diff| %.3e  (%ld samples, %d frames)\n",
           corr, mad, n_gold, frames);
    printf("blob: weights_q8 %zu bytes + meta_q8 %zu bytes  %s\n",
           w_nb, meta_nb, pass ? "PASS" : "FAIL");
    free(meta); free(weights); free(z); free(gold); free(arena); free(audio);
    return pass ? 0 : 1;
}

int main(int argc, char **argv) {
    int i, failures = 0, ran = 0;
    int verbose = 0, argstart = 1;
    if (argc > 1 && strcmp(argv[1], "-s") == 0) { verbose = 1; argstart = 2; }
    if (argc <= argstart) {
        fprintf(stderr, "usage: %s [-s] <golden-dir> [more dirs...]\n"
                        "  -s  diff dequantized stages vs fp32 goldens\n",
                argv[0]);
        return 2;
    }
    for (i = argstart; i < argc; i++) {
        failures += run_dir(argv[i], verbose);
        ran++;
    }
    printf("%d/%d dirs passed (gate: corr > %.2f)\n", ran - failures, ran,
           GATE_CORR);
    return failures ? 1 : 0;
}
