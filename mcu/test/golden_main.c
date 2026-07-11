/* golden_main.c -- bit-exactness gate: every port must pass this against
 * the shipped golden vectors (corr >= 0.98 vs the PyTorch float model). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snt_tts.h"

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

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "golden";
    size_t nb;
    void *front = xload(dir, "front_q8.bin", NULL);
    void *dec = xload(dir, "model_q8.bin", NULL);
    int32_t *ids = (int32_t *)xload(dir, "e2e_ids.bin", &nb);
    int n_ids = (int)(nb / 4);
    int32_t *durs = (int32_t *)xload(dir, "e2e_durs.bin", NULL);
    CorrSink sink = {0};
    sink.gold = (const float *)xload(dir, "e2e_audio.bin", &nb);
    sink.n_gold = nb / 4;

    static unsigned char arena[320 * 1024] __attribute__((aligned(16)));
    snt_config cfg = {front, dec, arena, sizeof arena, durs};
    snt_stats st;
    int rc = snt_synthesize(&cfg, ids, n_ids, corr_cb, &sink, &st);
    double n = (double)sink.pos;
    double cov = sink.sab - sink.sa * sink.sb / n;
    double cr = cov / sqrt((sink.saa - sink.sa * sink.sa / n) *
                           (sink.sbb - sink.sb * sink.sb / n) + 1e-30);
    printf("frames %d samples %d elapsed %lld us\n", st.frames, st.samples,
           (long long)st.elapsed_us);
    printf("golden corr %.6f (%zu samples)\n", cr, sink.pos);
    int pass = (rc == 0 && cr > 0.98);
    printf(pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}
