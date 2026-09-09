/* nano_pcm_hash_main.c -- host PCM identity harness for the E12-nano runtime.
 *
 * Renders every row of a fixture directory through snt_nano_synthesize() with
 * exactly the configuration the golden gate uses (frozen durations from the
 * fixture, sha256(row_id) noise seed from rows.txt) and emits, per row:
 *
 *   - the raw float32 PCM as <out_dir>/rNN.f32, byte-for-byte as the callback
 *     delivered it, so an external SHA-256 can be taken over the same bytes;
 *   - FNV-1a 64 over those same bytes, computed in-process.
 *
 * The point is to answer a question waveform correlation cannot: are two
 * builds of the runtime emitting THE SAME BYTES, or merely similar audio.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snt_nano.h"

#define FNV64_OFFSET 1469598103934665603ULL
#define FNV64_PRIME  1099511628211ULL

typedef struct {
    FILE *out;
    uint64_t fnv;
    long samples;
    int write_failed;
} Sink;

static void *xload(const char *dir, const char *name, size_t *bytes) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *fh = fopen(path, "rb");
    if (!fh) {
        fprintf(stderr, "FATAL: cannot open %s\n", path);
        exit(2);
    }
    if (fseek(fh, 0, SEEK_END) != 0) {
        fprintf(stderr, "FATAL: cannot seek %s\n", path);
        exit(2);
    }
    long sz = ftell(fh);
    if (sz < 0) {
        fprintf(stderr, "FATAL: cannot size %s\n", path);
        exit(2);
    }
    rewind(fh);
    void *buf = malloc((size_t)sz);
    if (!buf) {
        fprintf(stderr, "FATAL: out of memory for %s (%ld bytes)\n", path, sz);
        exit(2);
    }
    if (fread(buf, 1, (size_t)sz, fh) != (size_t)sz) {
        fprintf(stderr, "FATAL: short read on %s\n", path);
        exit(2);
    }
    if (fclose(fh) != 0) {
        fprintf(stderr, "FATAL: cannot close %s\n", path);
        exit(2);
    }
    if (bytes) *bytes = (size_t)sz;
    return buf;
}

static int sink_cb(const float *pcm, int n, void *user) {
    Sink *s = (Sink *)user;
    const unsigned char *p = (const unsigned char *)pcm;
    size_t nb = (size_t)n * sizeof(float);
    for (size_t i = 0; i < nb; i++) {
        s->fnv ^= (uint64_t)p[i];
        s->fnv *= FNV64_PRIME;
    }
    if (fwrite(pcm, 1, nb, s->out) != nb) {
        s->write_failed = 1;
        return 1;                 /* abort synthesis: the evidence is broken */
    }
    s->samples += n;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <fixture-dir> <out-dir>\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];
    const char *out_dir = argv[2];

    /* int8 fixtures ship front_q8/model_q8; --weights f32 exports ship
     * front_f32/model_f32. The build must match: snt_nano.c refuses the other
     * element type at compile time via NANO_WEIGHT_FORMAT. Same switch as
     * test/nano_golden_main.c, so this harness covers every lineage the
     * golden gate covers rather than the int8 ones alone. */
#ifdef SNT_NANO_W_F32
    void *front = xload(dir, "front_f32.bin", NULL);
    void *model = xload(dir, "model_f32.bin", NULL);
#else
    void *front = xload(dir, "front_q8.bin", NULL);
    void *model = xload(dir, "model_q8.bin", NULL);
#endif

    char rows_path[512];
    snprintf(rows_path, sizeof rows_path, "%s/rows.txt", dir);
    FILE *rf = fopen(rows_path, "r");
    if (!rf) {
        fprintf(stderr, "FATAL: cannot open %s\n", rows_path);
        return 2;
    }

    static unsigned char arena[768 * 1024] __attribute__((aligned(16)));

    char id[128];
    int tokens, frames, samples;
    unsigned long long seed;
    int idx = 0, failures = 0;

    printf("# row  index  rc  frames  samples  arena_peak  fnv1a64\n");
    while (fscanf(rf, "%127s %d %d %d %llu", id, &tokens, &frames, &samples,
                  &seed) == 5) {
        char name[64], out_path[640];
        size_t nb;
        snprintf(name, sizeof name, "r%02d_ids.bin", idx);
        int32_t *ids = (int32_t *)xload(dir, name, &nb);
        int n_ids = (int)(nb / sizeof(int32_t));
        snprintf(name, sizeof name, "r%02d_durs.bin", idx);
        int32_t *durs = (int32_t *)xload(dir, name, NULL);

        snprintf(out_path, sizeof out_path, "%s/r%02d.f32", out_dir, idx);
        Sink sink;
        memset(&sink, 0, sizeof sink);
        sink.fnv = FNV64_OFFSET;
        sink.out = fopen(out_path, "wb");
        if (!sink.out) {
            fprintf(stderr, "FATAL: cannot create %s\n", out_path);
            return 2;
        }

        snt_nano_config cfg;
        cfg.front_blob = front;
        cfg.dec_blob = model;
        cfg.arena = arena;
        cfg.arena_size = sizeof arena;
        cfg.dur_override = durs;
        cfg.noise_seed = (uint64_t)seed;

        snt_nano_stats st;
        memset(&st, 0, sizeof st);
        int rc = snt_nano_synthesize(&cfg, ids, n_ids, sink_cb, &sink, &st);

        if (fclose(sink.out) != 0) {
            fprintf(stderr, "FATAL: cannot close %s\n", out_path);
            return 2;
        }
        if (sink.write_failed) {
            fprintf(stderr, "FATAL: PCM write failed for %s\n", id);
            return 2;
        }
        if (rc != 0) {
            fprintf(stderr, "ERROR: %s synthesize rc=%d\n", id, rc);
            failures++;
        }
        if (rc == 0 && (st.frames != frames || sink.samples != samples)) {
            fprintf(stderr,
                    "ERROR: %s length mismatch: got %d frames / %ld samples, "
                    "fixture says %d / %d\n",
                    id, st.frames, sink.samples, frames, samples);
            failures++;
        }
        printf("%-14s %2d %3d %7d %9ld %11zu  %016llx\n", id, idx, rc, st.frames,
               sink.samples, st.arena_peak, (unsigned long long)sink.fnv);
        fflush(stdout);

        free(ids);
        free(durs);
        idx++;
    }
    if (fclose(rf) != 0) {
        fprintf(stderr, "FATAL: cannot close %s\n", rows_path);
        return 2;
    }
    free(front);
    free(model);

    if (idx == 0) {
        fprintf(stderr, "FATAL: %s listed no rows\n", rows_path);
        return 2;
    }
    if (failures) {
        fprintf(stderr, "FATAL: %d row failure(s)\n", failures);
        return 1;
    }
    return 0;
}
