/* nano_trace_main.c -- visualization trace driver for the vendored E13b engine.
 *
 * Runs the real int8 device math (same snt_nano.c the ESP32 compiles, built
 * here with -DSNT_NANO_TRACE) over the golden fixture rows and dumps every
 * pipeline stage to raw little-endian files the site can fetch:
 *
 *   <out>/<row_id>/ids.i32        phoneme ids [N]            (input)
 *   <out>/<row_id>/durs.i32       student durations [N]      (frames/phoneme)
 *   <out>/<row_id>/dur_hidden.f32 duration trunk [26 x N]
 *   <out>/<row_id>/tok_hidden.f32 acoustic token-rate [31 x N]
 *   <out>/<row_id>/frame_hidden.f32 acoustic frame-rate [31 x T]
 *   <out>/<row_id>/mel.f32        mel spectrogram [100 x T]  (streamed, ring order)
 *   <out>/<row_id>/noise.f32      Gaussian conditioning [4 x T]
 *   <out>/<row_id>/stem.f32       decoder stem [62 x T]
 *   <out>/<row_id>/trunk_b0..3.f32 ConvNeXt trunk after each block [62 x T]
 *   <out>/<row_id>/spec_re.f32    spectrum real [513 x T]
 *   <out>/<row_id>/spec_im.f32    spectrum imag [513 x T]
 *   <out>/<row_id>/pcm.f32        24 kHz waveform [(T-1)*256]
 *   <out>/<row_id>/manifest.json  shapes + stats
 *
 * All matrices are row-major [channel][time], matching the engine's
 * column-per-frame layout (x[c*T + t]).
 *
 * Modes:
 *   freeze_durs=1  use the fixture's frozen durations (validation vs the
 *                  golden reference audio, Pearson corr must be >= 0.98)
 *   freeze_durs=0  full-student path (what the paper reports at inference)
 *
 * usage: nano_trace <fixture_dir> <out_dir> <freeze_durs 0|1>
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "snt_nano.h"
#include "snt_nano_trace.h"

#define MAX_ROWS 32

/* ---- trace sink (the engine links against these) ---------------------- */

static char g_out_dir[512];
static FILE *g_append;   /* at most one append stream open at a time */

void snt_trace_set_dir(const char *dir) {
    snprintf(g_out_dir, sizeof g_out_dir, "%s", dir);
    mkdir(g_out_dir, 0755);
}

static void trace_write(const char *name, const void *data, long n, size_t esz,
                        const char *ext, int append) {
    char path[768];
    snprintf(path, sizeof path, "%s/%s.%s", g_out_dir, name, ext);
    FILE *fh = fopen(path, append ? "ab" : "wb");
    if (!fh) { fprintf(stderr, "trace: cannot open %s\n", path); return; }
    fwrite(data, esz, (size_t)n, fh);
    fclose(fh);
}

void snt_trace_f32(const char *name, const float *data, long n) {
    trace_write(name, data, n, sizeof(float), "f32", 0);
}
void snt_trace_i32(const char *name, const int32_t *data, long n) {
    trace_write(name, data, n, sizeof(int32_t), "i32", 0);
}
void snt_trace_f32_append(const char *name, const float *data, long n) {
    (void)g_append;
    trace_write(name, data, n, sizeof(float), "f32", 1);
}

/* ---- fixture loading --------------------------------------------------- */

static void *xload(const char *dir, const char *name, size_t *bytes, int required) {
    char path[768];
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
    float *buf;
    size_t n, cap;
    const float *gold;
    size_t n_gold, pos;
    double sa, sb, saa, sbb, sab;
} PcmSink;

static int pcm_cb(const float *pcm, int n, void *user) {
    PcmSink *s = (PcmSink *)user;
    if (s->n + (size_t)n > s->cap) {
        s->cap = (s->n + (size_t)n) * 2 + 4096;
        s->buf = (float *)realloc(s->buf, s->cap * sizeof(float));
        if (!s->buf) return 1;
    }
    memcpy(s->buf + s->n, pcm, (size_t)n * sizeof(float));
    s->n += (size_t)n;
    for (int i = 0; i < n && s->pos < s->n_gold; i++, s->pos++) {
        double a = pcm[i], b = s->gold[s->pos];
        s->sa += a; s->sb += b;
        s->saa += a * a; s->sbb += b * b; s->sab += a * b;
    }
    return 0;
}

typedef struct {
    char row_id[64];
    int tokens, frames, samples;
    unsigned long long seed;
} RowMeta;

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "engine/fixtures/en_us_e13b";
    const char *out = argc > 2 ? argv[2] : "public/traces";
    int freeze = argc > 3 ? atoi(argv[3]) : 0;

    void *front = xload(dir, "front_q8.bin", NULL, 1);
    void *dec = xload(dir, "model_q8.bin", NULL, 1);

    char path[768];
    snprintf(path, sizeof path, "%s/rows.txt", dir);
    FILE *mf = fopen(path, "r");
    if (!mf) { fprintf(stderr, "missing %s\n", path); return 1; }
    RowMeta rows[MAX_ROWS];
    int n_rows = 0;
    while (n_rows < MAX_ROWS &&
           fscanf(mf, "%63s %d %d %d %llu", rows[n_rows].row_id,
                  &rows[n_rows].tokens, &rows[n_rows].frames,
                  &rows[n_rows].samples, &rows[n_rows].seed) == 5)
        n_rows++;
    fclose(mf);
    printf("[trace] %d rows from %s, freeze_durs=%d\n", n_rows, dir, freeze);

    static unsigned char arena[768 * 1024] __attribute__((aligned(16)));
    double min_corr = 2.0;
    mkdir(out, 0755);

    for (int r = 0; r < n_rows; r++) {
        char name[128];
        size_t nb;
        snprintf(name, sizeof name, "r%02d_ids.bin", r);
        int32_t *ids = (int32_t *)xload(dir, name, &nb, 1);
        int n_ids = (int)(nb / 4);
        snprintf(name, sizeof name, "r%02d_durs.bin", r);
        int32_t *durs = (int32_t *)xload(dir, name, NULL, 1);
        snprintf(name, sizeof name, "r%02d_audio.bin", r);
        size_t gab;
        float *gold = (float *)xload(dir, name, &gab, 1);

        char rowdir[768];
        snprintf(rowdir, sizeof rowdir, "%s/%s", out, rows[r].row_id);
        snt_trace_set_dir(rowdir);
        /* ids are an input, not an engine stage: dump them here */
        snt_trace_i32("ids", ids, n_ids);

        PcmSink sink;
        memset(&sink, 0, sizeof sink);
        sink.gold = gold;
        sink.n_gold = gab / 4;

        snt_nano_config cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.front_blob = front;
        cfg.dec_blob = dec;
        cfg.arena = arena;
        cfg.arena_size = sizeof arena;
        cfg.dur_override = freeze ? durs : NULL;
        cfg.noise_seed = (uint64_t)rows[r].seed;

        snt_nano_stats st;
        memset(&st, 0, sizeof st);
        int rc = snt_nano_synthesize(&cfg, ids, n_ids, pcm_cb, &sink, &st);
        if (rc != 0) {
            printf("[trace] %-14s rc=%d -- SKIP\n", rows[r].row_id, rc);
            free(ids); free(durs); free(gold); free(sink.buf);
            continue;
        }
        snt_trace_f32("pcm", sink.buf, (long)sink.n);

        double nn = (double)sink.pos;
        double cov = sink.sab - sink.sa * sink.sb / nn;
        double cr = cov / sqrt((sink.saa - sink.sa * sink.sa / nn) *
                               (sink.sbb - sink.sb * sink.sb / nn) + 1e-30);
        if (isfinite(cr) && cr < min_corr) min_corr = cr;

        /* per-row manifest: shapes the site needs, plus fidelity stats */
        snprintf(path, sizeof path, "%s/manifest.json", rowdir);
        FILE *jf = fopen(path, "w");
        if (jf) {
            fprintf(jf,
                "{\n"
                "  \"row_id\": \"%s\",\n"
                "  \"n_tokens\": %d,\n"
                "  \"frames\": %d,\n"
                "  \"samples\": %d,\n"
                "  \"seed\": %llu,\n"
                "  \"freeze_durs\": %d,\n"
                "  \"got_frames\": %d,\n"
                "  \"got_samples\": %d,\n"
                "  \"corr_vs_fixture\": %.6f,\n"
                "  \"arena_peak\": %zu,\n"
                "  \"elapsed_us\": %lld,\n"
                "  \"tensors\": {\n"
                "    \"ids\":         {\"file\": \"ids.i32\",         \"dtype\": \"i32\", \"shape\": [%d]},\n"
                "    \"durs\":        {\"file\": \"durs.i32\",        \"dtype\": \"i32\", \"shape\": [%d]},\n"
                "    \"dur_hidden\":  {\"file\": \"dur_hidden.f32\",  \"dtype\": \"f32\", \"shape\": [26, %d]},\n"
                "    \"tok_hidden\":  {\"file\": \"tok_hidden.f32\",  \"dtype\": \"f32\", \"shape\": [31, %d]},\n"
                "    \"frame_hidden\":{\"file\": \"frame_hidden.f32\",\"dtype\": \"f32\", \"shape\": [31, %d]},\n"
                "    \"mel\":         {\"file\": \"mel.f32\",         \"dtype\": \"f32\", \"shape\": [100, %d]},\n"
                "    \"noise\":       {\"file\": \"noise.f32\",       \"dtype\": \"f32\", \"shape\": [4, %d]},\n"
                "    \"stem\":        {\"file\": \"stem.f32\",        \"dtype\": \"f32\", \"shape\": [62, %d]},\n"
                "    \"trunk_b0\":    {\"file\": \"trunk_b0.f32\",    \"dtype\": \"f32\", \"shape\": [62, %d]},\n"
                "    \"trunk_b1\":    {\"file\": \"trunk_b1.f32\",    \"dtype\": \"f32\", \"shape\": [62, %d]},\n"
                "    \"trunk_b2\":    {\"file\": \"trunk_b2.f32\",    \"dtype\": \"f32\", \"shape\": [62, %d]},\n"
                "    \"trunk_b3\":    {\"file\": \"trunk_b3.f32\",    \"dtype\": \"f32\", \"shape\": [62, %d]},\n"
                "    \"spec_re\":     {\"file\": \"spec_re.f32\",     \"dtype\": \"f32\", \"shape\": [513, %d]},\n"
                "    \"spec_im\":     {\"file\": \"spec_im.f32\",     \"dtype\": \"f32\", \"shape\": [513, %d]},\n"
                "    \"pcm\":         {\"file\": \"pcm.f32\",         \"dtype\": \"f32\", \"shape\": [%d]}\n"
                "  }\n"
                "}\n",
                rows[r].row_id, rows[r].tokens, rows[r].frames, rows[r].samples,
                rows[r].seed, freeze, st.frames, st.samples, cr,
                st.arena_peak, (long long)st.elapsed_us,
                n_ids, n_ids, n_ids, n_ids,
                st.frames, st.frames, st.frames, st.frames, st.frames,
                st.frames, st.frames, st.frames, st.frames, st.samples);
            fclose(jf);
        }
        printf("[trace] %-14s N=%d T=%d pcm=%zu corr=%.6f arena=%zu\n",
               rows[r].row_id, n_ids, st.frames, sink.n, cr, st.arena_peak);
        free(ids); free(durs); free(gold); free(sink.buf);
    }

    printf("[trace] done. min corr vs fixture = %.6f %s\n", min_corr,
           min_corr >= 0.98 ? "(PASS, >= 0.98)" : "(BELOW GOLDEN GATE 0.98)");
    free(front); free(dec);
    return 0;
}
