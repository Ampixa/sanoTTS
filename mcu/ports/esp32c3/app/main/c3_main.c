/* c3_main.c -- ESP32-C3 golden + timing app for saanotts-mcu (Tier S).
 * Run 1: timing with a no-op PCM sink. Run 2: correctness (corr vs the
 * embedded float-model golden; float math in the cb is soft-float on the
 * C3, hence kept out of the timed run). */
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "esp_heap_caps.h"
#include "snt_tts.h"

extern const uint8_t front_start[] asm("_binary_front_q8_bin_start");
extern const uint8_t dec_start[] asm("_binary_model_q8_bin_start");
extern const uint8_t ids_start[] asm("_binary_e2e_ids_bin_start");
extern const uint8_t ids_end[] asm("_binary_e2e_ids_bin_end");
extern const uint8_t durs_start[] asm("_binary_e2e_durs_bin_start");
extern const uint8_t gold_start[] asm("_binary_e2e_audio_bin_start");
extern const uint8_t gold_end[] asm("_binary_e2e_audio_bin_end");

static int null_cb(const float *pcm, int n, void *user) {
    (void)pcm; (void)n; (void)user;
    return 0;
}

typedef struct {
    const float *gold;
    size_t n_gold, pos;
    float sa, sb, saa, sbb, sab;
} Corr;

static int corr_cb(const float *pcm, int n, void *user) {
    Corr *c = (Corr *)user;
    for (int i = 0; i < n && c->pos < c->n_gold; i++, c->pos++) {
        float a = pcm[i], b = c->gold[c->pos];
        c->sa += a; c->sb += b;
        c->saa += a * a; c->sbb += b * b; c->sab += a * b;
    }
    return 0;
}

void app_main(void) {
    printf("saanotts-mcu on ESP32-C3 (Tier S scalar)\n");
    printf("free heap: %u\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT));
    size_t arena_size = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT) - 8192;
    void *arena = heap_caps_malloc(arena_size, MALLOC_CAP_8BIT);
    printf("arena: %u bytes\n", (unsigned)arena_size);

    const int32_t *ids = (const int32_t *)ids_start;
    int n_ids = (int)((size_t)(ids_end - ids_start) / 4);
    snt_config cfg = {front_start, dec_start, arena, arena_size,
                      (const int32_t *)durs_start};
    snt_stats st;

    int rc = snt_synthesize(&cfg, ids, n_ids, null_cb, NULL, &st);
    double audio_ms = st.samples * 1000.0 / 22050.0;
    printf("RESULT rc %d frames %d | %.1f ms compute for %.1f ms audio -> %.2fx RT\n",
           rc, st.frames, st.elapsed_us / 1000.0, audio_ms,
           st.elapsed_us / 1000.0 / audio_ms);

    Corr c = {(const float *)gold_start, (size_t)(gold_end - gold_start) / 4, 0,
              0, 0, 0, 0, 0};
    rc = snt_synthesize(&cfg, ids, n_ids, corr_cb, &c, &st);
    double n = (double)c.pos;
    double cov = (double)c.sab - (double)c.sa * c.sb / n;
    double cr = cov / sqrt(((double)c.saa - (double)c.sa * c.sa / n) *
                           ((double)c.sbb - (double)c.sb * c.sb / n) + 1e-30);
    printf("RESULT corr %.6f (%u samples)\n", cr, (unsigned)c.pos);
    printf(cr > 0.98 ? "PASS\n" : "FAIL\n");
}
