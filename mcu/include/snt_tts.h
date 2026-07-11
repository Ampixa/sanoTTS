/* snt_tts.h -- public API of the saanotts-mcu TTS runtime. */
#ifndef SNT_TTS_H
#define SNT_TTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* PCM sink: called with completed samples as synthesis progresses
 * (float today; int16 variant with the streaming build). Return 0 to
 * continue, nonzero to abort synthesis. */
typedef int (*snt_pcm_cb)(const float *pcm, int n_samples, void *user);

typedef struct {
    const void *front_blob;  /* duration + acoustic weights (flash ok)   */
    const void *dec_blob;    /* decoder weights (flash ok)               */
    void *arena;             /* caller-owned working memory, 16-aligned  */
    size_t arena_size;       /* see model header for the documented peak */
    const int32_t *dur_override; /* NULL in production; golden tests pass
                                    reference durations for frame-exact
                                    comparison against the float model  */
} snt_config;

typedef struct {
    int frames;              /* latent frames synthesized                */
    int samples;             /* PCM samples emitted                      */
    int64_t elapsed_us;      /* 0 if the port has no clock               */
} snt_stats;

/* Synthesize one utterance from Piper phoneme IDs. Blocking; emits PCM
 * through cb as frames complete. Returns 0 on success. */
int snt_synthesize(const snt_config *cfg,
                   const int32_t *phoneme_ids, int n_ids,
                   snt_pcm_cb cb, void *user, snt_stats *stats_out);

#ifdef __cplusplus
}
#endif
#endif
