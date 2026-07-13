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

/* ---- SanoTTS Arduino-library addition (not upstream mcu/ API) --------
 * Sibilant fricative-noise injection: fixes the whistly/metallic
 * /s z sh zh/ artifact of the deterministic acoustic student. See the
 * block comment above sib_is_sibilant() in snt_tts.c for the full story;
 * ported from mcu/ports/esp32s3/firmware/main/fsd_e2e.c in the saanoTTS
 * repo, which never fed this fix back into the portable core.
 *
 * tea_std: pointer to FSD_CODE_DIM (40) per-channel teacher-latent std
 *   floats, voice-specific, from that voice's sibilant-injection
 *   calib.npz (tools/calibrate_sibilant_noise.py). Must stay alive for
 *   the lifetime of subsequent snt_synthesize() calls (no copy is made).
 * sib_ids / n_sib_ids: that voice's phoneme ids that count as sibilants
 *   (also voice- and phoneme-set-specific -- do not reuse another
 *   voice's ids).
 * beta: 0 disables injection (default state before this is ever called).
 *   ~0.9 matches the host-verified Kristin/r7 calibration; treat it as a
 *   per-voice dial, not a universal constant.
 *
 * Global, not per-call: configure once after begin(), before
 * synthesize(). Not thread-safe against a concurrent synthesize() call. */
void snt_sibilant_configure(const float *tea_std, const int32_t *sib_ids,
                            int n_sib_ids, float beta);

#ifdef __cplusplus
}
#endif
#endif
