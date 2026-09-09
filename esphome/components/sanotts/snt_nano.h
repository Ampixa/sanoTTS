/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Ampixa
 *
 * The sanoTTS inference runtime is MIT; see LICENSE.MIT for the exact file
 * list and why the split is sound. The repository as a whole is GPL-3.0,
 * because the grapheme-to-phoneme layer embeds espeak-ng. This file does not.
 */
/* snt_nano.h -- public API of the E12-nano runtime.
 *
 * Same shape as snt_tts.h (the R7 line): the caller hands the library two
 * flash-mapped blobs and ONE arena; the library never allocates. Platform
 * speed lives entirely behind snt_port.h.
 *
 * Pipeline: phoneme ids -> duration student -> acoustic student -> mel-100
 *           -> ConvNeXt1D decoder -> magnitude/phase -> iSTFT -> DC block -> PCM
 */
#ifndef SNT_NANO_H
#define SNT_NANO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const void *front_blob;   /* front_q8.bin: duration + acoustic          */
    const void *dec_blob;     /* model_q8.bin: TinyVocos decoder            */
    void *arena;              /* caller-owned scratch, 16-byte aligned      */
    size_t arena_size;
    const int32_t *dur_override; /* NULL, or n_ids frame counts (test hook) */
    uint64_t noise_seed;      /* sha256(row_id)[:8] big-endian; MT19937 uses
                               * only the low 32 bits, exactly as ATen does */
} snt_nano_config;

typedef struct {
    int frames;               /* mel frames T                              */
    int samples;              /* PCM samples emitted, == (T-1)*hop         */
    int64_t elapsed_us;
    size_t arena_peak;        /* high-water mark of the bump allocator     */
} snt_nano_stats;

/* Called with each finished PCM run. Return non-zero to abort. */
typedef int (*snt_nano_pcm_cb)(const float *pcm, int n, void *user);

/* 0 on success. Negative values are hard errors (see snt_nano.c). */
int snt_nano_synthesize(const snt_nano_config *cfg,
                        const int32_t *phoneme_ids, int n_ids,
                        snt_nano_pcm_cb cb, void *user,
                        snt_nano_stats *stats_out);

/* Exposed for the golden test: the deterministic noise generator and the
 * fixed-point sincos are the two genuinely new numeric primitives in this
 * lineage, so the harness checks them directly rather than only through the
 * end-to-end correlation. */
int snt_nano_seeded_noise(uint64_t seed, int channels, int frames, float *out);
int snt_nano_uniform_stream(uint64_t seed, int n, float *out);
void snt_nano_sincos(float phi, float *cos_out, float *sin_out);
int snt_nano_sha256_seed(const char *text, uint64_t *seed_out);

#ifdef __cplusplus
}
#endif
#endif
