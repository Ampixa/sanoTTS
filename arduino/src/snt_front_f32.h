/* snt_front_f32.h -- portable fp32 reference of the saanoTTS "front half":
 * phoneme ids -> per-token frame counts (DurationStudent,
 * tools/train_roota_piper_duration_student.py) -> expanded features ->
 * latent [out_channels, frames] (ContextualLatentStudent "token_context",
 * tools/train_roota_piper_latent_student.py, plus the optional
 * CalibratedLatentStudent output adapter). The latent output is channel-major
 * and feeds snt_piperlite_synthesize directly.
 *
 * Duration semantics mirror the dashboard inference path
 * (duration_source="student"): exp(log_dur) -> clamp_min(1) -> * length_scale
 * -> round-half-to-even -> clamp(1, max_duration). length_scale is a runtime
 * parameter; goldens gate exact integer parity with PyTorch.
 *
 * Weights/meta come from tools/export_front_golden.py: front_weights_f32.bin
 * (fixed tensor order, documented in the exporter) plus meta.bin (dims +
 * per-slot offset table), so this code hardcodes no shapes.
 */
#ifndef SNT_FRONT_F32_H
#define SNT_FRONT_F32_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNT_FRONT_MAGIC 0x534E4652L /* 'SNFR' */
#define SNT_FRONT_MAX_TENSORS 96

/* adapter_mode values (match the exporter) */
#define SNT_FRONT_ADAPTER_NONE 0
#define SNT_FRONT_ADAPTER_AFFINE 1
#define SNT_FRONT_ADAPTER_DEPTHWISE 2
#define SNT_FRONT_ADAPTER_LOWRANK 3
#define SNT_FRONT_ADAPTER_DEPTHWISE_LOWRANK 4

typedef struct {
    /* duration student dims (from meta.bin) */
    int d_vocab, d_hidden, d_depth, d_kernel;
    int d_max_tokens, d_max_duration;
    /* acoustic student dims */
    int a_vocab, a_hidden, a_token_depth, a_depth, a_kernel, a_out;
    /* optional output adapter (calibrated checkpoints) */
    int adapter_mode, adapter_kernel, adapter_rank;
    int n_tensors;
    /* per-slot weight pointers, in exporter slot order */
    const float *w[SNT_FRONT_MAX_TENSORS];
    /* optional bring-up tap: called with fp32 intermediates
     * ("dur_feats", "dur_log", "tok_ctx", "frame_feats", "latent_base",
     * "latent"). NULL in production. */
    void (*stage_cb)(const char *name, const float *data, int ch, int len,
                     void *user);
    void *stage_user;
} snt_front_model;

/* Parse meta.bin + bind weight pointers. weights must stay alive for the
 * model's lifetime (flash ok). Returns 0 on success, negative on a malformed
 * or out-of-range meta blob. */
int snt_front_init(snt_front_model *m,
                   const void *meta, size_t meta_bytes,
                   const float *weights, size_t weight_floats);

/* Working-memory floats needed by snt_front_durations for n_tokens ids. */
size_t snt_front_duration_arena_floats(const snt_front_model *m, int n_tokens);

/* ids[n_tokens] -> dur_out[n_tokens] integer frame counts (each in
 * [1, d_max_duration]). length_scale stretches/compresses speech rate
 * (1.0 = trained rate). Returns the total frame count (sum of dur_out) on
 * success, negative on bad args / out-of-range id / undersized arena. */
long snt_front_durations(const snt_front_model *m,
                         const int32_t *ids, int n_tokens,
                         float length_scale,
                         int32_t *dur_out,
                         float *arena, size_t arena_floats);

/* Working-memory floats needed by snt_front_latent for n_tokens ids
 * expanding to `frames` total frames. */
size_t snt_front_latent_arena_floats(const snt_front_model *m,
                                     int n_tokens, long frames);

/* ids[n_tokens] + durations[n_tokens] (from snt_front_durations; every entry
 * >= 1, summing to `frames`) -> latent_out[a_out * frames], channel-major
 * [a_out][frames] -- the exact layout snt_piperlite_synthesize consumes as z.
 * Returns 0 on success, negative on bad args / undersized arena. */
int snt_front_latent(const snt_front_model *m,
                     const int32_t *ids, const int32_t *durations,
                     int n_tokens, long frames,
                     float *latent_out,
                     float *arena, size_t arena_floats);

#ifdef __cplusplus
}
#endif
#endif
