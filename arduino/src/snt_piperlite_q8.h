/* snt_piperlite_q8.h -- int8 inference of the piperlite decoder.
 *
 * Quantization scheme (blobs from tools/export_piperlite_q8.py):
 *   - weights: symmetric per-output-channel int8 (max|row|/127), biases fp32
 *     -- the int8 payload is the distribution blob
 *   - activations: int16 lane holding 12-bit values (sat +/-2047), symmetric
 *     per-tensor STATIC scales calibrated offline over real eval128 latents
 *     (one scale per named tensor: z, pre, per stage act/up/branch-t1/y1/t2/
 *     mix, post-conv input "ap"). Pure int8 activations were measured first
 *     and top out at ~0.985 audio corr (heavy-tailed tensors, error spread
 *     across all stages); 12 bits keeps every int32 accumulator safe.
 *   - convolutions accumulate int16 x int8 -> int32, then requantize with an
 *     fp32 multiplier (s_in * s_w[oc] / s_out); residual adds happen at the
 *     requant point in the shared output scale; the residual-bank mean (/3)
 *     is folded into the bank-output requant
 *   - the final post conv requantizes straight to fp32 (pre_tanh is never
 *     quantized -- the 0.01-slope/tanh sensitivity), tanh + the optional
 *     post filter run in fp32 with weights dequantized on the fly.
 *
 * meta_q8.bin carries dims, the activation-scale table, per-tensor kind /
 * offset / size and per-channel weight-scale offsets, so this code hardcodes
 * no shapes. weights_q8.bin is pure int8 payload (the distribution blob).
 */
#ifndef SNT_PIPERLITE_Q8_H
#define SNT_PIPERLITE_Q8_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNT_PIPERLITE_Q8_MAGIC 0x534E5051L /* 'SNPQ' */
#define SNT_PIPERLITE_Q8_HOP 256
#define SNT_PIPERLITE_Q8_MAX_TENSORS 64
#define SNT_PIPERLITE_Q8_ACTS 39

typedef struct {
    int in_ch, c0, c1, c2, c3;
    int pf_channels, pf_layers, pf_kernel;
    float pf_scale;
    int n_tensors;
    /* slot order identical to the fp32 exporter; per slot exactly one of
     * wq (int8 weights, with wscale = per-out-channel scales) or f32
     * (bias / pf unit scalar) is non-NULL. */
    const int8_t *wq[SNT_PIPERLITE_Q8_MAX_TENSORS];
    const float *wscale[SNT_PIPERLITE_Q8_MAX_TENSORS];
    const float *f32[SNT_PIPERLITE_Q8_MAX_TENSORS];
    /* activation scales, exporter-documented order:
     * [0]=z [1]=pre; stage s in 0..2 at base 2+12s: +0 act, +1 up,
     * +2+3b..+4+3b = branch b t1/y1/t2, +11 mix; [38]=ap */
    float act[SNT_PIPERLITE_Q8_ACTS];
    /* bring-up tap: dequantized fp32 view of each stage tensor, same names
     * as the fp32 runtime. NULL in production. */
    void (*stage_cb)(const char *name, const float *data, int ch, int len,
                     void *user);
    void *stage_user;
} snt_piperlite_q8_model;

/* Parse meta_q8.bin + bind pointers into the int8 blob (flash ok).
 * Returns 0 on success, negative on malformed/out-of-range meta. */
int snt_piperlite_q8_init(snt_piperlite_q8_model *m,
                          const void *meta, size_t meta_bytes,
                          const int8_t *weights, size_t weight_bytes);

/* Arena bytes needed for `frames` latent frames (16-aligned base). */
size_t snt_piperlite_q8_arena_bytes(const snt_piperlite_q8_model *m,
                                    int frames);

/* z is fp32 [in_ch, frames] channel-major; audio_out receives
 * frames*SNT_PIPERLITE_Q8_HOP fp32 samples. No malloc inside. */
int snt_piperlite_q8_synthesize(const snt_piperlite_q8_model *m,
                                const float *z, int frames,
                                float *audio_out,
                                void *arena, size_t arena_bytes);

#ifdef __cplusplus
}
#endif
#endif
