/* snt_kernels_ref.c -- scalar reference kernels (Tier S default).
 * These define the EXACT integer semantics every optimized port must
 * reproduce: plain int32 accumulation, no saturation, no rounding.
 */
#include "snt_port.h"

#include "snt_arch.h"

/* On ESP32-S3 these three live in snt_kernels_esp32s3.c, which implements
 * them with PIE SIMD. Exactly one definition of each exists in any build. */
#if !SANOTTS_S3_SIMD
int32_t snt_dot_s8(const int8_t *a, const int8_t *b, int len) {
    int32_t acc = 0;
    for (int i = 0; i < len; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

void snt_matvec_s8(const int8_t *act, const int8_t *w, int32_t *out,
                   int rows, int len) {
    for (int r = 0; r < rows; r++)
        out[r] = snt_dot_s8(act, w + (long)r * len, len);
}
#endif /* !SANOTTS_S3_SIMD */

/* The int16 x int8 kernels have no SIMD variant, so they are always built. */
int32_t snt_dot_s16s8(const int16_t *a, const int8_t *b, int len) {
    int32_t acc = 0;
    for (int i = 0; i < len; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

void snt_matvec_s16s8(const int16_t *act, const int8_t *w, int32_t *out,
                      int rows, int len) {
    for (int r = 0; r < rows; r++)
        out[r] = snt_dot_s16s8(act, w + (long)r * len, len);
}

#if !SANOTTS_S3_SIMD
int snt_weights_resident(const void *p) {
    (void)p;
    return 1; /* hosts and flat-memory MCUs: everything readable */
}
#endif /* !SANOTTS_S3_SIMD */

