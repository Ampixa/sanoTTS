/* snt_kernels_ref.c -- scalar reference kernels (Tier S default).
 * These define the EXACT integer semantics every optimized port must
 * reproduce: plain int32 accumulation, no saturation, no rounding.
 */
#include "snt_port.h"

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

int snt_weights_resident(const void *p) {
    (void)p;
    return 1; /* hosts and flat-memory MCUs: everything readable */
}
