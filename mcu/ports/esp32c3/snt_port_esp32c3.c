/* snt_port_esp32c3.c -- ESP32-C3 port (Tier S): scalar kernels, single
 * core, no FPU. Correctness-identical to every other port by contract;
 * speed comes later (int16 activation refactor is the known lever).
 */
#include <stdint.h>
#include "esp_timer.h"
#include "snt_port.h"

int32_t snt_dot_s8(const int8_t *a, const int8_t *b, int len) {
    int32_t acc = 0;
    /* 4x unroll: RV32IMC has no SIMD; keep the loop tight for the C3 */
    int i = 0;
    for (; i + 4 <= len; i += 4)
        acc += (int32_t)a[i] * b[i] + (int32_t)a[i + 1] * b[i + 1]
             + (int32_t)a[i + 2] * b[i + 2] + (int32_t)a[i + 3] * b[i + 3];
    for (; i < len; i++) acc += (int32_t)a[i] * b[i];
    return acc;
}

void snt_matvec_s8(const int8_t *act, const int8_t *w, int32_t *out,
                   int rows, int len) {
    for (int r = 0; r < rows; r++)
        out[r] = snt_dot_s8(act, w + (long)r * len, len);
}

/* C3 address space is flat for scalar reads; nothing needs staging,
 * but returning 0 for flash keeps the residency copies, which still
 * pay for themselves (flash cache misses on random access). */
int32_t snt_dot_s16s8(const int16_t *a, const int8_t *b, int len) {
    int32_t acc = 0;
    int i = 0;
    for (; i + 4 <= len; i += 4)
        acc += (int32_t)a[i] * b[i] + (int32_t)a[i + 1] * b[i + 1]
             + (int32_t)a[i + 2] * b[i + 2] + (int32_t)a[i + 3] * b[i + 3];
    for (; i < len; i++) acc += (int32_t)a[i] * b[i];
    return acc;
}

void snt_matvec_s16s8(const int16_t *act, const int8_t *w, int32_t *out,
                      int rows, int len) {
    for (int r = 0; r < rows; r++)
        out[r] = snt_dot_s16s8(act, w + (long)r * len, len);
}

int snt_weights_resident(const void *p) {
    uint32_t a = (uint32_t)p;
    return a >= 0x3FC80000u && a < 0x3FD00000u; /* C3 internal SRAM */
}

void snt_par_run(snt_par_fn f, int n, void *ctx) { f(0, n, ctx); }
int snt_scratch_id(void) { return 0; }
int64_t snt_now_us(void) { return esp_timer_get_time(); }
