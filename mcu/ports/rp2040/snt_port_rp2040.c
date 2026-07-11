/* snt_port_rp2040.c -- Raspberry Pi RP2040 port (Tier S, dual-core).
 *
 * Dual Cortex-M0+ @133 MHz: no FPU, no SIMD, but single-cycle 32x32
 * multipliers and a REAL second core - snt_par_run maps onto core 1 via
 * the inter-core FIFO (blocking, no busy-wait: the FIFO IRQ wakes it).
 * Weights XIP from QSPI flash through the 16 KB cache; the arena
 * residency machinery matters here exactly as on ESP32.
 *
 * Status: compiles against pico-sdk; UNMEASURED pending hardware.
 * Projection from the ESP32-C3 baseline (same class, single core,
 * 160 MHz): ~2x the C3's per-core pace from the second core, i.e.
 * roughly 5-7x RT on the bundled voice with current optimizations.
 */
#include <stdint.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "snt_port.h"

int32_t snt_dot_s8(const int8_t *a, const int8_t *b, int len) {
    int32_t acc = 0;
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

int32_t snt_dot_s16s8(const int16_t *a, const int8_t *b, int len) {
    int32_t acc = 0;
    for (int i = 0; i < len; i++) acc += (int32_t)a[i] * b[i];
    return acc;
}

void snt_matvec_s16s8(const int16_t *act, const int8_t *w, int32_t *out,
                      int rows, int len) {
    for (int r = 0; r < rows; r++)
        out[r] = snt_dot_s16s8(act, w + (long)r * len, len);
}

/* SRAM is 0x20000000-0x20042000; XIP flash reads are scalar-safe but
 * slow on cache miss - stage what fits, exactly like the ESP32-C3. */
int snt_weights_resident(const void *p) {
    uint32_t a = (uint32_t)p;
    return a >= 0x20000000u && a < 0x20042000u;
}

static volatile snt_par_fn s_fn;
static void *volatile s_ctx;
static volatile int s_lo, s_hi, s_done;
static int s_up = 0;

static void worker_main(void) {
    for (;;) {
        (void)multicore_fifo_pop_blocking(); /* sleeps in IRQ, no bus noise */
        s_fn(s_lo, s_hi, (void *)s_ctx);
        __sync_synchronize();
        s_done = 1;
    }
}

int snt_scratch_id(void) { return (int)get_core_num(); }

void snt_port_rp2040_start_worker(void) {
    if (s_up) return;
    multicore_launch_core1(worker_main);
    s_up = 1;
}

void snt_par_run(snt_par_fn f, int n, void *ctx) {
    if (!s_up || n < 8) { f(0, n, ctx); return; }
    int mid = n / 2;
    s_fn = f; s_ctx = ctx; s_lo = mid; s_hi = n; s_done = 0;
    __sync_synchronize();
    multicore_fifo_push_blocking(1);
    f(0, mid, ctx);
    while (!s_done) { tight_loop_contents(); }
    __sync_synchronize();
}

int64_t snt_now_us(void) { return (int64_t)time_us_64(); }
