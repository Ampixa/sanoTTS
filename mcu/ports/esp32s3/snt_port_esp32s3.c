/* snt_port_esp32s3.c -- ESP32-S3 port: PIE matvec kernels, SRAM residency
 * dispatch, dual-core notification worker. This is the code path measured
 * at 0.22x RT; every construct here was validated on hardware in the lab
 * harness (esp32c3/fsd/fsd_e2e.c).
 */
#include <stdint.h>
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "snt_port.h"

extern void sn_matvec_s8_c3(const int8_t *act, const int8_t *w, int32_t *out, int rows);
extern void sn_matvec_s8_c5(const int8_t *act, const int8_t *w, int32_t *out, int rows);
extern void sn_matvec_s8_g(const int8_t *act, const int8_t *w, int32_t *out, int rows, int chunks);
extern int32_t esp_nn_dot_s8_aligned_esp32s3(const int8_t *a, const int8_t *b, int32_t len);

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

/* PIE vector loads from flash-XIP return garbage silently (measured,
 * corr 0.011): SIMD only on internal SRAM. */
int snt_weights_resident(const void *p) {
    uint32_t a = (uint32_t)p;
    return a >= 0x3FC80000u && a < 0x3FD00000u;
}

int32_t snt_dot_s8(const int8_t *a, const int8_t *b, int len) {
    if (snt_weights_resident(b)) return esp_nn_dot_s8_aligned_esp32s3(a, b, len);
    int32_t acc = 0;
    for (int i = 0; i < len; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

void snt_matvec_s8(const int8_t *act, const int8_t *w, int32_t *out,
                   int rows, int len) {
    if (snt_weights_resident(w)) {
        int chunks = len >> 4;
        if (chunks == 3) { sn_matvec_s8_c3(act, w, out, rows); return; }
        if (chunks == 5) { sn_matvec_s8_c5(act, w, out, rows); return; }
        sn_matvec_s8_g(act, w, out, rows, chunks);
        return;
    }
    for (int r = 0; r < rows; r++)
        out[r] = snt_dot_s8(act, w + (long)r * len, len);
}

/* dual-core worker: MUST block while idle (busy-wait costs ~10% memory-bus
 * contention on every single-core section - measured) */
static volatile snt_par_fn s_fn;
static void *volatile s_ctx;
static volatile int s_lo, s_hi, s_done;
static TaskHandle_t s_worker;
static int s_up = 0;

static void worker_task(void *arg) {
    (void)arg;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        __sync_synchronize();
        s_fn(s_lo, s_hi, (void *)s_ctx);
        __sync_synchronize();
        s_done = 1;
    }
}

int snt_scratch_id(void) { return xPortGetCoreID(); }

void snt_port_esp32s3_start_worker(void) {
    if (s_up) return;
    xTaskCreatePinnedToCore(worker_task, "sntwork", 4096, NULL,
                            configMAX_PRIORITIES - 2, &s_worker, 1);
    s_up = 1;
}

void snt_par_run(snt_par_fn f, int n, void *ctx) {
    if (!s_up || n < 8) { f(0, n, ctx); return; }
    int mid = n / 2;
    s_fn = f; s_ctx = ctx; s_lo = mid; s_hi = n; s_done = 0;
    __sync_synchronize();
    xTaskNotifyGive(s_worker);
    f(0, mid, ctx);
    while (!s_done) { }
    __sync_synchronize();
}

int64_t snt_now_us(void) { return esp_timer_get_time(); }
