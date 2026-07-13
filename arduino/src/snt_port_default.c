/* snt_port_default.c -- default SanoTTS Arduino port: the 3 remaining
 * pieces of the snt_port.h surface not already covered by
 * snt_kernels_ref.c (the 4 kernels + snt_weights_resident).
 *
 * This is the Tier-S (scalar, single-core) port: correctness-identical to
 * every other mcu/ port by contract, matching mcu/ports/host/snt_port_host.c
 * and mcu/ports/wasm/snt_port_wasm.c (serial par_run, one scratch bank),
 * with Arduino's micros() standing in for the platform clock. It runs
 * unmodified on every architecture this library declares support for
 * (esp32, esp32s3, esp32c3, rp2040, avr, samd, ...): no PIE asm, no
 * esp-nn, no CMSIS-NN.
 *
 * Optional ESP32 second-core worker: define SANOTTS_ESP32_DUALCORE
 * (build_flags / -D on PlatformIO, or a #define before including
 * SanoTTS.h then re-including this file's logic is NOT how the Arduino
 * library build works -- instead pass -D SANOTTS_ESP32_DUALCORE=1 as a
 * global build flag) to run snt_par_run's split half on FreeRTOS core 1
 * via xTaskCreatePinnedToCore + task-notify, exactly the pattern in
 * mcu/ports/esp32s3/snt_port_esp32s3.c. This is a genuine speed win (two
 * cores instead of one) using ONLY portable scalar kernels -- it is NOT
 * the esp-nn PIE SIMD path that gets the reference port to 0.22x RT. That
 * path requires vendoring espressif__esp-nn plus a hand Xtensa-PIE .S file
 * as an ESP-IDF component and is deliberately NOT part of this Arduino
 * library; see arduino/README.md and mcu/ports/esp32s3/README.md for how
 * to wire it into a full ESP-IDF (not Arduino) build if you need it.
 *
 * Call snt_port_dualcore_start() once from setup() (no-op, and declared
 * unconditionally so sketches compile either way) before your first
 * SanoTTS::synthesize() if you want the worker; without it (or on any
 * other architecture) snt_par_run runs everything on the calling core --
 * correct, just single-threaded.
 */
#include <stdint.h>
#include "snt_port.h"

#if defined(ARDUINO)
#include <Arduino.h>
#endif

#if defined(ARDUINO_ARCH_ESP32) && defined(SANOTTS_ESP32_DUALCORE)

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/* Workers MUST block while idle (busy-wait poisons the memory bus ~10%
 * on every single-core section -- measured on the esp32s3 reference
 * port); ulTaskNotifyTake blocks, so this holds on Arduino-ESP32 too. */
static volatile snt_par_fn s_fn;
static void *volatile s_ctx;
static volatile int s_lo, s_hi, s_done;
static TaskHandle_t s_worker;
static int s_up = 0;

static void snt_worker_task(void *arg) {
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

void snt_port_dualcore_start(void) {
    if (s_up) return;
    xTaskCreatePinnedToCore(snt_worker_task, "snttts_worker", 4096, NULL,
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

#else /* default: serial, single scratch bank, every other architecture */

void snt_par_run(snt_par_fn f, int n, void *ctx) { f(0, n, ctx); }
int snt_scratch_id(void) { return 0; }
/* Declared unconditionally (see header comment) so a sketch can call it
 * on non-ESP32 boards or without the dualcore flag and get a harmless
 * no-op instead of a link error. */
void snt_port_dualcore_start(void) { }

#endif

/* Profiling only (snt_stats.elapsed_us, or 0 if a port has no clock).
 * Arduino's micros() wraps at ~71 minutes on a 32-bit counter; fine here
 * since it is never used for anything but a single utterance's timing. */
int64_t snt_now_us(void) {
#if defined(ARDUINO)
    return (int64_t)micros();
#else
    return 0;
#endif
}
