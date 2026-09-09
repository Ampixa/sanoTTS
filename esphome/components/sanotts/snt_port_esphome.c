/* snt_port_esphome.c -- the ESP32-S3/ESP-IDF half of the snt_port.h contract
 * that is NOT covered by snt_kernels_esp32s3.c.
 *
 * snt_port.h needs six things from a port. Three of them (snt_dot_s8,
 * snt_matvec_s8, snt_weights_resident) come from snt_kernels_esp32s3.c on this
 * target and from snt_kernels_ref.c everywhere else. Two more
 * (snt_dot_s16s8/snt_matvec_s16s8) always come from snt_kernels_ref.c. This
 * file supplies the remaining two: snt_par_run and snt_now_us.
 *
 * DUAL CORE IS DELIBERATELY NOT USED HERE, and that is a measured decision, not
 * laziness. BOARDS.md records the second core as marginally SLOWER for this
 * workload on this part (0.3863 vs 0.3828 xRT, and 1.5856 vs 1.5825 scalar --
 * both within run-to-run spread). More importantly, this component runs inside
 * ESPHome: core 1 is where the WiFi/lwIP stacks do their work, and a synthesis
 * worker pinned there at configMAX_PRIORITIES-2 would starve them for the whole
 * utterance. A serial snt_par_run is correct by the contract's own definition
 * (the default implementation IS f(0, n, ctx)), so this costs nothing but the
 * speed the measurement says is not there.
 *
 * SPDX-License-Identifier: MIT
 */
#include "snt_port.h"

#include "esp_timer.h"

void snt_par_run(snt_par_fn f, int n, void *ctx) {
    /* Serial. See the header comment: the second core is not a win here and
     * belongs to the network stacks. */
    f(0, n, ctx);
}

int64_t snt_now_us(void) { return esp_timer_get_time(); }

/* snt_nano.c indexes a two-bank scratch array with this. With the serial
 * snt_par_run above, only bank 0 is ever live -- returning xPortGetCoreID()
 * here would be a correctness bug waiting for the day a caller runs
 * synthesis from a task pinned to core 1. */
int snt_scratch_id(void) { return 0; }
