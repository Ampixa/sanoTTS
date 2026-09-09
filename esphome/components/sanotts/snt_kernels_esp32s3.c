/* snt_kernels_esp32s3.c -- ESP32-S3 int8 kernels via Xtensa LX7 PIE SIMD.
 *
 * Replaces the scalar int8 kernels in snt_kernels_ref.c when, and only when,
 * the target really is an ESP32-S3. Everything else in the library is
 * unchanged, and snt_kernels_ref.c compiles its scalar versions whenever this
 * file is inactive, so exactly one definition of each symbol always exists.
 *
 * Why this is worth vendoring: the same 294k stack measures 1.58 xRT with the
 * portable C kernels and 0.185 xRT with these -- 8.5x, from the kernels alone.
 * The PIE unit retires 16 int8 MACs per issue against the C loop's one.
 *
 * TWO HARD CONSTRAINTS, both measured rather than assumed:
 *
 *  1. PIE vector loads against flash-XIP addresses return garbage SILENTLY
 *     (measured corr 0.011 -- a plausible-looking waveform, not a crash). So
 *     snt_weights_resident() answers false for anything outside internal
 *     SRAM, and the runtime stages weights into the arena before dispatching.
 *     If the arena itself is not in SRAM, nothing is staged and every MAC
 *     takes the scalar path -- correct, just slow.
 *
 *  2. The assembly's contract: act and w 16-byte aligned, len a multiple of
 *     16, rows >= 1, w rows contiguous with stride == len, and at least 16
 *     readable bytes past the final row because the fused load prefetches one
 *     chunk beyond the end. The runtime's res_copy staging honours all of it
 *     (that is what RES_PAD is for); operands that do not are sent to the
 *     scalar loop rather than trusted.
 *
 * esp-nn is deliberately NOT required. The reference ESP-IDF port calls
 * esp_nn_dot_s8_aligned_esp32s3 for single-row dots, but esp-nn is not part
 * of the Arduino ESP32 core, and a dependency the user must vendor by hand
 * would defeat the point of a one-click library. Single-row dots run through
 * the generic matvec kernel instead, which is the same inner loop.
 */
#include "snt_arch.h"

#if SANOTTS_S3_SIMD

#include <stdint.h>

#include "snt_port.h"

extern void sn_matvec_s8_c3(const int8_t *act, const int8_t *w, int32_t *out, int rows);
extern void sn_matvec_s8_c5(const int8_t *act, const int8_t *w, int32_t *out, int rows);
extern void sn_matvec_s8_g(const int8_t *act, const int8_t *w, int32_t *out,
                           int rows, int chunks);

/* Internal SRAM only; see constraint 1 above. Range is the ESP32-S3 data bus
 * window for internal RAM, matching mcu/ports/esp32s3/snt_port_esp32s3.c. */
int snt_weights_resident(const void *p) {
    uint32_t a = (uint32_t)p;
    return a >= 0x3FC80000u && a < 0x3FD00000u;
}

/* ---- residency accounting -------------------------------------------------
 * ADDED for the ESPHome port (not present in the Arduino library copy).
 *
 * simd_ok() silently falls back to the scalar loop whenever an operand is not
 * SRAM-resident or not 16-byte aligned, and a scalar fallback is CORRECT --
 * just 4x slower. That makes "did the vector unit actually run?" invisible in
 * the output, and it is exactly the failure mode a too-small arena produces:
 * stage_buf() cannot allocate its resident copy, hands back the flash pointer,
 * and every MAC quietly takes the slow path. These counters turn that into a
 * measurement instead of an inference. Mirrors the accounting in
 * mcu/ports/esp32s3/snt_port_esp32s3.c.
 */
int64_t g_snt_macs_simd, g_snt_macs_scalar;
int64_t g_snt_calls_simd, g_snt_calls_scalar;

void snt_res_reset(void) {
    g_snt_macs_simd = g_snt_macs_scalar = 0;
    g_snt_calls_simd = g_snt_calls_scalar = 0;
}

/* Both operands must satisfy the assembly's alignment and length contract.
 * Checking is cheap next to the work, and a silent wrong answer is not. */
static inline int simd_ok(const void *act, const void *w, int len) {
    return snt_weights_resident(w) && (len & 15) == 0 && len > 0 &&
           (((uint32_t)act | (uint32_t)w) & 15u) == 0;
}

void snt_matvec_s8(const int8_t *act, const int8_t *w, int32_t *out,
                   int rows, int len) {
    if (rows > 0 && simd_ok(act, w, len)) {
        const int chunks = len >> 4;
        g_snt_calls_simd++;
        g_snt_macs_simd += (int64_t)rows * len;
        if (chunks == 3) { sn_matvec_s8_c3(act, w, out, rows); return; }
        if (chunks == 5) { sn_matvec_s8_c5(act, w, out, rows); return; }
        sn_matvec_s8_g(act, w, out, rows, chunks);
        return;
    }
    g_snt_calls_scalar++;
    g_snt_macs_scalar += (int64_t)rows * len;
    for (int r = 0; r < rows; r++) {
        int32_t acc = 0;
        for (int i = 0; i < len; i++)
            acc += (int32_t)act[i] * (int32_t)w[(long)r * len + i];
        out[r] = acc;
    }
}

int32_t snt_dot_s8(const int8_t *a, const int8_t *b, int len) {
    if (simd_ok(a, b, len)) {
        int32_t acc = 0;
        g_snt_calls_simd++;
        g_snt_macs_simd += len;
        sn_matvec_s8_g(a, b, &acc, 1, len >> 4);
        return acc;
    }
    g_snt_calls_scalar++;
    g_snt_macs_scalar += len;
    int32_t acc = 0;
    for (int i = 0; i < len; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

#endif /* SANOTTS_S3_SIMD */
