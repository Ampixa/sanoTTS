/* snt_port.h -- the COMPLETE porting surface of saanotts-mcu.
 *
 * A port provides the symbols below (or accepts the weak/scalar
 * defaults) and must pass test/golden_main.c bit-exactness afterwards.
 * The scalar reference kernels in src/snt_kernels_ref.c define the
 * integer semantics every optimized kernel must reproduce exactly.
 */
#ifndef SNT_PORT_H
#define SNT_PORT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- kernels ---------------------------------------------------------
 * Contract (both):
 *   - len is a multiple of 16; both pointers 16-byte aligned
 *   - matvec weights are row-contiguous with stride == len and have
 *     >= 16 bytes of readable padding after the last row (prefetch)
 *   - exact int32 accumulation, no saturation, no requantization
 */
int32_t snt_dot_s8(const int8_t *a, const int8_t *b, int len);
void snt_matvec_s8(const int8_t *act, const int8_t *w, int32_t *out,
                   int rows, int len);

/* int16 activations x int8 weights: for activation tensors whose int8
 * grid measurably fails (residual-chain accumulation). Same layout and
 * padding contract; exact int32 accumulation (|act|<=2^15, len<=512:
 * no overflow). Optional: ports without it inherit the scalar default. */
int32_t snt_dot_s16s8(const int16_t *a, const int8_t *b, int len);
void snt_matvec_s16s8(const int16_t *act, const int8_t *w, int32_t *out,
                      int rows, int len);

/* May the SIMD kernels read from this address? (e.g. internal SRAM vs
 * flash-XIP). The core stages weights into resident buffers when this
 * returns 0; kernels are then only ever called on resident operands. */
int snt_weights_resident(const void *p);

/* ---- optional parallelism --------------------------------------------
 * Run f over [0,n) split across cores, returning only when ALL ranges
 * completed (barrier). Every call site is column-disjoint by design; a
 * port needs no model knowledge. The default runs f(0, n, ctx) serially.
 * Workers MUST block while idle (busy-wait poisons the memory bus). */
typedef void (*snt_par_fn)(int lo, int hi, void *ctx);
void snt_par_run(snt_par_fn f, int n, void *ctx);

/* ---- time (profiling only; may return 0) ---- */
int64_t snt_now_us(void);

#ifdef __cplusplus
}
#endif
#endif
