/* snt_port_rvv.c -- RISC-V Vector (RVV 1.0) port of the saanotts-mcu
 * porting surface (see include/snt_port.h).
 *
 * Kernels are written with <riscv_vector.h> intrinsics, fully
 * vl-agnostic (no VLEN assumption): all loops strip-mine with
 * __riscv_vsetvl_* so the same binary is correct on VLEN=128
 * (SpacemiT K1 / CanMV K230) and any larger implementation.
 *
 * LMUL choice
 * -----------
 *   snt_dot_s8:    e8m2 inputs -> i16m4 products -> i32m8 accumulator.
 *                  A single dot product has no register-reuse pressure,
 *                  so the widest practical LMUL is used to minimise
 *                  strip count and vsetvl overhead. Register budget:
 *                  2 (a) + 2 (b) + 4 (prod) + 8 (acc) = 16 of 32.
 *
 *   snt_matvec_s8: e8m1 inputs -> i16m2 products -> i32m4 accumulators,
 *                  rows blocked by 4. The activation strip is loaded
 *                  ONCE per column strip and reused against 4 weight
 *                  rows (4 independent i32m4 accumulators live across
 *                  the column loop). Register budget:
 *                  1 (act) + 1 (w, reused) + 2 (prod, reused)
 *                  + 4*4 (acc) = 20 of 32. e8m1 (rather than m2) is
 *                  what makes a 4-row block fit; a 2-row m2 block would
 *                  amortise activation loads only 2x.
 *
 * Exactness: products of two int8 always fit in int16
 * (-128*-128 = 16384), and vwadd.wv accumulates them into int32 lanes
 * with ordinary wraparound -- bit-identical to the scalar reference in
 * src/snt_kernels_ref.c. The accumulate steps use the tail-undisturbed
 * (_tu) intrinsics so a short final strip (vl < VLMAX) cannot clobber
 * accumulator tail lanes under the default tail-agnostic policy.
 *
 * The whole RVV section is guarded by __riscv_v_intrinsic; without it
 * the file degrades to the scalar reference loops, so it compiles on
 * any host.
 */
#include <time.h>

#include "snt_port.h"

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>

int32_t snt_dot_s8(const int8_t *a, const int8_t *b, int len) {
    size_t vlmax32 = __riscv_vsetvlmax_e32m8();
    vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vlmax32);
    size_t n = (size_t)len;
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e8m2(n);
        vint8m2_t va = __riscv_vle8_v_i8m2(a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(b, vl);
        vint16m4_t p = __riscv_vwmul_vv_i16m4(va, vb, vl);
        /* tail-undisturbed: lanes >= vl keep their partial sums */
        acc = __riscv_vwadd_wv_i32m8_tu(acc, acc, p, vl);
        a += vl;
        b += vl;
        n -= vl;
    }
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
    vint32m1_t s = __riscv_vredsum_vs_i32m8_i32m1(acc, z, vlmax32);
    return __riscv_vmv_x_s_i32m1_i32(s);
}

/* Reduce one i32m4 accumulator to a scalar. */
static inline int32_t snt_rvv_hsum_i32m4(vint32m4_t acc, size_t vlmax32) {
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
    vint32m1_t s = __riscv_vredsum_vs_i32m4_i32m1(acc, z, vlmax32);
    return __riscv_vmv_x_s_i32m1_i32(s);
}

void snt_matvec_s8(const int8_t *act, const int8_t *w, int32_t *out,
                   int rows, int len) {
    size_t vlmax32 = __riscv_vsetvlmax_e32m4();
    int r = 0;

    /* 4-row blocks: load each activation strip once, use it 4 times. */
    for (; r + 4 <= rows; r += 4) {
        const int8_t *w0 = w + (long)r * len;
        const int8_t *w1 = w0 + len;
        const int8_t *w2 = w1 + len;
        const int8_t *w3 = w2 + len;
        vint32m4_t acc0 = __riscv_vmv_v_x_i32m4(0, vlmax32);
        vint32m4_t acc1 = __riscv_vmv_v_x_i32m4(0, vlmax32);
        vint32m4_t acc2 = __riscv_vmv_v_x_i32m4(0, vlmax32);
        vint32m4_t acc3 = __riscv_vmv_v_x_i32m4(0, vlmax32);
        size_t i = 0, n = (size_t)len;
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e8m1(n);
            vint8m1_t va = __riscv_vle8_v_i8m1(act + i, vl);
            vint16m2_t p;
            p = __riscv_vwmul_vv_i16m2(va, __riscv_vle8_v_i8m1(w0 + i, vl), vl);
            acc0 = __riscv_vwadd_wv_i32m4_tu(acc0, acc0, p, vl);
            p = __riscv_vwmul_vv_i16m2(va, __riscv_vle8_v_i8m1(w1 + i, vl), vl);
            acc1 = __riscv_vwadd_wv_i32m4_tu(acc1, acc1, p, vl);
            p = __riscv_vwmul_vv_i16m2(va, __riscv_vle8_v_i8m1(w2 + i, vl), vl);
            acc2 = __riscv_vwadd_wv_i32m4_tu(acc2, acc2, p, vl);
            p = __riscv_vwmul_vv_i16m2(va, __riscv_vle8_v_i8m1(w3 + i, vl), vl);
            acc3 = __riscv_vwadd_wv_i32m4_tu(acc3, acc3, p, vl);
            i += vl;
            n -= vl;
        }
        out[r + 0] = snt_rvv_hsum_i32m4(acc0, vlmax32);
        out[r + 1] = snt_rvv_hsum_i32m4(acc1, vlmax32);
        out[r + 2] = snt_rvv_hsum_i32m4(acc2, vlmax32);
        out[r + 3] = snt_rvv_hsum_i32m4(acc3, vlmax32);
    }

    /* remainder rows (rows % 4) */
    for (; r < rows; r++) {
        const int8_t *wr = w + (long)r * len;
        vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vlmax32);
        size_t i = 0, n = (size_t)len;
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e8m1(n);
            vint8m1_t va = __riscv_vle8_v_i8m1(act + i, vl);
            vint8m1_t vw = __riscv_vle8_v_i8m1(wr + i, vl);
            vint16m2_t p = __riscv_vwmul_vv_i16m2(va, vw, vl);
            acc = __riscv_vwadd_wv_i32m4_tu(acc, acc, p, vl);
            i += vl;
            n -= vl;
        }
        out[r] = snt_rvv_hsum_i32m4(acc, vlmax32);
    }
}

#else /* !__riscv_v_intrinsic: scalar fallback, identical semantics */

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

#endif /* __riscv_v_intrinsic */

/* ---- non-kernel port shims ------------------------------------------- */

int snt_weights_resident(const void *p) {
    (void)p;
    return 1; /* flat memory: everything readable by the kernels */
}

void snt_par_run(snt_par_fn f, int n, void *ctx) {
    f(0, n, ctx); /* serial stub */
}

int snt_scratch_id(void) { return 0; }

int64_t snt_now_us(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (int64_t)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}
