//
// saanoTTS int8 matrix-vector kernels for ESP32-S3 (Xtensa LX7 PIE).
//
// Motivation (measured): calling esp_nn_dot_s8 once per output row spends
// ~90% of trunk time on call overhead and activation reloads. These kernels
// keep the activation vector in Q registers (specialized lens) or stream it
// (generic), and exploit row-contiguous weights so the fused MAC+load weight
// stream never resets across rows.
//
// Contract (all variants):
//   - act and w 16-byte aligned; len multiple of 16; rows >= 1
//   - w rows contiguous with stride == len
//   - w must have >= 16 bytes of readable padding after the last row
//     (the fused load prefetches one chunk past the end)
//   - out: int32 per row
//
    .text
    .align  4

// ============================================================
// sn_matvec_s8_c3: len == 48 (3 chunks). act preloaded in q0..q2.
// a2: act, a3: w, a4: out (int32*), a5: rows
// ============================================================
    .type   sn_matvec_s8_c3, @function
    .align  4
    .global sn_matvec_s8_c3
sn_matvec_s8_c3:
    entry   a1, 32
    ee.vld.128.ip   q0, a2, 16
    ee.vld.128.ip   q1, a2, 16
    ee.vld.128.ip   q2, a2, 16
    ee.vld.128.ip   q7, a3, 16          // prime first weight chunk
    loopgtz a5, .Lc3_end
    ee.zero.accx
    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q0, q7
    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q1, q7
    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q2, q7   // loads next row chunk0
    nop
    rur.accx_0 a6
    s32i    a6, a4, 0
    addi    a4, a4, 4
.Lc3_end:
    retw.n
    .size   sn_matvec_s8_c3, . - sn_matvec_s8_c3

// ============================================================
// sn_matvec_s8_c5: len == 80 (5 chunks). act preloaded in q0..q4.
// a2: act, a3: w, a4: out (int32*), a5: rows
// ============================================================
    .type   sn_matvec_s8_c5, @function
    .align  4
    .global sn_matvec_s8_c5
sn_matvec_s8_c5:
    entry   a1, 32
    ee.vld.128.ip   q0, a2, 16
    ee.vld.128.ip   q1, a2, 16
    ee.vld.128.ip   q2, a2, 16
    ee.vld.128.ip   q3, a2, 16
    ee.vld.128.ip   q4, a2, 16
    ee.vld.128.ip   q7, a3, 16
    loopgtz a5, .Lc5_end
    ee.zero.accx
    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q0, q7
    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q1, q7
    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q2, q7
    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q3, q7
    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q4, q7
    nop
    rur.accx_0 a6
    s32i    a6, a4, 0
    addi    a4, a4, 4
.Lc5_end:
    retw.n
    .size   sn_matvec_s8_c5, . - sn_matvec_s8_c5

// ============================================================
// sn_matvec_s8_g: generic len (chunks >= 1). act streamed per row.
// a2: act, a3: w, a4: out (int32*), a5: rows, a6: chunks (len/16)
// (inner loopgtz per row; weight stream continuous across rows)
// ============================================================
    .type   sn_matvec_s8_g, @function
    .align  4
    .global sn_matvec_s8_g
sn_matvec_s8_g:
    entry   a1, 32
    ee.vld.128.ip   q7, a3, 16          // prime weight stream
.Lg_row:
    mov     a7, a2                       // reset act pointer
    ee.zero.accx
    loopgtz a6, .Lg_inner_end
    ee.vld.128.ip   q6, a7, 16
    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q6, q7
.Lg_inner_end:
    nop
    rur.accx_0 a8
    s32i    a8, a4, 0
    addi    a4, a4, 4
    addi    a5, a5, -1
    bnez    a5, .Lg_row
    retw.n
    .size   sn_matvec_s8_g, . - sn_matvec_s8_g
