/* snt_matvec_esp32s3_asm.c -- the Xtensa LX7 PIE SIMD int8 matvec kernels,
 * carried as file-scope inline assembly inside a .c translation unit.
 *
 * WHY IT IS NOT A .S FILE. ESPHome copies an external component's sources
 * into the generated ESP-IDF project, and it only recognises the extensions
 * in esphome/const.py's SOURCE_FILE_EXTENSIONS -- {".cpp", ".hpp", ".h",
 * ".c", ".tcc", ".ino"}. A .S file is silently NOT copied, and the generated
 * src/CMakeLists.txt globs only C/C++ anyway, so it would not be assembled
 * even if it were. Putting the assembly in a basic file-scope asm block is
 * what lets an ESPHome external component ship hand-written vector kernels
 * with no ESP-IDF component plumbing at all.
 *
 * The instructions are a verbatim transcription of
 * arduino/src/snt_matvec_esp32s3.S (itself vendored from
 * mcu/ports/esp32s3/sn_matvec_s8_esp32s3.S). Only the comment syntax
 * changed: the original's `//` line comments are dropped here rather than
 * passed to the assembler, because inside an asm string they are not
 * reliably a comment introducer on every gas version.
 *
 * Contract (all three variants), unchanged from the original:
 *   - act and w 16-byte aligned; len a multiple of 16; rows >= 1
 *   - w rows contiguous with stride == len
 *   - w has >= 16 readable bytes past the last row (the fused load
 *     prefetches one chunk beyond the end)
 *   - out: one int32 per row
 *
 * SPDX-License-Identifier: MIT
 */
#include "snt_arch.h"

#if SANOTTS_S3_SIMD

/* clang-format off */
__asm__(
"    .text                                                           \n"
"    .align  4                                                       \n"
/* sn_matvec_s8_c3: len == 48 (3 chunks). act preloaded in q0..q2.
 * a2: act, a3: w, a4: out (int32*), a5: rows */
"    .type   sn_matvec_s8_c3, @function                              \n"
"    .align  4                                                       \n"
"    .global sn_matvec_s8_c3                                         \n"
"sn_matvec_s8_c3:                                                    \n"
"    entry   a1, 32                                                  \n"
"    ee.vld.128.ip   q0, a2, 16                                      \n"
"    ee.vld.128.ip   q1, a2, 16                                      \n"
"    ee.vld.128.ip   q2, a2, 16                                      \n"
"    ee.vld.128.ip   q7, a3, 16                                      \n"
"    loopgtz a5, .Lc3_end                                            \n"
"    ee.zero.accx                                                    \n"
"    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q0, q7                     \n"
"    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q1, q7                     \n"
"    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q2, q7                     \n"
"    nop                                                             \n"
"    rur.accx_0 a6                                                   \n"
"    s32i    a6, a4, 0                                               \n"
"    addi    a4, a4, 4                                               \n"
".Lc3_end:                                                           \n"
"    retw.n                                                          \n"
"    .size   sn_matvec_s8_c3, . - sn_matvec_s8_c3                    \n"
/* sn_matvec_s8_c5: len == 80 (5 chunks). act preloaded in q0..q4. */
"    .type   sn_matvec_s8_c5, @function                              \n"
"    .align  4                                                       \n"
"    .global sn_matvec_s8_c5                                         \n"
"sn_matvec_s8_c5:                                                    \n"
"    entry   a1, 32                                                  \n"
"    ee.vld.128.ip   q0, a2, 16                                      \n"
"    ee.vld.128.ip   q1, a2, 16                                      \n"
"    ee.vld.128.ip   q2, a2, 16                                      \n"
"    ee.vld.128.ip   q3, a2, 16                                      \n"
"    ee.vld.128.ip   q4, a2, 16                                      \n"
"    ee.vld.128.ip   q7, a3, 16                                      \n"
"    loopgtz a5, .Lc5_end                                            \n"
"    ee.zero.accx                                                    \n"
"    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q0, q7                     \n"
"    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q1, q7                     \n"
"    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q2, q7                     \n"
"    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q3, q7                     \n"
"    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q4, q7                     \n"
"    nop                                                             \n"
"    rur.accx_0 a6                                                   \n"
"    s32i    a6, a4, 0                                               \n"
"    addi    a4, a4, 4                                               \n"
".Lc5_end:                                                           \n"
"    retw.n                                                          \n"
"    .size   sn_matvec_s8_c5, . - sn_matvec_s8_c5                    \n"
/* sn_matvec_s8_g: generic len (chunks >= 1). act streamed per row.
 * a2: act, a3: w, a4: out, a5: rows, a6: chunks (len/16) */
"    .type   sn_matvec_s8_g, @function                               \n"
"    .align  4                                                       \n"
"    .global sn_matvec_s8_g                                          \n"
"sn_matvec_s8_g:                                                     \n"
"    entry   a1, 32                                                  \n"
"    ee.vld.128.ip   q7, a3, 16                                      \n"
".Lg_row:                                                            \n"
"    mov     a7, a2                                                  \n"
"    ee.zero.accx                                                    \n"
"    loopgtz a6, .Lg_inner_end                                       \n"
"    ee.vld.128.ip   q6, a7, 16                                      \n"
"    ee.vmulas.s8.accx.ld.ip  q7, a3, 16, q6, q7                     \n"
".Lg_inner_end:                                                      \n"
"    nop                                                             \n"
"    rur.accx_0 a8                                                   \n"
"    s32i    a8, a4, 0                                               \n"
"    addi    a4, a4, 4                                               \n"
"    addi    a5, a5, -1                                              \n"
"    bnez    a5, .Lg_row                                             \n"
"    retw.n                                                          \n"
"    .size   sn_matvec_s8_g, . - sn_matvec_s8_g                      \n"
);
/* clang-format on */

#endif /* SANOTTS_S3_SIMD */
