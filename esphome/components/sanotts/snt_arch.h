/* snt_arch.h -- decide, in one place, whether this translation unit gets the
 * ESP32-S3 PIE SIMD kernels or the portable scalar ones.
 *
 * The library's promise is that it builds unmodified on every Arduino core,
 * so the SIMD path must be opt-out-safe and invisible everywhere else. It
 * turns on only when the compiler is genuinely targeting an ESP32-S3, which
 * we learn from the ESP-IDF sdkconfig the Arduino core generates. Define
 * SANOTTS_NO_S3_SIMD to force the scalar path (useful for A/B measurement).
 */
#ifndef SNT_ARCH_H
#define SNT_ARCH_H

#if defined(__has_include)
#  if __has_include("sdkconfig.h")
#    include "sdkconfig.h"
#  endif
#endif

#if defined(CONFIG_IDF_TARGET_ESP32S3) && defined(__XTENSA__) && !defined(SANOTTS_NO_S3_SIMD)
#  define SANOTTS_S3_SIMD 1
#else
#  define SANOTTS_S3_SIMD 0
#endif

#endif
