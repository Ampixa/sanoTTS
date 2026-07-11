#pragma once
/* Minimal espeak-ng config for ESP32 phoneme-only (no audio/async/mbrola). */
#define HAVE_MKSTEMP 0
#define USE_ASYNC 0
#define USE_KLATT 0
#define USE_LIBPCAUDIO 0
#define USE_LIBSONIC 0
#define USE_MBROLA 0
#define USE_SPEECHPLAYER 0
#define PACKAGE_VERSION "1.52.0"
/* Override the broken default fallback macro (a comma-expr that fails under C23).
 * Cosmetic: we set the real data path at runtime via espeak_ng_InitializePath(). */
#define PATH_ESPEAK_DATA "/espeak-ng-data"
