#pragma once
/* Minimal espeak-ng build configuration: translator only, no audio output,
 * no async, no MBROLA, no Klatt. Same set as
 * mcu/ports/esp32s3/firmware/components/espeak-ng/config.h, which is the
 * configuration the working ESP32-S3 firmware ships.
 *
 * Reached because the sources are compiled with -DHAVE_CONFIG_H and the
 * component root is on the include path.
 */

#define HAVE_MKSTEMP 0
#define USE_ASYNC 0
#define USE_KLATT 0
#define USE_LIBPCAUDIO 0
#define USE_LIBSONIC 0
#define USE_MBROLA 0
#define USE_SPEECHPLAYER 0

#define PACKAGE_VERSION "1.52.0"

/* Upstream's fallback for this macro is a comma expression that does not
 * survive a C23 preprocessor. Defining it here sidesteps that. The value is
 * cosmetic: the real data path is set at runtime by
 * espeak_ng_InitializePath(), which nano_g2p_init_path() calls with the SPIFFS
 * mount point ("/espeak" on the device, a real directory on the host). */
#define PATH_ESPEAK_DATA "/espeak-ng-data"
