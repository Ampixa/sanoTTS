/* nano_g2p.h -- on-device text -> phoneme ids for the `en_us_e12nano` voice.
 *
 * Turns arbitrary English text into the 62-symbol token id sequence the
 * 294,642-parameter nano stack was trained on, using espeak-ng 1.52.0 as the
 * grapheme-to-phoneme engine and the misaki E2M rewrite table to reach the
 * model's alphabet.
 *
 * The output shape matches the golden fixture
 * mcu/test/fixtures/en_us_e12nano/r00_ids.bin: a DENSE int32 sequence
 *
 *     <bos> t0 t1 ... tN-1 <eos>
 *
 * with NO blank/pad interleaving. That is the opposite of the Piper/Kristin
 * path in mcu/ports/esp32s3/firmware/main/esp_g2p.c, which emits
 * <bos> pad t0 pad t1 pad ... -- do not copy that framing here.
 *
 * LICENCE WARNING: this component links espeak-ng, which is GPL-3.0-or-later.
 * See README.md before shipping a binary.
 */

#ifndef NANO_G2P_H
#define NANO_G2P_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- limits --- */

/* Longest input accepted, in bytes of UTF-8. A sequence of 207 tokens is the
 * model's own cap (NANO_MAX_TOKENS in nano_g2p_table.h) and corresponds to
 * roughly 40 words, so 512 bytes of text is already more than can ever be
 * tokenised; the cap exists to bound the buffers, not to be generous. */
#define NANO_G2P_TEXT_MAX 512

/* Longest intermediate phoneme string, in bytes of UTF-8. Phoneme characters
 * are 1-3 bytes each and the espeak-side string still carries length marks,
 * ties and stress before E2M strips them. */
#define NANO_G2P_PS_MAX 1024

/* Punctuation splits one utterance into at most this many chunks. */
#define NANO_G2P_MAX_LINES 32
#define NANO_G2P_MAX_MARKS 32

/* ------------------------------------------------------- return codes --- */

#define NANO_G2P_OK 0
#define NANO_G2P_ERR_NOT_READY (-1)   /* nano_g2p_init() not called or failed */
#define NANO_G2P_ERR_ARG (-2)         /* NULL pointer or non-positive capacity */
#define NANO_G2P_ERR_MOUNT (-3)       /* SPIFFS mount failed (ESP-IDF only) */
#define NANO_G2P_ERR_ESPEAK_INIT (-4) /* espeak_ng_Initialize() failed */
#define NANO_G2P_ERR_VOICE (-5)       /* espeak_ng_SetVoiceByName("en-us") failed */
#define NANO_G2P_ERR_TEXT_LONG (-6)   /* input exceeds NANO_G2P_TEXT_MAX */
#define NANO_G2P_ERR_OVERFLOW (-7)    /* an internal buffer filled up */
#define NANO_G2P_ERR_UTF8 (-8)        /* input is not well-formed UTF-8 */
#define NANO_G2P_ERR_EMPTY (-9)       /* no symbol survived the vocabulary filter */
#define NANO_G2P_ERR_TOO_MANY_TOKENS (-10) /* > NANO_MAX_TOKENS including BOS/EOS */
#define NANO_G2P_ERR_ESPEAK (-11)     /* espeak refused to advance its text pointer */

/* Human-readable name for a return code. Never NULL. */
const char *nano_g2p_strerror(int rc);

/* ------------------------------------------------------------- api ------ */

/* Mount the espeak data and start the translator.
 *
 * On ESP-IDF this registers the SPIFFS partition labelled `espeak` at
 * `/espeak` and then calls espeak_ng_InitializePath("/espeak"). On any other
 * platform it skips the mount and uses `data_path` directly, which is what the
 * host parity harness does.
 *
 * Idempotent: a second successful call is a no-op. Returns NANO_G2P_OK or a
 * negative NANO_G2P_ERR_*.
 */
int nano_g2p_init_path(const char *data_path);

/* nano_g2p_init_path() with this port's default data path ("/espeak"). */
int nano_g2p_init(void);

/* Text -> dense phoneme ids.
 *
 * Writes `<bos> ... <eos>` into `out` and returns the number of ids written,
 * or a negative NANO_G2P_ERR_*. `cap` is the capacity of `out` in int32_t
 * elements; NANO_MAX_TOKENS (207) is always enough.
 *
 * The full chain, matching pypkg/sanotts/nano_frontend.py symbol for symbol:
 *   phonemizer Punctuation.preserve
 *     -> espeak-ng IPA with U+0361 ties, per chunk
 *     -> phonemizer EspeakBackend._postprocess_line
 *     -> phonemizer Punctuation.restore
 *     -> nano_frontend.postprocess_line (rewrites the tie to '^')
 *     -> misaki EspeakFallback E2M
 *     -> 62-symbol vocabulary filter with <bos>/<eos>
 */
int nano_g2p_text_to_ids(const char *text, int32_t *out, int cap);

/* The intermediate phoneme string, for debugging and for the host parity
 * harness. Writes a NUL-terminated UTF-8 string into `out` and returns its
 * length in bytes, or a negative NANO_G2P_ERR_*. */
int nano_g2p_text_to_phonemes(const char *text, char *out, int cap);

/* Bytes of static RAM this module occupies (its scratch workspace), so the
 * firmware can print an honest ledger line instead of an estimate. */
size_t nano_g2p_workspace_bytes(void);

#ifdef __cplusplus
}
#endif

#endif /* NANO_G2P_H */
