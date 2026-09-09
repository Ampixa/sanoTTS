/* nano_g2p_table.h -- GENERATED, DO NOT EDIT BY HAND.
 *
 * Regenerate with:   python3 esphome/g2p/gen_nano_g2p_table.py
 * Verify with:       python3 esphome/g2p/gen_nano_g2p_table.py --check
 * Generated:         2026-09-09
 *
 * The 62-symbol token vocabulary of the `en_us_e12nano` voice (294,642
 * params, Kokoro af_heart lineage) plus the espeak-IPA -> misaki-character
 * rewrite table needed to reach that alphabet from raw espeak-ng output.
 *
 * This is NOT the 157-entry Piper table in
 * mcu/ports/esp32s3/firmware/main/cp_id_table.h -- that one belongs to the
 * Kristin lineage and is wrong for this voice.
 *
 * SOURCES (sha256 at generation time):
 *   artifacts/kokoro-corpus-af_heart-20260713/kokoro_vocab.json
 *     c2086245349a5a398dd0e4697e3d57178a096e0dbd055f647f10d12c21d8eceb
 *   pypkg/sanotts/nano_frontend.py
 *     d0de0009e01b5295c45d7bff34f0a25124d49ec323cabbdbb00ad6d853e9f304
 *   web/trellis_frontend.js
 *     1d8523bab2e96f03ea98131a3d575d92851d3870899435ba3a306804238e49b3
 *   src/saanotts/kokoro_frontend.py
 *     aef6eaafcc1ce49bd98d1fd091223dc2fc6ca5021377936101e5a4a1987768b7
 *
 * The rewrite rules are transcribed from nano_frontend.py's E2M table and
 * apply_e2m() tail, which web/trellis_frontend.js records as a verbatim dump
 * of misaki.espeak.EspeakFallback.E2M. They are not reconstructed from
 * memory and must not be 'improved': they are the model's input contract.
 */

#ifndef NANO_G2P_TABLE_H
#define NANO_G2P_TABLE_H

#include <stdint.h>

/* Special tokens. Not codepoints, so they are not in NANO_CP_ID. */
#define NANO_ID_PAD 0
#define NANO_ID_BOS 1
#define NANO_ID_EOS 2

#define NANO_VOCAB_SIZE 62

/* configs.front.max_tokens, including <bos> and <eos>. Mirrors
 * nano_frontend.DEFAULT_MAX_TOKENS / trellis_frontend.js DEFAULT_MAX_TOKENS. */
#define NANO_MAX_TOKENS 207

typedef struct {
    uint32_t cp;   /* Unicode codepoint */
    int16_t  id;   /* token id in the 62-entry corpus vocabulary */
} nano_cp_id_t;

/* 59 single-codepoint symbols, sorted by codepoint for binary search.
 * ASCII 'A I O T W Y' are misaki's diphthong/flap letters, not letters:
 *   A = /eɪ/   I = /aɪ/   O = /oʊ/   W = /aʊ/   Y = /ɔɪ/   T = alveolar flap
 * U+1D4A MODIFIER LETTER SMALL SCHWA and U+1D7B SMALL CAPITAL I WITH STROKE
 * are misaki-only symbols with no standard IPA spelling. */
static const nano_cp_id_t NANO_CP_ID[59] = {
    { 0x0020,  3 },  /* U+0020  ' ' */
    { 0x0021,  4 },  /* U+0021  '!' */
    { 0x0022,  5 },  /* U+0022  '"' */
    { 0x0028,  6 },  /* U+0028  '(' */
    { 0x0029,  7 },  /* U+0029  ')' */
    { 0x002C,  8 },  /* U+002C  ',' */
    { 0x002E,  9 },  /* U+002E  '.' */
    { 0x003A, 10 },  /* U+003A  ':' */
    { 0x003B, 11 },  /* U+003B  ';' */
    { 0x003F, 12 },  /* U+003F  '?' */
    { 0x0041, 13 },  /* U+0041  'A' */
    { 0x0049, 14 },  /* U+0049  'I' */
    { 0x004F, 15 },  /* U+004F  'O' */
    { 0x0054, 16 },  /* U+0054  'T' */
    { 0x0057, 17 },  /* U+0057  'W' */
    { 0x0059, 18 },  /* U+0059  'Y' */
    { 0x0062, 19 },  /* U+0062  'b' */
    { 0x0064, 20 },  /* U+0064  'd' */
    { 0x0066, 21 },  /* U+0066  'f' */
    { 0x0068, 22 },  /* U+0068  'h' */
    { 0x0069, 23 },  /* U+0069  'i' */
    { 0x006A, 24 },  /* U+006A  'j' */
    { 0x006B, 25 },  /* U+006B  'k' */
    { 0x006C, 26 },  /* U+006C  'l' */
    { 0x006D, 27 },  /* U+006D  'm' */
    { 0x006E, 28 },  /* U+006E  'n' */
    { 0x0070, 29 },  /* U+0070  'p' */
    { 0x0073, 30 },  /* U+0073  's' */
    { 0x0074, 31 },  /* U+0074  't' */
    { 0x0075, 32 },  /* U+0075  'u' */
    { 0x0076, 33 },  /* U+0076  'v' */
    { 0x0077, 34 },  /* U+0077  'w' */
    { 0x007A, 35 },  /* U+007A  'z' */
    { 0x00E6, 36 },  /* U+00E6  'æ' */
    { 0x00F0, 37 },  /* U+00F0  'ð' */
    { 0x014B, 38 },  /* U+014B  'ŋ' */
    { 0x0250, 39 },  /* U+0250  'ɐ' */
    { 0x0251, 40 },  /* U+0251  'ɑ' */
    { 0x0254, 41 },  /* U+0254  'ɔ' */
    { 0x0259, 42 },  /* U+0259  'ə' */
    { 0x025B, 43 },  /* U+025B  'ɛ' */
    { 0x025C, 44 },  /* U+025C  'ɜ' */
    { 0x0261, 45 },  /* U+0261  'ɡ' */
    { 0x026A, 46 },  /* U+026A  'ɪ' */
    { 0x0279, 47 },  /* U+0279  'ɹ' */
    { 0x0283, 48 },  /* U+0283  'ʃ' */
    { 0x028A, 49 },  /* U+028A  'ʊ' */
    { 0x028C, 50 },  /* U+028C  'ʌ' */
    { 0x0292, 51 },  /* U+0292  'ʒ' */
    { 0x02A4, 52 },  /* U+02A4  'ʤ' */
    { 0x02A7, 53 },  /* U+02A7  'ʧ' */
    { 0x02C8, 54 },  /* U+02C8  'ˈ' */
    { 0x02CC, 55 },  /* U+02CC  'ˌ' */
    { 0x03B8, 56 },  /* U+03B8  'θ' */
    { 0x1D4A, 57 },  /* U+1D4A  'ᵊ' */
    { 0x1D7B, 58 },  /* U+1D7B  'ᵻ' */
    { 0x2014, 59 },  /* U+2014  '—' */
    { 0x201C, 60 },  /* U+201C  '“' */
    { 0x201D, 61 },  /* U+201D  '”' */
};
#define NANO_CP_ID_COUNT 59

typedef struct {
    const char *from;   /* UTF-8 needle */
    const char *to;     /* UTF-8 replacement, may be empty */
} nano_rewrite_t;

/* espeak's tie, requested from espeak_TextToPhonemes() via
 * (espeakPHONEMES_IPA | espeakPHONEMES_TIE | (0x0361 << 8)); phonemizer
 * rewrites it to '^' before misaki sees it, and every diphthong rule below
 * matches on '^'. Ask espeak for untied IPA and none of them fire. */
#define NANO_TIE_ESPEAK "\xcd\xa1"   /* U+0361 COMBINING DOUBLE INVERTED BREVE */
#define NANO_TIE_MISAKI "^"
#define NANO_TIE_CODEPOINT 0x0361u

#define NANO_SYLLABIC "\xcc\xa9"     /* U+0329 COMBINING VERTICAL LINE BELOW */
#define NANO_SYLLABIC_CODEPOINT 0x0329u
#define NANO_SCHWA_MODIFIER "\xe1\xb5\x8a"  /* U+1D4A, the regex replacement */

/* misaki EspeakFallback.E2M, applied in order, each as a full
 * left-to-right non-overlapping pass. Order is load-bearing. */
static const nano_rewrite_t NANO_E2M[20] = {
    { "\xca\x94\xcb\x8cn\xcc\xa9", "\xca\x94n" },  /* U+0294 U+02CC U+006E U+0329 -> U+0294 U+006E */
    { "\xca\x94n\xcc\xa9", "\xca\x94n" },  /* U+0294 U+006E U+0329 -> U+0294 U+006E */
    { "a^\xc9\xaa", "I" },  /* U+0061 U+005E U+026A -> U+0049 */
    { "a^\xca\x8a", "W" },  /* U+0061 U+005E U+028A -> U+0057 */
    { "d^\xca\x92", "\xca\xa4" },  /* U+0064 U+005E U+0292 -> U+02A4 */
    { "e^\xc9\xaa", "A" },  /* U+0065 U+005E U+026A -> U+0041 */
    { "t^\xca\x83", "\xca\xa7" },  /* U+0074 U+005E U+0283 -> U+02A7 */
    { "\xc9\x94^\xc9\xaa", "Y" },  /* U+0254 U+005E U+026A -> U+0059 */
    { "\xc9\x99^l", "\xe1\xb5\x8al" },  /* U+0259 U+005E U+006C -> U+1D4A U+006C */
    { "\xca\xb2o", "jo" },  /* U+02B2 U+006F -> U+006A U+006F */
    { "\xca\xb2\xc9\x99", "j\xc9\x99" },  /* U+02B2 U+0259 -> U+006A U+0259 */
    { "e", "A" },  /* U+0065 -> U+0041 */
    { "\xca\xb2", "" },  /* U+02B2 -> (delete) */
    { "\xc9\x9a", "\xc9\x99\xc9\xb9" },  /* U+025A -> U+0259 U+0279 */
    { "r", "\xc9\xb9" },  /* U+0072 -> U+0279 */
    { "x", "k" },  /* U+0078 -> U+006B */
    { "\xc3\xa7", "k" },  /* U+00E7 -> U+006B */
    { "\xc9\x90", "\xc9\x99" },  /* U+0250 -> U+0259 */
    { "\xc9\xac", "l" },  /* U+026C -> U+006C */
    { "\xcc\x83", "" },  /* U+0303 -> (delete) */
};
#define NANO_E2M_COUNT 20

/* apply_e2m()'s tail, applied AFTER the syllabic regex rewrite. */
static const nano_rewrite_t NANO_E2M_TAIL[9] = {
    { "o^\xca\x8a", "O" },  /* U+006F U+005E U+028A -> U+004F */
    { "\xc9\x9c\xcb\x90\xc9\xb9", "\xc9\x9c\xc9\xb9" },  /* U+025C U+02D0 U+0279 -> U+025C U+0279 */
    { "\xc9\x9c\xcb\x90", "\xc9\x9c\xc9\xb9" },  /* U+025C U+02D0 -> U+025C U+0279 */
    { "\xc9\xaa\xc9\x99", "i\xc9\x99" },  /* U+026A U+0259 -> U+0069 U+0259 */
    { "\xcb\x90", "" },  /* U+02D0 -> (delete) */
    { "o", "\xc9\x94" },  /* U+006F -> U+0254 */
    { "\xc9\xbe", "T" },  /* U+027E -> U+0054 */
    { "\xca\x94", "t" },  /* U+0294 -> U+0074 */
    { "^", "" },  /* U+005E -> (delete) */
};
#define NANO_E2M_TAIL_COUNT 9

/* pypkg/sanotts/frontend.py PUNCTUATION_MARKS, as codepoints.
 * phonemizer splits these out of the text BEFORE espeak sees it and splices
 * them back into the phoneme string afterwards. Skip that and every comma
 * and full stop -- which this vocabulary has ids for and the model was
 * trained to pause on -- is lost.
 *
 * This is the package's OVERRIDE, not phonemizer's default mark set and not
 * the DEFAULT_MARKS in web/trellis_frontend.js. Note it includes the
 * apostrophe and the hyphen, so "it's" and "twenty-two" are split into
 * separate espeak calls. That is measured behaviour of the shipped front
 * end, not an oversight to correct here. */
static const uint32_t NANO_PUNCT_MARKS[11] = {
    0x0021, 0x0022, 0x0027, 0x0028, 0x0029, 0x002C, 0x002D, 0x002E,
    0x003A, 0x003B, 0x003F,
};
#define NANO_PUNCT_MARK_COUNT 11

#endif /* NANO_G2P_TABLE_H */
