/* nano_lex_tables.h -- GENERATED, DO NOT EDIT BY HAND.
 *
 * Regenerate with:   python3 esphome/g2p/lex/gen_lex_tables.py
 * Verify with:       python3 esphome/g2p/lex/gen_lex_tables.py --check
 * Generated:         2026-09-09
 *
 * misaki's American English pronunciation dictionaries, packed for
 * in-place lookup out of flash. Every array in nano_lex_tables.c is
 * `const` and therefore lands in .rodata; nothing here is copied to RAM.
 *
 * LICENCE: the dictionary data is misaki's us_gold.json / us_silver.json,
 * Apache-2.0, (c) hexgrad and the misaki contributors. Full provenance,
 * including the upstream commit, is in
 * pypkg/sanotts/g2p_data/NOTICE.md and pypkg/sanotts/g2p_data/LICENSE.misaki.txt.
 *
 * SOURCES (sha256 at generation time):
 *   pypkg/sanotts/g2p_data/us_gold.json
 *     dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f
 *     3000469 B, 90201 entries
 *   pypkg/sanotts/g2p_data/us_silver.json
 *     de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475
 *     3099517 B, 93361 entries
 *
 * PACKED SIZE:
 *   gold     90201 entries -> blob  1451195 B + index   45104 B =  1496299 B
 *   silver   93361 entries -> blob  1501167 B + index   46684 B =  1547851 B
 *   total   3044150 B
 */

#ifndef NANO_LEX_TABLES_H
#define NANO_LEX_TABLES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compile with -DNANO_LEX_WITH_SILVER=0 to drop us_silver.json entirely.
 * That saves 1547851 B of flash and costs accuracy on the words only silver has. */
#ifndef NANO_LEX_WITH_SILVER
#define NANO_LEX_WITH_SILVER 1
#endif

/* Every control byte and phoneme code in the blob is stored biased by '0'
 * so the whole table is printable ASCII and can be a string literal. */
#define NANO_LEX_BIAS '0'
#define NANO_LEX_UNBIAS(b) ((uint8_t)((uint8_t)(b) - (uint8_t)NANO_LEX_BIAS))

#define NANO_LEX_BLOCK 8
#define NANO_LEX_MAX_KEY 45
#define NANO_LEX_MAX_VALUE 59

/* vcode conventions, mirrored from gen_lex_tables.py's docstring. */
#define NANO_LEX_VCODE_NULL 0
#define NANO_LEX_VCODE_PLAIN_BASE 1
#define NANO_LEX_VCODE_TAGGED_BASE 61

/* ---- phoneme alphabet ------------------------------------------------ */

#define NANO_LEX_PH_COUNT 62

/* Codes 1..59 are the model vocabulary in token-id order, so
 * token id == code + 2 for those. Codes 60.. are intermediate-only
 * symbols with no id; NANO_LEX_PH_ID gives them -1. */
#define NLG_PH_0020 1  /* U+0020 */
#define NLG_PH_0021 2  /* U+0021 */
#define NLG_PH_0022 3  /* U+0022 */
#define NLG_PH_0028 4  /* U+0028 */
#define NLG_PH_0029 5  /* U+0029 */
#define NLG_PH_002C 6  /* U+002C */
#define NLG_PH_002E 7  /* U+002E */
#define NLG_PH_003A 8  /* U+003A */
#define NLG_PH_003B 9  /* U+003B */
#define NLG_PH_003F 10  /* U+003F */
#define NLG_PH_0041 11  /* U+0041 */
#define NLG_PH_0049 12  /* U+0049 */
#define NLG_PH_004F 13  /* U+004F */
#define NLG_PH_0054 14  /* U+0054 */
#define NLG_PH_0057 15  /* U+0057 */
#define NLG_PH_0059 16  /* U+0059 */
#define NLG_PH_0062 17  /* U+0062 */
#define NLG_PH_0064 18  /* U+0064 */
#define NLG_PH_0066 19  /* U+0066 */
#define NLG_PH_0068 20  /* U+0068 */
#define NLG_PH_0069 21  /* U+0069 */
#define NLG_PH_006A 22  /* U+006A */
#define NLG_PH_006B 23  /* U+006B */
#define NLG_PH_006C 24  /* U+006C */
#define NLG_PH_006D 25  /* U+006D */
#define NLG_PH_006E 26  /* U+006E */
#define NLG_PH_0070 27  /* U+0070 */
#define NLG_PH_0073 28  /* U+0073 */
#define NLG_PH_0074 29  /* U+0074 */
#define NLG_PH_0075 30  /* U+0075 */
#define NLG_PH_0076 31  /* U+0076 */
#define NLG_PH_0077 32  /* U+0077 */
#define NLG_PH_007A 33  /* U+007A */
#define NLG_PH_00E6 34  /* U+00E6 */
#define NLG_PH_00F0 35  /* U+00F0 */
#define NLG_PH_014B 36  /* U+014B */
#define NLG_PH_0250 37  /* U+0250 */
#define NLG_PH_0251 38  /* U+0251 */
#define NLG_PH_0254 39  /* U+0254 */
#define NLG_PH_0259 40  /* U+0259 */
#define NLG_PH_025B 41  /* U+025B */
#define NLG_PH_025C 42  /* U+025C */
#define NLG_PH_0261 43  /* U+0261 */
#define NLG_PH_026A 44  /* U+026A */
#define NLG_PH_0279 45  /* U+0279 */
#define NLG_PH_0283 46  /* U+0283 */
#define NLG_PH_028A 47  /* U+028A */
#define NLG_PH_028C 48  /* U+028C */
#define NLG_PH_0292 49  /* U+0292 */
#define NLG_PH_02A4 50  /* U+02A4 */
#define NLG_PH_02A7 51  /* U+02A7 */
#define NLG_PH_02C8 52  /* U+02C8 */
#define NLG_PH_02CC 53  /* U+02CC */
#define NLG_PH_03B8 54  /* U+03B8 */
#define NLG_PH_1D4A 55  /* U+1D4A */
#define NLG_PH_1D7B 56  /* U+1D7B */
#define NLG_PH_2014 57  /* U+2014 */
#define NLG_PH_201C 58  /* U+201C */
#define NLG_PH_201D 59  /* U+201D */
#define NLG_PH_027E 60  /* U+027E */
#define NLG_PH_0294 61  /* U+0294 */
#define NLG_PH_2026 62  /* U+2026 */

/* code -> 62-symbol token id, or -1 when the symbol has no id. */
extern const int16_t NANO_LEX_PH_ID[63];
/* code -> UTF-8 spelling, for diagnostics only. */
extern const char *const NANO_LEX_PH_UTF8[63];
/* code -> Unicode codepoint; index 0 is unused. */
extern const uint32_t NANO_LEX_PH_CP[63];
/* code -> bitmask of NANO_LEX_F_* */
extern const uint8_t NANO_LEX_PH_FLAGS[63];

#define NANO_LEX_F_VOWEL      0x01u
#define NANO_LEX_F_CONSONANT  0x02u
#define NANO_LEX_F_DIPHTHONG  0x04u
#define NANO_LEX_F_PUNCT      0x08u  /* misaki PUNCTS */
#define NANO_LEX_F_NONQ_PUNCT 0x10u  /* misaki NON_QUOTE_PUNCTS */
#define NANO_LEX_F_US_TAU     0x20u

/* ---- Penn tags ------------------------------------------------------- */

typedef enum {
    NLG_TAG_NONE = 0,
    NLG_TAG_HASH = 1,
    NLG_TAG_DOLLAR = 2,
    NLG_TAG_CLOSEQ = 3,
    NLG_TAG_COMMA = 4,
    NLG_TAG_LRB = 5,
    NLG_TAG_RRB = 6,
    NLG_TAG_PERIOD = 7,
    NLG_TAG_COLON = 8,
    NLG_TAG_OPENQ = 9,
    NLG_TAG_DQUOTE = 10,
    NLG_TAG_NFP = 11,
    NLG_TAG_HYPH = 12,
    NLG_TAG_ADD = 13,
    NLG_TAG_CC = 14,
    NLG_TAG_CD = 15,
    NLG_TAG_DT = 16,
    NLG_TAG_EX = 17,
    NLG_TAG_IN = 18,
    NLG_TAG_JJ = 19,
    NLG_TAG_MD = 20,
    NLG_TAG_NN = 21,
    NLG_TAG_NNP = 22,
    NLG_TAG_NNS = 23,
    NLG_TAG_PDT = 24,
    NLG_TAG_PRP = 25,
    NLG_TAG_PRPS = 26,
    NLG_TAG_RB = 27,
    NLG_TAG_RBR = 28,
    NLG_TAG_RBS = 29,
    NLG_TAG_TO = 30,
    NLG_TAG_VB = 31,
    NLG_TAG_VBD = 32,
    NLG_TAG_VBG = 33,
    NLG_TAG_VBN = 34,
    NLG_TAG_VBP = 35,
    NLG_TAG_VBZ = 36,
    NLG_TAG_WDT = 37,
    NLG_TAG_WP = 38,
    NLG_TAG_WPS = 39,
    NLG_TAG_WRB = 40,
    NLG_TAG_ADJ = 41,
    NLG_TAG_ADV = 42,
    NLG_TAG_NOUN = 43,
    NLG_TAG_VERB = 44,
    NLG_TAG_NONEKEY = 45,
    NLG_TAG_DEFAULT = 46,
    NLG_TAG_COUNT = 47
} nlg_tag_t;

extern const char *const NANO_LEX_TAG_NAME[NLG_TAG_COUNT];
/* get_parent_tag: VB.. -> VERB, NN.. -> NOUN, RB../ADV -> ADV, JJ../ADJ -> ADJ. */
extern const uint8_t NANO_LEX_TAG_PARENT[NLG_TAG_COUNT];

/* ---- packed dictionaries --------------------------------------------- */

typedef struct {
    const char     *blob;        /* biased, printable, NUL-terminated */
    const uint32_t *block_off;   /* byte offset of each block's first record */
    uint32_t        blob_bytes;
    uint32_t        block_count;
    uint32_t        entry_count;
} nano_lex_dict_t;

extern const nano_lex_dict_t NANO_LEX_GOLD;
#if NANO_LEX_WITH_SILVER
extern const nano_lex_dict_t NANO_LEX_SILVER;
#endif

/* ---- word lists the tokeniser needs ---------------------------------- */

typedef struct {
    const char *word;
    uint8_t     tag;
} nano_lex_word_tag_t;

#define NANO_LEX_CLOSED_COUNT 159
/* nano_g2p._CLOSED_CLASS, sorted by word for binary search. */
extern const nano_lex_word_tag_t NANO_LEX_CLOSED[NANO_LEX_CLOSED_COUNT];

#define NANO_LEX_ABBREV_COUNT 44
/* nano_g2p._ABBREVIATIONS, lower-cased and sorted. */
extern const char *const NANO_LEX_ABBREV[NANO_LEX_ABBREV_COUNT];

#define NANO_LEX_PERFECT_AUX_COUNT 11
/* nano_g2p._PERFECT_AUXILIARIES, sorted. */
extern const char *const NANO_LEX_PERFECT_AUX[NANO_LEX_PERFECT_AUX_COUNT];

/* nano_g2p._PUNCT_TAG_BY_CHAR: codepoint -> tag, sorted by codepoint. */
typedef struct {
    uint32_t cp;
    uint8_t  tag;
} nano_lex_punct_tag_t;
#define NANO_LEX_PUNCT_TAG_COUNT 24
extern const nano_lex_punct_tag_t NANO_LEX_PUNCT_TAG[NANO_LEX_PUNCT_TAG_COUNT];

/* nano_g2p._PREFIX_CHARS / _SUFFIX_CHARS as sorted codepoint arrays. */
#define NANO_LEX_PREFIX_COUNT 16
#define NANO_LEX_SUFFIX_COUNT 14
extern const uint32_t NANO_LEX_PREFIX_CP[NANO_LEX_PREFIX_COUNT];
extern const uint32_t NANO_LEX_SUFFIX_CP[NANO_LEX_SUFFIX_COUNT];

/* The 62-symbol vocabulary's specials, mirrored from nano_frontend.py. */
#define NANO_LEX_ID_PAD 0
#define NANO_LEX_ID_BOS 1
#define NANO_LEX_ID_EOS 2
#define NANO_LEX_MAX_TOKENS 207
#define NANO_LEX_MAX_PS_CHARS 510

#ifdef __cplusplus
}
#endif

#endif /* NANO_LEX_TABLES_H */
