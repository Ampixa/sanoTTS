/* nano_lex_g2p.h -- espeak-free English text -> e12nano phoneme ids.
 *
 * A C99 port of the dictionary-first path in pypkg/sanotts/nano_g2p.py, which
 * is itself a port of misaki's `misaki/en.py` with spaCy and num2words
 * replaced. Nothing here links against espeak-ng, so nothing here is GPL: the
 * pronunciation data is misaki's us_gold.json / us_silver.json, Apache-2.0.
 * See esphome/g2p/lex/README.md and pypkg/sanotts/g2p_data/NOTICE.md.
 *
 * MEMORY
 *   All tables live in nano_lex_tables.c and are `const`, so they link into
 *   .rodata and are read in place out of flash; none of the ~3 MB of
 *   dictionary is copied to RAM. The module's own scratch is a single static
 *   block whose exact size nano_lex_g2p_workspace_bytes() reports; there is no
 *   malloc anywhere, and no allocation of any kind at run time.
 *
 * REENTRANCY
 *   Not reentrant. The static workspace is shared, so exactly one task may be
 *   inside nano_lex_g2p_text_to_ids() at a time. Callers that need more must
 *   serialise with their own mutex.
 *
 * STACK
 *   Written to stay under 2 KB of stack; the measured figure for the host
 *   build is in README.md. No recursion is unbounded: the only recursive
 *   function is the cardinal number speller, which recurses at most once.
 */

#ifndef NANO_LEX_G2P_H
#define NANO_LEX_G2P_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes. Every failure path returns exactly one of these; none of them
 * is shared between two causes, and nano_lex_g2p_strerror() names each. */
#define NANO_LEX_OK              0
#define NANO_LEX_E_NULL_ARG     (-1)   /* text or out was NULL */
#define NANO_LEX_E_BAD_CAP      (-2)   /* cap < 2, so BOS+EOS cannot fit */
#define NANO_LEX_E_EMPTY_TEXT   (-3)   /* text was empty or all whitespace */
#define NANO_LEX_E_BAD_UTF8     (-4)   /* input is not well-formed UTF-8 */
#define NANO_LEX_E_TEXT_LONG    (-5)   /* over NANO_LEX_MAX_CHARS codepoints */
#define NANO_LEX_E_TOKENS       (-6)   /* over NANO_LEX_MAX_TOKEN tokens */
#define NANO_LEX_E_ARENA        (-7)   /* phoneme workspace exhausted */
#define NANO_LEX_E_NO_SYMBOLS   (-8)   /* nothing survived the vocabulary filter */
#define NANO_LEX_E_CAP          (-9)   /* output did not fit `cap`; see below */
#define NANO_LEX_E_NUMBER       (-10)  /* a numeral this speller cannot say */
#define NANO_LEX_E_TABLE        (-11)  /* the packed table failed its own checks */
#define NANO_LEX_E_INTERNAL     (-12)  /* an invariant this port maintains broke */

/* Text -> phoneme ids for the en_us_e12nano voice.
 *
 * Writes the dense id sequence <bos> ... <eos> (id 1 first, id 2 last, no pad
 * and no blank interleaving) exactly as mcu/test/fixtures/en_us_e12nano/
 * r00_ids.bin is framed, and returns the number of ids written.
 *
 * Never writes more than `cap` ids. If the full sequence does not fit, the
 * function writes a well-framed prefix of exactly `cap` ids -- the first
 * cap-1 ids of the real sequence followed by <eos> -- and returns
 * NANO_LEX_E_CAP. That is a truncation the caller can detect and, if it wants
 * to, still speak; it is never silent and never overruns.
 *
 * On any other error nothing is written and a negative code is returned.
 */
int nano_lex_g2p_text_to_ids(const char *text, int32_t *out, int cap);

/* Human-readable name for a code from this module. Never NULL, including for
 * codes it does not recognise. */
const char *nano_lex_g2p_strerror(int rc);

/* Bytes of static workspace this module occupies (.bss), for the memory
 * ledger. Does not include the const tables, which are in flash. */
size_t nano_lex_g2p_workspace_bytes(void);

/* ---- everything below is diagnostic, and not part of the ids contract ---- */

/* Counters describing the last nano_lex_g2p_text_to_ids() or
 * nano_lex_g2p_text_to_phonemes() call. `oov_words` is the one that matters:
 * this port has no neural fallback, so a word that is in neither dictionary
 * and survives no stemmer contributes no phonemes at all. That is a defined,
 * countable outcome rather than a silent skip -- read it and log it. */
typedef struct {
    uint16_t tokens;          /* tokens after retokenize() */
    uint16_t lexicon_words;   /* word groups the lexicon was asked about */
    uint16_t oov_words;       /* of those, ones that produced no phonemes */
    uint16_t chunks;          /* chunks en_chunks() produced */
    uint16_t ids;             /* ids the last call would have written */
    uint16_t dropped;         /* phonemes outside the 62-symbol vocabulary */
    uint16_t arena_peak;      /* high-water mark of the phoneme workspace */
} nano_lex_g2p_stats_t;

void nano_lex_g2p_get_stats(nano_lex_g2p_stats_t *out);

/* The phoneme string the ids were built from, UTF-8, NUL-terminated. This is
 * `nano_g2p.phonemize_to_string()`: the chunks concatenated with no separator.
 * Returns the number of bytes written excluding the NUL, or a negative code.
 * Used by the host parity harness; a firmware build can drop it. */
int nano_lex_g2p_text_to_phonemes(const char *text, char *out, size_t cap);

/* The 62-symbol vocabulary's special ids, mirrored from nano_frontend.py and
 * repeated here so a caller need not include the generated table header. */
#ifndef NANO_LEX_ID_PAD
#define NANO_LEX_ID_PAD 0
#define NANO_LEX_ID_BOS 1
#define NANO_LEX_ID_EOS 2
#endif

/* configs.front.max_tokens for this voice, including <bos> and <eos>: the cap
 * a caller normally passes as `cap`. */
#ifndef NANO_LEX_MAX_TOKENS
#define NANO_LEX_MAX_TOKENS 207
#endif

/* Workspace limits, exposed so a caller can reject input before calling. */
#define NANO_LEX_MAX_CHARS 512   /* input codepoints, after UTF-8 decoding */
#define NANO_LEX_MAX_TOKEN 320   /* tokens after sub-tokenisation */

#ifdef __cplusplus
}
#endif

#endif /* NANO_LEX_G2P_H */
