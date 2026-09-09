/* nano_g2p.c -- text -> e12nano phoneme ids, on-chip.
 *
 * A C transcription of pypkg/sanotts/nano_frontend.py's `phonemize()`, which
 * is the front end the `sanotts` package ships and the one measured in
 * experiments/evidence/nano-frontend-ab-20260904.json. Every stage below names
 * the Python function it reproduces. The rules are the model's input contract:
 * a "nicer" rule is a wrong rule.
 *
 * The chain, in order:
 *
 *   1. phonemizer Punctuation.preserve       -- split punctuation out
 *   2. espeak_TextToPhonemes per chunk       -- IPA with U+0361 ties
 *   3. phonemizer EspeakBackend._postprocess_line
 *   4. phonemizer Punctuation.restore        -- splice punctuation back
 *   5. nano_frontend.postprocess_line        -- rewrites U+0361 -> '^'
 *   6. misaki EspeakFallback E2M             -- '^'-matching diphthong rules
 *   7. 62-symbol vocabulary filter, <bos> ... <eos>
 *
 * Step 2 MUST ask espeak for ties. Every diphthong rule in step 6 matches on
 * "a^ɪ", "e^ɪ", "o^ʊ", "a^ʊ", "ɔ^ɪ", "d^ʒ", "t^ʃ"; ask for plain IPA and not
 * one of them fires, and the model receives `ɪ`/`ʊ`/`ʃ`/`ʒ` everywhere it was
 * trained on `I`/`A`/`O`/`W`/`Y`/`ʤ`/`ʧ`.
 *
 * No dynamic allocation: one static workspace, sized by the constants in
 * nano_g2p.h and reported by nano_g2p_workspace_bytes().
 *
 * LICENCE: links espeak-ng, GPL-3.0-or-later. See README.md.
 */

#include "nano_g2p.h"

#include <stdio.h>
#include <string.h>

#include "nano_g2p_table.h"

#include "espeak-ng/espeak_ng.h"
#include "espeak-ng/speak_lib.h"

#ifdef ESP_PLATFORM
#include "esp_err.h"
#include "esp_spiffs.h"
#endif

/* Byte-for-byte the value phonemizer passes to espeak_TextToPhonemes() when
 * EspeakBackend is built with tie=True:
 *   phonemizer/backend/espeak/wrapper.py
 *     phonemes_mode = 0x02 | 0x01 << 7 | ord('͡') << 8
 * which is (espeakPHONEMES_IPA | espeakPHONEMES_TIE | (0x0361 << 8)).
 * Same constant as mcu/ports/wasm/snt_g2p_wasm.c. */
#define NANO_IPA_TIE_MODE \
    (espeakPHONEMES_IPA | espeakPHONEMES_TIE | ((int)NANO_TIE_CODEPOINT << 8))

/* espeak's own clause loop must terminate; this bounds a runaway. */
#define NANO_CLAUSE_GUARD 256

/* The apostrophe is in the shipped front end's punctuation set
 * (pypkg/sanotts/frontend.py PUNCTUATION_MARKS), so phonemizer splits every
 * contraction in two and espeak reads the orphaned tail as a letter name:
 *
 *     "It's unlocked."   -> ɪt'ˈɛs ʌnlˈɑkt.     ("it ess unlocked")
 *     "Don't forget"     -> dˈɑn'tˈi fəɹɡˈɛt    ("don tee forget")
 *     "You're home."     -> ju'ɹˌi hˈOm.        ("you ar home")
 *
 * That is measured behaviour of the front end the model ships with, not a bug
 * in this port -- esphome/g2p/host/verify_parity.py shows the C and the Python
 * agreeing on it byte for byte. It matters here because Home Assistant
 * announcements are contraction-heavy.
 *
 * Setting this to 0 keeps contractions whole. HOST-REF: measured on the 330-row
 * evalset in artifacts/nano-g2p-espeak-free-20260904 against the real misaki
 * reference, where only 35 of 330 rows contain an apostrophe at all:
 *
 *     1 (default, shipped-exact)  token error rate 6.0936%, 1895/31098 edits
 *     0 (contractions kept whole) token error rate 5.6756%, 1765/31098 edits
 *
 * So 0 is measurably CLOSER to what the model was trained on. It is not the
 * default only because 1 reproduces the shipped front end bit for bit, which is
 * the property the parity harness checks. Flip it deliberately, and re-run
 * verify_parity.py, which will then report a controlled mismatch against
 * shipped-espeak.jsonl and an improvement against misaki-reference.jsonl. */
#ifndef NANO_G2P_SPLIT_ON_APOSTROPHE
#define NANO_G2P_SPLIT_ON_APOSTROPHE 1
#endif

/* --------------------------------------------------------------- utf-8 --- */

/* Decode one codepoint. Returns bytes consumed (1..4), or 0 if the sequence is
 * malformed, truncated, overlong, a surrogate, or beyond U+10FFFF. Malformed
 * input is an error, never a silent substitution: the caller turns a 0 into
 * NANO_G2P_ERR_UTF8. */
static int utf8_next(const char *s, const char *end, uint32_t *cp)
{
    if (s >= end) {
        return 0;
    }
    const unsigned char *u = (const unsigned char *)s;
    unsigned char c0 = u[0];

    if (c0 < 0x80u) {
        *cp = c0;
        return 1;
    }
    int need;
    uint32_t value;
    uint32_t lowest;
    if ((c0 & 0xE0u) == 0xC0u) {
        need = 1;
        value = c0 & 0x1Fu;
        lowest = 0x80u;
    } else if ((c0 & 0xF0u) == 0xE0u) {
        need = 2;
        value = c0 & 0x0Fu;
        lowest = 0x800u;
    } else if ((c0 & 0xF8u) == 0xF0u) {
        need = 3;
        value = c0 & 0x07u;
        lowest = 0x10000u;
    } else {
        return 0; /* continuation byte or 0xF8..0xFF as a lead */
    }
    if (end - s < (long)(need + 1)) {
        return 0; /* truncated sequence */
    }
    for (int i = 1; i <= need; i++) {
        if ((u[i] & 0xC0u) != 0x80u) {
            return 0;
        }
        value = (value << 6) | (uint32_t)(u[i] & 0x3Fu);
    }
    if (value < lowest || value > 0x10FFFFu || (value >= 0xD800u && value <= 0xDFFFu)) {
        return 0;
    }
    *cp = value;
    return need + 1;
}

/* Python's `\s` for `str`, i.e. str.isspace(). Deliberately NOT JavaScript's
 * `\s`: JS adds U+FEFF and omits U+001C..U+001F and U+0085. The Python set is
 * the reference because the shipped front end is the Python one. */
static int is_ws(uint32_t cp)
{
    switch (cp) {
    case 0x0009u:
    case 0x000Au:
    case 0x000Bu:
    case 0x000Cu:
    case 0x000Du:
    case 0x001Cu:
    case 0x001Du:
    case 0x001Eu:
    case 0x001Fu:
    case 0x0020u:
    case 0x0085u:
    case 0x00A0u:
    case 0x1680u:
    case 0x2028u:
    case 0x2029u:
    case 0x202Fu:
    case 0x205Fu:
    case 0x3000u:
        return 1;
    default:
        return (cp >= 0x2000u && cp <= 0x200Au) ? 1 : 0;
    }
}

static int is_punct_mark(uint32_t cp)
{
#if !NANO_G2P_SPLIT_ON_APOSTROPHE
    if (cp == 0x0027u) { /* APOSTROPHE -- see the note at the top of this file */
        return 0;
    }
#endif
    int lo = 0;
    int hi = NANO_PUNCT_MARK_COUNT;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (NANO_PUNCT_MARKS[mid] < cp) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return (lo < NANO_PUNCT_MARK_COUNT && NANO_PUNCT_MARKS[lo] == cp) ? 1 : 0;
}

/* 62-entry vocabulary lookup. Returns the token id, or -1 for a symbol the
 * vocabulary has no id for (Kokoro's `drop_like_kokoro` policy). */
static int vocab_lookup(uint32_t cp)
{
    int lo = 0;
    int hi = NANO_CP_ID_COUNT;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (NANO_CP_ID[mid].cp < cp) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return (lo < NANO_CP_ID_COUNT && NANO_CP_ID[lo].cp == cp) ? (int)NANO_CP_ID[lo].id : -1;
}

/* ------------------------------------------------------ string builder --- */

typedef struct {
    char *buf;
    int cap; /* usable bytes, excluding the NUL terminator slot */
    int len;
    int overflow;
} sb_t;

static void sb_init(sb_t *sb, char *buf, int cap)
{
    sb->buf = buf;
    sb->cap = cap - 1; /* always keep room for the NUL */
    sb->len = 0;
    sb->overflow = 0;
    sb->buf[0] = '\0';
}

static void sb_putn(sb_t *sb, const char *src, int n)
{
    if (sb->overflow) {
        return;
    }
    if (n < 0 || sb->len + n > sb->cap) {
        sb->overflow = 1;
        return;
    }
    memcpy(sb->buf + sb->len, src, (size_t)n);
    sb->len += n;
    sb->buf[sb->len] = '\0';
}

static void sb_puts(sb_t *sb, const char *src)
{
    sb_putn(sb, src, (int)strlen(src));
}

/* ------------------------------------------------------ byte utilities --- */

/* First occurrence of `needle` in `hay`. UTF-8 is self-synchronizing, so a
 * byte-level search can never match across a character boundary spuriously:
 * every byte of a multi-byte sequence is >= 0x80 and no valid sequence is a
 * proper suffix-prefix of another. Returns the offset or -1. */
static int find_bytes(const char *hay, int hlen, const char *needle, int nlen)
{
    if (nlen == 0 || nlen > hlen) {
        return -1;
    }
    for (int i = 0; i + nlen <= hlen; i++) {
        if (hay[i] == needle[0] && memcmp(hay + i, needle, (size_t)nlen) == 0) {
            return i;
        }
    }
    return -1;
}

/* Python str.replace(needle, repl): one left-to-right non-overlapping pass.
 * NOT repeated to a fixed point -- "   ".replace("  ", " ") is "  ", and every
 * caller below depends on that. */
static void replace_all(sb_t *out, const char *src, int slen, const char *needle, const char *repl)
{
    int nlen = (int)strlen(needle);
    if (nlen == 0) {
        sb_putn(out, src, slen);
        return;
    }
    int rlen = (int)strlen(repl);
    int i = 0;
    while (i < slen) {
        if (slen - i >= nlen && src[i] == needle[0] && memcmp(src + i, needle, (size_t)nlen) == 0) {
            sb_putn(out, repl, rlen);
            i += nlen;
        } else {
            sb_putn(out, src + i, 1);
            i++;
        }
    }
}

/* re.sub(r'_+', '_', s) */
static void collapse_underscores(sb_t *out, const char *src, int slen)
{
    int i = 0;
    while (i < slen) {
        if (src[i] == '_') {
            sb_putn(out, "_", 1);
            while (i < slen && src[i] == '_') {
                i++;
            }
        } else {
            sb_putn(out, src + i, 1);
            i++;
        }
    }
}

/* re.sub(r'\(.+?\)', '', s) -- phonemizer's RemoveFlags language-switch policy.
 * `.` does not match '\n' and `+?` needs at least one character, so "()" is
 * left alone and a flag cannot span a newline. */
static void remove_lang_flags(sb_t *out, const char *src, int slen)
{
    int i = 0;
    while (i < slen) {
        if (src[i] == '(') {
            int j = i + 1;
            int close = -1;
            while (j < slen && src[j] != '\n') {
                if (src[j] == ')' && j > i + 1) {
                    close = j;
                    break;
                }
                j++;
            }
            if (close >= 0) {
                i = close + 1;
                continue;
            }
        }
        sb_putn(out, src + i, 1);
        i++;
    }
}

/* Python str.strip(): drop leading and trailing whitespace codepoints.
 * Adjusts *off and *len in place; returns 0 on malformed UTF-8. */
static int strip_ws(const char *src, int *off, int *len)
{
    int start = *off;
    int end = *off + *len;
    while (start < end) {
        uint32_t cp;
        int n = utf8_next(src + start, src + end, &cp);
        if (n == 0) {
            return 0;
        }
        if (!is_ws(cp)) {
            break;
        }
        start += n;
    }
    /* Walk forward from `start` remembering the last non-whitespace run end;
     * scanning backwards through UTF-8 is error-prone and this is short. */
    int last_end = start;
    int i = start;
    while (i < end) {
        uint32_t cp;
        int n = utf8_next(src + i, src + end, &cp);
        if (n == 0) {
            return 0;
        }
        i += n;
        if (!is_ws(cp)) {
            last_end = i;
        }
    }
    *off = start;
    *len = last_end - start;
    return 1;
}

/* ------------------------------------------------------------ workspace --- */

typedef struct {
    int16_t off;
    int16_t len;
} span_t;

typedef struct {
    /* Punctuation.preserve() output */
    char text_arena[NANO_G2P_TEXT_MAX];
    int text_used;
    span_t lines[NANO_G2P_MAX_LINES];
    int n_lines;

    char mark_arena[NANO_G2P_TEXT_MAX];
    int mark_used;
    span_t marks[NANO_G2P_MAX_MARKS];
    char mark_pos[NANO_G2P_MAX_MARKS]; /* 'B' | 'I' | 'E' | 'A' */
    int n_marks;

    /* One NUL-terminated chunk, handed to espeak. */
    char chunk[NANO_G2P_TEXT_MAX];

    /* _postprocess_line() output, one entry per preserved chunk. */
    char ph_arena[NANO_G2P_PS_MAX];
    int ph_used;
    span_t ph[NANO_G2P_MAX_LINES];
    int n_ph;

    /* Punctuation.restore()'s text[0] accumulator, and scratch. */
    char acc[NANO_G2P_PS_MAX];
    char scratch[NANO_G2P_PS_MAX];
    char ping[NANO_G2P_PS_MAX];
    char pong[NANO_G2P_PS_MAX];
} nano_ws_t;

static nano_ws_t g_ws;
static int g_ready;

size_t nano_g2p_workspace_bytes(void)
{
    return sizeof(g_ws);
}

const char *nano_g2p_strerror(int rc)
{
    switch (rc) {
    case NANO_G2P_OK:
        return "ok";
    case NANO_G2P_ERR_NOT_READY:
        return "nano_g2p_init() not called or failed";
    case NANO_G2P_ERR_ARG:
        return "invalid argument";
    case NANO_G2P_ERR_MOUNT:
        return "espeak SPIFFS partition would not mount";
    case NANO_G2P_ERR_ESPEAK_INIT:
        return "espeak_ng_Initialize failed";
    case NANO_G2P_ERR_VOICE:
        return "could not select the en-us voice";
    case NANO_G2P_ERR_TEXT_LONG:
        return "input text longer than NANO_G2P_TEXT_MAX";
    case NANO_G2P_ERR_OVERFLOW:
        return "internal buffer exhausted";
    case NANO_G2P_ERR_UTF8:
        return "input is not well-formed UTF-8";
    case NANO_G2P_ERR_EMPTY:
        return "no symbol survived the vocabulary filter";
    case NANO_G2P_ERR_TOO_MANY_TOKENS:
        return "phoneme sequence exceeds NANO_MAX_TOKENS";
    case NANO_G2P_ERR_ESPEAK:
        return "espeak did not advance its text pointer";
    default:
        return "unknown error";
    }
}

/* --------------------------------------------- 1. Punctuation.preserve --- */

/* One match of phonemizer's `(\s*[marks]+\s*)+`, searching from `from`.
 * Writes the match bounds and returns 1, or returns 0 when there is none.
 * Returns -1 on malformed UTF-8. */
static int next_mark_match(const char *s, int n, int from, int *m_start, int *m_end)
{
    for (int p = from; p <= n; p++) {
        int q = p;
        int matched = 0;
        for (;;) {
            int r = q;
            /* \s* */
            while (r < n) {
                uint32_t cp;
                int k = utf8_next(s + r, s + n, &cp);
                if (k == 0) {
                    return -1;
                }
                if (!is_ws(cp)) {
                    break;
                }
                r += k;
            }
            /* [marks]+ -- at least one, or this iteration fails */
            int marks_seen = 0;
            while (r < n) {
                uint32_t cp;
                int k = utf8_next(s + r, s + n, &cp);
                if (k == 0) {
                    return -1;
                }
                if (!is_punct_mark(cp)) {
                    break;
                }
                r += k;
                marks_seen++;
            }
            if (marks_seen == 0) {
                break;
            }
            /* \s* */
            while (r < n) {
                uint32_t cp;
                int k = utf8_next(s + r, s + n, &cp);
                if (k == 0) {
                    return -1;
                }
                if (!is_ws(cp)) {
                    break;
                }
                r += k;
            }
            q = r;
            matched = 1;
        }
        if (matched) {
            *m_start = p;
            *m_end = q;
            return 1;
        }
    }
    return 0;
}

/* phonemizer Punctuation._preserve_line + Punctuation.preserve, specialised to
 * a single input utterance (so every mark carries index 0). */
static int do_preserve(nano_ws_t *ws, const char *text, int tlen)
{
    ws->text_used = 0;
    ws->n_lines = 0;
    ws->mark_used = 0;
    ws->n_marks = 0;

    /* Collect the matches first: positions B/E are decided by comparing each
     * match's text against the whole line, exactly as phonemizer does with
     * line.startswith()/line.endswith(). */
    int starts[NANO_G2P_MAX_MARKS];
    int ends[NANO_G2P_MAX_MARKS];
    int n_match = 0;
    int cursor = 0;
    for (;;) {
        int ms, me;
        int rc = next_mark_match(text, tlen, cursor, &ms, &me);
        if (rc < 0) {
            return NANO_G2P_ERR_UTF8;
        }
        if (rc == 0) {
            break;
        }
        if (n_match >= NANO_G2P_MAX_MARKS) {
            return NANO_G2P_ERR_OVERFLOW;
        }
        starts[n_match] = ms;
        ends[n_match] = me;
        n_match++;
        /* re.finditer advances past the match; a zero-width match is
         * impossible here because [marks]+ consumes at least one byte. */
        cursor = (me > ms) ? me : ms + 1;
    }

    if (n_match == 0) {
        /* `return [line], []` */
        if (tlen > NANO_G2P_TEXT_MAX) {
            return NANO_G2P_ERR_OVERFLOW;
        }
        memcpy(ws->text_arena, text, (size_t)tlen);
        ws->text_used = tlen;
        ws->lines[0].off = 0;
        ws->lines[0].len = (int16_t)tlen;
        ws->n_lines = (tlen > 0) ? 1 : 0; /* preserve() filters empty lines */
        return NANO_G2P_OK;
    }

    /* "the line is made only of punctuation marks" */
    if (n_match == 1 && starts[0] == 0 && ends[0] == tlen) {
        memcpy(ws->mark_arena, text, (size_t)tlen);
        ws->mark_used = tlen;
        ws->marks[0].off = 0;
        ws->marks[0].len = (int16_t)tlen;
        ws->mark_pos[0] = 'A';
        ws->n_marks = 1;
        ws->n_lines = 0;
        return NANO_G2P_OK;
    }

    for (int i = 0; i < n_match; i++) {
        int mlen = ends[i] - starts[i];
        char pos = 'I';
        /* `if match == matches[0] and line.startswith(match.group())` */
        if (i == 0 && mlen <= tlen && memcmp(text, text + starts[i], (size_t)mlen) == 0) {
            pos = 'B';
        } else if (i == n_match - 1 && mlen <= tlen &&
                   memcmp(text + tlen - mlen, text + starts[i], (size_t)mlen) == 0) {
            pos = 'E';
        }
        if (ws->mark_used + mlen > NANO_G2P_TEXT_MAX) {
            return NANO_G2P_ERR_OVERFLOW;
        }
        memcpy(ws->mark_arena + ws->mark_used, text + starts[i], (size_t)mlen);
        ws->marks[i].off = (int16_t)ws->mark_used;
        ws->marks[i].len = (int16_t)mlen;
        ws->mark_pos[i] = pos;
        ws->mark_used += mlen;
    }
    ws->n_marks = n_match;

    /* Split the line: for each mark in order, cut at its FIRST occurrence in
     * what is left. phonemizer searches the remaining line from the start, not
     * from the match offset, so this must too. */
    const char *rest = text;
    int rest_len = tlen;
    for (int i = 0; i < ws->n_marks; i++) {
        const char *mark = ws->mark_arena + ws->marks[i].off;
        int mlen = ws->marks[i].len;
        int at = find_bytes(rest, rest_len, mark, mlen);
        const char *prefix = rest;
        int prefix_len;
        if (at < 0) {
            /* `line.split(mark)` found nothing: split[0] is the whole line and
             * split[1:] is empty, so the remainder becomes "". */
            prefix_len = rest_len;
            rest = rest + rest_len;
            rest_len = 0;
        } else {
            prefix_len = at;
            rest = rest + at + mlen;
            rest_len = rest_len - at - mlen;
        }
        if (prefix_len > 0) { /* preserve() drops empty chunks */
            if (ws->n_lines >= NANO_G2P_MAX_LINES) {
                return NANO_G2P_ERR_OVERFLOW;
            }
            if (ws->text_used + prefix_len > NANO_G2P_TEXT_MAX) {
                return NANO_G2P_ERR_OVERFLOW;
            }
            memcpy(ws->text_arena + ws->text_used, prefix, (size_t)prefix_len);
            ws->lines[ws->n_lines].off = (int16_t)ws->text_used;
            ws->lines[ws->n_lines].len = (int16_t)prefix_len;
            ws->n_lines++;
            ws->text_used += prefix_len;
        }
    }
    /* the trailing remainder */
    if (rest_len > 0) {
        if (ws->n_lines >= NANO_G2P_MAX_LINES) {
            return NANO_G2P_ERR_OVERFLOW;
        }
        if (ws->text_used + rest_len > NANO_G2P_TEXT_MAX) {
            return NANO_G2P_ERR_OVERFLOW;
        }
        memcpy(ws->text_arena + ws->text_used, rest, (size_t)rest_len);
        ws->lines[ws->n_lines].off = (int16_t)ws->text_used;
        ws->lines[ws->n_lines].len = (int16_t)rest_len;
        ws->n_lines++;
        ws->text_used += rest_len;
    }
    return NANO_G2P_OK;
}

/* ------------------------------------------------- 2. espeak per chunk --- */

/* phonemizer EspeakWrapper.text_to_phonemes(): call espeak_TextToPhonemes()
 * until it nulls the text pointer, drop empty clause results, join the rest
 * with a single space. */
static int espeak_ipa(const char *chunk, sb_t *out)
{
    const void *tp = chunk;
    int guard = 0;
    int clauses = 0;
    while (tp != NULL) {
        if (guard++ > NANO_CLAUSE_GUARD) {
            return NANO_G2P_ERR_ESPEAK;
        }
        const char *ph = espeak_TextToPhonemes(&tp, espeakCHARS_UTF8, NANO_IPA_TIE_MODE);
        if (ph == NULL || *ph == '\0') {
            continue; /* python skips falsy results */
        }
        if (clauses > 0) {
            sb_putn(out, " ", 1);
        }
        sb_puts(out, ph);
        clauses++;
    }
    return out->overflow ? NANO_G2P_ERR_OVERFLOW : NANO_G2P_OK;
}

/* --------------------------------- 3./5. the two _postprocess_line steps --- */

/* Shared head of both post-processing functions:
 *   line.strip().replace('\n', ' ').replace('  ', ' ')
 *   re.sub(r'_+', '_', line)
 *   line.replace('_ ', ' ')
 * Result lands in `dst` (capacity NANO_G2P_PS_MAX), via `tmp` as scratch. */
static int pp_head(const char *src, int slen, char *dst, char *tmp)
{
    int off = 0;
    int len = slen;
    if (!strip_ws(src, &off, &len)) {
        return NANO_G2P_ERR_UTF8;
    }

    sb_t a;
    sb_init(&a, tmp, NANO_G2P_PS_MAX);
    replace_all(&a, src + off, len, "\n", " ");
    if (a.overflow) {
        return NANO_G2P_ERR_OVERFLOW;
    }

    sb_t b;
    sb_init(&b, dst, NANO_G2P_PS_MAX);
    replace_all(&b, a.buf, a.len, "  ", " ");
    if (b.overflow) {
        return NANO_G2P_ERR_OVERFLOW;
    }

    sb_init(&a, tmp, NANO_G2P_PS_MAX);
    collapse_underscores(&a, b.buf, b.len);
    if (a.overflow) {
        return NANO_G2P_ERR_OVERFLOW;
    }

    sb_init(&b, dst, NANO_G2P_PS_MAX);
    replace_all(&b, a.buf, a.len, "_ ", " ");
    return b.overflow ? NANO_G2P_ERR_OVERFLOW : b.len;
}

/* Split on single ' ' (empty fields included, exactly like str.split(' ')),
 * strip each field, apply `per_word`, and append field + ' '. This is the
 * shared tail of both post-processing functions; `strip=False` and a non-NULL
 * tie mean no '_' is appended and no trailing separator is removed. */
typedef void (*word_fn)(sb_t *out, const char *word, int wlen);

/* step 3: EspeakBackend._process_tie with _tie == '͡' -> word.replace('_', '')
 * (separator.phone is the empty string). */
static void word_drop_underscores(sb_t *out, const char *word, int wlen)
{
    for (int i = 0; i < wlen; i++) {
        if (word[i] != '_') {
            sb_putn(out, word + i, 1);
        }
    }
}

/* step 5: nano_frontend.postprocess_line -> word.replace('͡', '^') */
static void word_tie_to_caret(sb_t *out, const char *word, int wlen)
{
    replace_all(out, word, wlen, NANO_TIE_ESPEAK, NANO_TIE_MISAKI);
}

static int pp_tail(const char *src, int slen, char *dst, word_fn per_word)
{
    sb_t out;
    sb_init(&out, dst, NANO_G2P_PS_MAX);
    if (slen == 0) {
        return 0; /* `if not line: return ''` */
    }
    int i = 0;
    while (i <= slen) {
        int j = i;
        while (j < slen && src[j] != ' ') {
            j++;
        }
        int off = i;
        int len = j - i;
        if (!strip_ws(src, &off, &len)) {
            return NANO_G2P_ERR_UTF8;
        }
        per_word(&out, src + off, len);
        sb_putn(&out, " ", 1);
        if (j >= slen) {
            break;
        }
        i = j + 1;
    }
    return out.overflow ? NANO_G2P_ERR_OVERFLOW : out.len;
}

/* ------------------------------------------------ 4. Punctuation.restore --- */

/* phonemizer Punctuation.restore(), specialised to a single input utterance.
 *
 * Every mark then carries index 0, and `pos` only ever advances past 0 in the
 * 'E' / 'A' branches or when the mark list no longer matches -- and each of
 * those branches pushes to the output. Since the caller consumes only
 * `restore(...)[0]`, this returns as soon as the first element exists, which
 * makes the whole list-rewriting dance collapse into one accumulator.
 *
 * `text[0]` is `acc`; `text[1:]` are ws->ph[ti .. n_ph-1].
 */
static int do_restore(nano_ws_t *ws, char *dst)
{
    sb_t acc;
    sb_init(&acc, ws->acc, NANO_G2P_PS_MAX);

    int ti = 0;
    int have_acc = 0;
    if (ws->n_ph > 0) {
        sb_putn(&acc, ws->ph_arena + ws->ph[0].off, ws->ph[0].len);
        ti = 1;
        have_acc = 1;
    }

    sb_t out;
    sb_init(&out, dst, NANO_G2P_PS_MAX);

    int mi = 0;
    int pos = 0;
    int guard = 0;

    while (have_acc || mi < ws->n_marks) {
        if (guard++ > 4 * (NANO_G2P_MAX_LINES + NANO_G2P_MAX_MARKS) + 8) {
            return NANO_G2P_ERR_OVERFLOW; /* cannot happen; refuses to spin */
        }
        if (mi >= ws->n_marks) {
            /* `if not marks:` -- flush the remaining text with separators. */
            if (!(acc.len > 0 && acc.buf[acc.len - 1] == ' ')) {
                sb_putn(&acc, " ", 1);
            }
            sb_putn(&out, acc.buf, acc.len);
            return out.overflow ? NANO_G2P_ERR_OVERFLOW : out.len;
        }
        if (!have_acc) {
            /* `elif not text:` -- emit the remaining marks joined. */
            for (int k = mi; k < ws->n_marks; k++) {
                sb_putn(&out, ws->mark_arena + ws->marks[k].off, ws->marks[k].len);
            }
            return out.overflow ? NANO_G2P_ERR_OVERFLOW : out.len;
        }

        const char *mark = ws->mark_arena + ws->marks[mi].off;
        int mlen = ws->marks[mi].len;
        char mpos = ws->mark_pos[mi];

        if (pos != 0) {
            /* `else: punctuated_text.append(text[0])` */
            sb_putn(&out, acc.buf, acc.len);
            return out.overflow ? NANO_G2P_ERR_OVERFLOW : out.len;
        }

        mi++;
        /* `if sep.word and text[0].endswith(sep.word): text[0] = text[0][:-1]` */
        if (acc.len > 0 && acc.buf[acc.len - 1] == ' ') {
            acc.len--;
            acc.buf[acc.len] = '\0';
        }

        if (mpos == 'B') {
            /* text[0] = mark + text[0] */
            sb_t tmp;
            sb_init(&tmp, ws->scratch, NANO_G2P_PS_MAX);
            sb_putn(&tmp, mark, mlen);
            sb_putn(&tmp, acc.buf, acc.len);
            if (tmp.overflow) {
                return NANO_G2P_ERR_OVERFLOW;
            }
            sb_init(&acc, ws->acc, NANO_G2P_PS_MAX);
            sb_putn(&acc, tmp.buf, tmp.len);
        } else if (mpos == 'E' || mpos == 'A') {
            if (mpos == 'E') {
                sb_putn(&out, acc.buf, acc.len);
            }
            sb_putn(&out, mark, mlen);
            if (!(mlen > 0 && mark[mlen - 1] == ' ')) {
                sb_putn(&out, " ", 1); /* strip is False */
            }
            return out.overflow ? NANO_G2P_ERR_OVERFLOW : out.len;
        } else { /* 'I' */
            sb_putn(&acc, mark, mlen);
            if (ti < ws->n_ph) {
                /* text[0] = first_word + mark + text[1], list shrinks by one */
                sb_putn(&acc, ws->ph_arena + ws->ph[ti].off, ws->ph[ti].len);
                ti++;
            }
            /* else: `len(text) == 1` corner case -- mark appended, nothing to
             * merge with, list length unchanged. */
        }
        if (acc.overflow) {
            return NANO_G2P_ERR_OVERFLOW;
        }
    }
    /* Both lists empty: phonemizer returns [], and the caller treats an empty
     * result as a hard failure. */
    return NANO_G2P_ERR_EMPTY;
}

/* --------------------------------------------------------- 6. apply_e2m --- */

/* re.sub(r'(\S)̩', r'ᵊ\1', ps).replace(chr(809), '')
 *
 * Written as an index scan rather than two passes because regex matches are
 * non-overlapping: a match consumes both codepoints and scanning resumes after
 * them, so "a̩̩" gives "ᵊa" and not "ᵊᵊa". */
static int syllabic_rewrite(sb_t *out, const char *src, int slen)
{
    int i = 0;
    while (i < slen) {
        uint32_t cp;
        int n = utf8_next(src + i, src + slen, &cp);
        if (n == 0) {
            return NANO_G2P_ERR_UTF8;
        }
        uint32_t next_cp = 0;
        int n2 = 0;
        if (i + n < slen) {
            n2 = utf8_next(src + i + n, src + slen, &next_cp);
            if (n2 == 0) {
                return NANO_G2P_ERR_UTF8;
            }
        }
        if (n2 != 0 && next_cp == NANO_SYLLABIC_CODEPOINT && !is_ws(cp)) {
            sb_puts(out, NANO_SCHWA_MODIFIER);
            sb_putn(out, src + i, n);
            i += n + n2;
            continue;
        }
        if (cp == NANO_SYLLABIC_CODEPOINT) {
            i += n; /* the trailing .replace(chr(809), '') */
            continue;
        }
        sb_putn(out, src + i, n);
        i += n;
    }
    return out->overflow ? NANO_G2P_ERR_OVERFLOW : out->len;
}

/* misaki EspeakFallback.__call__ tail (british=False, version=None). */
static int apply_e2m(nano_ws_t *ws, const char *src, int slen, char **result, int *result_len)
{
    int off = 0;
    int len = slen;
    if (!strip_ws(src, &off, &len)) {
        return NANO_G2P_ERR_UTF8;
    }

    char *cur = ws->ping;
    char *other = ws->pong;
    sb_t sb;
    sb_init(&sb, cur, NANO_G2P_PS_MAX);
    sb_putn(&sb, src + off, len);
    if (sb.overflow) {
        return NANO_G2P_ERR_OVERFLOW;
    }
    int cur_len = sb.len;

    for (int i = 0; i < NANO_E2M_COUNT; i++) {
        sb_init(&sb, other, NANO_G2P_PS_MAX);
        replace_all(&sb, cur, cur_len, NANO_E2M[i].from, NANO_E2M[i].to);
        if (sb.overflow) {
            return NANO_G2P_ERR_OVERFLOW;
        }
        char *swap = cur;
        cur = other;
        other = swap;
        cur_len = sb.len;
    }

    sb_init(&sb, other, NANO_G2P_PS_MAX);
    int rc = syllabic_rewrite(&sb, cur, cur_len);
    if (rc < 0) {
        return rc;
    }
    char *swap = cur;
    cur = other;
    other = swap;
    cur_len = rc;

    for (int i = 0; i < NANO_E2M_TAIL_COUNT; i++) {
        sb_init(&sb, other, NANO_G2P_PS_MAX);
        replace_all(&sb, cur, cur_len, NANO_E2M_TAIL[i].from, NANO_E2M_TAIL[i].to);
        if (sb.overflow) {
            return NANO_G2P_ERR_OVERFLOW;
        }
        swap = cur;
        cur = other;
        other = swap;
        cur_len = sb.len;
    }

    *result = cur;
    *result_len = cur_len;
    return NANO_G2P_OK;
}

/* ------------------------------------------------------- the pipeline --- */

static int phonemize(const char *text, char **result, int *result_len)
{
    if (!g_ready) {
        return NANO_G2P_ERR_NOT_READY;
    }
    if (text == NULL) {
        return NANO_G2P_ERR_ARG;
    }
    size_t tlen = strlen(text);
    if (tlen == 0) {
        return NANO_G2P_ERR_EMPTY;
    }
    if (tlen >= NANO_G2P_TEXT_MAX) {
        return NANO_G2P_ERR_TEXT_LONG;
    }

    nano_ws_t *ws = &g_ws;

    int rc = do_preserve(ws, text, (int)tlen);
    if (rc != NANO_G2P_OK) {
        return rc;
    }

    ws->ph_used = 0;
    ws->n_ph = 0;
    for (int i = 0; i < ws->n_lines; i++) {
        int clen = ws->lines[i].len;
        memcpy(ws->chunk, ws->text_arena + ws->lines[i].off, (size_t)clen);
        ws->chunk[clen] = '\0';

        sb_t raw;
        sb_init(&raw, ws->ping, NANO_G2P_PS_MAX);
        rc = espeak_ipa(ws->chunk, &raw);
        if (rc != NANO_G2P_OK) {
            return rc;
        }

        int head = pp_head(raw.buf, raw.len, ws->pong, ws->scratch);
        if (head < 0) {
            return head;
        }
        sb_t flags;
        sb_init(&flags, ws->scratch, NANO_G2P_PS_MAX);
        remove_lang_flags(&flags, ws->pong, head);
        if (flags.overflow) {
            return NANO_G2P_ERR_OVERFLOW;
        }
        int tail = pp_tail(flags.buf, flags.len, ws->ping, word_drop_underscores);
        if (tail < 0) {
            return tail;
        }

        if (ws->n_ph >= NANO_G2P_MAX_LINES || ws->ph_used + tail > NANO_G2P_PS_MAX) {
            return NANO_G2P_ERR_OVERFLOW;
        }
        memcpy(ws->ph_arena + ws->ph_used, ws->ping, (size_t)tail);
        ws->ph[ws->n_ph].off = (int16_t)ws->ph_used;
        ws->ph[ws->n_ph].len = (int16_t)tail;
        ws->n_ph++;
        ws->ph_used += tail;
    }

    int restored = do_restore(ws, ws->pong);
    if (restored < 0) {
        return restored;
    }

    int head = pp_head(ws->pong, restored, ws->ping, ws->scratch);
    if (head < 0) {
        return head;
    }
    int tail = pp_tail(ws->ping, head, ws->pong, word_tie_to_caret);
    if (tail < 0) {
        return tail;
    }

    /* apply_e2m reads pong and ping-pongs between ping/pong internally; copy
     * the input somewhere neither of those aliases. */
    memcpy(ws->acc, ws->pong, (size_t)tail);
    return apply_e2m(ws, ws->acc, tail, result, result_len);
}

/* --------------------------------------------------------- public api --- */

int nano_g2p_init_path(const char *data_path)
{
    if (g_ready) {
        return NANO_G2P_OK;
    }
    if (data_path == NULL) {
        return NANO_G2P_ERR_ARG;
    }

#ifdef ESP_PLATFORM
    esp_vfs_spiffs_conf_t conf = {
        .base_path = data_path,
        .partition_label = "espeak",
        .max_files = 6,
        .format_if_mount_failed = false,
    };
    esp_err_t err = esp_vfs_spiffs_register(&conf);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        /* ESP_ERR_INVALID_STATE means somebody already mounted it, which is
         * fine; anything else is fatal and must not be swallowed. */
        fprintf(stderr, "nano_g2p: spiffs mount of '%s' failed: %s\n", data_path, esp_err_to_name(err));
        return NANO_G2P_ERR_MOUNT;
    }
    size_t total = 0;
    size_t used = 0;
    esp_err_t info = esp_spiffs_info("espeak", &total, &used);
    if (info != ESP_OK) {
        fprintf(stderr, "nano_g2p: esp_spiffs_info failed: %s\n", esp_err_to_name(info));
        return NANO_G2P_ERR_MOUNT;
    }
    fprintf(stderr, "nano_g2p: spiffs mounted, %u/%u bytes used\n", (unsigned)used, (unsigned)total);
#endif

    espeak_ng_InitializePath(data_path);
    espeak_ng_ERROR_CONTEXT ctx = NULL;
    espeak_ng_STATUS status = espeak_ng_Initialize(&ctx);
    if (status != ENS_OK) {
        fprintf(stderr, "nano_g2p: espeak_ng_Initialize failed (0x%x)\n", (unsigned)status);
        return NANO_G2P_ERR_ESPEAK_INIT;
    }
    /* misaki's EspeakFallback(british=False) is en-us. The Piper port used
     * plain "en"; that is a different dictionary and the wrong one here. */
    status = espeak_ng_SetVoiceByName("en-us");
    if (status != ENS_OK) {
        fprintf(stderr, "nano_g2p: espeak_ng_SetVoiceByName(\"en-us\") failed (0x%x)\n", (unsigned)status);
        return NANO_G2P_ERR_VOICE;
    }
    g_ready = 1;
    fprintf(stderr, "nano_g2p: ready (espeak en-us, workspace %u bytes)\n", (unsigned)sizeof(g_ws));
    return NANO_G2P_OK;
}

int nano_g2p_init(void)
{
    return nano_g2p_init_path("/espeak");
}

int nano_g2p_text_to_phonemes(const char *text, char *out, int cap)
{
    if (out == NULL || cap <= 0) {
        return NANO_G2P_ERR_ARG;
    }
    char *ps = NULL;
    int ps_len = 0;
    int rc = phonemize(text, &ps, &ps_len);
    if (rc != NANO_G2P_OK) {
        return rc;
    }
    if (ps_len + 1 > cap) {
        return NANO_G2P_ERR_OVERFLOW;
    }
    memcpy(out, ps, (size_t)ps_len);
    out[ps_len] = '\0';
    return ps_len;
}

int nano_g2p_text_to_ids(const char *text, int32_t *out, int cap)
{
    if (out == NULL || cap <= 0) {
        return NANO_G2P_ERR_ARG;
    }
    char *ps = NULL;
    int ps_len = 0;
    int rc = phonemize(text, &ps, &ps_len);
    if (rc != NANO_G2P_OK) {
        return rc;
    }

    /* kokoro_frontend.phonemes_to_token_ids: keep what the vocabulary knows,
     * drop the rest (Kokoro's own policy), wrap in <bos>/<eos>. DENSE -- no
     * pad interleaving, matching mcu/test/fixtures/en_us_e12nano/r00_ids.bin. */
    int n = 0;
    if (cap < 2) {
        return NANO_G2P_ERR_OVERFLOW;
    }
    out[n++] = NANO_ID_BOS;

    int kept = 0;
    int i = 0;
    while (i < ps_len) {
        uint32_t cp;
        int k = utf8_next(ps + i, ps + ps_len, &cp);
        if (k == 0) {
            return NANO_G2P_ERR_UTF8;
        }
        i += k;
        int id = vocab_lookup(cp);
        if (id < 0) {
            continue; /* dropped, exactly as Kokoro drops it */
        }
        if (n + 1 >= cap) {
            return NANO_G2P_ERR_OVERFLOW;
        }
        out[n++] = (int32_t)id;
        kept++;
    }
    if (kept == 0) {
        return NANO_G2P_ERR_EMPTY;
    }
    if (n + 1 > cap) {
        return NANO_G2P_ERR_OVERFLOW;
    }
    out[n++] = NANO_ID_EOS;
    if (n > NANO_MAX_TOKENS) {
        return NANO_G2P_ERR_TOO_MANY_TOKENS;
    }
    return n;
}
