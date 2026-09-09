/* nano_lex_g2p.c -- the dictionary-first English G2P of pypkg/sanotts/nano_g2p.py
 * in C99, with no espeak-ng, no malloc, and no libc.
 *
 * Every function that corresponds to one in nano_g2p.py (and through it to
 * misaki/en.py) names it, so the three can be diffed side by side. Where this
 * file deviates it says so in a comment beginning "DEVIATION:"; README.md
 * lists all of them in one place.
 *
 * The neural out-of-vocabulary model is NOT ported. This file therefore
 * behaves exactly as `NanoG2P` with `fallback = None`: a word that reaches the
 * end of the back-off loop unresolved contributes no phonemes, and the event
 * is counted in nano_lex_g2p_stats_t.oov_words rather than being swallowed.
 *
 * All lookup tables are in nano_lex_tables.c and are `const`, so they link
 * into .rodata and are read in place out of flash. The only RAM this module
 * owns is the single `g_ws` block; nano_lex_g2p_workspace_bytes() returns its
 * size and there is no other static, no thread-local and no heap use.
 */

#include "nano_lex_g2p.h"
#include "nano_lex_tables.h"

/* --------------------------------------------------------------------------
 * Tiny libc, written out so the module builds freestanding.
 * -------------------------------------------------------------------------- */

static void nlg_copy(void *dst, const void *src, size_t n)
{
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    while (n--) {
        *d++ = *s++;
    }
}

static int nlg_cmp(const void *a, const void *b, size_t n)
{
    const unsigned char *p = (const unsigned char *)a;
    const unsigned char *q = (const unsigned char *)b;
    while (n--) {
        if (*p != *q) {
            return (int)*p - (int)*q;
        }
        p++;
        q++;
    }
    return 0;
}

static size_t nlg_slen(const char *s)
{
    size_t n = 0;
    while (s[n] != '\0') {
        n++;
    }
    return n;
}

static void nlg_zero(void *dst, size_t n)
{
    unsigned char *d = (unsigned char *)dst;
    while (n--) {
        *d++ = 0;
    }
}

/* --------------------------------------------------------------------------
 * Sizes
 * -------------------------------------------------------------------------- */

#define NLG_MAX_WORD    64     /* codepoints in one lexicon query; keys max 45 */
#define NLG_PS_ARENA    3072   /* phoneme workspace, bytes; the high-water
                                * mark over the 382-sentence parity corpora is
                                * 142 bytes, and overflow is a clean
                                * NANO_LEX_E_ARENA rather than corruption */
#define NLG_NUM_PARTS   96     /* spans get_number may accumulate */
#define NLG_NUM_TEXT    384    /* longest spelled-out numeral */
#define NLG_MAX_PIECES  64     /* sub-tokens one chunk may split into */

/* misaki's stresses are all multiples of 0.5, so they are carried doubled and
 * compared as integers: -0.5 is -1, 2 is 4. 127 stands for Python's None. */
#define NLG_S2_NONE     127

typedef struct {
    uint16_t off;
    uint16_t len;
    uint8_t  present;   /* 0 is Python's None, which differs from length 0 */
} nlg_ps_t;

static const nlg_ps_t NLG_PS_NONE = { 0, 0, 0 };

typedef struct {
    uint16_t text_off;
    uint16_t text_len;
    nlg_ps_t ps;
    uint8_t  tag;
    uint8_t  ws;
    uint8_t  is_head;
    uint8_t  prespace;
    uint8_t  alias_to;
    uint8_t  currency;
    uint8_t  num_flags;
    int8_t   rating;
    int8_t   stress2;
} nlg_token_t;

/* What tokenize() produces, before sub-tokenisation. Kept separate and narrow
 * so retokenize() does not need a second full token array. */
typedef struct {
    uint16_t text_off;
    uint16_t text_len;
    uint8_t  tag;
    uint8_t  ws;
} nlg_src_t;

typedef struct {
    uint16_t start;
    uint16_t count;
    uint8_t  is_list;
} nlg_group_t;

/* future_vowel is a tri-state: -1 None, 0 False, 1 True. */
typedef struct {
    int8_t  future_vowel;
    uint8_t future_to;
} nlg_ctx_t;

typedef struct {
    uint16_t cp[NLG_MAX_WORD];
    uint16_t len;
} nlg_word_t;

/* The whole of this module's RAM, in one block. */
static struct {
    uint16_t     cp[NANO_LEX_MAX_CHARS];
    nlg_src_t    src[NANO_LEX_MAX_TOKEN];
    nlg_token_t  tok[NANO_LEX_MAX_TOKEN];
    nlg_group_t  grp[NANO_LEX_MAX_TOKEN];
    uint8_t      arena[NLG_PS_ARENA];
    uint8_t      chunk[NANO_LEX_MAX_PS_CHARS + 2];
    uint16_t     merge_cp[NLG_MAX_WORD];
    nlg_ps_t     num_parts[NLG_NUM_PARTS];
    int8_t       num_ratings[NLG_NUM_PARTS];
    /* Scratch that would otherwise be large stack frames. Each is used by
     * exactly one phase, and the phases do not overlap, but they are kept
     * separate so the ownership is obvious from the name. */
    uint16_t     piece_off[NLG_MAX_PIECES];
    uint16_t     piece_len[NLG_MAX_PIECES];
    uint16_t     stress_idx[NLG_MAX_PIECES];
    uint16_t     stress_weight[NLG_MAX_PIECES];
    uint8_t      stress_primary[NLG_MAX_PIECES];
    uint8_t      stress_used[NLG_MAX_PIECES];
    uint16_t     num_clean[NLG_MAX_WORD];
    /* One nlg_word_t slot per call-graph level.
     *
     * These are static rather than local because a 64-codepoint word is 132
     * bytes and eight of them on the stack at once is most of a FreeRTOS
     * task's budget. It is safe only because none of these functions is ever
     * on the stack twice at the same time: the call graph is
     *
     *   lexicon_call -> get_word -> {get_special_case -> lookup_ascii,
     *                                stem_s|stem_ed|stem_ing, is_known}
     *                            -> lookup -> get_NNP
     *                -> get_number -> {extend_num -> num_lookup_word, stem_s}
     *                -> append_currency -> stem_s
     *
     * and it has no cycles. is_known, grown_find, lookup and the three
     * stemmers are each entered from several places but never from
     * themselves, directly or indirectly. Adding a recursive call anywhere in
     * that graph would silently corrupt these slots, so do not. */
    nlg_word_t   w_token;      /* lexicon_call */
    nlg_word_t   w_lowered;    /* get_word */
    nlg_word_t   w_alt;        /* get_word */
    nlg_word_t   w_number;     /* get_number */
    nlg_word_t   w_lookup;     /* lookup */
    nlg_word_t   w_grow;       /* grown_find */
    nlg_word_t   w_known;      /* is_known */
    nlg_word_t   w_stem;       /* stem_s / stem_ed / stem_ing */
    nlg_word_t   w_numword;    /* num_lookup_word */
    nlg_word_t   w_ascii;      /* lookup_ascii */
    nlg_word_t   w_tmp;        /* the fixed-spelling lookups */
    char         num_text[NLG_NUM_TEXT];
    char         num_scratch[NLG_NUM_TEXT];
    uint16_t     cp_len;
    uint16_t     src_len;
    uint16_t     tok_len;
    uint16_t     grp_len;
    uint16_t     arena_len;
    uint16_t     arena_peak;
    int          num_count;
    nano_lex_g2p_stats_t stats;
} g_ws;

/* --------------------------------------------------------------------------
 * Arena
 * -------------------------------------------------------------------------- */

static void nlg_arena_reset(void)
{
    g_ws.arena_len = 0;
    g_ws.arena_peak = 0;
}

static uint16_t nlg_arena_mark(void)
{
    return g_ws.arena_len;
}

static void nlg_arena_rewind(uint16_t mark)
{
    g_ws.arena_len = mark;
}

/* Slide a just-built string down to `mark`, discarding the temporaries the
 * builder left underneath it. Without this the back-off loop's repeated
 * lookups would grow the arena quadratically in the size of a word group. */
static void nlg_arena_keep(uint16_t mark, nlg_ps_t *ps)
{
    uint16_t i;
    if (!ps->present) {
        g_ws.arena_len = mark;
        return;
    }
    if (ps->off != mark) {
        for (i = 0; i < ps->len; i++) {
            g_ws.arena[mark + i] = g_ws.arena[ps->off + i];
        }
        ps->off = mark;
    }
    g_ws.arena_len = (uint16_t)(mark + ps->len);
}

typedef struct {
    uint16_t off;
    uint16_t len;
    int      rc;
} nlg_sb_t;

static void nlg_sb_begin(nlg_sb_t *sb)
{
    sb->off = g_ws.arena_len;
    sb->len = 0;
    sb->rc = NANO_LEX_OK;
}

static void nlg_sb_push(nlg_sb_t *sb, uint8_t code)
{
    if (sb->rc != NANO_LEX_OK) {
        return;
    }
    if (g_ws.arena_len >= NLG_PS_ARENA) {
        sb->rc = NANO_LEX_E_ARENA;
        return;
    }
    g_ws.arena[g_ws.arena_len++] = code;
    if (g_ws.arena_len > g_ws.arena_peak) {
        g_ws.arena_peak = g_ws.arena_len;
    }
    sb->len++;
}

static void nlg_sb_push_span(nlg_sb_t *sb, uint16_t off, uint16_t len)
{
    uint16_t i;
    for (i = 0; i < len; i++) {
        nlg_sb_push(sb, g_ws.arena[off + i]);
    }
}

static void nlg_sb_push_ps(nlg_sb_t *sb, const nlg_ps_t *ps)
{
    if (ps->present) {
        nlg_sb_push_span(sb, ps->off, ps->len);
    }
}

static int nlg_sb_finish(nlg_sb_t *sb, nlg_ps_t *out)
{
    if (sb->rc != NANO_LEX_OK) {
        return sb->rc;
    }
    out->off = sb->off;
    out->len = sb->len;
    out->present = 1;
    return NANO_LEX_OK;
}

static uint8_t nlg_ps_at(const nlg_ps_t *ps, uint16_t i)
{
    return g_ws.arena[ps->off + i];
}

static int nlg_ps_has(const nlg_ps_t *ps, uint8_t code)
{
    uint16_t i;
    if (!ps->present) {
        return 0;
    }
    for (i = 0; i < ps->len; i++) {
        if (g_ws.arena[ps->off + i] == code) {
            return 1;
        }
    }
    return 0;
}

static int nlg_ps_has_flag(const nlg_ps_t *ps, uint8_t flag)
{
    uint16_t i;
    if (!ps->present) {
        return 0;
    }
    for (i = 0; i < ps->len; i++) {
        if (NANO_LEX_PH_FLAGS[g_ws.arena[ps->off + i]] & flag) {
            return 1;
        }
    }
    return 0;
}

/* --------------------------------------------------------------------------
 * Codepoint classification
 *
 * DEVIATION: Python asks Unicode for isalpha/upper/lower; this is exact for
 * ASCII and approximate above it. nlg_is_letter treats every codepoint >= 0x80
 * as a letter unless it falls in one of the punctuation, symbol or combining
 * ranges listed, and case mapping is ASCII-only. A word holding a non-ASCII
 * letter can never be a dictionary key (misaki's LEXICON_ORDS is ASCII), so
 * the only thing this can change is how such a word reaches the
 * out-of-vocabulary path -- which it does either way.
 * -------------------------------------------------------------------------- */

static int nlg_in_range(uint32_t cp, uint32_t lo, uint32_t hi)
{
    return cp >= lo && cp <= hi;
}

static int nlg_is_space(uint32_t cp)
{
    if (cp == 0x20u || (cp >= 0x09u && cp <= 0x0Du)) {
        return 1;
    }
    if (cp == 0x85u || cp == 0xA0u || cp == 0x1680u || cp == 0x2028u || cp == 0x2029u) {
        return 1;
    }
    if (nlg_in_range(cp, 0x2000u, 0x200Au) || cp == 0x202Fu || cp == 0x205Fu ||
        cp == 0x3000u) {
        return 1;
    }
    return 0;
}

static int nlg_is_digit_cp(uint32_t cp)
{
    return cp >= '0' && cp <= '9';
}

static int nlg_is_upper(uint32_t cp)
{
    return cp >= 'A' && cp <= 'Z';
}

static int nlg_is_lower(uint32_t cp)
{
    return cp >= 'a' && cp <= 'z';
}

static uint16_t nlg_lower(uint32_t cp)
{
    return (uint16_t)(nlg_is_upper(cp) ? cp + 32u : cp);
}

static uint16_t nlg_upper(uint32_t cp)
{
    return (uint16_t)(nlg_is_lower(cp) ? cp - 32u : cp);
}

static int nlg_is_letter(uint32_t cp)
{
    if (cp < 0x80u) {
        return nlg_is_upper(cp) || nlg_is_lower(cp);
    }
    if (nlg_in_range(cp, 0x0080u, 0x00BFu) || cp == 0x00D7u || cp == 0x00F7u) {
        return 0;
    }
    if (nlg_in_range(cp, 0x0300u, 0x036Fu)) {
        return 0;
    }
    if (nlg_in_range(cp, 0x2000u, 0x2BFFu) || nlg_in_range(cp, 0x3000u, 0x303Fu)) {
        return 0;
    }
    if (nlg_in_range(cp, 0xFE00u, 0xFE6Fu) || nlg_in_range(cp, 0xFF00u, 0xFF20u)) {
        return 0;
    }
    if (nlg_in_range(cp, 0xFFF0u, 0xFFFFu)) {
        return 0;
    }
    return 1;
}

static int nlg_is_alnum(uint32_t cp)
{
    return nlg_is_letter(cp) || nlg_is_digit_cp(cp);
}

static int nlg_all_digits(const uint16_t *cp, uint16_t len)
{
    uint16_t i;
    if (len == 0) {
        return 0;
    }
    for (i = 0; i < len; i++) {
        if (!nlg_is_digit_cp(cp[i])) {
            return 0;
        }
    }
    return 1;
}

/* all(97 <= ord(c.lower()) <= 122 for c in text); empty text is True. */
static int nlg_all_ascii_alpha(const uint16_t *cp, uint16_t len)
{
    uint16_t i;
    for (i = 0; i < len; i++) {
        uint16_t c = nlg_lower(cp[i]);
        if (c < 'a' || c > 'z') {
            return 0;
        }
    }
    return 1;
}

static uint8_t nlg_cp_to_ph(uint32_t cp)
{
    uint8_t code;
    for (code = 1; code <= NANO_LEX_PH_COUNT; code++) {
        if (NANO_LEX_PH_CP[code] == cp) {
            return code;
        }
    }
    return 0;
}

/* misaki SUBTOKEN_JUNKS: ' , - . _ / and the two curly apostrophes. */
static int nlg_is_junk(uint32_t cp)
{
    return cp == '\'' || cp == ',' || cp == '-' || cp == '.' || cp == '_' ||
           cp == '/' || cp == 0x2018u || cp == 0x2019u;
}

/* --------------------------------------------------------------------------
 * UTF-8
 * -------------------------------------------------------------------------- */

static int nlg_decode_utf8(const char *text)
{
    const unsigned char *p = (const unsigned char *)text;
    uint16_t n = 0;
    while (*p) {
        uint32_t cp;
        unsigned extra;
        unsigned i;
        if (*p < 0x80u) {
            cp = *p++;
            extra = 0;
        } else if ((*p & 0xE0u) == 0xC0u) {
            cp = (uint32_t)(*p++ & 0x1Fu);
            extra = 1;
        } else if ((*p & 0xF0u) == 0xE0u) {
            cp = (uint32_t)(*p++ & 0x0Fu);
            extra = 2;
        } else if ((*p & 0xF8u) == 0xF0u) {
            cp = (uint32_t)(*p++ & 0x07u);
            extra = 3;
        } else {
            return NANO_LEX_E_BAD_UTF8;
        }
        for (i = 0; i < extra; i++) {
            if ((*p & 0xC0u) != 0x80u) {
                return NANO_LEX_E_BAD_UTF8;
            }
            cp = (cp << 6) | (uint32_t)(*p++ & 0x3Fu);
        }
        if (n >= NANO_LEX_MAX_CHARS) {
            return NANO_LEX_E_TEXT_LONG;
        }
        /* DEVIATION: astral-plane codepoints (emoji) fold to U+FFFD. They are
         * non-letters either way and carry no phonemes. */
        g_ws.cp[n++] = (uint16_t)(cp > 0xFFFFu ? 0xFFFDu : cp);
    }
    g_ws.cp_len = n;
    return NANO_LEX_OK;
}

/* --------------------------------------------------------------------------
 * Packed dictionary access
 * -------------------------------------------------------------------------- */

typedef struct {
    uint8_t     vcode;
    const char *payload;
} nlg_entry_t;

static int nlg_payload_len(const char *payload, uint8_t vcode, uint16_t *len)
{
    if (vcode == NANO_LEX_VCODE_NULL) {
        *len = 0;
        return NANO_LEX_OK;
    }
    if (vcode < NANO_LEX_VCODE_TAGGED_BASE) {
        *len = (uint16_t)(vcode - NANO_LEX_VCODE_PLAIN_BASE);
        return NANO_LEX_OK;
    }
    {
        uint8_t variants = (uint8_t)(vcode - NANO_LEX_VCODE_TAGGED_BASE + 1u);
        uint16_t total = 0;
        uint8_t i;
        for (i = 0; i < variants; i++) {
            uint8_t vlen1 = NANO_LEX_UNBIAS(payload[total + 1]);
            total = (uint16_t)(total + 2u + (vlen1 ? (uint16_t)(vlen1 - 1u) : 0u));
        }
        *len = total;
        return NANO_LEX_OK;
    }
}

static int nlg_cmp_head(const char *blob, uint32_t off, const char *key, uint16_t klen)
{
    uint8_t hlen = NANO_LEX_UNBIAS(blob[off]);
    uint16_t n = (hlen < klen) ? hlen : klen;
    int cmp = nlg_cmp(blob + off + 1, key, n);
    if (cmp != 0) {
        return cmp;
    }
    if (hlen == klen) {
        return 0;
    }
    return (hlen < klen) ? -1 : 1;
}

/* Binary search the block index on each block's head key, then scan that
 * block's front-coded records. Returns 1 hit, 0 miss, negative on a table that
 * fails its own structural checks. */
static int nlg_dict_find(const nano_lex_dict_t *dict, const char *key, uint16_t klen,
                         nlg_entry_t *out)
{
    uint32_t lo = 0;
    uint32_t hi;
    uint32_t block;
    uint32_t pos;
    uint32_t first;
    uint32_t i;
    char cur[NANO_LEX_MAX_KEY];
    uint16_t curlen = 0;

    if (klen == 0 || klen > NANO_LEX_MAX_KEY) {
        return 0;
    }
    hi = dict->block_count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2u;
        if (nlg_cmp_head(dict->blob, dict->block_off[mid], key, klen) <= 0) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo == 0) {
        return 0;
    }
    block = lo - 1u;
    pos = dict->block_off[block];
    first = block * (uint32_t)NANO_LEX_BLOCK;

    for (i = 0; i < (uint32_t)NANO_LEX_BLOCK && first + i < dict->entry_count; i++) {
        uint8_t vcode;
        uint16_t plen;
        int rc;
        if (i == 0) {
            curlen = NANO_LEX_UNBIAS(dict->blob[pos++]);
            if (curlen == 0 || curlen > NANO_LEX_MAX_KEY) {
                return NANO_LEX_E_TABLE;
            }
            nlg_copy(cur, dict->blob + pos, curlen);
            pos += curlen;
        } else {
            uint8_t shared = NANO_LEX_UNBIAS(dict->blob[pos]);
            uint8_t slen = NANO_LEX_UNBIAS(dict->blob[pos + 1]);
            pos += 2u;
            if ((uint16_t)shared + slen > NANO_LEX_MAX_KEY || shared > curlen) {
                return NANO_LEX_E_TABLE;
            }
            nlg_copy(cur + shared, dict->blob + pos, slen);
            curlen = (uint16_t)(shared + slen);
            pos += slen;
        }
        vcode = NANO_LEX_UNBIAS(dict->blob[pos++]);
        rc = nlg_payload_len(dict->blob + pos, vcode, &plen);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (curlen == klen && nlg_cmp(cur, key, klen) == 0) {
            out->vcode = vcode;
            out->payload = dict->blob + pos;
            return 1;
        }
        pos += plen;
    }
    return 0;
}

/* Point at an entry's chosen value without copying it. `*is_null` is Python's
 * None; `*bytes` is a run of biased phoneme codes inside the blob. */
static int nlg_entry_bytes(const nlg_entry_t *entry, uint8_t tag, int have_ctx,
                           int8_t future_vowel, const char **bytes, uint16_t *len,
                           int *is_null)
{
    const char *p = entry->payload;
    uint8_t vcode = entry->vcode;

    *bytes = 0;
    *len = 0;
    *is_null = 0;

    if (vcode == NANO_LEX_VCODE_NULL) {
        *is_null = 1;
        return NANO_LEX_OK;
    }
    if (vcode < NANO_LEX_VCODE_TAGGED_BASE) {
        *bytes = p;
        *len = (uint16_t)(vcode - NANO_LEX_VCODE_PLAIN_BASE);
        return NANO_LEX_OK;
    }
    {
        uint8_t variants = (uint8_t)(vcode - NANO_LEX_VCODE_TAGGED_BASE + 1u);
        uint8_t wanted = tag;
        uint8_t i;
        const char *scan;
        int have_wanted = 0;
        int have_none_key = 0;

        scan = p;
        for (i = 0; i < variants; i++) {
            uint8_t vtag = NANO_LEX_UNBIAS(scan[0]);
            uint8_t vlen1 = NANO_LEX_UNBIAS(scan[1]);
            if (vtag == NLG_TAG_NONEKEY) {
                have_none_key = 1;
            }
            if (vtag == wanted) {
                have_wanted = 1;
            }
            scan += 2 + (vlen1 ? (vlen1 - 1) : 0);
        }
        if (have_ctx && future_vowel < 0 && have_none_key) {
            wanted = NLG_TAG_NONEKEY;
        } else if (!have_wanted) {
            wanted = NANO_LEX_TAG_PARENT[tag];
        }
        scan = p;
        for (i = 0; i < variants; i++) {
            uint8_t vtag = NANO_LEX_UNBIAS(scan[0]);
            uint8_t vlen1 = NANO_LEX_UNBIAS(scan[1]);
            uint16_t vlen = vlen1 ? (uint16_t)(vlen1 - 1u) : 0u;
            if (vtag == wanted) {
                if (vlen1 == 0) {
                    *is_null = 1;
                } else {
                    *bytes = scan + 2;
                    *len = vlen;
                }
                return NANO_LEX_OK;
            }
            scan += 2 + vlen;
        }
        /* dict.get(tag, entry["DEFAULT"]); the generator writes DEFAULT first. */
        if (NANO_LEX_UNBIAS(p[0]) != NLG_TAG_DEFAULT) {
            return NANO_LEX_E_TABLE;
        }
        {
            uint8_t vlen1 = NANO_LEX_UNBIAS(p[1]);
            if (vlen1 == 0) {
                *is_null = 1;
            } else {
                *bytes = p + 2;
                *len = (uint16_t)(vlen1 - 1u);
            }
        }
        return NANO_LEX_OK;
    }
}

static int nlg_entry_value(const nlg_entry_t *entry, uint8_t tag, int have_ctx,
                           int8_t future_vowel, nlg_ps_t *out)
{
    const char *bytes;
    uint16_t len;
    int is_null;
    int rc = nlg_entry_bytes(entry, tag, have_ctx, future_vowel, &bytes, &len, &is_null);
    *out = NLG_PS_NONE;
    if (rc != NANO_LEX_OK || is_null) {
        return rc;
    }
    {
        nlg_sb_t sb;
        uint16_t i;
        nlg_sb_begin(&sb);
        for (i = 0; i < len; i++) {
            nlg_sb_push(&sb, NANO_LEX_UNBIAS(bytes[i]));
        }
        return nlg_sb_finish(&sb, out);
    }
}

/* --------------------------------------------------------------------------
 * Words
 * -------------------------------------------------------------------------- */

static void nlg_word_clear(nlg_word_t *w)
{
    w->len = 0;
}

static int nlg_word_push(nlg_word_t *w, uint16_t cp)
{
    if (w->len >= NLG_MAX_WORD) {
        return 0;
    }
    w->cp[w->len++] = cp;
    return 1;
}

static void nlg_word_from_ascii(nlg_word_t *w, const char *s)
{
    nlg_word_clear(w);
    while (*s) {
        (void)nlg_word_push(w, (uint16_t)(unsigned char)*s++);
    }
}

static int nlg_word_ascii(const nlg_word_t *w, char *buf, uint16_t *len)
{
    uint16_t i;
    if (w->len == 0 || w->len > NANO_LEX_MAX_KEY) {
        return 0;
    }
    for (i = 0; i < w->len; i++) {
        if (w->cp[i] > 0x7Fu) {
            return 0;
        }
        buf[i] = (char)w->cp[i];
    }
    *len = w->len;
    return 1;
}

static int nlg_word_eq_ascii(const nlg_word_t *w, const char *s)
{
    size_t n = nlg_slen(s);
    uint16_t i;
    if (w->len != n) {
        return 0;
    }
    for (i = 0; i < w->len; i++) {
        if (w->cp[i] != (uint16_t)(unsigned char)s[i]) {
            return 0;
        }
    }
    return 1;
}

static int nlg_word_ends_ascii(const nlg_word_t *w, const char *s)
{
    size_t n = nlg_slen(s);
    uint16_t i;
    if (w->len < n) {
        return 0;
    }
    for (i = 0; i < n; i++) {
        if (w->cp[w->len - n + i] != (uint16_t)(unsigned char)s[i]) {
            return 0;
        }
    }
    return 1;
}

static int nlg_word_is_lower(const nlg_word_t *w)
{
    uint16_t i;
    for (i = 0; i < w->len; i++) {
        if (nlg_lower(w->cp[i]) != w->cp[i]) {
            return 0;
        }
    }
    return 1;
}

static int nlg_word_is_upper(const nlg_word_t *w)
{
    uint16_t i;
    for (i = 0; i < w->len; i++) {
        if (nlg_upper(w->cp[i]) != w->cp[i]) {
            return 0;
        }
    }
    return 1;
}

static int nlg_word_is_alpha(const nlg_word_t *w)
{
    uint16_t i;
    if (w->len == 0) {
        return 0;
    }
    for (i = 0; i < w->len; i++) {
        if (!nlg_is_letter(w->cp[i])) {
            return 0;
        }
    }
    return 1;
}

static void nlg_word_to_lower(nlg_word_t *w)
{
    uint16_t i;
    for (i = 0; i < w->len; i++) {
        w->cp[i] = nlg_lower(w->cp[i]);
    }
}

static void nlg_word_capitalize(nlg_word_t *w)
{
    uint16_t i;
    if (w->len == 0) {
        return;
    }
    w->cp[0] = nlg_upper(w->cp[0]);
    for (i = 1; i < w->len; i++) {
        w->cp[i] = nlg_lower(w->cp[i]);
    }
}

static int nlg_word_same(const nlg_word_t *a, const nlg_word_t *b)
{
    uint16_t i;
    if (a->len != b->len) {
        return 0;
    }
    for (i = 0; i < a->len; i++) {
        if (a->cp[i] != b->cp[i]) {
            return 0;
        }
    }
    return 1;
}

/* --------------------------------------------------------------------------
 * misaki/en.py: apply_stress
 * -------------------------------------------------------------------------- */

#define NLG_PRIMARY   NLG_PH_02C8
#define NLG_SECONDARY NLG_PH_02CC

static int nlg_is_stress(uint8_t code)
{
    return code == NLG_PRIMARY || code == NLG_SECONDARY;
}

static uint16_t nlg_stress_weight(const nlg_ps_t *ps)
{
    uint16_t i;
    uint16_t total = 0;
    if (!ps->present) {
        return 0;
    }
    for (i = 0; i < ps->len; i++) {
        total = (uint16_t)(total +
            ((NANO_LEX_PH_FLAGS[g_ws.arena[ps->off + i]] & NANO_LEX_F_DIPHTHONG) ? 2u : 1u));
    }
    return total;
}

/* The `restress` closure, in the only shape it is ever reached with: one
 * leading stress mark on an otherwise mark-free string that is known to hold a
 * vowel. Under those conditions the sort collapses to "move the mark to just
 * before the first vowel". Both call sites check the preconditions. */
static int nlg_restress(uint8_t mark, nlg_ps_t *ps, nlg_ps_t *out)
{
    uint16_t vowel = 0;
    uint16_t i;
    nlg_sb_t sb;
    for (i = 0; i < ps->len; i++) {
        if (NANO_LEX_PH_FLAGS[nlg_ps_at(ps, i)] & NANO_LEX_F_VOWEL) {
            vowel = i;
            break;
        }
    }
    if (i == ps->len) {
        return NANO_LEX_E_INTERNAL;
    }
    nlg_sb_begin(&sb);
    for (i = 0; i < vowel; i++) {
        nlg_sb_push(&sb, nlg_ps_at(ps, i));
    }
    nlg_sb_push(&sb, mark);
    for (i = vowel; i < ps->len; i++) {
        nlg_sb_push(&sb, nlg_ps_at(ps, i));
    }
    return nlg_sb_finish(&sb, out);
}

static int nlg_apply_stress(nlg_ps_t *ps, int8_t stress2)
{
    int has_primary;
    int has_secondary;
    int has_any;

    if (!ps->present || stress2 == NLG_S2_NONE) {
        return NANO_LEX_OK;
    }
    has_primary = nlg_ps_has(ps, NLG_PRIMARY);
    has_secondary = nlg_ps_has(ps, NLG_SECONDARY);
    has_any = has_primary || has_secondary;

    if (stress2 < -2) {
        nlg_sb_t sb;
        uint16_t i;
        nlg_sb_begin(&sb);
        for (i = 0; i < ps->len; i++) {
            if (!nlg_is_stress(nlg_ps_at(ps, i))) {
                nlg_sb_push(&sb, nlg_ps_at(ps, i));
            }
        }
        return nlg_sb_finish(&sb, ps);
    }
    if (stress2 == -2 || ((stress2 == 0 || stress2 == -1) && has_primary)) {
        nlg_sb_t sb;
        uint16_t i;
        nlg_sb_begin(&sb);
        for (i = 0; i < ps->len; i++) {
            uint8_t code = nlg_ps_at(ps, i);
            if (code == NLG_SECONDARY) {
                continue;
            }
            nlg_sb_push(&sb, code == NLG_PRIMARY ? (uint8_t)NLG_SECONDARY : code);
        }
        return nlg_sb_finish(&sb, ps);
    }
    if ((stress2 == 0 || stress2 == 1 || stress2 == 2) && !has_any) {
        if (!nlg_ps_has_flag(ps, NANO_LEX_F_VOWEL)) {
            return NANO_LEX_OK;
        }
        return nlg_restress(NLG_SECONDARY, ps, ps);
    }
    if (stress2 >= 2 && !has_primary && has_secondary) {
        nlg_sb_t sb;
        uint16_t i;
        nlg_sb_begin(&sb);
        for (i = 0; i < ps->len; i++) {
            uint8_t code = nlg_ps_at(ps, i);
            nlg_sb_push(&sb, code == NLG_SECONDARY ? (uint8_t)NLG_PRIMARY : code);
        }
        return nlg_sb_finish(&sb, ps);
    }
    if (stress2 > 2 && !has_any) {
        if (!nlg_ps_has_flag(ps, NANO_LEX_F_VOWEL)) {
            return NANO_LEX_OK;
        }
        return nlg_restress(NLG_PRIMARY, ps, ps);
    }
    return NANO_LEX_OK;
}

/* --------------------------------------------------------------------------
 * Lexicon
 * -------------------------------------------------------------------------- */

static const char *const NLG_CURRENCY_UNIT[4][2] = {
    { "", "" }, { "dollar", "cent" }, { "pound", "pence" }, { "euro", "cent" }
};

static int nlg_bsearch_str(const char *const *table, int count, const char *key, uint16_t klen)
{
    int lo = 0;
    int hi = count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        size_t n = nlg_slen(table[mid]);
        size_t m = (n < klen) ? n : klen;
        int cmp = nlg_cmp(table[mid], key, m);
        if (cmp == 0) {
            cmp = (n == klen) ? 0 : ((n < klen) ? -1 : 1);
        }
        if (cmp == 0) {
            return mid;
        }
        if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
}

/* Lexicon.grow_dictionary, evaluated lazily instead of materialised.
 *
 * grow_dictionary adds, for every lower-case key, its capitalised spelling,
 * and for every capitalised key, its lower-case spelling, with the original
 * entries winning any collision. Searching the originals and then the single
 * alias shape that could have produced a hit answers every query identically
 * without doubling the 3 MB of flash. */
static int nlg_grown_find(const nano_lex_dict_t *dict, const nlg_word_t *w,
                          nlg_entry_t *entry, int *found)
{
    char buf[NANO_LEX_MAX_KEY];
    uint16_t klen;
    nlg_word_t *alt = &g_ws.w_grow;
    int rc;

    *found = 0;
    if (nlg_word_ascii(w, buf, &klen)) {
        rc = nlg_dict_find(dict, buf, klen, entry);
        if (rc < 0) {
            return rc;
        }
        if (rc == 1) {
            *found = 1;
            return NANO_LEX_OK;
        }
    }
    if (w->len < 2) {
        return NANO_LEX_OK;
    }
    *alt = *w;
    if (nlg_word_is_lower(w)) {
        nlg_word_capitalize(alt);
        if (nlg_word_same(alt, w)) {
            return NANO_LEX_OK;    /* grow_dictionary's `key != key.capitalize()` */
        }
    } else {
        nlg_word_to_lower(alt);
        nlg_word_capitalize(alt);
        if (!nlg_word_same(alt, w)) {
            return NANO_LEX_OK;    /* not `key == key.lower().capitalize()` */
        }
        nlg_word_to_lower(alt);
    }
    if (!nlg_word_ascii(alt, buf, &klen)) {
        return NANO_LEX_OK;
    }
    rc = nlg_dict_find(dict, buf, klen, entry);
    if (rc < 0) {
        return rc;
    }
    *found = (rc == 1);
    return NANO_LEX_OK;
}

static int nlg_in_gold(const nlg_word_t *w, int *found)
{
    nlg_entry_t entry;
    return nlg_grown_find(&NANO_LEX_GOLD, w, &entry, found);
}

static int nlg_in_silver(const nlg_word_t *w, int *found)
{
#if NANO_LEX_WITH_SILVER
    nlg_entry_t entry;
    return nlg_grown_find(&NANO_LEX_SILVER, w, &entry, found);
#else
    (void)w;
    *found = 0;
    return NANO_LEX_OK;
#endif
}

static const char *nlg_symbol_word(const nlg_word_t *w)
{
    if (w->len != 1) {
        return 0;
    }
    switch (w->cp[0]) {
    case '%': return "percent";
    case '&': return "and";
    case '+': return "plus";
    case '@': return "at";
    default:  return 0;
    }
}

static int nlg_lookup(const nlg_word_t *word, uint8_t tag, int8_t stress2,
                      const nlg_ctx_t *ctx, nlg_ps_t *ps, int8_t *rating);
static int nlg_stem_s(const nlg_word_t *word, uint8_t tag, int8_t stress2,
                      const nlg_ctx_t *ctx, nlg_ps_t *ps, int8_t *rating);
static int nlg_stem_ed(const nlg_word_t *word, uint8_t tag, int8_t stress2,
                       const nlg_ctx_t *ctx, nlg_ps_t *ps, int8_t *rating);
static int nlg_stem_ing(const nlg_word_t *word, uint8_t tag, int8_t stress2,
                        const nlg_ctx_t *ctx, nlg_ps_t *ps, int8_t *rating);

/* misaki/en.py: Lexicon.get_NNP -- spell the word out letter by letter. */
static int nlg_get_NNP(const nlg_word_t *word, nlg_ps_t *ps, int8_t *rating)
{
    nlg_sb_t sb;
    uint16_t i;
    int any = 0;
    uint16_t mark = nlg_arena_mark();

    *ps = NLG_PS_NONE;
    *rating = -1;
    nlg_sb_begin(&sb);
    for (i = 0; i < word->len; i++) {
        nlg_entry_t entry;
        const char *bytes;
        uint16_t len;
        char key[1];
        int is_null;
        int found;
        int rc;
        uint16_t k;
        if (!nlg_is_letter(word->cp[i])) {
            continue;
        }
        any = 1;
        /* grow_dictionary skips keys shorter than two characters, so a single
         * letter is a plain gold hit with no alias to consider. */
        if (nlg_upper(word->cp[i]) > 0x7Fu) {
            nlg_arena_rewind(mark);
            return NANO_LEX_OK;
        }
        key[0] = (char)nlg_upper(word->cp[i]);
        rc = nlg_dict_find(&NANO_LEX_GOLD, key, 1, &entry);
        if (rc < 0) {
            nlg_arena_rewind(mark);
            return rc;
        }
        found = (rc == 1);
        if (!found) {
            nlg_arena_rewind(mark);
            return NANO_LEX_OK;
        }
        rc = nlg_entry_bytes(&entry, NLG_TAG_NONE, 0, -1, &bytes, &len, &is_null);
        if (rc != NANO_LEX_OK) {
            nlg_arena_rewind(mark);
            return rc;
        }
        if (is_null) {
            nlg_arena_rewind(mark);
            return NANO_LEX_OK;
        }
        for (k = 0; k < len; k++) {
            nlg_sb_push(&sb, NANO_LEX_UNBIAS(bytes[k]));
        }
    }
    if (!any) {
        nlg_arena_rewind(mark);
        return NANO_LEX_OK;
    }
    {
        nlg_ps_t spelled;
        int rc = nlg_sb_finish(&sb, &spelled);
        uint16_t j;
        int32_t last_secondary = -1;
        if (rc != NANO_LEX_OK) {
            nlg_arena_rewind(mark);
            return rc;
        }
        rc = nlg_apply_stress(&spelled, 0);
        if (rc != NANO_LEX_OK) {
            nlg_arena_rewind(mark);
            return rc;
        }
        for (j = 0; j < spelled.len; j++) {
            if (nlg_ps_at(&spelled, j) == NLG_SECONDARY) {
                last_secondary = (int32_t)j;
            }
        }
        if (last_secondary > 0) {
            g_ws.arena[spelled.off + last_secondary] = NLG_PRIMARY;
        }
        nlg_arena_keep(mark, &spelled);
        *ps = spelled;
        *rating = 3;
        return NANO_LEX_OK;
    }
}

/* misaki/en.py: Lexicon.is_known */
static int nlg_is_known(const nlg_word_t *word, int *known)
{
    int found;
    int rc;
    uint16_t i;
    nlg_word_t *lowered = &g_ws.w_known;

    *known = 0;
    rc = nlg_in_gold(word, &found);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    if (found || nlg_symbol_word(word) != 0) {
        *known = 1;
        return NANO_LEX_OK;
    }
    rc = nlg_in_silver(word, &found);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    if (found) {
        *known = 1;
        return NANO_LEX_OK;
    }
    if (!nlg_word_is_alpha(word)) {
        return NANO_LEX_OK;
    }
    for (i = 0; i < word->len; i++) {
        uint16_t c = word->cp[i];
        int in_ords = (c == 39u) || (c == 45u) ||
                      (c >= 65u && c <= 90u) || (c >= 97u && c <= 122u);
        if (!in_ords) {
            return NANO_LEX_OK;
        }
    }
    if (word->len == 1) {
        *known = 1;
        return NANO_LEX_OK;
    }
    if (nlg_word_is_upper(word)) {
        *lowered = *word;
        nlg_word_to_lower(lowered);
        rc = nlg_in_gold(lowered, &found);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (found) {
            *known = 1;
            return NANO_LEX_OK;
        }
    }
    for (i = 1; i < word->len; i++) {
        if (nlg_upper(word->cp[i]) != word->cp[i]) {
            return NANO_LEX_OK;
        }
    }
    *known = 1;
    return NANO_LEX_OK;
}

/* misaki/en.py: Lexicon.lookup */
static int nlg_lookup(const nlg_word_t *word, uint8_t tag, int8_t stress2,
                      const nlg_ctx_t *ctx, nlg_ps_t *ps, int8_t *rating)
{
    nlg_word_t *w = &g_ws.w_lookup;
    int is_NNP = -1;
    nlg_entry_t entry;
    int found = 0;
    int rc;
    uint16_t mark = nlg_arena_mark();

    *ps = NLG_PS_NONE;
    *rating = 4;
    *w = *word;

    if (nlg_word_is_upper(w)) {
        int in_gold;
        rc = nlg_in_gold(w, &in_gold);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (!in_gold) {
            nlg_word_to_lower(w);
            is_NNP = (tag == NLG_TAG_NNP) ? 1 : 0;
        }
    }
    rc = nlg_grown_find(&NANO_LEX_GOLD, w, &entry, &found);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    if (!found && is_NNP != 1) {
        *rating = 3;
#if NANO_LEX_WITH_SILVER
        rc = nlg_grown_find(&NANO_LEX_SILVER, w, &entry, &found);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
#endif
    }
    if (found) {
        rc = nlg_entry_value(&entry, tag, ctx != 0,
                             ctx ? ctx->future_vowel : (int8_t)-1, ps);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
    }
    if (!ps->present || (is_NNP == 1 && !nlg_ps_has(ps, NLG_PRIMARY))) {
        nlg_ps_t spelled;
        int8_t spelled_rating;
        uint16_t inner = nlg_arena_mark();
        rc = nlg_get_NNP(w, &spelled, &spelled_rating);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (spelled.present) {
            nlg_arena_keep(mark, &spelled);
            *ps = spelled;
            *rating = spelled_rating;
            return NANO_LEX_OK;
        }
        nlg_arena_rewind(inner);
    }
    rc = nlg_apply_stress(ps, stress2);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    nlg_arena_keep(mark, ps);
    return NANO_LEX_OK;
}

/* misaki/en.py: Lexicon._s */
static int nlg_suffix_s(nlg_ps_t *ps)
{
    uint8_t last;
    nlg_sb_t sb;
    if (!ps->present || ps->len == 0) {
        *ps = NLG_PS_NONE;
        return NANO_LEX_OK;
    }
    last = nlg_ps_at(ps, ps->len - 1);
    nlg_sb_begin(&sb);
    nlg_sb_push_ps(&sb, ps);
    if (last == NLG_PH_0070 || last == NLG_PH_0074 || last == NLG_PH_006B ||
        last == NLG_PH_0066 || last == NLG_PH_03B8) {
        nlg_sb_push(&sb, NLG_PH_0073);
    } else if (last == NLG_PH_0073 || last == NLG_PH_007A || last == NLG_PH_0283 ||
               last == NLG_PH_0292 || last == NLG_PH_02A7 || last == NLG_PH_02A4) {
        nlg_sb_push(&sb, NLG_PH_1D7B);
        nlg_sb_push(&sb, NLG_PH_007A);
    } else {
        nlg_sb_push(&sb, NLG_PH_007A);
    }
    return nlg_sb_finish(&sb, ps);
}

/* misaki/en.py: Lexicon._ed */
static int nlg_suffix_ed(nlg_ps_t *ps)
{
    uint8_t last;
    nlg_sb_t sb;
    if (!ps->present || ps->len == 0) {
        *ps = NLG_PS_NONE;
        return NANO_LEX_OK;
    }
    last = nlg_ps_at(ps, ps->len - 1);
    nlg_sb_begin(&sb);
    if (last == NLG_PH_0070 || last == NLG_PH_006B || last == NLG_PH_0066 ||
        last == NLG_PH_03B8 || last == NLG_PH_0283 || last == NLG_PH_0073 ||
        last == NLG_PH_02A7) {
        nlg_sb_push_ps(&sb, ps);
        nlg_sb_push(&sb, NLG_PH_0074);
    } else if (last == NLG_PH_0064) {
        nlg_sb_push_ps(&sb, ps);
        nlg_sb_push(&sb, NLG_PH_1D7B);
        nlg_sb_push(&sb, NLG_PH_0064);
    } else if (last != NLG_PH_0074) {
        nlg_sb_push_ps(&sb, ps);
        nlg_sb_push(&sb, NLG_PH_0064);
    } else if (ps->len < 2) {
        nlg_sb_push_ps(&sb, ps);
        nlg_sb_push(&sb, NLG_PH_026A);
        nlg_sb_push(&sb, NLG_PH_0064);
    } else if (NANO_LEX_PH_FLAGS[nlg_ps_at(ps, ps->len - 2)] & NANO_LEX_F_US_TAU) {
        nlg_sb_push_span(&sb, ps->off, (uint16_t)(ps->len - 1));
        nlg_sb_push(&sb, NLG_PH_027E);
        nlg_sb_push(&sb, NLG_PH_1D7B);
        nlg_sb_push(&sb, NLG_PH_0064);
    } else {
        nlg_sb_push_ps(&sb, ps);
        nlg_sb_push(&sb, NLG_PH_1D7B);
        nlg_sb_push(&sb, NLG_PH_0064);
    }
    return nlg_sb_finish(&sb, ps);
}

/* misaki/en.py: Lexicon._ing */
static int nlg_suffix_ing(nlg_ps_t *ps)
{
    nlg_sb_t sb;
    if (!ps->present || ps->len == 0) {
        *ps = NLG_PS_NONE;
        return NANO_LEX_OK;
    }
    nlg_sb_begin(&sb);
    if (ps->len > 1 && nlg_ps_at(ps, ps->len - 1) == NLG_PH_0074 &&
        (NANO_LEX_PH_FLAGS[nlg_ps_at(ps, ps->len - 2)] & NANO_LEX_F_US_TAU)) {
        nlg_sb_push_span(&sb, ps->off, (uint16_t)(ps->len - 1));
        nlg_sb_push(&sb, NLG_PH_027E);
        nlg_sb_push(&sb, NLG_PH_026A);
        nlg_sb_push(&sb, NLG_PH_014B);
    } else {
        nlg_sb_push_ps(&sb, ps);
        nlg_sb_push(&sb, NLG_PH_026A);
        nlg_sb_push(&sb, NLG_PH_014B);
    }
    return nlg_sb_finish(&sb, ps);
}

static int nlg_stem_s(const nlg_word_t *word, uint8_t tag, int8_t stress2,
                      const nlg_ctx_t *ctx, nlg_ps_t *ps, int8_t *rating)
{
    nlg_word_t *stem = &g_ws.w_stem;
    int known;
    int rc;
    uint16_t mark = nlg_arena_mark();

    *ps = NLG_PS_NONE;
    *rating = -1;
    if (word->len < 3 || !nlg_word_ends_ascii(word, "s")) {
        return NANO_LEX_OK;
    }
    if (!nlg_word_ends_ascii(word, "ss")) {
        *stem = *word;
        stem->len = (uint16_t)(word->len - 1);
        rc = nlg_is_known(stem, &known);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (known) {
            goto have_stem;
        }
    }
    if (nlg_word_ends_ascii(word, "'s") ||
        (word->len > 4 && nlg_word_ends_ascii(word, "es") &&
         !nlg_word_ends_ascii(word, "ies"))) {
        *stem = *word;
        stem->len = (uint16_t)(word->len - 2);
        rc = nlg_is_known(stem, &known);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (known) {
            goto have_stem;
        }
    }
    if (word->len > 4 && nlg_word_ends_ascii(word, "ies")) {
        *stem = *word;
        stem->len = (uint16_t)(word->len - 3);
        (void)nlg_word_push(stem, 'y');
        rc = nlg_is_known(stem, &known);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (known) {
            goto have_stem;
        }
    }
    return NANO_LEX_OK;

have_stem:
    rc = nlg_lookup(stem, tag, stress2, ctx, ps, rating);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    rc = nlg_suffix_s(ps);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    nlg_arena_keep(mark, ps);
    return NANO_LEX_OK;
}

static int nlg_stem_ed(const nlg_word_t *word, uint8_t tag, int8_t stress2,
                       const nlg_ctx_t *ctx, nlg_ps_t *ps, int8_t *rating)
{
    nlg_word_t *stem = &g_ws.w_stem;
    int known;
    int rc;
    uint16_t mark = nlg_arena_mark();

    *ps = NLG_PS_NONE;
    *rating = -1;
    if (word->len < 4 || !nlg_word_ends_ascii(word, "d")) {
        return NANO_LEX_OK;
    }
    if (!nlg_word_ends_ascii(word, "dd")) {
        *stem = *word;
        stem->len = (uint16_t)(word->len - 1);
        rc = nlg_is_known(stem, &known);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (known) {
            goto have_stem;
        }
    }
    if (word->len > 4 && nlg_word_ends_ascii(word, "ed") &&
        !nlg_word_ends_ascii(word, "eed")) {
        *stem = *word;
        stem->len = (uint16_t)(word->len - 2);
        rc = nlg_is_known(stem, &known);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (known) {
            goto have_stem;
        }
    }
    return NANO_LEX_OK;

have_stem:
    rc = nlg_lookup(stem, tag, stress2, ctx, ps, rating);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    rc = nlg_suffix_ed(ps);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    nlg_arena_keep(mark, ps);
    return NANO_LEX_OK;
}

/* re.search(r"([bcdgklmnprstvxz])\1ing$|cking$", word) */
static int nlg_doubled_ing(const nlg_word_t *w)
{
    static const char doubles[] = "bcdgklmnprstvxz";
    uint16_t n = w->len;
    uint16_t i;
    if (n < 5 || !nlg_word_ends_ascii(w, "ing")) {
        return 0;
    }
    if (nlg_word_ends_ascii(w, "cking")) {
        return 1;
    }
    if (w->cp[n - 4] != w->cp[n - 5]) {
        return 0;
    }
    for (i = 0; doubles[i]; i++) {
        if (w->cp[n - 4] == (uint16_t)(unsigned char)doubles[i]) {
            return 1;
        }
    }
    return 0;
}

static int nlg_stem_ing(const nlg_word_t *word, uint8_t tag, int8_t stress2,
                        const nlg_ctx_t *ctx, nlg_ps_t *ps, int8_t *rating)
{
    nlg_word_t *stem = &g_ws.w_stem;
    int known;
    int rc;
    uint16_t mark = nlg_arena_mark();

    *ps = NLG_PS_NONE;
    *rating = -1;
    if (word->len < 5 || !nlg_word_ends_ascii(word, "ing")) {
        return NANO_LEX_OK;
    }
    if (word->len > 5) {
        *stem = *word;
        stem->len = (uint16_t)(word->len - 3);
        rc = nlg_is_known(stem, &known);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (known) {
            goto have_stem;
        }
    }
    *stem = *word;
    stem->len = (uint16_t)(word->len - 3);
    (void)nlg_word_push(stem, 'e');
    rc = nlg_is_known(stem, &known);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    if (known) {
        goto have_stem;
    }
    if (word->len > 5 && nlg_doubled_ing(word)) {
        *stem = *word;
        stem->len = (uint16_t)(word->len - 4);
        rc = nlg_is_known(stem, &known);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (known) {
            goto have_stem;
        }
    }
    return NANO_LEX_OK;

have_stem:
    rc = nlg_lookup(stem, tag, stress2, ctx, ps, rating);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    rc = nlg_suffix_ing(ps);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    nlg_arena_keep(mark, ps);
    return NANO_LEX_OK;
}

/* --------------------------------------------------------------------------
 * Numbers -- the English subset of num2words misaki asks for
 * -------------------------------------------------------------------------- */

static const char *const NLG_ONES[20] = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen"
};
static const char *const NLG_TENS[10] = {
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety"
};
static const char *const NLG_SCALES[6] = {
    "", "thousand", "million", "billion", "trillion", "quadrillion"
};

typedef struct {
    char  *buf;
    size_t cap;
    size_t len;
    int    rc;
} nlg_txt_t;

static void nlg_txt_begin(nlg_txt_t *t, char *buf, size_t cap)
{
    t->buf = buf;
    t->cap = cap;
    t->len = 0;
    t->rc = NANO_LEX_OK;
    if (cap > 0) {
        buf[0] = '\0';
    }
}

static void nlg_txt_puts(nlg_txt_t *t, const char *s)
{
    if (t->rc != NANO_LEX_OK) {
        return;
    }
    while (*s) {
        if (t->len + 1 >= t->cap) {
            t->rc = NANO_LEX_E_NUMBER;
            return;
        }
        t->buf[t->len++] = *s++;
    }
    t->buf[t->len] = '\0';
}

static void nlg_txt_putc(nlg_txt_t *t, char c)
{
    char one[2];
    one[0] = c;
    one[1] = '\0';
    nlg_txt_puts(t, one);
}

static void nlg_under_hundred(nlg_txt_t *t, int64_t v)
{
    if (v < 20) {
        nlg_txt_puts(t, NLG_ONES[v]);
        return;
    }
    nlg_txt_puts(t, NLG_TENS[v / 10]);
    if (v % 10 != 0) {
        nlg_txt_puts(t, "-");
        nlg_txt_puts(t, NLG_ONES[v % 10]);
    }
}

static void nlg_under_thousand(nlg_txt_t *t, int64_t v)
{
    int64_t hundreds = v / 100;
    int64_t rest = v % 100;
    if (hundreds == 0) {
        nlg_under_hundred(t, rest);
        return;
    }
    nlg_txt_puts(t, NLG_ONES[hundreds]);
    nlg_txt_puts(t, " hundred");
    if (rest != 0) {
        nlg_txt_puts(t, " and ");
        nlg_under_hundred(t, rest);
    }
}

/* nano_g2p.cardinal_words. Recurses at most once, for the negative sign. */
static void nlg_cardinal(nlg_txt_t *t, int64_t v)
{
    int64_t groups[7];
    int ngroups = 0;
    int64_t rest;
    int power;
    int written = 0;
    int last_nonzero = -1;
    int64_t tail;

    if (v < 0) {
        nlg_txt_puts(t, "minus ");
        nlg_cardinal(t, -v);
        return;
    }
    if (v < 1000) {
        nlg_under_thousand(t, v);
        return;
    }
    rest = v;
    while (rest != 0) {
        if (ngroups >= 7) {
            t->rc = NANO_LEX_E_NUMBER;
            return;
        }
        groups[ngroups++] = rest % 1000;
        rest /= 1000;
    }
    if (ngroups > 6) {
        t->rc = NANO_LEX_E_NUMBER;   /* larger than _SCALES covers */
        return;
    }
    tail = groups[0];
    for (power = ngroups - 1; power >= 0; power--) {
        if (groups[power] != 0) {
            last_nonzero = power;
        }
    }
    for (power = ngroups - 1; power >= 0; power--) {
        if (groups[power] == 0) {
            continue;
        }
        if (written > 0) {
            if (power == last_nonzero && tail > 0 && tail < 100) {
                nlg_txt_puts(t, " and ");
            } else {
                nlg_txt_puts(t, ", ");
            }
        }
        nlg_under_thousand(t, groups[power]);
        if (power > 0) {
            nlg_txt_puts(t, " ");
            nlg_txt_puts(t, NLG_SCALES[power]);
        }
        written++;
    }
}

static void nlg_ordinalise(nlg_txt_t *t, const char *word, size_t n)
{
    static const char *const from[7] = { "one", "two", "three", "five", "eight",
                                         "nine", "twelve" };
    static const char *const to[7] = { "first", "second", "third", "fifth",
                                       "eighth", "ninth", "twelfth" };
    size_t k;
    int i;
    for (i = 0; i < 7; i++) {
        if (nlg_slen(from[i]) == n && nlg_cmp(from[i], word, n) == 0) {
            nlg_txt_puts(t, to[i]);
            return;
        }
    }
    if (n > 0 && word[n - 1] == 'y') {
        for (k = 0; k + 1 < n; k++) {
            nlg_txt_putc(t, word[k]);
        }
        nlg_txt_puts(t, "ieth");
        return;
    }
    for (k = 0; k < n; k++) {
        nlg_txt_putc(t, word[k]);
    }
    nlg_txt_puts(t, "th");
}

/* nano_g2p.ordinal_words: the cardinal with its last [a-z]+ run ordinalised.
 * `scratch` is g_ws.num_scratch, so it must not be the same buffer as `t`. */
static void nlg_ordinal(nlg_txt_t *t, int64_t v)
{
    nlg_txt_t inner;
    size_t end;
    size_t start;
    size_t i;

    nlg_txt_begin(&inner, g_ws.num_scratch, sizeof g_ws.num_scratch);
    nlg_cardinal(&inner, v);
    if (inner.rc != NANO_LEX_OK) {
        t->rc = inner.rc;
        return;
    }
    end = inner.len;
    while (end > 0 && !(g_ws.num_scratch[end - 1] >= 'a' &&
                        g_ws.num_scratch[end - 1] <= 'z')) {
        end--;
    }
    if (end == 0) {
        t->rc = NANO_LEX_E_NUMBER;
        return;
    }
    start = end;
    while (start > 0 && g_ws.num_scratch[start - 1] >= 'a' &&
           g_ws.num_scratch[start - 1] <= 'z') {
        start--;
    }
    for (i = 0; i < start; i++) {
        nlg_txt_putc(t, g_ws.num_scratch[i]);
    }
    nlg_ordinalise(t, g_ws.num_scratch + start, end - start);
    for (i = end; i < inner.len; i++) {
        nlg_txt_putc(t, g_ws.num_scratch[i]);
    }
}

/* nano_g2p.year_words */
static void nlg_year(nlg_txt_t *t, int64_t v)
{
    int64_t high;
    int64_t low;
    if (v < 0) {
        t->rc = NANO_LEX_E_NUMBER;
        return;
    }
    high = v / 100;
    low = v % 100;
    if (high == 0 || (high % 10 == 0 && low < 10) || high >= 100) {
        nlg_cardinal(t, v);
        return;
    }
    if (low == 0) {
        nlg_cardinal(t, high);
        nlg_txt_puts(t, " hundred");
        return;
    }
    nlg_cardinal(t, high);
    nlg_txt_puts(t, low < 10 ? " oh-" : " ");
    nlg_cardinal(t, low);
}

static int nlg_parse_int(const uint16_t *cp, uint16_t len, int64_t *out)
{
    int64_t value = 0;
    uint16_t i;
    if (len == 0 || len > 18) {
        return NANO_LEX_E_NUMBER;
    }
    for (i = 0; i < len; i++) {
        if (!nlg_is_digit_cp(cp[i])) {
            return NANO_LEX_E_NUMBER;
        }
        value = value * 10 + (int64_t)(cp[i] - '0');
    }
    *out = value;
    return NANO_LEX_OK;
}

/* nano_g2p.decimal_words */
static void nlg_decimal(nlg_txt_t *t, const uint16_t *cp, uint16_t len)
{
    uint16_t dot = len;
    uint16_t i;
    for (i = 0; i < len; i++) {
        if (cp[i] == '.') {
            dot = i;
            break;
        }
    }
    if (dot == 0) {
        nlg_txt_puts(t, "zero");
    } else {
        int64_t whole;
        if (nlg_parse_int(cp, dot, &whole) != NANO_LEX_OK) {
            t->rc = NANO_LEX_E_NUMBER;
            return;
        }
        nlg_cardinal(t, whole);
    }
    if (dot >= len) {
        return;   /* no '.' at all: str.partition leaves the fraction empty */
    }
    if (dot + 1 >= len) {
        return;   /* a trailing '.' likewise */
    }
    nlg_txt_puts(t, " point");
    for (i = (uint16_t)(dot + 1); i < len; i++) {
        if (!nlg_is_digit_cp(cp[i])) {
            t->rc = NANO_LEX_E_NUMBER;
            return;
        }
        nlg_txt_puts(t, " ");
        nlg_txt_puts(t, NLG_ONES[cp[i] - '0']);
    }
}

/* --------------------------------------------------------------------------
 * Lexicon.get_number
 * -------------------------------------------------------------------------- */

static int nlg_num_push(const nlg_ps_t *ps, int8_t rating)
{
    if (g_ws.num_count >= NLG_NUM_PARTS) {
        return NANO_LEX_E_INTERNAL;
    }
    g_ws.num_parts[g_ws.num_count] = *ps;
    g_ws.num_ratings[g_ws.num_count] = rating;
    g_ws.num_count++;
    return NANO_LEX_OK;
}

static int nlg_num_lookup_word(const char *word, size_t n, int8_t stress2)
{
    nlg_word_t *w = &g_ws.w_numword;
    nlg_ps_t ps;
    int8_t rating;
    size_t i;
    int rc;
    nlg_word_clear(w);
    for (i = 0; i < n; i++) {
        if (!nlg_word_push(w, (uint16_t)(unsigned char)word[i])) {
            return NANO_LEX_E_INTERNAL;
        }
    }
    rc = nlg_lookup(w, NLG_TAG_NONE, stress2, 0, &ps, &rating);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    return nlg_num_push(&ps, rating);
}

/* The `extend_num` closure inside Lexicon.get_number.
 *
 * DEVIATION (with no effect): the closure has three branches keyed on
 * `num_flags`, which misaki sets in a preprocess step nano_g2p.py reduces to a
 * left strip. Nothing in the Python port ever assigns a non-empty num_flags,
 * so "and" is always dropped and the "a"/"n" branches are dead. This
 * implements the num_flags == 0 case and refuses to run otherwise rather than
 * pretending to handle flags it has no code for. */
static int nlg_extend_num_text(const char *text, size_t len, uint8_t num_flags)
{
    size_t i = 0;
    if (num_flags != 0) {
        return NANO_LEX_E_INTERNAL;
    }
    while (i < len) {
        size_t start;
        size_t n;
        if (!(text[i] >= 'a' && text[i] <= 'z')) {
            i++;
            continue;
        }
        start = i;
        while (i < len && text[i] >= 'a' && text[i] <= 'z') {
            i++;
        }
        n = i - start;
        if (n == 3 && nlg_cmp(text + start, "and", 3) == 0) {
            continue;
        }
        {
            int is_point = (n == 5 && nlg_cmp(text + start, "point", 5) == 0);
            int rc = nlg_num_lookup_word(text + start, n,
                                         is_point ? (int8_t)-4 : NLG_S2_NONE);
            if (rc != NANO_LEX_OK) {
                return rc;
            }
        }
    }
    return NANO_LEX_OK;
}

static int nlg_extend_num_value(int64_t value, uint8_t num_flags)
{
    nlg_txt_t t;
    nlg_txt_begin(&t, g_ws.num_text, sizeof g_ws.num_text);
    nlg_cardinal(&t, value);
    if (t.rc != NANO_LEX_OK) {
        return t.rc;
    }
    return nlg_extend_num_text(g_ws.num_text, t.len, num_flags);
}

/* misaki/en.py: Lexicon.is_currency */
static int nlg_is_currency(const uint16_t *cp, uint16_t len)
{
    uint16_t dots = 0;
    uint16_t dot = len;
    uint16_t i;
    for (i = 0; i < len; i++) {
        if (cp[i] == '.') {
            dots++;
            if (dot == len) {
                dot = i;
            }
        }
    }
    if (dots == 0) {
        return 1;
    }
    if (dots > 1) {
        return 0;
    }
    {
        uint16_t cents = (uint16_t)(len - dot - 1);
        int all_zero = cents > 0;
        for (i = (uint16_t)(dot + 1); i < len; i++) {
            if (cp[i] != '0') {
                all_zero = 0;
            }
        }
        return cents < 3 || all_zero;
    }
}

/* misaki/en.py: Lexicon.is_number */
static int nlg_is_number(const uint16_t *cp, uint16_t len, int is_head)
{
    static const char *const suffixes[9] = { "ing", "'d", "ed", "'s",
                                             "st", "nd", "rd", "th", "s" };
    uint16_t n = len;
    uint16_t i;
    int any_digit = 0;

    for (i = 0; i < len; i++) {
        if (nlg_is_digit_cp(cp[i])) {
            any_digit = 1;
            break;
        }
    }
    if (!any_digit) {
        return 0;
    }
    for (i = 0; i < 9; i++) {
        size_t sn = nlg_slen(suffixes[i]);
        size_t k;
        int match = n >= sn;
        for (k = 0; match && k < sn; k++) {
            if (cp[n - sn + k] != (uint16_t)(unsigned char)suffixes[i][k]) {
                match = 0;
            }
        }
        if (match) {
            n = (uint16_t)(n - sn);
            break;
        }
    }
    for (i = 0; i < n; i++) {
        if (nlg_is_digit_cp(cp[i]) || cp[i] == ',' || cp[i] == '.') {
            continue;
        }
        if (is_head && i == 0 && cp[i] == '-') {
            continue;
        }
        return 0;
    }
    return 1;
}

static int nlg_get_number(const nlg_word_t *word_in, uint8_t currency, int is_head,
                          uint8_t num_flags, nlg_ps_t *ps, int8_t *rating)
{
    nlg_word_t *wp = &g_ws.w_number;
    uint16_t suffix_start;
    uint16_t mark = nlg_arena_mark();
    uint16_t *clean = g_ws.num_clean;
    uint16_t clen = 0;
    uint16_t k;
    int rc = NANO_LEX_OK;
    int i;
    int is_ordinal_suffix = 0;
    int is_s_suffix = 0;
    int is_ed_suffix = 0;
    int is_ing_suffix = 0;
    int all_digits;
    int has_dot = 0;
    uint16_t dots = 0;
    int minus_seen = 0;

    *ps = NLG_PS_NONE;
    *rating = -1;
    g_ws.num_count = 0;
    *wp = *word_in;
    suffix_start = wp->len;

    while (suffix_start > 0 &&
           ((wp->cp[suffix_start - 1] >= 'a' && wp->cp[suffix_start - 1] <= 'z') ||
            wp->cp[suffix_start - 1] == '\'')) {
        suffix_start--;
    }
    {
        uint16_t slen = (uint16_t)(wp->len - suffix_start);
        const uint16_t *sfx = wp->cp + suffix_start;
        if (slen == 2) {
            is_ordinal_suffix =
                (sfx[0] == 's' && sfx[1] == 't') || (sfx[0] == 'n' && sfx[1] == 'd') ||
                (sfx[0] == 'r' && sfx[1] == 'd') || (sfx[0] == 't' && sfx[1] == 'h');
            is_ed_suffix = (sfx[0] == 'e' && sfx[1] == 'd') ||
                           (sfx[0] == '\'' && sfx[1] == 'd');
            is_s_suffix = (sfx[0] == '\'' && sfx[1] == 's');
        } else if (slen == 1) {
            is_s_suffix = (sfx[0] == 's');
        } else if (slen == 3) {
            is_ing_suffix = (sfx[0] == 'i' && sfx[1] == 'n' && sfx[2] == 'g');
        }
        wp->len = suffix_start;
    }

    if (wp->len > 0 && wp->cp[0] == '-') {
        rc = nlg_num_lookup_word("minus", 5, NLG_S2_NONE);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        for (i = 0; i + 1 < (int)wp->len; i++) {
            wp->cp[i] = wp->cp[i + 1];
        }
        wp->len--;
        minus_seen = 1;
    }

    all_digits = nlg_all_digits(wp->cp, wp->len);
    for (k = 0; k < wp->len; k++) {
        if (wp->cp[k] == '.') {
            has_dot = 1;
            dots++;
        }
        if (wp->cp[k] != ',') {
            clean[clen++] = wp->cp[k];
        }
    }

    if (all_digits && is_ordinal_suffix) {
        int64_t value;
        nlg_txt_t t;
        rc = nlg_parse_int(wp->cp, wp->len, &value);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        nlg_txt_begin(&t, g_ws.num_text, sizeof g_ws.num_text);
        nlg_ordinal(&t, value);
        if (t.rc != NANO_LEX_OK) {
            return t.rc;
        }
        rc = nlg_extend_num_text(g_ws.num_text, t.len, num_flags);
    } else if (!minus_seen && wp->len == 4 && currency == 0 && all_digits) {
        int64_t value;
        nlg_txt_t t;
        rc = nlg_parse_int(wp->cp, wp->len, &value);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        nlg_txt_begin(&t, g_ws.num_text, sizeof g_ws.num_text);
        nlg_year(&t, value);
        if (t.rc != NANO_LEX_OK) {
            return t.rc;
        }
        rc = nlg_extend_num_text(g_ws.num_text, t.len, num_flags);
    } else if (!is_head && !has_dot) {
        if (clen == 0) {
            return NANO_LEX_E_NUMBER;   /* Python would index num[0] and raise */
        }
        if (clean[0] == '0' || clen > 3) {
            for (k = 0; k < clen; k++) {
                if (!nlg_is_digit_cp(clean[k])) {
                    return NANO_LEX_E_NUMBER;
                }
                rc = nlg_extend_num_value((int64_t)(clean[k] - '0'), num_flags);
                if (rc != NANO_LEX_OK) {
                    return rc;
                }
            }
        } else if (clen == 3 && !(clean[1] == '0' && clean[2] == '0')) {
            int64_t v;
            rc = nlg_parse_int(clean, 1, &v);
            if (rc != NANO_LEX_OK) {
                return rc;
            }
            rc = nlg_extend_num_value(v, num_flags);
            if (rc != NANO_LEX_OK) {
                return rc;
            }
            if (clean[1] == '0') {
                rc = nlg_num_lookup_word("O", 1, -4);
                if (rc != NANO_LEX_OK) {
                    return rc;
                }
                rc = nlg_parse_int(clean + 2, 1, &v);
                if (rc != NANO_LEX_OK) {
                    return rc;
                }
                rc = nlg_extend_num_value(v, num_flags);
            } else {
                rc = nlg_parse_int(clean + 1, 2, &v);
                if (rc != NANO_LEX_OK) {
                    return rc;
                }
                rc = nlg_extend_num_value(v, num_flags);
            }
        } else {
            int64_t v;
            rc = nlg_parse_int(clean, clen, &v);
            if (rc != NANO_LEX_OK) {
                return rc;
            }
            rc = nlg_extend_num_value(v, num_flags);
        }
    } else if (dots > 1 || !is_head) {
        uint16_t seg_start = 0;
        int first = 1;
        for (k = 0; k <= clen; k++) {
            if (k == clen || clean[k] == '.') {
                uint16_t n = (uint16_t)(k - seg_start);
                const uint16_t *seg = clean + seg_start;
                uint16_t j;
                int rest_nonzero = 0;
                seg_start = (uint16_t)(k + 1);
                if (n == 0) {
                    first = 0;
                    continue;
                }
                for (j = 1; j < n; j++) {
                    if (seg[j] != '0') {
                        rest_nonzero = 1;
                    }
                }
                if (seg[0] == '0' || (n != 2 && rest_nonzero)) {
                    for (j = 0; j < n; j++) {
                        if (!nlg_is_digit_cp(seg[j])) {
                            return NANO_LEX_E_NUMBER;
                        }
                        rc = nlg_extend_num_value((int64_t)(seg[j] - '0'), num_flags);
                        if (rc != NANO_LEX_OK) {
                            return rc;
                        }
                    }
                } else {
                    int64_t v;
                    rc = nlg_parse_int(seg, n, &v);
                    if (rc != NANO_LEX_OK) {
                        return rc;
                    }
                    rc = nlg_extend_num_value(v, num_flags);
                    if (rc != NANO_LEX_OK) {
                        return rc;
                    }
                }
                first = 0;
                (void)first;
            }
        }
    } else if (currency != 0 && nlg_is_currency(wp->cp, wp->len)) {
        int64_t amounts[2] = { 0, 0 };
        int npairs = 0;
        uint16_t seg_start = 0;
        for (k = 0; k <= clen && npairs < 2; k++) {
            if (k == clen || clean[k] == '.') {
                uint16_t n = (uint16_t)(k - seg_start);
                int64_t v = 0;
                if (n > 0) {
                    rc = nlg_parse_int(clean + seg_start, n, &v);
                    if (rc != NANO_LEX_OK) {
                        return rc;
                    }
                }
                amounts[npairs++] = v;
                seg_start = (uint16_t)(k + 1);
            }
        }
        {
            int first_unit = 0;
            int last_unit = npairs - 1;
            if (npairs > 1) {
                if (amounts[1] == 0) {
                    last_unit = 0;
                } else if (amounts[0] == 0) {
                    first_unit = 1;
                }
            }
            for (i = first_unit; i <= last_unit; i++) {
                const char *unit = NLG_CURRENCY_UNIT[currency][i];
                if (i > first_unit) {
                    rc = nlg_num_lookup_word("and", 3, NLG_S2_NONE);
                    if (rc != NANO_LEX_OK) {
                        return rc;
                    }
                }
                rc = nlg_extend_num_value(amounts[i], num_flags);
                if (rc != NANO_LEX_OK) {
                    return rc;
                }
                if (amounts[i] != 1 && amounts[i] != -1 &&
                    !(nlg_slen(unit) == 5 && nlg_cmp(unit, "pence", 5) == 0)) {
                    nlg_word_t *plural = &g_ws.w_tmp;
                    nlg_ps_t sps;
                    int8_t srating;
                    nlg_word_from_ascii(plural, unit);
                    (void)nlg_word_push(plural, 's');
                    rc = nlg_stem_s(plural, NLG_TAG_NONE, NLG_S2_NONE, 0, &sps, &srating);
                    if (rc != NANO_LEX_OK) {
                        return rc;
                    }
                    rc = nlg_num_push(&sps, srating);
                } else {
                    rc = nlg_num_lookup_word(unit, nlg_slen(unit), NLG_S2_NONE);
                }
                if (rc != NANO_LEX_OK) {
                    return rc;
                }
            }
        }
    } else {
        nlg_txt_t t;
        nlg_txt_begin(&t, g_ws.num_text, sizeof g_ws.num_text);
        if (nlg_all_digits(clean, clen)) {
            int64_t v;
            rc = nlg_parse_int(clean, clen, &v);
            if (rc != NANO_LEX_OK) {
                return rc;
            }
            if (is_ordinal_suffix) {
                nlg_ordinal(&t, v);
            } else {
                nlg_cardinal(&t, v);
            }
        } else if (!has_dot) {
            return NANO_LEX_E_NUMBER;
        } else if (clen > 0 && clean[0] == '.') {
            nlg_txt_puts(&t, "point");
            for (k = 1; k < clen; k++) {
                if (!nlg_is_digit_cp(clean[k])) {
                    return NANO_LEX_E_NUMBER;
                }
                nlg_txt_puts(&t, " ");
                nlg_txt_puts(&t, NLG_ONES[clean[k] - '0']);
            }
        } else {
            nlg_decimal(&t, clean, clen);
        }
        if (t.rc != NANO_LEX_OK) {
            return t.rc;
        }
        rc = nlg_extend_num_text(g_ws.num_text, t.len, num_flags);
    }
    if (rc != NANO_LEX_OK) {
        return rc;
    }

    if (g_ws.num_count == 0) {
        nlg_arena_rewind(mark);
        return NANO_LEX_OK;
    }
    {
        nlg_sb_t sb;
        int wrote = 0;
        int8_t best = 127;
        int any_rating = 0;
        nlg_ps_t joined;
        nlg_sb_begin(&sb);
        for (i = 0; i < g_ws.num_count; i++) {
            const nlg_ps_t *part = &g_ws.num_parts[i];
            if (g_ws.num_ratings[i] >= 0) {
                any_rating = 1;
                if (g_ws.num_ratings[i] < best) {
                    best = g_ws.num_ratings[i];
                }
            }
            if (!part->present || part->len == 0) {
                continue;
            }
            if (wrote) {
                nlg_sb_push(&sb, NLG_PH_0020);
            }
            nlg_sb_push_span(&sb, part->off, part->len);
            wrote = 1;
        }
        rc = nlg_sb_finish(&sb, &joined);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (is_s_suffix) {
            rc = nlg_suffix_s(&joined);
        } else if (is_ed_suffix) {
            rc = nlg_suffix_ed(&joined);
        } else if (is_ing_suffix) {
            rc = nlg_suffix_ing(&joined);
        }
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        nlg_arena_keep(mark, &joined);
        *ps = joined;
        *rating = any_rating ? best : (int8_t)-1;
    }
    return NANO_LEX_OK;
}

/* misaki/en.py: Lexicon.append_currency */
static int nlg_append_currency(nlg_ps_t *ps, uint8_t currency)
{
    nlg_word_t *plural = &g_ws.w_tmp;
    nlg_ps_t unit;
    int8_t rating;
    int rc;
    uint16_t mark;
    if (currency == 0 || !ps->present) {
        return NANO_LEX_OK;
    }
    mark = nlg_arena_mark();
    nlg_word_from_ascii(plural, NLG_CURRENCY_UNIT[currency][0]);
    (void)nlg_word_push(plural, 's');
    rc = nlg_stem_s(plural, NLG_TAG_NONE, NLG_S2_NONE, 0, &unit, &rating);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    if (!unit.present) {
        nlg_arena_rewind(mark);
        return NANO_LEX_OK;
    }
    {
        nlg_sb_t sb;
        nlg_ps_t out;
        nlg_sb_begin(&sb);
        nlg_sb_push_ps(&sb, ps);
        nlg_sb_push(&sb, NLG_PH_0020);
        nlg_sb_push_ps(&sb, &unit);
        rc = nlg_sb_finish(&sb, &out);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        *ps = out;
    }
    return NANO_LEX_OK;
}

/* --------------------------------------------------------------------------
 * get_special_case / get_word / Lexicon.__call__
 * -------------------------------------------------------------------------- */

static int nlg_lookup_ascii(const char *word, uint8_t tag, int8_t stress2,
                            const nlg_ctx_t *ctx, nlg_ps_t *ps, int8_t *rating)
{
    nlg_word_t *w = &g_ws.w_ascii;
    nlg_word_from_ascii(w, word);
    return nlg_lookup(w, tag, stress2, ctx, ps, rating);
}

/* "." in word.strip(".") and word.replace(".","").isalpha()
 *   and len(max(word.split("."), key=len)) < 3 */
static int nlg_is_dotted_initialism(const nlg_word_t *w)
{
    uint16_t i;
    uint16_t s = 0;
    uint16_t e = w->len;
    int inner_dot = 0;
    uint16_t longest = 0;
    uint16_t run = 0;
    int any = 0;

    while (s < e && w->cp[s] == '.') {
        s++;
    }
    while (e > s && w->cp[e - 1] == '.') {
        e--;
    }
    for (i = s; i < e; i++) {
        if (w->cp[i] == '.') {
            inner_dot = 1;
        }
    }
    if (!inner_dot) {
        return 0;
    }
    for (i = 0; i < w->len; i++) {
        if (w->cp[i] == '.') {
            continue;
        }
        any = 1;
        if (!nlg_is_letter(w->cp[i])) {
            return 0;
        }
    }
    if (!any) {
        return 0;
    }
    for (i = 0; i <= w->len; i++) {
        if (i == w->len || w->cp[i] == '.') {
            if (run > longest) {
                longest = run;
            }
            run = 0;
        } else {
            run++;
        }
    }
    return longest < 3;
}

/* misaki/en.py: Lexicon.get_special_case. The `tag == "ADD"` branch is left
 * out deliberately: nano_g2p._tag never returns spaCy's ADD tag. */
static int nlg_get_special_case(const nlg_word_t *w, uint8_t tag, int8_t stress2,
                                const nlg_ctx_t *ctx, nlg_ps_t *ps, int8_t *rating,
                                int *hit)
{
    const char *symbol;
    int rc;
    nlg_sb_t sb;

    *hit = 0;
    *ps = NLG_PS_NONE;
    *rating = -1;

    symbol = nlg_symbol_word(w);
    if (symbol != 0) {
        *hit = 1;
        return nlg_lookup_ascii(symbol, NLG_TAG_NONE, NLG_S2_NONE, ctx, ps, rating);
    }
    if (nlg_is_dotted_initialism(w)) {
        *hit = 1;
        return nlg_get_NNP(w, ps, rating);
    }
    if (nlg_word_eq_ascii(w, "a") || nlg_word_eq_ascii(w, "A")) {
        nlg_sb_begin(&sb);
        if (tag == NLG_TAG_DT) {
            nlg_sb_push(&sb, NLG_PH_0250);
        } else {
            nlg_sb_push(&sb, NLG_PRIMARY);
            nlg_sb_push(&sb, NLG_PH_0041);
        }
        *hit = 1;
        *rating = 4;
        return nlg_sb_finish(&sb, ps);
    }
    if (nlg_word_eq_ascii(w, "am") || nlg_word_eq_ascii(w, "Am") ||
        nlg_word_eq_ascii(w, "AM")) {
        *hit = 1;
        if (tag == NLG_TAG_NN || tag == NLG_TAG_NNP || tag == NLG_TAG_NNS) {
            return nlg_get_NNP(w, ps, rating);
        }
        if (!ctx || ctx->future_vowel < 0 || !nlg_word_eq_ascii(w, "am") ||
            (stress2 != NLG_S2_NONE && stress2 > 0)) {
            nlg_word_t *am = &g_ws.w_tmp;
            nlg_entry_t entry;
            int found;
            nlg_word_from_ascii(am, "am");
            rc = nlg_grown_find(&NANO_LEX_GOLD, am, &entry, &found);
            if (rc != NANO_LEX_OK) {
                return rc;
            }
            if (!found) {
                return NANO_LEX_E_TABLE;
            }
            *rating = 4;
            return nlg_entry_value(&entry, NLG_TAG_NONE, 0, -1, ps);
        }
        nlg_sb_begin(&sb);
        nlg_sb_push(&sb, NLG_PH_0250);
        nlg_sb_push(&sb, NLG_PH_006D);
        *rating = 4;
        return nlg_sb_finish(&sb, ps);
    }
    if (nlg_word_eq_ascii(w, "an") || nlg_word_eq_ascii(w, "An") ||
        nlg_word_eq_ascii(w, "AN")) {
        *hit = 1;
        if (nlg_word_eq_ascii(w, "AN") &&
            (tag == NLG_TAG_NN || tag == NLG_TAG_NNP || tag == NLG_TAG_NNS)) {
            return nlg_get_NNP(w, ps, rating);
        }
        nlg_sb_begin(&sb);
        nlg_sb_push(&sb, NLG_PH_0250);
        nlg_sb_push(&sb, NLG_PH_006E);
        *rating = 4;
        return nlg_sb_finish(&sb, ps);
    }
    if (nlg_word_eq_ascii(w, "I") && tag == NLG_TAG_PRP) {
        nlg_sb_begin(&sb);
        nlg_sb_push(&sb, NLG_SECONDARY);
        nlg_sb_push(&sb, NLG_PH_0049);
        *hit = 1;
        *rating = 4;
        return nlg_sb_finish(&sb, ps);
    }
    if ((nlg_word_eq_ascii(w, "by") || nlg_word_eq_ascii(w, "By") ||
         nlg_word_eq_ascii(w, "BY")) && NANO_LEX_TAG_PARENT[tag] == NLG_TAG_ADV) {
        nlg_sb_begin(&sb);
        nlg_sb_push(&sb, NLG_PH_0062);
        nlg_sb_push(&sb, NLG_PRIMARY);
        nlg_sb_push(&sb, NLG_PH_0049);
        *hit = 1;
        *rating = 4;
        return nlg_sb_finish(&sb, ps);
    }
    if (nlg_word_eq_ascii(w, "to") || nlg_word_eq_ascii(w, "To") ||
        (nlg_word_eq_ascii(w, "TO") && (tag == NLG_TAG_TO || tag == NLG_TAG_IN))) {
        *hit = 1;
        *rating = 4;
        if (!ctx || ctx->future_vowel < 0) {
            nlg_word_t *to = &g_ws.w_tmp;
            nlg_entry_t entry;
            int found;
            nlg_word_from_ascii(to, "to");
            rc = nlg_grown_find(&NANO_LEX_GOLD, to, &entry, &found);
            if (rc != NANO_LEX_OK) {
                return rc;
            }
            if (!found) {
                return NANO_LEX_E_TABLE;
            }
            return nlg_entry_value(&entry, NLG_TAG_NONE, 0, -1, ps);
        }
        nlg_sb_begin(&sb);
        nlg_sb_push(&sb, NLG_PH_0074);
        nlg_sb_push(&sb, ctx->future_vowel == 1 ? (uint8_t)NLG_PH_028A
                                                : (uint8_t)NLG_PH_0259);
        return nlg_sb_finish(&sb, ps);
    }
    if (nlg_word_eq_ascii(w, "in") || nlg_word_eq_ascii(w, "In") ||
        (nlg_word_eq_ascii(w, "IN") && tag != NLG_TAG_NNP)) {
        nlg_sb_begin(&sb);
        if (!ctx || ctx->future_vowel < 0 || tag != NLG_TAG_IN) {
            nlg_sb_push(&sb, NLG_PRIMARY);
        }
        nlg_sb_push(&sb, NLG_PH_026A);
        nlg_sb_push(&sb, NLG_PH_006E);
        *hit = 1;
        *rating = 4;
        return nlg_sb_finish(&sb, ps);
    }
    if (nlg_word_eq_ascii(w, "the") || nlg_word_eq_ascii(w, "The") ||
        (nlg_word_eq_ascii(w, "THE") && tag == NLG_TAG_DT)) {
        nlg_sb_begin(&sb);
        nlg_sb_push(&sb, NLG_PH_00F0);
        nlg_sb_push(&sb, (ctx && ctx->future_vowel == 1) ? (uint8_t)NLG_PH_0069
                                                         : (uint8_t)NLG_PH_0259);
        *hit = 1;
        *rating = 4;
        return nlg_sb_finish(&sb, ps);
    }
    if (tag == NLG_TAG_IN) {
        int is_vs = 0;
        if (w->len == 2 || (w->len == 3 && w->cp[2] == '.')) {
            is_vs = nlg_lower(w->cp[0]) == 'v' && nlg_lower(w->cp[1]) == 's';
        }
        if (is_vs) {
            *hit = 1;
            return nlg_lookup_ascii("versus", NLG_TAG_NONE, NLG_S2_NONE, ctx, ps, rating);
        }
    }
    if (nlg_word_eq_ascii(w, "used") || nlg_word_eq_ascii(w, "Used") ||
        nlg_word_eq_ascii(w, "USED")) {
        nlg_word_t *used = &g_ws.w_tmp;
        nlg_entry_t entry;
        int found;
        uint8_t want = NLG_TAG_DEFAULT;
        if ((tag == NLG_TAG_VBD || tag == NLG_TAG_JJ) && ctx && ctx->future_to) {
            want = NLG_TAG_VBD;
        }
        nlg_word_from_ascii(used, "used");
        rc = nlg_grown_find(&NANO_LEX_GOLD, used, &entry, &found);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (!found) {
            return NANO_LEX_E_TABLE;
        }
        *hit = 1;
        *rating = 4;
        return nlg_entry_value(&entry, want, 0, -1, ps);
    }
    return NANO_LEX_OK;
}

/* misaki/en.py: Lexicon.get_word */
static int nlg_get_word(const nlg_word_t *word_in, uint8_t tag, int8_t stress2,
                        const nlg_ctx_t *ctx, nlg_ps_t *ps, int8_t *rating)
{
    const nlg_word_t *wp = word_in;
    nlg_word_t *lowered = &g_ws.w_lowered;
    int hit = 0;
    int known;
    int rc;
    uint16_t mark = nlg_arena_mark();

    rc = nlg_get_special_case(wp, tag, stress2, ctx, ps, rating, &hit);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    if (hit && ps->present) {
        nlg_arena_keep(mark, ps);
        return NANO_LEX_OK;
    }
    nlg_arena_rewind(mark);
    *ps = NLG_PS_NONE;
    *rating = -1;

    *lowered = *word_in;
    nlg_word_to_lower(lowered);

    if (wp->len > 1 && !nlg_word_same(wp, lowered) &&
        (tag != NLG_TAG_NNP || wp->len > 7)) {
        uint16_t i;
        int stripped_alpha = 0;   /* word.replace("'", "").isalpha() */
        for (i = 0; i < wp->len; i++) {
            if (wp->cp[i] == '\'') {
                continue;
            }
            if (!nlg_is_letter(wp->cp[i])) {
                stripped_alpha = 0;
                break;
            }
            stripped_alpha = 1;
        }
        if (stripped_alpha) {
            int tail_lower = 1;
            for (i = 1; i < wp->len; i++) {
                if (nlg_lower(wp->cp[i]) != wp->cp[i]) {
                    tail_lower = 0;
                    break;
                }
            }
            if (nlg_word_is_upper(wp) || tail_lower) {
                int in_gold;
                int in_silver;
                rc = nlg_in_gold(wp, &in_gold);
                if (rc != NANO_LEX_OK) {
                    return rc;
                }
                rc = nlg_in_silver(wp, &in_silver);
                if (rc != NANO_LEX_OK) {
                    return rc;
                }
                if (!in_gold && !in_silver) {
                    int lower_known = 0;
                    rc = nlg_in_gold(lowered, &lower_known);
                    if (rc != NANO_LEX_OK) {
                        return rc;
                    }
                    if (!lower_known) {
                        rc = nlg_in_silver(lowered, &lower_known);
                        if (rc != NANO_LEX_OK) {
                            return rc;
                        }
                    }
                    if (!lower_known) {
                        nlg_ps_t probe;
                        int8_t probe_rating;
                        uint16_t probe_mark = nlg_arena_mark();
                        rc = nlg_stem_s(lowered, tag, stress2, ctx, &probe, &probe_rating);
                        if (rc != NANO_LEX_OK) {
                            return rc;
                        }
                        lower_known = probe.present;
                        if (!lower_known) {
                            rc = nlg_stem_ed(lowered, tag, stress2, ctx, &probe,
                                             &probe_rating);
                            if (rc != NANO_LEX_OK) {
                                return rc;
                            }
                            lower_known = probe.present;
                        }
                        if (!lower_known) {
                            rc = nlg_stem_ing(lowered, tag, stress2, ctx, &probe,
                                              &probe_rating);
                            if (rc != NANO_LEX_OK) {
                                return rc;
                            }
                            lower_known = probe.present;
                        }
                        nlg_arena_rewind(probe_mark);
                    }
                    if (lower_known) {
                        wp = lowered;
                    }
                }
            }
        }
    }

    rc = nlg_is_known(wp, &known);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    if (known) {
        rc = nlg_lookup(wp, tag, stress2, ctx, ps, rating);
        if (rc == NANO_LEX_OK) {
            nlg_arena_keep(mark, ps);
        }
        return rc;
    }
    if (nlg_word_ends_ascii(wp, "s'")) {
        nlg_word_t *alt = &g_ws.w_alt;
        *alt = *wp;
        alt->len = (uint16_t)(wp->len - 2);
        (void)nlg_word_push(alt, '\'');
        (void)nlg_word_push(alt, 's');
        rc = nlg_is_known(alt, &known);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (known) {
            rc = nlg_lookup(alt, tag, stress2, ctx, ps, rating);
            if (rc == NANO_LEX_OK) {
                nlg_arena_keep(mark, ps);
            }
            return rc;
        }
    }
    if (nlg_word_ends_ascii(wp, "'")) {
        nlg_word_t *alt = &g_ws.w_alt;
        *alt = *wp;
        alt->len = (uint16_t)(wp->len - 1);
        rc = nlg_is_known(alt, &known);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (known) {
            rc = nlg_lookup(alt, tag, stress2, ctx, ps, rating);
            if (rc == NANO_LEX_OK) {
                nlg_arena_keep(mark, ps);
            }
            return rc;
        }
    }
    rc = nlg_stem_s(wp, tag, stress2, ctx, ps, rating);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    if (ps->present) {
        nlg_arena_keep(mark, ps);
        return NANO_LEX_OK;
    }
    nlg_arena_rewind(mark);
    rc = nlg_stem_ed(wp, tag, stress2, ctx, ps, rating);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    if (ps->present) {
        nlg_arena_keep(mark, ps);
        return NANO_LEX_OK;
    }
    nlg_arena_rewind(mark);
    rc = nlg_stem_ing(wp, tag, (int8_t)(stress2 == NLG_S2_NONE ? 1 : stress2),
                      ctx, ps, rating);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    if (ps->present) {
        nlg_arena_keep(mark, ps);
        return NANO_LEX_OK;
    }
    nlg_arena_rewind(mark);
    *ps = NLG_PS_NONE;
    *rating = -1;
    return NANO_LEX_OK;
}

/* misaki/en.py: Lexicon.__call__ */
static int nlg_lexicon_call(const nlg_token_t *token, const nlg_ctx_t *ctx,
                            const uint16_t *text_cp, uint16_t text_len,
                            nlg_ps_t *ps, int8_t *rating)
{
    nlg_word_t *wp = &g_ws.w_token;
    int8_t stress2;
    int rc;
    uint16_t i;
    uint16_t mark = nlg_arena_mark();

    *ps = NLG_PS_NONE;
    *rating = -1;

    if (token->alias_to) {
        nlg_word_from_ascii(wp, "to");
    } else {
        nlg_word_clear(wp);
        for (i = 0; i < text_len; i++) {
            uint16_t c = text_cp[i];
            /* DEVIATION: Lexicon.__call__ normalises the curly apostrophes,
             * then runs NFKC and numeric_if_needed. NFKC is not implemented;
             * it is the identity on every ASCII string, which is every string
             * that can reach the dictionaries, and numeric_if_needed only ever
             * rewrites non-ASCII digits. */
            if (c == 0x2018u || c == 0x2019u) {
                c = '\'';
            }
            if (!nlg_word_push(wp, c)) {
                /* DEVIATION: a token longer than NLG_MAX_WORD codepoints is
                 * treated as out-of-vocabulary rather than growing the
                 * workspace. No dictionary key is longer than 45. */
                return NANO_LEX_OK;
            }
        }
    }

    if (nlg_word_is_lower(wp)) {
        stress2 = NLG_S2_NONE;
    } else if (nlg_word_is_upper(wp)) {
        stress2 = 4;    /* cap_stresses[1] == 2 */
    } else {
        stress2 = 1;    /* cap_stresses[0] == 0.5 */
    }

    rc = nlg_get_word(wp, token->tag, stress2, ctx, ps, rating);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    if (ps->present) {
        rc = nlg_append_currency(ps, token->currency);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        rc = nlg_apply_stress(ps, token->stress2);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        nlg_arena_keep(mark, ps);
        return NANO_LEX_OK;
    }
    nlg_arena_rewind(mark);
    if (nlg_is_number(wp->cp, wp->len, token->is_head)) {
        rc = nlg_get_number(wp, token->currency, token->is_head,
                            token->num_flags, ps, rating);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        rc = nlg_apply_stress(ps, token->stress2);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        nlg_arena_keep(mark, ps);
        return NANO_LEX_OK;
    }
    nlg_arena_rewind(mark);
    *ps = NLG_PS_NONE;
    *rating = -1;
    return NANO_LEX_OK;
}

/* --------------------------------------------------------------------------
 * Tokeniser -- nano_g2p.py's stand-in for spaCy
 * -------------------------------------------------------------------------- */

static int nlg_cp_in(uint32_t cp, const uint32_t *table, int count)
{
    int lo = 0;
    int hi = count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (table[mid] == cp) {
            return 1;
        }
        if (table[mid] < cp) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return 0;
}

static uint8_t nlg_punct_tag_by_char(uint32_t cp)
{
    int lo = 0;
    int hi = NANO_LEX_PUNCT_TAG_COUNT - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (NANO_LEX_PUNCT_TAG[mid].cp == cp) {
            return NANO_LEX_PUNCT_TAG[mid].tag;
        }
        if (NANO_LEX_PUNCT_TAG[mid].cp < cp) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return 0xFFu;
}

#define NLG_POS_PREFIX 0
#define NLG_POS_CORE   1
#define NLG_POS_SUFFIX 2

/* nano_g2p._punct_tag; 0xFF means "not punctuation". */
static uint8_t nlg_punct_tag(const uint16_t *cp, uint16_t len, int position, int quote_open)
{
    uint16_t i;
    int all_quote = len > 0;
    int all_apostrophe = len > 0;
    uint8_t only = 0xFFu;
    int mixed = 0;

    for (i = 0; i < len; i++) {
        if (nlg_is_alnum(cp[i])) {
            return 0xFFu;
        }
    }
    if (len == 1 && (cp[0] == '%' || cp[0] == '&' || cp[0] == '+' || cp[0] == '@')) {
        return NLG_TAG_NN;
    }
    for (i = 0; i < len; i++) {
        if (!(cp[i] == '"' || cp[i] == 0x201Cu || cp[i] == 0x201Du)) {
            all_quote = 0;
        }
        if (cp[i] != '\'') {
            all_apostrophe = 0;
        }
    }
    if (all_quote) {
        if (len == 1 && cp[0] == 0x201Cu) {
            return NLG_TAG_OPENQ;
        }
        if (len == 1 && cp[0] == 0x201Du) {
            return NLG_TAG_CLOSEQ;
        }
        if (position == NLG_POS_PREFIX) {
            return NLG_TAG_OPENQ;
        }
        if (position == NLG_POS_SUFFIX) {
            return NLG_TAG_CLOSEQ;
        }
        return quote_open ? NLG_TAG_OPENQ : NLG_TAG_CLOSEQ;
    }
    if (all_apostrophe) {
        return NLG_TAG_CLOSEQ;
    }
    for (i = 0; i < len; i++) {
        uint8_t tag = nlg_punct_tag_by_char(cp[i]);
        if (i == 0) {
            only = tag;
        } else if (tag != only) {
            mixed = 1;
        }
    }
    if (!mixed && only != 0xFFu) {
        return only;
    }
    return NLG_TAG_NFP;
}

static int nlg_ascii_lower_buf(const uint16_t *cp, uint16_t len, char *buf)
{
    uint16_t i;
    if (len == 0 || len > NANO_LEX_MAX_KEY) {
        return 0;
    }
    for (i = 0; i < len; i++) {
        uint16_t c = nlg_lower(cp[i]);
        if (c > 0x7Fu) {
            return 0;
        }
        buf[i] = (char)c;
    }
    return 1;
}

/* nano_g2p._tag */
static uint8_t nlg_tag_for(const uint16_t *cp, uint16_t len, int position,
                           int sentence_start, int quote_open)
{
    uint8_t punct = nlg_punct_tag(cp, len, position, quote_open);
    char buf[NANO_LEX_MAX_KEY];

    if (punct != 0xFFu) {
        return punct;
    }
    if (nlg_is_number(cp, len, 1)) {
        return NLG_TAG_CD;
    }
    if (len == 1 && cp[0] == 'I') {
        return NLG_TAG_PRP;
    }
    if (nlg_ascii_lower_buf(cp, len, buf)) {
        int lo = 0;
        int hi = NANO_LEX_CLOSED_COUNT - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            size_t n = nlg_slen(NANO_LEX_CLOSED[mid].word);
            size_t m = (n < len) ? n : len;
            int cmp = nlg_cmp(NANO_LEX_CLOSED[mid].word, buf, m);
            if (cmp == 0) {
                cmp = (n == len) ? 0 : ((n < len) ? -1 : 1);
            }
            if (cmp == 0) {
                return NANO_LEX_CLOSED[mid].tag;
            }
            if (cmp < 0) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
    }
    if (len > 0 && nlg_is_upper(cp[0]) && !sentence_start) {
        return NLG_TAG_NNP;
    }
    return NLG_TAG_NN;
}

/* nano_g2p._keeps_final_period */
static int nlg_keeps_final_period(const uint16_t *cp, uint16_t len)
{
    char buf[NANO_LEX_MAX_KEY];
    nlg_word_t *w = &g_ws.w_tmp;
    uint16_t i;

    if (len == 0 || cp[len - 1] != '.') {
        return 0;
    }
    if (nlg_ascii_lower_buf(cp, len, buf) &&
        nlg_bsearch_str(NANO_LEX_ABBREV, NANO_LEX_ABBREV_COUNT, buf, len) >= 0) {
        return 1;
    }
    nlg_word_clear(w);
    for (i = 0; i < len; i++) {
        if (!nlg_word_push(w, cp[i])) {
            return 0;
        }
    }
    return nlg_is_dotted_initialism(w);
}

static int nlg_push_src(uint16_t off, uint16_t len, uint8_t tag, int ws)
{
    nlg_src_t *s;
    if (g_ws.src_len >= NANO_LEX_MAX_TOKEN) {
        return NANO_LEX_E_TOKENS;
    }
    s = &g_ws.src[g_ws.src_len++];
    s->text_off = off;
    s->text_len = len;
    s->tag = tag;
    s->ws = (uint8_t)(ws ? 1 : 0);
    return NANO_LEX_OK;
}

/* nano_g2p._split_chunk fused with the body of tokenize(). */
static int nlg_split_chunk(uint16_t off, uint16_t len, int *sentence_start, int *quote_open)
{
    uint16_t prefix[16];
    uint16_t nprefix = 0;
    uint16_t suffix[16];
    uint16_t nsuffix = 0;
    uint16_t *cores_off = g_ws.piece_off;   /* retokenize() has not run yet */
    uint16_t *cores_len = g_ws.piece_len;
    uint16_t ncores = 0;
    uint16_t core_off = off;
    uint16_t core_len = len;
    int changed = 1;
    uint16_t emitted = 0;
    uint16_t total;
    uint16_t i;
    int rc;

    if (len == 0) {
        return NANO_LEX_OK;
    }
    while (changed && core_len > 1) {
        changed = 0;
        if (nlg_cp_in(g_ws.cp[core_off], NANO_LEX_PREFIX_CP, NANO_LEX_PREFIX_COUNT)) {
            if (nprefix >= 16) {
                return NANO_LEX_E_TOKENS;
            }
            prefix[nprefix++] = core_off;
            core_off++;
            core_len--;
            changed = 1;
            continue;
        }
        if (nlg_cp_in(g_ws.cp[core_off + core_len - 1], NANO_LEX_SUFFIX_CP,
                      NANO_LEX_SUFFIX_COUNT) &&
            !nlg_keeps_final_period(g_ws.cp + core_off, core_len)) {
            if (nsuffix >= 16) {
                return NANO_LEX_E_TOKENS;
            }
            for (i = nsuffix; i > 0; i--) {
                suffix[i] = suffix[i - 1];
            }
            suffix[0] = (uint16_t)(core_off + core_len - 1);
            nsuffix++;
            core_len--;
            changed = 1;
        }
    }

    if (core_len > 0) {
        uint16_t start = 0;
        i = 0;
        while (i < core_len) {
            if (g_ws.cp[core_off + i] == '-' && i > 0 &&
                (nlg_is_alnum(g_ws.cp[core_off + i - 1]) ||
                 g_ws.cp[core_off + i - 1] == '_')) {
                uint16_t run = 0;
                while (i + run < core_len && g_ws.cp[core_off + i + run] == '-') {
                    run++;
                }
                if (run >= 2 && i + run < core_len &&
                    (nlg_is_alnum(g_ws.cp[core_off + i + run]) ||
                     g_ws.cp[core_off + i + run] == '_')) {
                    if (ncores + 2 > NLG_MAX_PIECES) {
                        return NANO_LEX_E_TOKENS;
                    }
                    if (i > start) {
                        cores_off[ncores] = (uint16_t)(core_off + start);
                        cores_len[ncores] = (uint16_t)(i - start);
                        ncores++;
                    }
                    cores_off[ncores] = (uint16_t)(core_off + i);
                    cores_len[ncores] = run;
                    ncores++;
                    i = (uint16_t)(i + run);
                    start = i;
                    continue;
                }
                i = (uint16_t)(i + run);
                continue;
            }
            i++;
        }
        if (core_len > start) {
            if (ncores >= NLG_MAX_PIECES) {
                return NANO_LEX_E_TOKENS;
            }
            cores_off[ncores] = (uint16_t)(core_off + start);
            cores_len[ncores] = (uint16_t)(core_len - start);
            ncores++;
        }
    }

    total = (uint16_t)(nprefix + ncores + nsuffix);
    if (total == 0) {
        return NANO_LEX_OK;
    }
    for (i = 0; i < total; i++) {
        uint16_t poff;
        uint16_t plen;
        int position;
        uint8_t tag;
        if (i < nprefix) {
            poff = prefix[i];
            plen = 1;
            position = NLG_POS_PREFIX;
        } else if (i < (uint16_t)(nprefix + ncores)) {
            poff = cores_off[i - nprefix];
            plen = cores_len[i - nprefix];
            position = NLG_POS_CORE;
        } else {
            poff = suffix[i - nprefix - ncores];
            plen = 1;
            position = NLG_POS_SUFFIX;
        }
        tag = nlg_tag_for(g_ws.cp + poff, plen, position, *sentence_start, *quote_open);
        rc = nlg_push_src(poff, plen, tag, i + 1 == total);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        if (tag == NLG_TAG_OPENQ || tag == NLG_TAG_CLOSEQ) {
            *quote_open = (tag == NLG_TAG_CLOSEQ);
        }
        *sentence_start = (tag == NLG_TAG_PERIOD) ||
                          (*sentence_start && (tag == NLG_TAG_OPENQ || tag == NLG_TAG_LRB));
        emitted++;
    }
    (void)emitted;
    return NANO_LEX_OK;
}

/* nano_g2p._retag_verbs */
static void nlg_retag_verbs(void)
{
    uint8_t previous_tag = 0xFFu;
    int previous_is_aux = 0;
    uint16_t i;
    for (i = 0; i < g_ws.src_len; i++) {
        nlg_src_t *t = &g_ws.src[i];
        if (t->tag == NLG_TAG_NN || t->tag == NLG_TAG_JJ) {
            if (previous_tag == NLG_TAG_TO || previous_tag == NLG_TAG_MD) {
                t->tag = NLG_TAG_VB;
            } else if (previous_is_aux) {
                t->tag = NLG_TAG_VBN;
            }
        }
        if (t->tag != NLG_TAG_RB && t->tag != NLG_TAG_RBR && t->tag != NLG_TAG_RBS) {
            char buf[NANO_LEX_MAX_KEY];
            previous_tag = t->tag;
            previous_is_aux = 0;
            if (nlg_ascii_lower_buf(g_ws.cp + t->text_off, t->text_len, buf)) {
                previous_is_aux = nlg_bsearch_str(NANO_LEX_PERFECT_AUX,
                                                  NANO_LEX_PERFECT_AUX_COUNT,
                                                  buf, t->text_len) >= 0;
            }
        }
    }
}

/* nano_g2p._retag_that */
static void nlg_retag_that(void)
{
    uint16_t i;
    for (i = 0; i < g_ws.src_len; i++) {
        nlg_src_t *t = &g_ws.src[i];
        uint8_t next;
        if (t->tag != NLG_TAG_IN || t->text_len != 4) {
            continue;
        }
        if (nlg_lower(g_ws.cp[t->text_off]) != 't' ||
            nlg_lower(g_ws.cp[t->text_off + 1]) != 'h' ||
            nlg_lower(g_ws.cp[t->text_off + 2]) != 'a' ||
            nlg_lower(g_ws.cp[t->text_off + 3]) != 't') {
            continue;
        }
        if (i + 1 >= g_ws.src_len) {
            continue;
        }
        next = g_ws.src[i + 1].tag;
        if (next == NLG_TAG_NN || next == NLG_TAG_NNP || next == NLG_TAG_NNS ||
            next == NLG_TAG_JJ) {
            t->tag = NLG_TAG_DT;
        }
    }
}

static int nlg_tokenize(void)
{
    uint16_t i = 0;
    int sentence_start = 1;
    int quote_open = 1;
    int rc;

    g_ws.src_len = 0;
    while (i < g_ws.cp_len) {
        uint16_t start;
        while (i < g_ws.cp_len && nlg_is_space(g_ws.cp[i])) {
            i++;
        }
        if (i >= g_ws.cp_len) {
            break;
        }
        start = i;
        while (i < g_ws.cp_len && !nlg_is_space(g_ws.cp[i])) {
            i++;
        }
        rc = nlg_split_chunk(start, (uint16_t)(i - start), &sentence_start, &quote_open);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
    }
    if (g_ws.src_len > 0) {
        g_ws.src[g_ws.src_len - 1].ws = 0;
    }
    nlg_retag_verbs();
    nlg_retag_that();
    return NANO_LEX_OK;
}

/* --------------------------------------------------------------------------
 * subtokenize -- nano_g2p._SUBTOKEN_RE, hand-scanned
 *
 * The alternatives, in the order re.findall tries them at each position:
 *   1  ^['..]+                        leading quotes
 *   2  [A-Z](?=[A-Z][a-z])            the USAToday split
 *   3  (?:^-)?(?:\d?[,.]?\d)+         a number run
 *   4  [-_]+
 *   5  ['..]{2,}
 *   6  L*?(?:['..]L)*?[a-z](?=[A-Z])  the camelCase split, lazily
 *   7  L+(?:['..]L)*                  a word, apostrophes included
 *   8  [^\w'..-]                      one non-word character
 *   9  ['..]+$                        trailing quotes
 * A position matching none of them is skipped, exactly as findall does.
 * -------------------------------------------------------------------------- */

static int nlg_is_quote_cp(uint32_t cp)
{
    return cp == '\'' || cp == 0x2018u || cp == 0x2019u;
}

static int nlg_is_word_cp(uint32_t cp)
{
    return nlg_is_alnum(cp) || cp == '_';
}

static uint16_t nlg_subtoken_at(const uint16_t *cp, uint16_t len, uint16_t i)
{
    uint16_t j;

    if (i == 0 && nlg_is_quote_cp(cp[i])) {
        j = i;
        while (j < len && nlg_is_quote_cp(cp[j])) {
            j++;
        }
        return (uint16_t)(j - i);
    }
    if (nlg_is_upper(cp[i]) && i + 2 < len && nlg_is_upper(cp[i + 1]) &&
        nlg_is_lower(cp[i + 2])) {
        return 1;
    }
    {
        uint16_t p = i;
        uint16_t units = 0;
        if (i == 0 && cp[i] == '-') {
            p = (uint16_t)(i + 1);
        }
        for (;;) {
            if (p < len && nlg_is_digit_cp(cp[p])) {
                if (p + 2 < len && (cp[p + 1] == ',' || cp[p + 1] == '.') &&
                    nlg_is_digit_cp(cp[p + 2])) {
                    p = (uint16_t)(p + 3);
                } else if (p + 1 < len && nlg_is_digit_cp(cp[p + 1])) {
                    p = (uint16_t)(p + 2);
                } else {
                    p = (uint16_t)(p + 1);
                }
                units++;
                continue;
            }
            if (p + 1 < len && (cp[p] == ',' || cp[p] == '.') &&
                nlg_is_digit_cp(cp[p + 1])) {
                p = (uint16_t)(p + 2);
                units++;
                continue;
            }
            break;
        }
        if (units > 0) {
            return (uint16_t)(p - i);
        }
    }
    if (cp[i] == '-' || cp[i] == '_') {
        j = i;
        while (j < len && (cp[j] == '-' || cp[j] == '_')) {
            j++;
        }
        return (uint16_t)(j - i);
    }
    if (nlg_is_quote_cp(cp[i]) && i + 1 < len && nlg_is_quote_cp(cp[i + 1])) {
        j = i;
        while (j < len && nlg_is_quote_cp(cp[j])) {
            j++;
        }
        return (uint16_t)(j - i);
    }
    {
        uint16_t p = i;
        for (;;) {
            if (p < len && nlg_is_lower(cp[p]) && p + 1 < len && nlg_is_upper(cp[p + 1])) {
                return (uint16_t)(p + 1 - i);
            }
            if (p < len && nlg_is_letter(cp[p])) {
                p++;
                continue;
            }
            if (p + 1 < len && nlg_is_quote_cp(cp[p]) && nlg_is_letter(cp[p + 1])) {
                p = (uint16_t)(p + 2);
                continue;
            }
            break;
        }
    }
    if (nlg_is_letter(cp[i])) {
        j = i;
        while (j < len && nlg_is_letter(cp[j])) {
            j++;
        }
        while (j + 1 < len && nlg_is_quote_cp(cp[j]) && nlg_is_letter(cp[j + 1])) {
            j = (uint16_t)(j + 2);
        }
        return (uint16_t)(j - i);
    }
    if (!nlg_is_word_cp(cp[i]) && !nlg_is_quote_cp(cp[i]) && cp[i] != '-') {
        return 1;
    }
    if (nlg_is_quote_cp(cp[i])) {
        j = i;
        while (j < len && nlg_is_quote_cp(cp[j])) {
            j++;
        }
        if (j == len) {
            return (uint16_t)(j - i);
        }
    }
    return 0;
}

/* --------------------------------------------------------------------------
 * retokenize
 * -------------------------------------------------------------------------- */

static int nlg_tag_is_punct(uint8_t tag)
{
    return tag == NLG_TAG_DQUOTE || tag == NLG_TAG_HASH || tag == NLG_TAG_DOLLAR ||
           tag == NLG_TAG_CLOSEQ || tag == NLG_TAG_COMMA || tag == NLG_TAG_LRB ||
           tag == NLG_TAG_RRB || tag == NLG_TAG_PERIOD || tag == NLG_TAG_COLON ||
           tag == NLG_TAG_NFP || tag == NLG_TAG_OPENQ;
}

/* PUNCT_TAG_PHONEMES.get(tag, "".join(c for c in text if c in PUNCTS)) */
static int nlg_punct_tag_phonemes(uint8_t tag, const uint16_t *cp, uint16_t len,
                                  nlg_ps_t *ps)
{
    nlg_sb_t sb;
    uint16_t i;
    nlg_sb_begin(&sb);
    if (tag == NLG_TAG_LRB) {
        nlg_sb_push(&sb, NLG_PH_0028);
    } else if (tag == NLG_TAG_RRB) {
        nlg_sb_push(&sb, NLG_PH_0029);
    } else if (tag == NLG_TAG_OPENQ) {
        nlg_sb_push(&sb, NLG_PH_201C);
    } else if (tag == NLG_TAG_DQUOTE || tag == NLG_TAG_CLOSEQ) {
        nlg_sb_push(&sb, NLG_PH_201D);
    } else {
        for (i = 0; i < len; i++) {
            uint8_t code = nlg_cp_to_ph(cp[i]);
            if (code != 0 && (NANO_LEX_PH_FLAGS[code] & NANO_LEX_F_PUNCT)) {
                nlg_sb_push(&sb, code);
            }
        }
    }
    return nlg_sb_finish(&sb, ps);
}

static int nlg_new_group(uint16_t start, int is_list)
{
    if (g_ws.grp_len >= NANO_LEX_MAX_TOKEN) {
        return NANO_LEX_E_TOKENS;
    }
    g_ws.grp[g_ws.grp_len].start = start;
    g_ws.grp[g_ws.grp_len].count = 1;
    g_ws.grp[g_ws.grp_len].is_list = (uint8_t)(is_list ? 1 : 0);
    g_ws.grp_len++;
    return NANO_LEX_OK;
}

/* misaki/en.py: G2P.retokenize. G2P.fold_left is the identity here, because
 * nano_g2p.tokenize never emits a token with is_head False. */
static int nlg_retokenize(void)
{
    uint16_t i;
    uint8_t currency = 0;
    int rc;

    g_ws.tok_len = 0;
    g_ws.grp_len = 0;

    for (i = 0; i < g_ws.src_len; i++) {
        uint16_t *piece_off = g_ws.piece_off;
        uint16_t *piece_len = g_ws.piece_len;
        uint16_t npieces = 0;
        uint16_t pos = 0;
        uint16_t j;
        const uint16_t *cp = g_ws.cp + g_ws.src[i].text_off;
        uint16_t len = g_ws.src[i].text_len;
        uint16_t first_new;

        while (pos < len) {
            uint16_t n = nlg_subtoken_at(cp, len, pos);
            if (n == 0) {
                pos++;
                continue;
            }
            if (npieces >= NLG_MAX_PIECES) {
                return NANO_LEX_E_TOKENS;
            }
            piece_off[npieces] = (uint16_t)(g_ws.src[i].text_off + pos);
            piece_len[npieces] = n;
            npieces++;
            pos = (uint16_t)(pos + n);
        }
        if (npieces == 0) {
            piece_off[0] = g_ws.src[i].text_off;
            piece_len[0] = len;
            npieces = 1;
        }

        first_new = g_ws.tok_len;
        for (j = 0; j < npieces; j++) {
            nlg_token_t *t;
            if (g_ws.tok_len >= NANO_LEX_MAX_TOKEN) {
                return NANO_LEX_E_TOKENS;
            }
            t = &g_ws.tok[g_ws.tok_len++];
            nlg_zero(t, sizeof *t);
            t->text_off = piece_off[j];
            t->text_len = piece_len[j];
            t->tag = g_ws.src[i].tag;
            t->is_head = 1;
            t->rating = -1;
            t->stress2 = NLG_S2_NONE;
            t->ps = NLG_PS_NONE;
            t->ws = (uint8_t)(j + 1 == npieces ? g_ws.src[i].ws : 0);
        }

        for (j = 0; j < npieces; j++) {
            nlg_token_t *tk = &g_ws.tok[first_new + j];
            const uint16_t *tcp = g_ws.cp + tk->text_off;
            uint16_t tlen = tk->text_len;

            if (tk->tag == NLG_TAG_DOLLAR && tlen == 1 &&
                (tcp[0] == 0x24u || tcp[0] == 0xA3u || tcp[0] == 0x20ACu)) {
                nlg_sb_t sb;
                currency = (uint8_t)(tcp[0] == 0x24u ? 1 : (tcp[0] == 0xA3u ? 2 : 3));
                nlg_sb_begin(&sb);
                rc = nlg_sb_finish(&sb, &tk->ps);
                if (rc != NANO_LEX_OK) {
                    return rc;
                }
                tk->rating = 4;
            } else if (tk->tag == NLG_TAG_COLON && tlen == 1 &&
                       (tcp[0] == '-' || tcp[0] == 0x2013u)) {
                nlg_sb_t sb;
                nlg_sb_begin(&sb);
                nlg_sb_push(&sb, NLG_PH_2014);
                rc = nlg_sb_finish(&sb, &tk->ps);
                if (rc != NANO_LEX_OK) {
                    return rc;
                }
                tk->rating = 3;
            } else if (nlg_tag_is_punct(tk->tag) && !nlg_all_ascii_alpha(tcp, tlen)) {
                rc = nlg_punct_tag_phonemes(tk->tag, tcp, tlen, &tk->ps);
                if (rc != NANO_LEX_OK) {
                    return rc;
                }
                tk->rating = 4;
            } else if (currency != 0) {
                if (tk->tag != NLG_TAG_CD) {
                    currency = 0;
                } else if (j + 1 == npieces &&
                           (i + 1 == g_ws.src_len || g_ws.src[i + 1].tag != NLG_TAG_CD)) {
                    tk->currency = currency;
                }
            } else if (j > 0 && j + 1 < npieces && tlen == 1 && tcp[0] == '2') {
                const nlg_token_t *prev = &g_ws.tok[first_new + j - 1];
                const nlg_token_t *next = &g_ws.tok[first_new + j + 1];
                if (prev->text_len > 0 && next->text_len > 0 &&
                    nlg_is_letter(g_ws.cp[prev->text_off + prev->text_len - 1]) &&
                    nlg_is_letter(g_ws.cp[next->text_off])) {
                    tk->alias_to = 1;
                }
            }

            if (tk->alias_to || tk->ps.present) {
                rc = nlg_new_group((uint16_t)(first_new + j), 0);
            } else if (g_ws.grp_len > 0 && g_ws.grp[g_ws.grp_len - 1].is_list &&
                       !g_ws.tok[g_ws.grp[g_ws.grp_len - 1].start +
                                 g_ws.grp[g_ws.grp_len - 1].count - 1].ws) {
                tk->is_head = 0;
                g_ws.grp[g_ws.grp_len - 1].count++;
                rc = NANO_LEX_OK;
            } else {
                rc = nlg_new_group((uint16_t)(first_new + j), tk->ws ? 0 : 1);
            }
            if (rc != NANO_LEX_OK) {
                return rc;
            }
        }
    }
    return NANO_LEX_OK;
}

/* --------------------------------------------------------------------------
 * The main resolution loop
 * -------------------------------------------------------------------------- */

/* misaki/en.py: merge_tokens, for the fields the lexicon reads. The merged
 * text is assembled in g_ws.merge_cp; only one merged token is live at a
 * time. */
static void nlg_merge_for_lookup(uint16_t start, uint16_t count, nlg_token_t *out,
                                 uint16_t *text_len)
{
    uint16_t i;
    uint16_t k;
    uint16_t n = 0;
    uint16_t best_score = 0;
    uint8_t best_tag = g_ws.tok[start].tag;
    int8_t stress = NLG_S2_NONE;
    int stress_count = 0;
    uint8_t currency = 0;
    uint8_t num_flags = 0;

    for (i = 0; i < count; i++) {
        const nlg_token_t *t = &g_ws.tok[start + i];
        uint16_t score = 0;
        for (k = 0; k < t->text_len; k++) {
            uint16_t c = g_ws.cp[t->text_off + k];
            score = (uint16_t)(score + (nlg_lower(c) == c ? 1u : 2u));
            if (n < NLG_MAX_WORD) {
                g_ws.merge_cp[n++] = c;
            }
        }
        if (i == 0 || score > best_score) {
            best_score = score;
            best_tag = t->tag;
        }
        if (t->stress2 != NLG_S2_NONE) {
            if (stress_count == 0) {
                stress = t->stress2;
                stress_count = 1;
            } else if (stress != t->stress2) {
                stress_count = 2;
            }
        }
        if (t->currency > currency) {
            currency = t->currency;
        }
        num_flags = (uint8_t)(num_flags | t->num_flags);
    }
    nlg_zero(out, sizeof *out);
    out->tag = best_tag;
    out->ws = g_ws.tok[start + count - 1].ws;
    out->is_head = g_ws.tok[start].is_head;
    out->prespace = g_ws.tok[start].prespace;
    out->currency = currency;
    out->num_flags = num_flags;
    out->rating = -1;
    out->stress2 = (stress_count == 1) ? stress : (int8_t)NLG_S2_NONE;
    out->ps = NLG_PS_NONE;
    *text_len = n;
}

/* misaki/en.py: G2P.token_context */
static void nlg_token_context(nlg_ctx_t *ctx, const nlg_ps_t *ps,
                              const uint16_t *text_cp, uint16_t text_len, uint8_t tag)
{
    uint16_t i;
    if (ps->present && ps->len > 0) {
        for (i = 0; i < ps->len; i++) {
            uint8_t flags = NANO_LEX_PH_FLAGS[nlg_ps_at(ps, i)];
            if (flags & NANO_LEX_F_NONQ_PUNCT) {
                ctx->future_vowel = -1;
                break;
            }
            if (flags & NANO_LEX_F_VOWEL) {
                ctx->future_vowel = 1;
                break;
            }
            if (flags & NANO_LEX_F_CONSONANT) {
                ctx->future_vowel = 0;
                break;
            }
        }
    }
    ctx->future_to = 0;
    if (text_len == 2 && text_cp[1] == 'o' && (text_cp[0] == 't' || text_cp[0] == 'T')) {
        ctx->future_to = 1;
    } else if (text_len == 2 && text_cp[0] == 'T' && text_cp[1] == 'O' &&
               (tag == NLG_TAG_TO || tag == NLG_TAG_IN)) {
        ctx->future_to = 1;
    }
}

/* misaki/en.py: G2P.resolve_tokens */
static int nlg_resolve_tokens(uint16_t start, uint16_t count)
{
    int prespace = 0;
    uint16_t i;
    uint16_t k;
    int kinds = 0;          /* bit0 alpha, bit1 digit, bit2 other */
    int rc;

    for (i = 0; i < count; i++) {
        const nlg_token_t *t = &g_ws.tok[start + i];
        for (k = 0; k < t->text_len; k++) {
            uint16_t c = g_ws.cp[t->text_off + k];
            if (c == ' ' || c == '/') {
                prespace = 1;
            }
            if (nlg_is_junk(c)) {
                continue;
            }
            kinds |= nlg_is_letter(c) ? 1 : (nlg_is_digit_cp(c) ? 2 : 4);
        }
    }
    if (!prespace) {
        int distinct = ((kinds & 1) ? 1 : 0) + ((kinds & 2) ? 1 : 0) + ((kinds & 4) ? 1 : 0);
        prespace = distinct > 1;
    }

    for (i = 0; i < count; i++) {
        nlg_token_t *t = &g_ws.tok[start + i];
        if (!t->ps.present) {
            if (i + 1 == count && t->text_len == 1) {
                uint8_t code = nlg_cp_to_ph(g_ws.cp[t->text_off]);
                if (code != 0 && (NANO_LEX_PH_FLAGS[code] & NANO_LEX_F_NONQ_PUNCT)) {
                    nlg_sb_t sb;
                    nlg_sb_begin(&sb);
                    nlg_sb_push(&sb, code);
                    rc = nlg_sb_finish(&sb, &t->ps);
                    if (rc != NANO_LEX_OK) {
                        return rc;
                    }
                    t->rating = 3;
                    continue;
                }
            }
            {
                int all_junk = 1;
                for (k = 0; k < t->text_len; k++) {
                    if (!nlg_is_junk(g_ws.cp[t->text_off + k])) {
                        all_junk = 0;
                        break;
                    }
                }
                if (all_junk) {
                    nlg_sb_t sb;
                    nlg_sb_begin(&sb);
                    rc = nlg_sb_finish(&sb, &t->ps);
                    if (rc != NANO_LEX_OK) {
                        return rc;
                    }
                    t->rating = 3;
                }
            }
        } else if (i > 0) {
            t->prespace = (uint8_t)prespace;
        }
    }
    if (prespace) {
        return NANO_LEX_OK;
    }
    {
        uint16_t *idx = g_ws.stress_idx;
        uint8_t *has_primary = g_ws.stress_primary;
        uint16_t *weight = g_ws.stress_weight;
        uint16_t n = 0;
        uint16_t primaries = 0;
        for (i = 0; i < count && n < NLG_MAX_PIECES; i++) {
            nlg_token_t *t = &g_ws.tok[start + i];
            if (!t->ps.present || t->ps.len == 0) {
                continue;
            }
            idx[n] = i;
            has_primary[n] = (uint8_t)(nlg_ps_has(&t->ps, NLG_PRIMARY) ? 1 : 0);
            weight[n] = nlg_stress_weight(&t->ps);
            primaries = (uint16_t)(primaries + has_primary[n]);
            n++;
        }
        if (n == 2 && g_ws.tok[start + idx[0]].text_len == 1) {
            nlg_token_t *t = &g_ws.tok[start + idx[1]];
            uint16_t mark = nlg_arena_mark();
            rc = nlg_apply_stress(&t->ps, -1);
            if (rc != NANO_LEX_OK) {
                return rc;
            }
            (void)mark;
            return NANO_LEX_OK;
        }
        if (n < 2 || primaries <= (uint16_t)((n + 1) / 2)) {
            return NANO_LEX_OK;
        }
        /* sorted(indices)[:len(indices)//2] -- ascending on
         * (has_primary, stress_weight, position). n is small, so selection
         * sort keeps this allocation-free. */
        {
            uint16_t take = (uint16_t)(n / 2);
            uint8_t *used = g_ws.stress_used;
            uint16_t s;
            for (i = 0; i < n; i++) {
                used[i] = 0;
            }
            for (s = 0; s < take; s++) {
                uint16_t best = 0xFFFFu;
                for (i = 0; i < n; i++) {
                    if (used[i]) {
                        continue;
                    }
                    if (best == 0xFFFFu ||
                        has_primary[i] < has_primary[best] ||
                        (has_primary[i] == has_primary[best] &&
                         (weight[i] < weight[best] ||
                          (weight[i] == weight[best] && idx[i] < idx[best])))) {
                        best = i;
                    }
                }
                if (best == 0xFFFFu) {
                    break;
                }
                used[best] = 1;
                rc = nlg_apply_stress(&g_ws.tok[start + idx[best]].ps, -1);
                if (rc != NANO_LEX_OK) {
                    return rc;
                }
            }
        }
    }
    return NANO_LEX_OK;
}

/* misaki/en.py: G2P.__call__, with fallback = None. */
static int nlg_resolve(void)
{
    nlg_ctx_t ctx;
    int32_t gi;
    int rc;

    ctx.future_vowel = -1;
    ctx.future_to = 0;

    for (gi = (int32_t)g_ws.grp_len - 1; gi >= 0; gi--) {
        uint16_t start = g_ws.grp[gi].start;
        uint16_t count = g_ws.grp[gi].count;

        if (count == 1) {
            nlg_token_t *t = &g_ws.tok[start];
            if (!t->ps.present) {
                nlg_ps_t ps;
                int8_t rating;
                g_ws.stats.lexicon_words++;
                rc = nlg_lexicon_call(t, &ctx, g_ws.cp + t->text_off, t->text_len,
                                      &ps, &rating);
                if (rc != NANO_LEX_OK) {
                    return rc;
                }
                t->ps = ps;
                t->rating = rating;
                if (!ps.present) {
                    g_ws.stats.oov_words++;
                }
            }
            nlg_token_context(&ctx, &t->ps, g_ws.cp + t->text_off, t->text_len, t->tag);
            continue;
        }

        {
            int left = 0;
            int right = (int)count;
            int unresolved = 0;
            uint16_t i;

            g_ws.stats.lexicon_words++;
            while (left < right) {
                int mergeable = 1;
                nlg_ps_t ps;
                int8_t rating = -1;
                for (i = (uint16_t)left; i < (uint16_t)right; i++) {
                    if (g_ws.tok[start + i].alias_to || g_ws.tok[start + i].ps.present) {
                        mergeable = 0;
                        break;
                    }
                }
                ps = NLG_PS_NONE;
                if (mergeable) {
                    nlg_token_t merged;
                    uint16_t mlen;
                    nlg_merge_for_lookup((uint16_t)(start + left),
                                         (uint16_t)(right - left), &merged, &mlen);
                    rc = nlg_lexicon_call(&merged, &ctx, g_ws.merge_cp, mlen, &ps, &rating);
                    if (rc != NANO_LEX_OK) {
                        return rc;
                    }
                    if (ps.present) {
                        nlg_sb_t sb;
                        nlg_ps_t empty;
                        g_ws.tok[start + left].ps = ps;
                        g_ws.tok[start + left].rating = rating;
                        nlg_sb_begin(&sb);
                        rc = nlg_sb_finish(&sb, &empty);
                        if (rc != NANO_LEX_OK) {
                            return rc;
                        }
                        for (i = (uint16_t)(left + 1); i < (uint16_t)right; i++) {
                            g_ws.tok[start + i].ps = empty;
                            g_ws.tok[start + i].rating = rating;
                        }
                        nlg_token_context(&ctx, &ps, g_ws.merge_cp, mlen, merged.tag);
                        right = left;
                        left = 0;
                        continue;
                    }
                }
                if (left + 1 < right) {
                    left++;
                    continue;
                }
                right--;
                {
                    nlg_token_t *tk = &g_ws.tok[start + right];
                    if (!tk->ps.present) {
                        int all_junk = tk->text_len > 0;
                        uint16_t k;
                        for (k = 0; k < tk->text_len; k++) {
                            if (!nlg_is_junk(g_ws.cp[tk->text_off + k])) {
                                all_junk = 0;
                                break;
                            }
                        }
                        if (all_junk) {
                            nlg_sb_t sb;
                            nlg_sb_begin(&sb);
                            rc = nlg_sb_finish(&sb, &tk->ps);
                            if (rc != NANO_LEX_OK) {
                                return rc;
                            }
                            tk->rating = 3;
                        } else {
                            /* No neural fallback here; nano_g2p.py with
                             * fallback=None leaves the token unresolved too. */
                            unresolved = 1;
                        }
                    }
                }
                left = 0;
            }
            if (unresolved) {
                g_ws.stats.oov_words++;
            }
            rc = nlg_resolve_tokens(start, count);
            if (rc != NANO_LEX_OK) {
                return rc;
            }
        }
    }
    return NANO_LEX_OK;
}

/* --------------------------------------------------------------------------
 * Flatten, chunk, emit
 * -------------------------------------------------------------------------- */

/* merge_tokens(word, unk="") for the phoneme string alone. */
static int nlg_flatten(void)
{
    uint16_t gi;
    int rc;
    for (gi = 0; gi < g_ws.grp_len; gi++) {
        uint16_t start = g_ws.grp[gi].start;
        uint16_t count = g_ws.grp[gi].count;
        nlg_ps_t merged;
        uint16_t i;
        nlg_sb_t sb;

        if (count > 1) {
            nlg_sb_begin(&sb);
            for (i = 0; i < count; i++) {
                const nlg_token_t *t = &g_ws.tok[start + i];
                if (t->prespace && sb.len > 0 &&
                    g_ws.arena[sb.off + sb.len - 1] != NLG_PH_0020 &&
                    t->ps.present && t->ps.len > 0) {
                    nlg_sb_push(&sb, NLG_PH_0020);
                }
                nlg_sb_push_ps(&sb, &t->ps);
            }
            rc = nlg_sb_finish(&sb, &merged);
            if (rc != NANO_LEX_OK) {
                return rc;
            }
            g_ws.tok[start].ps = merged;
        } else if (!g_ws.tok[start].ps.present) {
            nlg_sb_begin(&sb);
            rc = nlg_sb_finish(&sb, &g_ws.tok[start].ps);
            if (rc != NANO_LEX_OK) {
                return rc;
            }
        }
        /* G2P.__call__'s tail, applied whenever version != '2.0'. */
        {
            nlg_ps_t *ps = &g_ws.tok[start].ps;
            for (i = 0; i < ps->len; i++) {
                uint8_t code = g_ws.arena[ps->off + i];
                if (code == NLG_PH_027E) {
                    g_ws.arena[ps->off + i] = NLG_PH_0054;
                } else if (code == NLG_PH_0294) {
                    g_ws.arena[ps->off + i] = NLG_PH_0074;
                }
            }
        }
    }
    return NANO_LEX_OK;
}

static const nlg_ps_t *nlg_flat_ps(uint16_t gi)
{
    return &g_ws.tok[g_ws.grp[gi].start].ps;
}

static int nlg_flat_ws(uint16_t gi)
{
    const nlg_group_t *g = &g_ws.grp[gi];
    return g_ws.tok[g->start + g->count - 1].ws != 0;
}

/* kokoro/pipeline.py: KPipeline.tokens_to_ps, over the flat range [a, b). */
static uint16_t nlg_tokens_to_ps_len(uint16_t a, uint16_t b)
{
    uint16_t n = 0;
    uint16_t gi;
    uint16_t lead = 1;
    uint16_t trailing = 0;
    for (gi = a; gi < b; gi++) {
        const nlg_ps_t *ps = nlg_flat_ps(gi);
        uint16_t i;
        for (i = 0; i < ps->len; i++) {
            uint8_t code = g_ws.arena[ps->off + i];
            if (lead && code == NLG_PH_0020) {
                continue;
            }
            lead = 0;
            if (code == NLG_PH_0020) {
                trailing++;
            } else {
                n = (uint16_t)(n + trailing + 1u);
                trailing = 0;
            }
        }
        if (nlg_flat_ws(gi)) {
            if (!lead) {
                trailing++;
            }
        }
    }
    return n;
}

static int nlg_tokens_to_ps(uint16_t a, uint16_t b, uint8_t *out, uint16_t cap,
                            uint16_t *out_len)
{
    uint16_t n = 0;
    uint16_t gi;
    int lead = 1;
    uint16_t trailing = 0;
    for (gi = a; gi < b; gi++) {
        const nlg_ps_t *ps = nlg_flat_ps(gi);
        uint16_t i;
        for (i = 0; i < ps->len; i++) {
            uint8_t code = g_ws.arena[ps->off + i];
            if (lead && code == NLG_PH_0020) {
                continue;
            }
            lead = 0;
            if (code == NLG_PH_0020) {
                trailing++;
            } else {
                while (trailing > 0) {
                    if (n >= cap) {
                        return NANO_LEX_E_ARENA;
                    }
                    out[n++] = NLG_PH_0020;
                    trailing--;
                }
                if (n >= cap) {
                    return NANO_LEX_E_ARENA;
                }
                out[n++] = code;
            }
        }
        if (nlg_flat_ws(gi) && !lead) {
            trailing++;
        }
    }
    *out_len = n;
    return NANO_LEX_OK;
}

/* kokoro/pipeline.py: KPipeline.waterfall_last, over the flat range [a, b). */
static uint16_t nlg_waterfall_last(uint16_t a, uint16_t b, uint32_t next_count)
{
    static const uint8_t groups[3][4] = {
        { NLG_PH_0021, NLG_PH_002E, NLG_PH_003F, NLG_PH_2026 },
        { NLG_PH_003A, NLG_PH_003B, 0, 0 },
        { NLG_PH_002C, NLG_PH_2014, 0, 0 }
    };
    static const uint8_t sizes[3] = { 4, 2, 2 };
    int g;
    for (g = 0; g < 3; g++) {
        int32_t found = -1;
        uint16_t gi;
        for (gi = a; gi < b; gi++) {
            const nlg_ps_t *ps = nlg_flat_ps(gi);
            uint8_t k;
            if (ps->len != 1) {
                continue;
            }
            for (k = 0; k < sizes[g]; k++) {
                if (g_ws.arena[ps->off] == groups[g][k]) {
                    found = (int32_t)gi;
                }
            }
        }
        if (found < 0) {
            continue;
        }
        {
            uint16_t z = (uint16_t)(found + 1 - a);
            if (a + z < b) {
                const nlg_ps_t *ps = nlg_flat_ps((uint16_t)(a + z));
                if (ps->len == 1 && (g_ws.arena[ps->off] == NLG_PH_0029 ||
                                     g_ws.arena[ps->off] == NLG_PH_201D)) {
                    z++;
                }
            }
            if (next_count - nlg_tokens_to_ps_len(a, (uint16_t)(a + z)) <=
                (uint32_t)NANO_LEX_MAX_PS_CHARS) {
                return z;
            }
        }
    }
    return (uint16_t)(b - a);
}

typedef struct {
    int32_t *out;
    int      cap;
    int      n;
    int      overflow;
} nlg_ids_t;

static void nlg_ids_push(nlg_ids_t *ids, int32_t value)
{
    if (ids->n >= ids->cap) {
        ids->overflow = 1;
        return;
    }
    ids->out[ids->n++] = value;
}

static int nlg_emit_chunk(uint16_t a, uint16_t b, nlg_ids_t *ids)
{
    uint16_t len = 0;
    uint16_t i;
    int kept = 0;
    int rc = nlg_tokens_to_ps(a, b, g_ws.chunk, (uint16_t)sizeof g_ws.chunk, &len);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    if (len == 0) {
        return NANO_LEX_OK;    /* `if ps:` in en_chunks */
    }
    if (len > NANO_LEX_MAX_PS_CHARS) {
        len = NANO_LEX_MAX_PS_CHARS;
    }
    g_ws.stats.chunks++;
    nlg_ids_push(ids, NANO_LEX_ID_BOS);
    for (i = 0; i < len; i++) {
        int16_t id = NANO_LEX_PH_ID[g_ws.chunk[i]];
        if (id < 0) {
            g_ws.stats.dropped++;
            continue;
        }
        nlg_ids_push(ids, id);
        kept++;
    }
    if (kept == 0) {
        return NANO_LEX_E_NO_SYMBOLS;
    }
    nlg_ids_push(ids, NANO_LEX_ID_EOS);
    return NANO_LEX_OK;
}

/* kokoro/pipeline.py: KPipeline.en_tokenize, reduced to chunk boundaries. */
static int nlg_en_chunks(nlg_ids_t *ids)
{
    uint16_t held = 0;
    uint32_t pcount = 0;
    uint16_t gi;
    int rc;

    for (gi = 0; gi < g_ws.grp_len; gi++) {
        const nlg_ps_t *ps = nlg_flat_ps(gi);
        uint32_t next_ps_len = (uint32_t)ps->len + (nlg_flat_ws(gi) ? 1u : 0u);
        uint32_t rstripped = (uint32_t)ps->len;
        uint32_t next_pcount;

        /* len(next_ps.rstrip()) */
        while (rstripped > 0 && g_ws.arena[ps->off + rstripped - 1] == NLG_PH_0020) {
            rstripped--;
        }
        next_pcount = pcount + rstripped;
        if (next_pcount > (uint32_t)NANO_LEX_MAX_PS_CHARS) {
            uint16_t z = nlg_waterfall_last(held, gi, next_pcount);
            rc = nlg_emit_chunk(held, (uint16_t)(held + z), ids);
            if (rc != NANO_LEX_OK) {
                return rc;
            }
            held = (uint16_t)(held + z);
            pcount = nlg_tokens_to_ps_len(held, gi);
            if (held == gi) {
                /* next_ps.lstrip() */
                uint32_t lead = 0;
                while (lead < (uint32_t)ps->len &&
                       g_ws.arena[ps->off + lead] == NLG_PH_0020) {
                    lead++;
                }
                next_ps_len -= lead;
            }
        }
        pcount += next_ps_len;
    }
    if (held < g_ws.grp_len) {
        rc = nlg_emit_chunk(held, g_ws.grp_len, ids);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
    }
    return NANO_LEX_OK;
}

/* --------------------------------------------------------------------------
 * Public entry points
 * -------------------------------------------------------------------------- */

static int nlg_run(const char *text)
{
    int rc;
    nlg_zero(&g_ws.stats, sizeof g_ws.stats);
    nlg_arena_reset();
    g_ws.src_len = 0;
    g_ws.tok_len = 0;
    g_ws.grp_len = 0;
    g_ws.cp_len = 0;

    rc = nlg_decode_utf8(text);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    rc = nlg_tokenize();
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    if (g_ws.src_len == 0) {
        return NANO_LEX_E_EMPTY_TEXT;
    }
    rc = nlg_retokenize();
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    rc = nlg_resolve();
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    rc = nlg_flatten();
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    g_ws.stats.tokens = g_ws.tok_len;
    g_ws.stats.arena_peak = g_ws.arena_peak;
    return NANO_LEX_OK;
}

int nano_lex_g2p_text_to_ids(const char *text, int32_t *out, int cap)
{
    nlg_ids_t ids;
    int rc;

    if (text == 0 || out == 0) {
        return NANO_LEX_E_NULL_ARG;
    }
    if (cap < 2) {
        return NANO_LEX_E_BAD_CAP;
    }
    rc = nlg_run(text);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    ids.out = out;
    ids.cap = cap;
    ids.n = 0;
    ids.overflow = 0;
    rc = nlg_en_chunks(&ids);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    g_ws.stats.ids = (uint16_t)ids.n;
    if (ids.n == 0) {
        return NANO_LEX_E_NO_SYMBOLS;
    }
    if (ids.overflow) {
        /* Keep the buffer well framed: the caller gets a real prefix that still
         * ends in <eos>, plus a code saying it is not the whole utterance. */
        out[cap - 1] = NANO_LEX_ID_EOS;
        return NANO_LEX_E_CAP;
    }
    return ids.n;
}

int nano_lex_g2p_text_to_phonemes(const char *text, char *out, size_t cap)
{
    int rc;
    size_t pos = 0;
    uint16_t held = 0;
    uint16_t gi;

    if (text == 0 || out == 0) {
        return NANO_LEX_E_NULL_ARG;
    }
    if (cap < 1) {
        return NANO_LEX_E_BAD_CAP;
    }
    rc = nlg_run(text);
    if (rc != NANO_LEX_OK) {
        return rc;
    }
    /* phonemize_to_string is "".join(en_chunks(...)), so the chunk boundaries
     * matter only for where the string is cut, not for its content. Re-run the
     * chunker and concatenate. */
    {
        uint32_t pcount = 0;
        held = 0;
        for (gi = 0; gi < g_ws.grp_len; gi++) {
            const nlg_ps_t *ps = nlg_flat_ps(gi);
            uint32_t next_ps_len = (uint32_t)ps->len + (nlg_flat_ws(gi) ? 1u : 0u);
            uint32_t rstripped = (uint32_t)ps->len;
            uint32_t next_pcount;
            while (rstripped > 0 && g_ws.arena[ps->off + rstripped - 1] == NLG_PH_0020) {
                rstripped--;
            }
            next_pcount = pcount + rstripped;
            if (next_pcount > (uint32_t)NANO_LEX_MAX_PS_CHARS) {
                uint16_t z = nlg_waterfall_last(held, gi, next_pcount);
                uint16_t len = 0;
                uint16_t i;
                rc = nlg_tokens_to_ps(held, (uint16_t)(held + z), g_ws.chunk,
                                      (uint16_t)sizeof g_ws.chunk, &len);
                if (rc != NANO_LEX_OK) {
                    return rc;
                }
                for (i = 0; i < len; i++) {
                    uint32_t cp = NANO_LEX_PH_CP[g_ws.chunk[i]];
                    unsigned char buf[4];
                    size_t n;
                    size_t k;
                    if (cp < 0x80u) {
                        buf[0] = (unsigned char)cp;
                        n = 1;
                    } else if (cp < 0x800u) {
                        buf[0] = (unsigned char)(0xC0u | (cp >> 6));
                        buf[1] = (unsigned char)(0x80u | (cp & 0x3Fu));
                        n = 2;
                    } else {
                        buf[0] = (unsigned char)(0xE0u | (cp >> 12));
                        buf[1] = (unsigned char)(0x80u | ((cp >> 6) & 0x3Fu));
                        buf[2] = (unsigned char)(0x80u | (cp & 0x3Fu));
                        n = 3;
                    }
                    if (pos + n + 1 > cap) {
                        return NANO_LEX_E_CAP;
                    }
                    for (k = 0; k < n; k++) {
                        out[pos++] = (char)buf[k];
                    }
                }
                held = (uint16_t)(held + z);
                pcount = nlg_tokens_to_ps_len(held, gi);
                if (held == gi) {
                    uint32_t lead = 0;
                    while (lead < (uint32_t)ps->len &&
                           g_ws.arena[ps->off + lead] == NLG_PH_0020) {
                        lead++;
                    }
                    next_ps_len -= lead;
                }
            }
            pcount += next_ps_len;
        }
    }
    {
        uint16_t len = 0;
        uint16_t i;
        rc = nlg_tokens_to_ps(held, g_ws.grp_len, g_ws.chunk,
                              (uint16_t)sizeof g_ws.chunk, &len);
        if (rc != NANO_LEX_OK) {
            return rc;
        }
        for (i = 0; i < len; i++) {
            uint32_t cp = NANO_LEX_PH_CP[g_ws.chunk[i]];
            unsigned char buf[4];
            size_t n;
            size_t k;
            if (cp < 0x80u) {
                buf[0] = (unsigned char)cp;
                n = 1;
            } else if (cp < 0x800u) {
                buf[0] = (unsigned char)(0xC0u | (cp >> 6));
                buf[1] = (unsigned char)(0x80u | (cp & 0x3Fu));
                n = 2;
            } else {
                buf[0] = (unsigned char)(0xE0u | (cp >> 12));
                buf[1] = (unsigned char)(0x80u | ((cp >> 6) & 0x3Fu));
                buf[2] = (unsigned char)(0x80u | (cp & 0x3Fu));
                n = 3;
            }
            if (pos + n + 1 > cap) {
                return NANO_LEX_E_CAP;
            }
            for (k = 0; k < n; k++) {
                out[pos++] = (char)buf[k];
            }
        }
    }
    out[pos] = '\0';
    return (int)pos;
}

const char *nano_lex_g2p_strerror(int rc)
{
    if (rc > 0) {
        return "ok";
    }
    switch (rc) {
    case NANO_LEX_OK:            return "ok";
    case NANO_LEX_E_NULL_ARG:    return "null argument";
    case NANO_LEX_E_BAD_CAP:     return "output capacity is too small for BOS and EOS";
    case NANO_LEX_E_EMPTY_TEXT:  return "text is empty or all whitespace";
    case NANO_LEX_E_BAD_UTF8:    return "input is not well-formed UTF-8";
    case NANO_LEX_E_TEXT_LONG:   return "text is longer than the workspace allows";
    case NANO_LEX_E_TOKENS:      return "text produced more tokens than the workspace allows";
    case NANO_LEX_E_ARENA:       return "phoneme workspace exhausted";
    case NANO_LEX_E_NO_SYMBOLS:  return "no phoneme survived the 62-symbol vocabulary";
    case NANO_LEX_E_CAP:         return "id sequence did not fit; output was truncated";
    case NANO_LEX_E_NUMBER:      return "numeral outside the range this speller covers";
    case NANO_LEX_E_TABLE:       return "packed dictionary failed its structural checks";
    case NANO_LEX_E_INTERNAL:    return "internal invariant violated";
    default:                     return "unknown nano_lex_g2p error code";
    }
}

size_t nano_lex_g2p_workspace_bytes(void)
{
    return sizeof g_ws;
}

void nano_lex_g2p_get_stats(nano_lex_g2p_stats_t *out)
{
    if (out != 0) {
        *out = g_ws.stats;
    }
}
