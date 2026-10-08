/* test_nano_lex_g2p.c -- host driver and self-test for nano_lex_g2p.
 *
 *   ./test_nano_lex_g2p              read one UTF-8 sentence per line from
 *                                    stdin, print "<ids csv>\t<phonemes>" or
 *                                    "!ERROR\t<name>" per line
 *   ./test_nano_lex_g2p --selftest   contract checks that need no oracle
 *   ./test_nano_lex_g2p --stack      report peak stack for one call
 *   ./test_nano_lex_g2p --sizes      report workspace bytes
 *
 * This file is host-only: it is the only place printf and stdio appear.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nano_lex_g2p.h"
#include "nano_lex_tables.h"

#define LINE_MAX_BYTES 8192
#define ID_CAP         4096
#define PHONEME_CAP    4096

static int run_stdin(void)
{
    static char line[LINE_MAX_BYTES];
    static int32_t ids[ID_CAP];
    static char phonemes[PHONEME_CAP];

    while (fgets(line, sizeof line, stdin) != NULL) {
        size_t n = strlen(line);
        int rc;
        int i;
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
            line[--n] = '\0';
        }
        nano_lex_g2p_stats_t st;
        rc = nano_lex_g2p_text_to_ids(line, ids, ID_CAP);
        if (rc < 0) {
            printf("!ERROR\t%d\t%s\n", rc, nano_lex_g2p_strerror(rc));
            continue;
        }
        /* Read the counters before the phoneme call, which re-runs the
         * pipeline and resets them. */
        nano_lex_g2p_get_stats(&st);
        for (i = 0; i < rc; i++) {
            printf(i == 0 ? "%d" : ",%d", (int)ids[i]);
        }
        putchar('\t');
        {
            int pn = nano_lex_g2p_text_to_phonemes(line, phonemes, sizeof phonemes);
            if (pn < 0) {
                printf("!PHONEME_ERROR");
            } else {
                fputs(phonemes, stdout);
            }
        }
        printf("\t%u,%u,%u,%u,%u\n", (unsigned)st.lexicon_words, (unsigned)st.oov_words,
               (unsigned)st.chunks, (unsigned)st.dropped, (unsigned)st.arena_peak);
    }
    return 0;
}

/* ---- self-test ---------------------------------------------------------- */

static int failures;

static void check(int condition, const char *what)
{
    if (!condition) {
        printf("FAIL %s\n", what);
        failures++;
    } else {
        printf("ok   %s\n", what);
    }
}

/* A deterministic pronunciation provider keeps these tests independent of any
 * external G2P model. It can also return malformed results to check the API. */
static int fallback_calls;
static int fallback_result = 1;
static uint8_t fallback_phone = NLG_PH_0062;
static uint16_t fallback_word[64];
static size_t fallback_length;
static size_t fallback_capacity;

static int test_fallback(const uint16_t *word, size_t length,
                         uint8_t *phones, size_t capacity)
{
    fallback_calls++;
    fallback_length = length;
    fallback_capacity = capacity;
    if (length <= sizeof fallback_word / sizeof fallback_word[0]) {
        memcpy(fallback_word, word, length * sizeof *word);
    }
    if (capacity) phones[0] = fallback_phone;
    return fallback_result;
}

static void fallback_selftest(void)
{
    static const uint16_t word[] = {'Z', 'o', 'g', 'b', 'l', 'i', 'p'};
    int32_t before[32], ids[32];
    nano_lex_g2p_stats_t st;
    int full, rc, i;

    rc = nano_lex_g2p_text_to_ids("Zogblip", ids, 32);
    check(rc == NANO_LEX_E_NO_SYMBOLS, "fallback is disabled by default");
    full = nano_lex_g2p_text_to_ids("mother-in-law", before, 32);
    nano_lex_g2p_set_fallback(test_fallback);
    rc = nano_lex_g2p_text_to_ids("mother-in-law", ids, 32);
    check(full > 0 && rc == full &&
          memcmp(before, ids, (size_t)full * sizeof *ids) == 0 && !fallback_calls,
          "dictionary compounds take precedence over fallback");

    rc = nano_lex_g2p_text_to_ids("Zogblip", ids, 32);
    check(fallback_calls == 1 && fallback_length == 7 &&
          memcmp(fallback_word, word, sizeof word) == 0,
          "fallback receives the unresolved word as UTF-16");
    check(rc == 3 && ids[0] == NANO_LEX_ID_BOS &&
          ids[1] == NLG_PH_0062 + 2 && ids[2] == NANO_LEX_ID_EOS,
          "fallback phonemes become framed model ids");
    nano_lex_g2p_get_stats(&st);
    check(st.oov_words == 0, "a successful fallback resolves the OOV word");

    fallback_calls = 0;
    rc = nano_lex_g2p_text_to_ids("ZogblipQuxwob", ids, 32);
    check(rc == 4 && fallback_calls == 2 &&
          ids[1] == NLG_PH_0062 + 2 && ids[2] == NLG_PH_0062 + 2,
          "unresolved groups fall back one word at a time");

    fallback_result = 0;
    rc = nano_lex_g2p_text_to_ids("Zogblip", ids, 32);
    nano_lex_g2p_get_stats(&st);
    check(rc == NANO_LEX_E_NO_SYMBOLS && st.oov_words == 1,
          "unsupported words remain unresolved");
    fallback_result = NANO_LEX_E_ARENA;
    ids[0] = -12345;
    rc = nano_lex_g2p_text_to_ids("Zogblip", ids, 32);
    check(rc == NANO_LEX_E_ARENA && ids[0] == -12345,
          "fallback errors propagate without writing ids");
    fallback_result = (int)fallback_capacity + 1;
    rc = nano_lex_g2p_text_to_ids("Zogblip", ids, 32);
    check(rc == NANO_LEX_E_CAP, "an oversized fallback count is rejected");
    fallback_result = 1;
    for (i = 0; i < 2; i++) {
        fallback_phone = i ? NANO_LEX_PH_COUNT + 1 : 0;
        rc = nano_lex_g2p_text_to_ids("Zogblip", ids, 32);
        check(rc == NANO_LEX_E_INTERNAL, "an invalid fallback phoneme is rejected");
    }
    nano_lex_g2p_set_fallback(NULL);
    rc = nano_lex_g2p_text_to_ids("Zogblip", ids, 32);
    check(rc == NANO_LEX_E_NO_SYMBOLS, "NULL restores dictionary-only behavior");
}

static void currency_selftest(void)
{
    static const struct { const char *text; const char *spoken; } cases[] = {
        {"$3.5", "three dollars and fifty cents"},
        {"$0.5", "fifty cents"},
        {"$.5", "fifty cents"},
        {"$1.2", "one dollar and twenty cents"},
        {"$-0.5", "minus fifty cents"},
        {"$3.0", "three dollars"},
        {"$3.05", "three dollars and five cents"},
        {"$3.50", "three dollars and fifty cents"},
        {"$1.01", "one dollar and one cent"},
    };
    int32_t actual[128], expected[128];
    size_t i;
    for (i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        char what[128];
        int n = nano_lex_g2p_text_to_ids(cases[i].text, actual, 128);
        int m = nano_lex_g2p_text_to_ids(cases[i].spoken, expected, 128);
        snprintf(what, sizeof what, "%s matches %s", cases[i].text, cases[i].spoken);
        check(n > 0 && n == m &&
              memcmp(actual, expected, (size_t)n * sizeof *actual) == 0, what);
    }
}

static int selftest(void)
{
    static int32_t ids[ID_CAP];
    int rc;

    rc = nano_lex_g2p_text_to_ids(NULL, ids, ID_CAP);
    check(rc == NANO_LEX_E_NULL_ARG, "NULL text is rejected");
    rc = nano_lex_g2p_text_to_ids("hello", NULL, ID_CAP);
    check(rc == NANO_LEX_E_NULL_ARG, "NULL out is rejected");
    rc = nano_lex_g2p_text_to_ids("hello", ids, 1);
    check(rc == NANO_LEX_E_BAD_CAP, "cap below 2 is rejected");
    rc = nano_lex_g2p_text_to_ids("   ", ids, ID_CAP);
    check(rc == NANO_LEX_E_EMPTY_TEXT, "whitespace-only text is rejected");
    rc = nano_lex_g2p_text_to_ids("caf\xc3", ids, ID_CAP);
    check(rc == NANO_LEX_E_BAD_UTF8, "truncated UTF-8 is rejected");

    rc = nano_lex_g2p_text_to_ids("The front door is unlocked.", ids, ID_CAP);
    check(rc > 2, "a plain sentence produces ids");
    check(rc > 0 && ids[0] == NANO_LEX_ID_BOS, "the sequence starts with <bos>");
    check(rc > 0 && ids[rc - 1] == NANO_LEX_ID_EOS, "the sequence ends with <eos>");
    {
        int i;
        int has_pad = 0;
        for (i = 1; i < rc - 1; i++) {
            if (ids[i] == NANO_LEX_ID_PAD || ids[i] == NANO_LEX_ID_BOS ||
                ids[i] == NANO_LEX_ID_EOS) {
                has_pad = 1;
            }
        }
        check(!has_pad, "the sequence is dense: no pad or interior bos/eos");
    }

    /* Truncation: ask for less than the sentence needs and check the buffer is
     * still framed and nothing past cap was touched. */
    {
        static int32_t small[16];
        int i;
        int full;
        full = nano_lex_g2p_text_to_ids("The front door is unlocked.", ids, ID_CAP);
        for (i = 0; i < 16; i++) {
            small[i] = -12345;
        }
        rc = nano_lex_g2p_text_to_ids("The front door is unlocked.", small, 8);
        check(rc == NANO_LEX_E_CAP, "an over-long sequence returns NANO_LEX_E_CAP");
        check(small[7] == NANO_LEX_ID_EOS, "the truncated buffer still ends with <eos>");
        check(small[8] == -12345, "nothing was written past cap");
        check(small[0] == NANO_LEX_ID_BOS, "the truncated buffer still starts with <bos>");
        check(full > 8, "the untruncated sequence really was longer than cap");
    }

    /* One golden sequence, checked against nano_g2p.phonemize() by
     * run_parity.py, so a table or tokeniser regression fails here first. */
    {
        static const int32_t expect[] = {
            1, 37, 42, 3, 21, 47, 54, 50, 28, 31, 3, 20, 54, 41, 47, 3, 46, 35,
            3, 55, 50, 28, 26, 54, 40, 25, 31, 9, 2
        };
        static const int n = (int)(sizeof expect / sizeof expect[0]);
        int same = (rc == n);
        int i;
        rc = nano_lex_g2p_text_to_ids("The front door is unlocked.", ids, ID_CAP);
        same = (rc == n);
        for (i = 0; same && i < n; i++) {
            same = (ids[i] == expect[i]);
        }
        check(same, "\"The front door is unlocked.\" matches the Python oracle's ids");
    }

    /* The remaining error paths, so none of them is only reachable in theory. */
    {
        static char big[NANO_LEX_MAX_CHARS + 64];
        int i;
        for (i = 0; i < (int)sizeof big - 1; i++) {
            big[i] = 'a';
        }
        big[sizeof big - 1] = '\0';
        rc = nano_lex_g2p_text_to_ids(big, ids, ID_CAP);
        check(rc == NANO_LEX_E_TEXT_LONG, "over-long text returns NANO_LEX_E_TEXT_LONG");
    }
    {
        static char many[400];
        int i;
        for (i = 0; i < (int)sizeof many - 1; i++) {
            many[i] = '?';
        }
        many[sizeof many - 1] = '\0';
        rc = nano_lex_g2p_text_to_ids(many, ids, ID_CAP);
        check(rc == NANO_LEX_E_TOKENS, "too many tokens returns NANO_LEX_E_TOKENS");
    }
    rc = nano_lex_g2p_text_to_ids("12345678901234567890 apples", ids, ID_CAP);
    check(rc == NANO_LEX_E_NUMBER, "a 20-digit numeral returns NANO_LEX_E_NUMBER");

    /* Distinct codes, distinct messages, and never NULL. */
    {
        int code;
        int distinct = 1;
        for (code = 0; code >= -13; code--) {
            int other;
            const char *a = nano_lex_g2p_strerror(code);
            if (a == NULL) {
                distinct = 0;
                break;
            }
            for (other = 0; other > code; other--) {
                if (nano_lex_g2p_strerror(other) == a && code != -13) {
                    distinct = 0;
                }
            }
        }
        check(distinct, "every error code has its own non-NULL message");
        check(nano_lex_g2p_strerror(-999) != NULL, "an unknown code still names itself");
    }

    currency_selftest();
    fallback_selftest();

    printf("\nworkspace %zu bytes\n", nano_lex_g2p_workspace_bytes());
    printf("%s (%d failures)\n", failures ? "SELFTEST FAILED" : "SELFTEST PASSED", failures);
    return failures ? 1 : 0;
}

/* ---- stack measurement --------------------------------------------------
 * Classic embedded stack painting: fill the region just below the current
 * stack pointer with a pattern, make the call, then see how far down the
 * pattern was destroyed. The paint is written through a volatile pointer with
 * an inline loop rather than memset(), because a call to memset would itself
 * run below the pointer and destroy the paint it had just laid down.
 * ------------------------------------------------------------------------- */

#define PAINT_BYTES (256u * 1024u)
#define PAINT_VALUE 0xA5u

static size_t measure_call(const char *text, int32_t *ids, int cap)
{
    volatile unsigned char *floor_;
    unsigned char here;
    size_t i;
    size_t used = 0;

    floor_ = (volatile unsigned char *)(&here) - PAINT_BYTES;
    for (i = 0; i < PAINT_BYTES; i++) {
        floor_[i] = (unsigned char)PAINT_VALUE;
    }
    (void)nano_lex_g2p_text_to_ids(text, ids, cap);
    /* The callee's frames sit immediately below &here, i.e. at the TOP of the
     * painted region, so the deepest point is the lowest surviving index. */
    for (i = 0; i < PAINT_BYTES; i++) {
        if (floor_[i] != (unsigned char)PAINT_VALUE) {
            used = PAINT_BYTES - i;
            break;
        }
    }
    return used;
}

static int stack_probe(void)
{
    static int32_t ids[ID_CAP];
    static const char *const texts[] = {
        "The front door is unlocked.",
        "It's twenty two degrees in the living room and the humidity is 48%.",
        ("Don't forget: the laundry finished 15 minutes ago, and you're low on "
         "detergent -- the washing machine reported 1,234 cycles since 2019."),
        "$1,234.56 was spent on the 3rd of December, 1999, at 4:05 p.m.",
        "Zzyzxwood's antidisestablishmentarianism-adjacent Fitzwilliam Euroclydon.",
        "The 1st, 2nd, 3rd and 21st readings were 0.5, 12.75 and 1,000,000.",
    };
    size_t worst = 0;
    size_t i;

    /* One warm-up call so nothing first-time is attributed to the deepest run. */
    (void)nano_lex_g2p_text_to_ids(texts[0], ids, ID_CAP);
    for (i = 0; i < sizeof texts / sizeof texts[0]; i++) {
        size_t used = measure_call(texts[i], ids, ID_CAP);
        printf("  %5zu bytes  %.48s\n", used, texts[i]);
        if (used > worst) {
            worst = used;
        }
    }
    printf("peak stack for one call, measured by painting: %zu bytes\n", worst);
    printf("(includes measure_call's own frame, so it is an over-estimate)\n");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "--selftest") == 0) {
        return selftest();
    }
    if (argc > 1 && strcmp(argv[1], "--stack") == 0) {
        return stack_probe();
    }
    if (argc > 1 && strcmp(argv[1], "--sizes") == 0) {
        printf("workspace_bytes %zu\n", nano_lex_g2p_workspace_bytes());
        return 0;
    }
    return run_stdin();
}
