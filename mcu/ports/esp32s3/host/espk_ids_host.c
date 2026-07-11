/* Host validation: espeak-ng text -> Kristin phoneme ids, replicating piper's
 * phonemize_espeak.py (IPA phonemes, strip (lang) flags, per-clause terminator
 * punctuation, NFD codepoints -> id_map, BOS/pad/EOS layout). The SAME logic
 * ports to the ESP32 (only espeak init/data path differs). Prints CSV ids. */
#include <stdio.h>
#include <string.h>
#include "espeak-ng/speak_lib.h"
#include "cp_id_table.h"

#define ID_BOS 1
#define ID_PAD 0
#define ID_EOS 2

static int lookup(unsigned cp) {
    int lo = 0, hi = 157;
    while (lo < hi) { int m = (lo + hi) / 2; if (CP_ID[m].cp < cp) lo = m + 1; else hi = m; }
    return (lo < 157 && CP_ID[lo].cp == cp) ? CP_ID[lo].id : -1;
}
static const char *nextcp(const char *s, unsigned *cp) {
    unsigned char c = (unsigned char)*s;
    if (c < 0x80) { *cp = c; return s + 1; }
    if ((c >> 5) == 6) { *cp = ((c & 0x1F) << 6) | (s[1] & 0x3F); return s + 2; }
    if ((c >> 4) == 14) { *cp = ((c & 0xF) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); return s + 3; }
    if ((c >> 3) == 30) { *cp = ((c & 7) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F); return s + 4; }
    *cp = c; return s + 1;
}

static int out[8192], n;
static void emit(int id) { if (id >= 0 && n < 8190) { out[n++] = id; out[n++] = ID_PAD; } }

int main(int argc, char **argv) {
    if (espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, "/opt/homebrew/share/espeak-ng-data", 0) < 0) {
        fprintf(stderr, "espeak init failed\n"); return 1;
    }
    espeak_SetVoiceByName("en");
    n = 0; out[n++] = ID_BOS; out[n++] = ID_PAD;
    const void *tp = argv[1];
    int guard = 0;
    while (tp && *(const char *)tp && guard++ < 256) {
        const char *p0 = (const char *)tp;
        const char *ph = espeak_TextToPhonemes(&tp, espeakCHARS_UTF8, espeakPHONEMES_IPA);
        const char *p1 = tp ? (const char *)tp : p0 + strlen(p0);
        if (ph) {
            const char *p = ph; unsigned cp; int paren = 0;
            while (*p) {
                p = nextcp(p, &cp);
                if (cp == '(') { paren = 1; continue; }   /* strip (lang) flags like piper */
                if (cp == ')') { paren = 0; continue; }
                if (paren) continue;
                emit(lookup(cp));
            }
        }
        /* recover the clause terminator punctuation from the consumed text span */
        char tc = 0;
        for (const char *q = p1 - 1; q >= p0; q--) {
            char c = *q;
            if (c == '.' || c == ',' || c == '?' || c == '!' || c == ':' || c == ';') { tc = c; break; }
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != '-') break;
        }
        if (tc) { emit(lookup((unsigned)tc)); if (tc == ',' || tc == ':' || tc == ';') emit(lookup(' ')); }
    }
    out[n++] = ID_EOS;
    for (int i = 0; i < n; i++) printf("%d%s", out[i], i + 1 < n ? "," : "\n");
    return 0;
}
