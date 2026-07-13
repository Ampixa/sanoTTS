/* snt_g2p_wasm.c -- WebAssembly (Emscripten) espeak-ng G2P shim.
 *
 * Same phonemize contract as the ESP32-S3 port's esp_g2p.c: espeak-ng
 * (translator only, no audio synth) turns text into IPA, each codepoint maps
 * to a Piper phoneme id via a binary-searched per-voice CP_ID table, and the
 * BOS/pad/EOS framing matches piper's phonemes_to_ids exactly. Only
 * init differs -- the browser mounts espeak-ng-data through Emscripten's
 * preloaded virtual FS (--preload-file ...@/espeak) instead of SPIFFS, so
 * there is no esp_vfs_spiffs_register() call here.
 *
 * Multi-language: snt_g2p_set_voice(espeak_voice, voice_slot) switches both
 * the espeak voice (G2P rules/dict) and the id table (that voice's Piper
 * phoneme_id_map, see cp_id_tables_multi.h). Default stays en-us + kristin
 * (slot 0) so the original English path is unchanged.
 *
 * Do not change the id-mapping/framing logic without re-running the parity
 * gate (verify_g2p_node.mjs) -- the tables are the training-time contract.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "espeak-ng/espeak_ng.h"
#include "espeak-ng/speak_lib.h"

#include "cp_id_tables_multi.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define SNT_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define SNT_EXPORT
#endif

#define ID_BOS 1
#define ID_PAD 0
#define ID_EOS 2

/* Active id table -- defaults to kristin (slot 0), the original contract. */
static const cp_id_t *g_tab = CP_ID_KRISTIN;
static int g_tab_n = (int)(sizeof(CP_ID_KRISTIN) / sizeof(CP_ID_KRISTIN[0]));

static int lookup(unsigned cp) {
    int lo = 0, hi = g_tab_n;
    while (lo < hi) { int m = (lo + hi) / 2; if (g_tab[m].cp < cp) lo = m + 1; else hi = m; }
    return (lo < g_tab_n && g_tab[lo].cp == cp) ? g_tab[lo].id : -1;
}

static const char *nextcp(const char *s, unsigned *cp) {
    unsigned char c = (unsigned char)*s;
    if (c < 0x80) { *cp = c; return s + 1; }
    if ((c >> 5) == 6) { *cp = ((c & 0x1F) << 6) | (s[1] & 0x3F); return s + 2; }
    if ((c >> 4) == 14) { *cp = ((c & 0xF) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); return s + 3; }
    if ((c >> 3) == 30) { *cp = ((c & 7) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F); return s + 4; }
    *cp = c; return s + 1;
}

static int g_ready = 0;

/* Idempotent: safe to call once from JS before the first phonemize call, or
 * to leave to the lazy init inside snt_g2p_text_to_ids(). */
SNT_EXPORT
int snt_g2p_init(void) {
    if (g_ready) return 0;

    espeak_ng_InitializePath("/espeak");
    espeak_ng_ERROR_CONTEXT ctx = NULL;
    espeak_ng_STATUS s = espeak_ng_Initialize(&ctx);
    if (s != ENS_OK) { printf("g2p: espeak init failed (0x%x)\n", (unsigned)s); return -2; }

    s = espeak_ng_SetVoiceByName("en-us");
    if (s != ENS_OK) {
        /* en-us not resolved for some reason -- fall back to the base "en"
         * voice the ESP32 port uses; still English G2P, same rule tables. */
        s = espeak_ng_SetVoiceByName("en");
        if (s != ENS_OK) { printf("g2p: set voice failed (0x%x)\n", (unsigned)s); return -3; }
    }
    g_ready = 1;
    printf("g2p: espeak ready (voice en-us)\n");
    return 0;
}

/* Select the espeak voice (G2P rules + dictionary) and the Piper id table
 * for one of the release voices. voice_slot indexes SNT_VOICE_TABS (see
 * cp_id_tables_multi.h for the slot order); espeak_voice is the espeak
 * voice name from that voice's Piper config ("en-us", "vi", "cmn", ...).
 * Passing them separately lets e.g. kristin (table slot 0) run under either
 * "en" (its config voice, exact Piper parity) or "en-us" (the historical
 * default). Returns 0 on success. */
SNT_EXPORT
int snt_g2p_set_voice(const char *espeak_voice, int voice_slot) {
    if (!g_ready) { int rc = snt_g2p_init(); if (rc != 0) return -1; }
    if (!espeak_voice || voice_slot < 0 || voice_slot >= SNT_NUM_VOICE_TABS) return -1;

    espeak_ng_STATUS s = espeak_ng_SetVoiceByName(espeak_voice);
    if (s != ENS_OK) {
        printf("g2p: set voice '%s' failed (0x%x)\n", espeak_voice, (unsigned)s);
        return -2;
    }
    g_tab = SNT_VOICE_TABS[voice_slot].tab;
    g_tab_n = SNT_VOICE_TABS[voice_slot].n;
    printf("g2p: voice '%s' + id table '%s' (slot %d, %d entries)\n",
           espeak_voice, SNT_VOICE_TABS[voice_slot].name, voice_slot, g_tab_n);
    return 0;
}

SNT_EXPORT
int snt_g2p_text_to_ids(const char *text, int32_t *out_ids, int max_ids) {
    if (!g_ready) { int rc = snt_g2p_init(); if (rc != 0) return -1; }
    if (!text || !out_ids || max_ids <= 0) return -1;

    int n = 0;
    if (n + 2 > max_ids) return -1;
    out_ids[n++] = ID_BOS; out_ids[n++] = ID_PAD;

    const void *tp = text;
    int guard = 0;
    while (tp && *(const char *)tp && guard++ < 256) {
        const char *p0 = (const char *)tp;
        const char *ph = espeak_TextToPhonemes(&tp, espeakCHARS_UTF8, espeakPHONEMES_IPA);
        const char *p1 = tp ? (const char *)tp : p0 + strlen(p0);
        if (ph) {
            const char *p = ph; unsigned cp; int paren = 0;
            while (*p) {
                p = nextcp(p, &cp);
                if (cp == '(') { paren = 1; continue; }
                if (cp == ')') { paren = 0; continue; }
                if (paren) continue;
                int id = lookup(cp);
                if (id >= 0 && n + 2 <= max_ids) { out_ids[n++] = id; out_ids[n++] = ID_PAD; }
            }
        }
        char tc = 0;
        for (const char *q = p1 - 1; q >= p0; q--) {
            char c = *q;
            if (c == '.' || c == ',' || c == '?' || c == '!' || c == ':' || c == ';') { tc = c; break; }
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != '-') break;
        }
        if (tc) {
            int id = lookup((unsigned)tc);
            if (id >= 0 && n + 2 <= max_ids) { out_ids[n++] = id; out_ids[n++] = ID_PAD; }
            if (tc == ',' || tc == ':' || tc == ';') { int sid = lookup(' '); if (sid >= 0 && n + 2 <= max_ids) { out_ids[n++] = sid; out_ids[n++] = ID_PAD; } }
        }
    }
    if (n + 1 <= max_ids) out_ids[n++] = ID_EOS;
    return n;
}
