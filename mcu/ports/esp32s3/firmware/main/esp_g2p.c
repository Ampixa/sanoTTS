/* On-device espeak-ng G2P wrapper. The phonemize logic is identical to the
 * host-validated espk_ids_host.c; only init (SPIFFS-mounted data) differs. */
#include <stdio.h>
#include <string.h>
#include "esp_spiffs.h"
#include "esp_err.h"
#include "espeak-ng/speak_lib.h"
#include "espeak-ng/espeak_ng.h"
#include "esp_g2p.h"
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

static int g_ready = 0;

int esp_g2p_init(void) {
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/espeak", .partition_label = "espeak",
        .max_files = 6, .format_if_mount_failed = false,
    };
    esp_err_t e = esp_vfs_spiffs_register(&conf);
    if (e != ESP_OK) { printf("g2p: spiffs mount failed (%d)\n", e); return -1; }
    size_t total = 0, used = 0;
    esp_spiffs_info("espeak", &total, &used);
    printf("g2p: spiffs mounted, %u/%u bytes used\n", (unsigned)used, (unsigned)total);

    espeak_ng_InitializePath("/espeak");
    espeak_ng_ERROR_CONTEXT ctx = NULL;
    espeak_ng_STATUS s = espeak_ng_Initialize(&ctx);
    if (s != ENS_OK) { printf("g2p: espeak init failed (0x%x)\n", (unsigned)s); return -2; }
    s = espeak_ng_SetVoiceByName("en");
    if (s != ENS_OK) { printf("g2p: set voice failed (0x%x)\n", (unsigned)s); return -3; }
    g_ready = 1;
    printf("g2p: espeak ready (voice en)\n");
    return 0;
}

int esp_g2p_text_to_ids(const char *text, int *out, int cap) {
    if (!g_ready) return -1;
    int n = 0;
    if (n + 2 > cap) return -1;
    out[n++] = ID_BOS; out[n++] = ID_PAD;
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
                if (id >= 0 && n + 2 <= cap) { out[n++] = id; out[n++] = ID_PAD; }
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
            if (id >= 0 && n + 2 <= cap) { out[n++] = id; out[n++] = ID_PAD; }
            if (tc == ',' || tc == ':' || tc == ';') { int sid = lookup(' '); if (sid >= 0 && n + 2 <= cap) { out[n++] = sid; out[n++] = ID_PAD; } }
        }
    }
    if (n + 1 <= cap) out[n++] = ID_EOS;
    return n;
}
