#pragma once
/* On-device espeak-ng G2P: text -> Kristin phoneme ids. Mirrors the host-validated
 * espk_ids_host.c (voice "en", IPA -> codepoint->id -> BOS/pad/EOS). */
int esp_g2p_init(void);                              /* mount spiffs + init espeak; 0 ok */
int esp_g2p_text_to_ids(const char *text, int *out, int cap);  /* returns n ids or -1 */
