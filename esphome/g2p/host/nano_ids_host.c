/* nano_ids_host.c -- host driver for the SAME nano_g2p.c that runs on the S3.
 *
 * Compiles the unmodified component source against a host libespeak-ng so the
 * espeak -> E2M -> 62-vocab chain can be measured against the repo's recorded
 * misaki reference before anything is flashed. Only nano_g2p_init_path()'s
 * SPIFFS branch differs between host and device, and it is #ifdef'd out here.
 *
 * Reads one UTF-8 sentence per line on stdin. Writes one TSV line per input:
 *
 *     <phoneme string> \t <id,id,id,...>
 *
 * and on failure:
 *
 *     !ERROR \t <code> \t <message>
 *
 * so a bad row is visible in the parity report instead of silently skewing it.
 *
 * Build: esphome/g2p/host/build_host.sh
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nano_g2p.h"
#include "nano_g2p_table.h"

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <espeak-ng-data-dir>\n", argv[0]);
        return 2;
    }

    int rc = nano_g2p_init_path(argv[1]);
    if (rc != NANO_G2P_OK) {
        fprintf(stderr, "nano_g2p_init_path(%s) failed: %d (%s)\n", argv[1], rc,
                nano_g2p_strerror(rc));
        return 1;
    }
    fprintf(stderr, "workspace: %zu bytes\n", nano_g2p_workspace_bytes());

    char line[4096];
    while (fgets(line, (int)sizeof(line), stdin) != NULL) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) {
            continue;
        }

        char phonemes[NANO_G2P_PS_MAX];
        int plen = nano_g2p_text_to_phonemes(line, phonemes, (int)sizeof(phonemes));
        if (plen < 0) {
            printf("!ERROR\t%d\t%s\n", plen, nano_g2p_strerror(plen));
            fflush(stdout);
            continue;
        }

        int32_t ids[NANO_MAX_TOKENS];
        int n = nano_g2p_text_to_ids(line, ids, (int)(sizeof(ids) / sizeof(ids[0])));
        if (n < 0) {
            printf("!ERROR\t%d\t%s\n", n, nano_g2p_strerror(n));
            fflush(stdout);
            continue;
        }

        printf("%s\t", phonemes);
        for (int i = 0; i < n; i++) {
            printf("%d%s", (int)ids[i], (i + 1 < n) ? "," : "");
        }
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
