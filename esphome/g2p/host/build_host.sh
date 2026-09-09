#!/usr/bin/env bash
# Build the host parity harness: the unmodified component nano_g2p.c linked
# against a host libespeak-ng. Proves the C chain on a real machine before any
# ESP-IDF toolchain is involved.
#
#   ./build_host.sh                 # uses the Homebrew espeak-ng
#   ESPEAK_PREFIX=/usr ./build_host.sh
#
# Homebrew's espeak-ng is 1.52.0, the same version the ESP32-S3 component
# vendors, so the phoneme output is comparable. Verify with `espeak-ng --version`.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
component="${here}/../idf/espeak_ng"
prefix="${ESPEAK_PREFIX:-/opt/homebrew}"
out="${here}/nano_ids_host"

if [ ! -f "${prefix}/include/espeak-ng/speak_lib.h" ]; then
    echo "error: no espeak-ng headers under ${prefix} (brew install espeak-ng)" >&2
    exit 1
fi
if [ ! -f "${component}/nano_g2p.c" ]; then
    echo "error: component source missing at ${component}/nano_g2p.c" >&2
    exit 1
fi

set -x
cc -std=gnu11 -O2 -Wall -Wextra -Wno-unused-parameter \
   -I "${component}/include" \
   -I "${component}" \
   -I "${prefix}/include" \
   -o "${out}" \
   "${here}/nano_ids_host.c" \
   "${component}/nano_g2p.c" \
   -L "${prefix}/lib" -lespeak-ng
set +x

echo "built ${out}"
echo "run: ${out} ${prefix}/share/espeak-ng-data < sentences.txt"
