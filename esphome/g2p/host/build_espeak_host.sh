#!/usr/bin/env bash
# Compile the component's VENDORED espeak-ng source set on the host, with the
# component's own config.h and the same -std=gnu11 the IDF build uses, and link
# it with nano_g2p.c into a standalone binary.
#
# Why: there is no xtensa toolchain on this machine, so "does the IDF component
# build?" cannot be answered directly. This answers the part that actually
# varies -- whether the 33-file source list is complete and self-consistent,
# whether config.h covers every USE_* the sources test, and whether the
# patch_speech.py change compiles -- using the identical sources and flags.
# What it does NOT prove is the xtensa link: section placement, IRAM/DRAM
# attributes and toolchain-specific warnings are not exercised here.
#
#   ./build_espeak_host.sh
#
# Run esphome/g2p/idf/espeak_ng/vendor_espeak.sh first.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
component="${here}/../idf/espeak_ng"
out="${here}/nano_ids_vendored"

if [ ! -d "${component}/libespeak-ng" ]; then
    echo "error: espeak-ng is not vendored yet; run ${component}/vendor_espeak.sh" >&2
    exit 1
fi

srcs=("${component}"/libespeak-ng/*.c "${component}"/ucd-tools/*.c)
n=${#srcs[@]}
echo "compiling ${n} vendored espeak-ng translation units + nano_g2p.c"

# -std=gnu11 because espeak-ng is old C: IDF's default gnu23 turns its implicit
# int->pointer conversions into hard errors. -w because upstream is not warning
# clean and its warnings are not ours to fix. Both match the IDF CMakeLists.
#
# include/compat is on the path HERE ONLY. macOS has no <endian.h>, which
# speech.h includes; ESP-IDF's newlib does provide it, which is why the IDF
# CMakeLists (and the working firmware component it was copied from) leave
# compat off. Adding it to the IDF build would shadow newlib headers.
set -x
cc -std=gnu11 -Os -w -DHAVE_CONFIG_H \
   -I "${component}" \
   -I "${component}/libespeak-ng" \
   -I "${component}/include" \
   -I "${component}/ucd-tools/include" \
   -I "${component}/include/compat" \
   -o "${out}" \
   "${here}/nano_ids_host.c" \
   "${component}/nano_g2p.c" \
   "${srcs[@]}" \
   -lm -lpthread
set +x

echo "built ${out}"
echo
echo "section sizes (HOST arm64/macOS -- indicative only, NOT the xtensa figure):"
size "${out}" || true
