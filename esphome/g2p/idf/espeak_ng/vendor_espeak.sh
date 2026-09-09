#!/usr/bin/env bash
# Vendor the espeak-ng 1.52.0 translator subset into this component.
#
# Only the 27 libespeak-ng translation units the CMakeLists compiles, the 6
# ucd-tools units they need, and the headers those pull in -- not the ~2500-file
# upstream tree. The audio synthesiser is not compiled out (synthesize.c and
# wavegen.c are still in the list) because the translator links against their
# symbols; see README.md, which also records what that costs.
#
#   ./vendor_espeak.sh                       # clones into a temp dir
#   ./vendor_espeak.sh /path/to/espeak-ng    # reuse an existing 1.52.0 checkout
#
# Re-running is safe: vendored directories are removed and recreated.
#
# LICENCE: everything this copies is GPL-3.0-or-later. See README.md.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tag="1.52.0"

# The exact translation units mcu/ports/esp32s3/firmware/components/espeak-ng
# compiles, kept in that order so the two lists can be diffed.
readonly SRCS=(
    common.c mnemonics.c error.c ieee80.c
    compiledata.c compiledict.c
    dictionary.c encoding.c intonation.c
    langopts.c numbers.c phoneme.c
    phonemelist.c readclause.c setlengths.c
    soundicon.c spect.c ssml.c
    synthdata.c synthesize.c tr_languages.c
    translate.c translateword.c voices.c
    wavegen.c speech.c espeak_api.c
)
readonly UCD_SRCS=(case.c categories.c ctype.c proplist.c scripts.c tostring.c)

cleanup_dir=""
cleanup() { [ -n "${cleanup_dir}" ] && rm -rf "${cleanup_dir}"; }
trap cleanup EXIT

if [ $# -ge 1 ]; then
    upstream="$1"
    if [ ! -d "${upstream}/src/libespeak-ng" ]; then
        echo "error: ${upstream} does not look like an espeak-ng checkout" >&2
        exit 1
    fi
else
    cleanup_dir="$(mktemp -d)"
    upstream="${cleanup_dir}/espeak-ng"
    echo "cloning espeak-ng ${tag} into ${upstream}"
    git clone --quiet --branch "${tag}" --depth 1 \
        https://github.com/espeak-ng/espeak-ng.git "${upstream}"
fi

echo "vendoring from ${upstream}"
rm -rf "${here}/libespeak-ng" "${here}/ucd-tools" "${here}/include/espeak-ng" \
       "${here}/include/compat"
mkdir -p "${here}/libespeak-ng" "${here}/ucd-tools/include"

for src in "${SRCS[@]}"; do
    if [ ! -f "${upstream}/src/libespeak-ng/${src}" ]; then
        echo "error: missing ${src} in ${upstream}; wrong tag?" >&2
        exit 1
    fi
    cp "${upstream}/src/libespeak-ng/${src}" "${here}/libespeak-ng/"
done
# Every private header, because the 27 units include each other's freely and
# the set is only ~250 KB.
cp "${upstream}"/src/libespeak-ng/*.h "${here}/libespeak-ng/"

for src in "${UCD_SRCS[@]}"; do
    if [ ! -f "${upstream}/src/ucd-tools/src/${src}" ]; then
        echo "error: missing ucd-tools/${src} in ${upstream}" >&2
        exit 1
    fi
    cp "${upstream}/src/ucd-tools/src/${src}" "${here}/ucd-tools/"
done
cp -R "${upstream}/src/ucd-tools/src/include/." "${here}/ucd-tools/include/"

mkdir -p "${here}/include"
cp -R "${upstream}/src/include/espeak-ng" "${here}/include/"
cp -R "${upstream}/src/include/compat" "${here}/include/"

# Upstream licence texts travel with the source. Not optional for GPL code.
cp "${upstream}/COPYING" "${here}/COPYING.espeak-ng"
cp "${upstream}/ChangeLog.md" "${here}/ChangeLog.espeak-ng.md" 2>/dev/null || true

python3 "${here}/patch_speech.py" "${here}/libespeak-ng/speech.c"

echo
echo "vendored:"
echo "  libespeak-ng  $(ls "${here}"/libespeak-ng/*.c | wc -l | tr -d ' ') .c, $(ls "${here}"/libespeak-ng/*.h | wc -l | tr -d ' ') .h"
echo "  ucd-tools     $(ls "${here}"/ucd-tools/*.c | wc -l | tr -d ' ') .c"
echo "  total         $(du -sh "${here}" | cut -f1)"
echo
echo "next: esphome/g2p/host/build_espeak_host.sh   # proves the file set compiles"
