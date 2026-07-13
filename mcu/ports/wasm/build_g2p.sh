#!/usr/bin/env bash
# Build the espeak-ng G2P (text -> Kristin phoneme ids) module to WebAssembly.
#
# Produces web/snt_g2p.{js,wasm,data} -- a SEPARATE Emscripten module from
# the synth runtime (build.sh / web/snt_tts.*). The two never touch: the
# browser page loads both and wires G2P output straight into the synth's
# id buffer.
#
# Sources are the same 27 libespeak-ng translator .c files + 6 ucd-tools
# files used by the ESP32-S3 port (see
# mcu/ports/esp32s3/firmware/components/espeak-ng/CMakeLists.txt), compiled
# -std=gnu11 for the same reason as that port: espeak-ng is old C
# (K&R-isms, implicit int->pointer) and gnu11 keeps those as warnings
# instead of hard errors under a stricter default.
#
# Requires Emscripten (emcc) on PATH:  https://emscripten.org
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
mcu="$(cd "$here/../.." && pwd)"          # .../mcu
repo="$(cd "$mcu/.." && pwd)"             # repo root
web="$repo/web"
esp="$here/espeak"
# Multi-language data set: the ESP32 minimal en set (piper-tts provenance,
# 4KB phondata stub) + vi/id/ne/hi/cmn dicts and voice files taken from
# piper-tts 1.4.2's bundled espeak-ng-data (byte-identical phontab), so the
# WASM ids match python PiperVoice exactly. See verify_g2p_node.mjs gate.
data="$here/espeak-data-multi"

command -v emcc >/dev/null 2>&1 || { echo "emcc not found on PATH -- install Emscripten"; exit 1; }
[ -d "$esp/libespeak-ng" ] || { echo "missing $esp/libespeak-ng -- see README.md setup steps"; exit 1; }
[ -f "$data/phontab" ] || { echo "missing $data/phontab -- espeak-ng-data not found"; exit 1; }

mkdir -p "$web"

lib="$esp/libespeak-ng"
ucd="$esp/ucd-tools"

LIBESPEAK_SRCS=(
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
UCD_SRCS=(case.c categories.c ctype.c proplist.c scripts.c tostring.c)

srcs=()
for f in "${LIBESPEAK_SRCS[@]}"; do srcs+=("$lib/$f"); done
for f in "${UCD_SRCS[@]}"; do srcs+=("$ucd/$f"); done

# -std=gnu11              espeak-ng old-C quirks (implicit int->ptr under c99/gnu23)
# -DHAVE_CONFIG_H         pulls in espeak/config.h (USE_*=0, PATH_ESPEAK_DATA)
# -I espeak/include       espeak-ng/*.h public headers
# -I espeak/ucd-tools/include   ucd/*.h
# -I espeak, -I lib       config.h + private libespeak-ng headers ("." + "libespeak-ng")
# --preload-file          stages the ~2.2MB 6-language data set at /espeak in
#                          the virtual FS (emitted as web/snt_g2p.data, fetched
#                          by web/snt_g2p.js at load time)
#
# Memory: FIXED at 32MB (no -sALLOW_MEMORY_GROWTH). The preload-file loader
# calls TextDecoder.decode() on views over the module heap, and Chrome
# rejects decode() over a *resizable* WebAssembly.Memory buffer
# ("The provided ArrayBuffer value must not be resizable"). The synth module
# (build.sh) keeps growth because it has no .data preload; here espeak's
# translator + the ~2.2MB data FS peak well under 32MB, so a fixed heap
# costs nothing and sidesteps the clash entirely.
emcc \
  -O2 -std=gnu11 -w -DHAVE_CONFIG_H \
  -I"$esp" -I"$esp/include" -I"$esp/ucd-tools/include" -I"$lib" \
  "${srcs[@]}" \
  "$here/snt_g2p_wasm.c" \
  --preload-file "$data@/espeak" \
  -sMODULARIZE=1 -sEXPORT_NAME=SaanoG2P \
  -sINITIAL_MEMORY=33554432 \
  -sEXPORTED_FUNCTIONS='_malloc,_free,_snt_g2p_init,_snt_g2p_set_voice,_snt_g2p_text_to_ids' \
  -sEXPORTED_RUNTIME_METHODS='cwrap,HEAP32,HEAPU8,stringToUTF8,lengthBytesUTF8' \
  -sENVIRONMENT=web,worker,node \
  -o "$web/snt_g2p.js"

echo "built $web/snt_g2p.js ($(wc -c <"$web/snt_g2p.wasm") bytes wasm, $(wc -c <"$web/snt_g2p.data") bytes data)"
