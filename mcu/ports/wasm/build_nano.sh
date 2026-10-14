#!/usr/bin/env bash
# Build one mel-100 / TinyVocos lineage (snt_nano.c) to WebAssembly.
#
#   bash mcu/ports/wasm/build_nano.sh <lineage> <voice> <ExportName> [int8|f32]
#
#   lineage     mcu/models/<lineage>/nano_q8_meta.h is compiled in (shapes and
#               byte offsets are compile-time constants in snt_nano.c)
#   voice       output name: web/snt_nano_<voice>.js (+ .wasm)
#   ExportName  the MODULARIZE'd factory the page calls, e.g. SaanoNanoHeart
#   int8|f32    weight element type of the blobs this module will be handed
#               (default int8). f32 adds -DSNT_NANO_W_F32 (float rows + float
#               activations); the header's NANO_WEIGHT_FORMAT makes a mismatch
#               a compile error rather than a silent misread.
#
# The two browser voices:
#   bash mcu/ports/wasm/build_nano.sh en_us_e13b    heartnano SaanoNanoHeartNano int8
#   bash mcu/ports/wasm/build_nano.sh en_us_r227f32 heart     SaanoNanoHeart     f32
#
# No weights are baked in: the page fetches web/voices/<voice>/*.bin at
# runtime and passes the raw bytes to snt_nano_wasm_synthesize.
#
# Memory: FIXED, no ALLOW_MEMORY_GROWTH, for the reason build_voices.sh gives
# (JS holds typed-array views into the heap across malloc-heavy calls; growth
# would detach them). 128 MB covers the largest blob pair (9.1 MB f32 for the
# release stack) plus a 16 MB arena and a 20 s output buffer several times.
#
# Requires Emscripten (emcc) on PATH: https://emscripten.org
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "usage: $0 <lineage> <voice> <ExportName> [int8|f32]" >&2
  exit 2
fi
lineage=$1; voice=$2; export_name=$3; weights=${4:-int8}

here="$(cd "$(dirname "$0")" && pwd)"
mcu="$(cd "$here/../.." && pwd)"          # .../mcu
repo="$(cd "$mcu/.." && pwd)"             # repo root
web="$repo/web"
model="$mcu/models/$lineage"

command -v emcc >/dev/null 2>&1 || { echo "emcc not found on PATH -- install Emscripten" >&2; exit 1; }
[ -f "$model/nano_q8_meta.h" ] || { echo "missing $model/nano_q8_meta.h" >&2; exit 1; }

case "$weights" in
  int8) wflag="" ;;
  f32)  wflag="-DSNT_NANO_W_F32" ;;
  *) echo "weights must be int8 or f32, got '$weights'" >&2; exit 2 ;;
esac

mkdir -p "$web"

# -O3, libm math (NOT FSD_FAST_MATH / SNT_NANO_FAST_MATH): the gate the
# fixtures are held to is `make test-nano` / `test-nano-wf32`, the libm path.
# shellcheck disable=SC2086
emcc \
  -O3 -std=c99 -D_GNU_SOURCE $wflag \
  -I"$mcu/include" -I"$mcu/src" -I"$model" \
  "$mcu/src/snt_nano.c" \
  "$mcu/src/snt_kernels_ref.c" \
  "$here/snt_nano_wasm.c" \
  -sMODULARIZE=1 -sEXPORT_NAME="$export_name" \
  -sINITIAL_MEMORY=134217728 \
  -sEXPORTED_FUNCTIONS='_malloc,_free,_snt_nano_wasm_synthesize,_snt_nano_wasm_set_length_scale,_snt_nano_wasm_seed_from_text,_snt_nano_wasm_sample_rate,_snt_nano_wasm_weight_format,_snt_nano_wasm_last_frames,_snt_nano_wasm_last_arena_peak,_snt_nano_wasm_last_rc' \
  -sEXPORTED_RUNTIME_METHODS='cwrap,HEAPU8,HEAP32,HEAPU32,HEAPF32,stringToUTF8,lengthBytesUTF8' \
  -sENVIRONMENT=web,worker,node \
  -o "$web/snt_nano_$voice.js"

echo "built $web/snt_nano_$voice.js ($(wc -c <"$web/snt_nano_$voice.wasm") bytes wasm, $weights weights, lineage $lineage)"
echo "verify with:  node $here/verify_nano_node.mjs $voice $lineage"
