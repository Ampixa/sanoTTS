#!/usr/bin/env bash
# Build the saanotts-mcu runtime to WebAssembly and stage the web demo.
#
# Produces web/snt_tts.js (+ .wasm) from the exact same core + reference
# kernels the host golden gate compiles, plus the WASM port. No model-
# specific code: the browser is Tier S with a flat address space.
#
# Requires Emscripten (emcc) on PATH:  https://emscripten.org
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
mcu="$(cd "$here/../.." && pwd)"          # .../mcu
repo="$(cd "$mcu/.." && pwd)"             # repo root
web="$repo/web"
golden="${GOLDEN:-$repo/esp32c3/fsd/golden}"

command -v emcc >/dev/null 2>&1 || { echo "emcc not found on PATH -- install Emscripten"; exit 1; }

mkdir -p "$web/assets"

# --- compile the runtime to WASM -------------------------------------------
# -O3                    the scalar kernels are the whole hot loop
# FSD_FAST_MATH          same math path the golden gate validates
# ALLOW_MEMORY_GROWTH    model blobs + arena + PCM are allocated from JS
# MODULARIZE/EXPORT_NAME factory function, no globals leaked onto window
emcc \
  -O3 -std=c99 -D_GNU_SOURCE -DFSD_FAST_MATH \
  -I"$mcu/include" -I"$mcu/src" \
  "$mcu/src/snt_tts.c" \
  "$mcu/src/snt_kernels_ref.c" \
  "$here/snt_port_wasm.c" \
  -sMODULARIZE=1 -sEXPORT_NAME=SaanoTTS \
  -sALLOW_MEMORY_GROWTH=1 \
  -sEXPORTED_FUNCTIONS='_malloc,_free,_snt_web_synthesize' \
  -sEXPORTED_RUNTIME_METHODS='cwrap,HEAPU8,HEAP32,HEAPF32' \
  -sENVIRONMENT=web,worker,node \
  -o "$web/snt_tts.js"

echo "built $web/snt_tts.js ($(wc -c <"$web/snt_tts.wasm") bytes wasm)"

# --- stage the model + golden assets the page fetches ----------------------
for f in front_q8.bin model_q8.bin e2e_ids.bin e2e_durs.bin e2e_audio.bin; do
  cp "$golden/$f" "$web/assets/$f"
done
echo "staged model + golden assets into $web/assets/"
echo "serve with:  (cd $web && python3 -m http.server 8000)  then open http://localhost:8000/"
