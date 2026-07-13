#!/usr/bin/env bash
# Build the all-voices front+decoder synth chain to WebAssembly.
#
# Produces web/snt_voice.js (+ .wasm) exporting SaanoVoice: a MODULARIZE'd
# Emscripten module wrapping snt_voice_synthesize (ids -> durations -> latent
# -> audio, chaining snt_front_f32.c + snt_piperlite.c) plus the debug-only
# snt_voice_synthesize_from_latent (decoder in isolation). No model weights
# are baked in or preloaded -- the page fetches each voice's
# web/voices/<key>/{front_f32.bin,dec_f32.bin} at runtime and passes the raw
# bytes in.
#
# Memory: FIXED at 256MB, no ALLOW_MEMORY_GROWTH. This module has no
# --preload-file (unlike snt_g2p.js), so growth would normally be safe, but
# a fixed generous heap sidesteps realloc-driven detached-buffer bugs when
# JS holds onto typed-array views into HEAPU8/HEAPF32 across an
# _malloc-heavy synth call, and 256MB comfortably covers the largest
# release voice (chinese/hindi front+dec blobs are ~3.4MB + ~2.5MB; the
# malloc'd scratch arenas inside snt_voice_synthesize are the dominant use,
# a few MB per call for a multi-second sentence).
#
# Requires Emscripten (emcc) on PATH: https://emscripten.org
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
mcu="$(cd "$here/../.." && pwd)"          # .../mcu
repo="$(cd "$mcu/.." && pwd)"             # repo root
web="$repo/web"

command -v emcc >/dev/null 2>&1 || { echo "emcc not found on PATH -- install Emscripten"; exit 1; }

front_src="$mcu/src/snt_front_f32.c"
if [ ! -f "$front_src" ]; then
  echo "missing $front_src -- the front half (agent B) hasn't landed yet." >&2
  echo "This build cannot produce a working snt_voice_synthesize without it." >&2
  exit 1
fi

mkdir -p "$web"

emcc \
  -O3 -std=c99 -D_GNU_SOURCE \
  -I"$mcu/include" -I"$mcu/src" \
  "$front_src" \
  "$mcu/src/snt_piperlite.c" \
  "$here/snt_voice_wasm.c" \
  -sMODULARIZE=1 -sEXPORT_NAME=SaanoVoice \
  -sINITIAL_MEMORY=268435456 \
  -sEXPORTED_FUNCTIONS='_malloc,_free,_snt_voice_synthesize,_snt_voice_synthesize_from_latent' \
  -sEXPORTED_RUNTIME_METHODS='cwrap,HEAPU8,HEAP32,HEAPF32' \
  -sENVIRONMENT=web,worker,node \
  -o "$web/snt_voice.js"

echo "built $web/snt_voice.js ($(wc -c <"$web/snt_voice.wasm") bytes wasm)"
echo "voice weight bundles: run tools/export_voice_bundle.py to (re)populate $web/voices/<key>/"
echo "serve with:  (cd $web && python3 -m http.server 8000)  then open http://localhost:8000/"
