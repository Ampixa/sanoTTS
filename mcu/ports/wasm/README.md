# WebAssembly port — sanoTTS in the browser

The full text→PCM stack compiled to WebAssembly and run entirely client-side:
no server, no network after the initial asset load, no WebGPU. The browser is
just another **Tier-S** target (flat address space, no second core, no SIMD we
rely on), so the port is tiny — the same `snt_tts.c` core and the same scalar
reference kernels the host golden gate compiles, plus three port shims and one
exported entry.

## What's here

| File | Role |
| --- | --- |
| `snt_port_wasm.c` | `snt_par_run` (serial), `snt_scratch_id`→0, `snt_now_us`, and the `snt_web_synthesize` browser entry |
| `build.sh` | `emcc` invocation → `web/snt_tts.{js,wasm}`; stages model + golden assets into `web/assets/` |
| `verify_node.mjs` | the WASM golden gate: runs the golden vector headless under Node, asserts corr > 0.98 |

The int8/int16 kernels and `snt_weights_resident` come from
`src/snt_kernels_ref.c` unchanged — everything WASM linear memory holds is
readable, so residency is trivially true and no staging is needed. Correctness
is therefore identical to the host port *by construction*: same core, same
kernels, same inputs.

## Build

Requires [Emscripten](https://emscripten.org) (`emcc` on `PATH`):

```bash
bash mcu/ports/wasm/build.sh
```

Produces `web/snt_tts.js` (~11 KB glue) + `web/snt_tts.wasm` (~46 KB) and copies
the R7 en_US model blobs + golden vectors into `web/assets/`.

## Validate (the golden gate)

```bash
node mcu/ports/wasm/verify_node.mjs
# samples 100096 (4.54s audio)  wall 0.083s  54.4x RT on this CPU
# golden corr 0.987425 vs PyTorch reference
# PASS
```

This is the WASM equivalent of `test/golden_main.c`: same 163-token utterance,
same reference durations, correlation against the same PyTorch float audio. It
drives the exact export the browser page calls, so a pass here means the page
produces bit-identical PCM.

## Run the demo

```bash
cd web && python3 -m http.server 8000
# open http://localhost:8000/
```

The page fetches the blobs, synthesizes in-page, plays the PCM through WebAudio,
and shows a **live Pearson correlation** of this run's output against the
PyTorch reference — the golden gate, computed in your browser.

## Honest framing

The speed the page reports is **this browser on your CPU** (a laptop runs this
~50–60× real time), *not* a microcontroller measurement. The demo's point is
that the identical bit-exact runtime executes client-side; the per-device MCU
numbers (ESP32-S3 at 0.22× real time, etc.) are in the top-level README and the
systems paper, and must not be conflated with the browser figure.

## API

```c
int snt_web_synthesize(const uint8_t *front, const uint8_t *dec,
                       const int32_t *ids, int n_ids,
                       const int32_t *durs,          /* NULL => predicted */
                       uint8_t *arena, int arena_size,
                       float *out, int out_cap);     /* returns #samples, <0 err */
```

All pointers are byte offsets into WASM linear memory. JavaScript `malloc()`s
each region, copies the blobs / IDs in, and reads the PCM back out of `out`.
Pass the reference durations for a frame-exact match to the golden audio, or
`0`/`NULL` to run the model's own duration head.

## The known gap: no text frontend

This runs **phoneme IDs → PCM**. The Devanagari/grapheme→phoneme frontend is a
separate piece and is not compiled here — the demo drives the runtime with the
embedded golden phoneme-ID sequence. Text-in from an arbitrary string is a
second, separable layer.

## The mel-100 lineage: `snt_nano.c` in the browser (heart, heart-nano)

`build_nano.sh` compiles `src/snt_nano.c` -- the 294k nano stacks and the
2,272,145-parameter release stack -- to one module PER LINEAGE, because that
runtime bakes every shape and byte offset in from the generated
`mcu/models/<lineage>/nano_q8_meta.h`:

```bash
bash mcu/ports/wasm/build_nano.sh en_us_e13b    heartnano SaanoNanoHeartNano int8
bash mcu/ports/wasm/build_nano.sh en_us_r227f32 heart     SaanoNanoHeart     f32
node mcu/ports/wasm/verify_nano_node.mjs heartnano en_us_e13b     # min corr 0.981, PASS
node mcu/ports/wasm/verify_nano_node.mjs heart     en_us_r227f32  # min corr 1.000000, PASS
```

| file | role |
| --- | --- |
| `snt_nano_wasm.c` | port shims + `snt_nano_wasm_synthesize` (front, dec, ids, optional frozen durations, 64-bit seed as two halves, arena, out) + `snt_nano_wasm_set_length_scale` |
| `build_nano.sh` | `emcc` -> `web/snt_nano_<voice>.js`; `f32` adds `-DSNT_NANO_W_F32` |
| `verify_nano_node.mjs` | the fixture gate under Node: every `mcu/test/fixtures/<lineage>` row with its frozen durations and seed, minimum Pearson vs the float PyTorch reference, 0.98 |

Two builds of the same C:

- **int8** (heart-nano): the device math path unchanged -- int8 weight blobs,
  dynamically quantised int8 activations, the scalar reference kernels. Same
  numbers as `make -C mcu test-nano` up to libm differences in the last digit.
- **f32** (heart): `tools/export_e12_nano_q8.py --weights f32` writes the same
  regions as float32 rows with unit scales, and `SNT_NANO_W_F32` reads them with
  float activations. Needed because the release stack measures **0.951** minimum
  correlation under int8 (below the 0.98 gate) and **1.000000** here. The header's
  `NANO_WEIGHT_FORMAT` makes an int8/f32 mismatch a compile error.

The phoneme path is the trellis one (`trellis_frontend.js`): misaki-normalised
espeak-ng IPA into the 62-symbol corpus vocabulary, which is the same object for
Trellis-RIFT and for every mel-100 stack (`artifacts/kokoro-corpus-af_heart-20260713/kokoro_vocab.json`,
checked entry by entry). The decoder-noise seed is `sha256(text)[:8]`, the
renderer's own convention, exposed as `snt_nano_wasm_seed_from_text`.

The speaking rate is `snt_nano_wasm_set_length_scale(scale)`, which divides every duration the
duration student predicts: 0 (the initial value) is the model's own pace, above 1 is slower and
below 1 is faster. The module keeps the value between calls, so a caller sets it before each
`snt_nano_wasm_synthesize`; it has no effect when frozen durations are passed.
