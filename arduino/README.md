# SanoTTS (Arduino / PlatformIO library)

On-device neural text-to-speech: the saanoTTS `mcu/` portable C99 int8
iSTFT engine (~745k parameters, ~680 KB weights), packaged as a standard
Arduino library and a PlatformIO package. Feed it Piper phoneme ids (not
text — see "Where phoneme ids come from" below); it synthesizes 22.05 kHz
PCM on-chip, no network, no cloud TTS call.

This is a **packaging** of the saanoTTS `mcu/` runtime, not a reimplementation.
The correctness story, the port architecture, and the honest per-chip
performance numbers all live upstream in the parent repo
(`docs/mcu-classes-and-porting.md`, `mcu/ports/*/README.md`) — this README
summarizes them for an Arduino/PlatformIO audience and is careful not to
repeat a number this library itself hasn't verified.

## Install

### Arduino IDE (zip)

1. Zip the `arduino/` directory's contents (`library.properties` must be
   at the zip root, not nested one level down — `cd arduino && zip -r
   ../SanoTTS.zip .`).
2. Sketch → Include Library → Add .ZIP Library... → select `SanoTTS.zip`.
3. `File → Examples → SanoTTS → SpeakGolden` to try the bundled example
   (after flashing the model blobs — see below, it will not run without
   them).

### PlatformIO

```ini
; platformio.ini
lib_deps =
    https://github.com/Ampixa/saanotts.git#master  ; whole-monorepo checkout
```

PlatformIO's dependency finder walks a git URL looking for
`library.json`/`library.properties`; since this library lives in a
subdirectory (`arduino/`) of the saanoTTS monorepo rather than at repo
root, a plain `lib_deps` git URL will pull the whole repo and PlatformIO's
library scanner needs to find `arduino/library.json` inside it — verify
this resolves for your PlatformIO version before relying on it, or vendor
the `arduino/` directory directly into your project's `lib/` folder
(`lib/SanoTTS/`) if it doesn't. The layout itself (`library.json` at the
directory's own root) is correct PlatformIO library format either way.

## Where phoneme ids come from

`SanoTTS::synthesize()` takes an array of **Piper phoneme ids** (`int32_t`),
not text. This library does no grapheme-to-phoneme (G2P) conversion.
Two ways to get ids:

1. **Host tool** (what `examples/SpeakGolden` does): phonemize offline on
   your dev machine (piper-phonemize, or espeak-ng), bake the id array
   into the sketch with `extras/gen_golden_ids.py`, or stream ids over
   serial/WiFi from a phonemizing server.
2. **On-chip espeak-ng G2P**: a real, working ESP32-S3 port exists in the
   parent repo but is **not part of this library (v1)** — it requires
   vendoring ~2500 files of espeak-ng source, an ESP-IDF-specific SPIFFS
   data-path patch, and `-std=gnu11` (Arduino's default C dialect for .c
   files may differ). See
   `mcu/ports/esp32s3/firmware/components/espeak-ng/README.md` in the
   saanoTTS repo if you want to wire it in yourself.

## What's in this library vs. what's a documented pointer only

| Feature | Status |
| --- | --- |
| 745k int8 fsd pipeline (duration → acoustic → iSTFT decoder) | **Shipped.** `src/snt_tts.c` + `src/snt_kernels_ref.c`, wrapped by the `SanoTTS` class. |
| Scalar (Tier-S) int8 kernels, every architecture | **Shipped.** Same reference kernels as `mcu/ports/host` and `mcu/ports/wasm` — correctness-identical by contract, no SIMD. |
| Optional ESP32 second-core worker | **Shipped**, opt-in. `src/snt_port_default.c`, behind `-DSANOTTS_ESP32_DUALCORE`; plain FreeRTOS task-notify, no esp-nn dependency. Call `tts.enableDualCore()`. |
| Sibilant fricative-noise injection (fixes whistly /s z sh zh/) | **Shipped**, off by default. Ported into `src/snt_tts.c` from the ESP32-S3 firmware reference (`mcu/ports/esp32s3/firmware/main/fsd_e2e.c`), which had never fed it back into the portable core — see "Sibilant injection" below. |
| Bigger-voice path: fp32 front + int8 (W8/A12) piperlite decoder | **Shipped** as source (`src/snt_front_f32.c`, `src/snt_piperlite_q8.c`) + a thin pass-through class `SanoTTSPiperLiteQ8`. ~1.0 MB decoder weights vs ~4 MB for the fp32 decoder variant (also shipped as source, `src/snt_piperlite.c`, not class-wrapped — call its C API directly on a PSRAM-class board). |
| int16 residual-chain activations (`SNT_INT_CHAIN`) | Present in the copied source (removing it risked silently drifting from upstream `mcu/`), but **do not define this macro** — the upstream implementation comment states it is disabled pending a precision redesign; it compiles, it is not correct. See `SanoTTS.h`'s macro table. |
| esp-nn PIE SIMD kernels + hand Xtensa asm (the ESP32-S3 **0.22× RT** measurement) | **Documented pointer only.** Requires vendoring `espressif__esp-nn` and a `.S` file as an ESP-IDF component — see `mcu/ports/esp32s3/README.md`. This library's ESP32-S3 experience out of the box is the scalar port above, not this number. |
| On-chip espeak-ng G2P | **Documented pointer only** (see above). |
| PSRAM weight-bank placement, internal-SRAM arena reservation, PSRAM-stack FreeRTOS worker tasks (`CONFIG_SPIRAM_ALLOW_STACK_EXTERNAL_MEMORY`) | **Documented pointer only** — firmware-level ESP-IDF build configuration from the full standalone-app reference, not something an Arduino sketch controls. See "ESP32-S3 specifics" below and `mcu/ports/esp32s3/README.md` / `mcu/ports/esp32s3/firmware/`. |

## Sibilant injection

The deterministic acoustic student regresses the *mean* 40-dim latent, so
broadband-noise sibilants (/s z ʃ ʒ/) collapse to a whistly tone. Injecting
per-channel Gaussian noise (scaled by that voice's own teacher-latent std)
at sibilant frames, before quantization, restores the hiss the decoder is
otherwise perfectly capable of rendering. Off by default; enable per-voice:

```cpp
tts.enableSibilantInjection(tea_std, sib_ids, n_sib_ids, 0.9f);
```

`tea_std` (40 floats) and `sib_ids` are **voice-specific** calibration —
they come from that voice's `sibilant-injection/calib.npz`
(`tools/calibrate_sibilant_noise.py` in the parent repo), not a universal
constant. The `en_US Kristin r7` values used by this library's own
verification (`extras/host_check_main.c`) are: sibilant ids `{31, 38, 96,
108}`, `beta ≈ 0.9` (host-verified: sibilant 2–8 kHz spectral flatness
0.597 → 0.686 against a teacher reference of 0.689).

## Compile-time flags

Full detail (including which are golden-gate-proven vs. experimental) is
in the block comment at the top of `src/SanoTTS.h`. Short version: define
`FSD_FAST_MATH` (the only combination this library's own correctness check
is proven under); everything else (`FSD_FROZEN_NORM`, `SNT_ACT_LUT`,
`SNT_INT_FFT`, `FSD_FULL_FFT`, `SNT_PROF`) is optional and should be
measured on your board before shipping with it on. **Never define
`SNT_INT_CHAIN`** — see the table above.

## Memory requirements

Weights: **~680 KB** for the shipped `en_US Kristin` 745k voice
(`front_q8.bin` ~280 KB + `model_q8.bin` ~400 KB) — flash/PROGMEM-resident;
the runtime reads them via pointer and never copies the whole thing into
RAM (it stages small resident working copies out of the arena as it goes).

Arena (working RAM): scales with utterance length. Measured on host by
binary-searching the golden fixture (a ~1.5 s / 134-frame utterance) down
to the smallest arena that doesn't abort: **it still succeeds at 100 KB**.
`SanoTTS::recommendedArenaBytes()` returns **320 KB** (matching the size
`mcu/test/golden_main.c`'s own gate is proven under) — a safe default with
headroom, not the true minimum for every utterance length. If you need to
support long utterances (`docs/mcu-classes-and-porting.md`'s whole-
utterance figure of ~300 KB is calibrated closer to the ESP32-S3 firmware's
~7 s cap), plan capacity against your actual max utterance length, not the
short golden demo.

| Board | Fits? | Basis |
| --- | --- | --- |
| **ESP32-S3** | ✓ | 512 KB internal SRAM comfortably clears the arena floor; weights are flash-XIP, no PSRAM required for the bare library. PSRAM becomes relevant once you add WiFi + on-chip G2P + larger buffers alongside it (the full firmware reference does; see below), or if you use the bigger-voice piperlite path. |
| **Plain ESP32** (original, Xtensa LX6) | ✓ (memory) | 520 KB internal SRAM, same margin as S3. **Not classified** in `docs/mcu-classes-and-porting.md`'s locked MCU-tier table (only S3/C3/P4/M55/M85/M7/M4/RP2040/Ethos are) — this library ships the same scalar kernels for it as every non-SIMD-ported architecture, so treat its *speed* as unmeasured, not the ESP32-S3's proven 0.22× RT figure (that number needs the esp-nn PIE port, not shipped here). |
| **ESP32-C3** | ✓ (memory) | Tier S, **measured 5.72× RT** upstream (offline, i.e. ~5.7× slower than the audio's own duration — not real-time) — the float glue dominates on a chip with no FPU-adjacent DSP, not the arena. |
| **RP2040** | Marginal | 264 KB **total** SRAM vs. a measured ~100–150 KB arena floor for a short utterance — the arena alone can fit, but that leaves little to nothing for the stack, the id buffer, PCM output, and USB/serial, on a chip with nothing else to borrow from. Tier S, "ported, unmeasured" upstream — no RTF number exists yet either. Treat as a proof-of-concept target, not a shipped one. |

## Model blobs

**As of this writing, there is no standalone GitHub Release asset that is
just the two MCU blob files** (`front_q8.bin` + `model_q8.bin`, ~680 KB) —
checked directly against the repo's actual releases
(`gh release list --repo Ampixa/saanotts`): the existing releases ship
either fp16 desktop/browser weights (a different format from these int8
MCU blobs) inside multi-MB tarballs, or a 236 MB generic preservation
snapshot. Do not assume a clean download URL exists; there isn't one yet.

Until a dedicated release is cut, get the blobs from a saanoTTS checkout:

- `mcu/test/fixtures/en_us_r7/front_q8.bin` + `model_q8.bin` — the exact
  pair `examples/SpeakGolden` and this library's own `extras/host_check.sh`
  are verified against (SHA256SUMS included alongside them).
- `releases/kristin-20260708/mcu/front_q8.bin` + `model_q8.bin` — the same
  voice's "corrected MCU package" plus `en_us_r7_calibration/` (scale
  headers + golden I/O vectors), if that local release directory has been
  published in the checkout you're working from (it is untracked in the
  repo as of this writing, so a fresh `git clone` alone will not have it).

Publishing a small, standalone `arduino-blobs-<voice>` release (just the
two files + a checksum) would be the natural next step to make this
library installable without a full monorepo clone.

## ESP32-S3 specifics (firmware-level, not library code)

The full standalone-app reference (`mcu/ports/esp32s3/firmware/`) that
achieves the measured 0.22× RT and runs WiFi + on-chip espeak-ng G2P +
this engine simultaneously relies on ESP-IDF build configuration this
Arduino library does not set for you:

- **PSRAM weight-bank placement** — large buffers (whole-utterance PCM,
  espeak's own working set) explicitly allocated in PSRAM
  (`heap_caps_malloc(..., MALLOC_CAP_SPIRAM)`), freeing internal SRAM for
  the arena and SIMD-resident weight staging.
- **Internal-SRAM arena reservation** — the compute arena is deliberately
  kept in internal SRAM (`MALLOC_CAP_INTERNAL`), because the SIMD dot
  kernels only run on internal-SRAM pointers (PIE vector loads from flash
  XIP silently return garbage — measured, corr 0.011 — and PSRAM is data-
  cache-mapped so it's scalar-safe but not SIMD-safe either without the
  same residency gate).
- **PSRAM-stack worker tasks** (`CONFIG_SPIRAM_ALLOW_STACK_EXTERNAL_MEMORY`)
  — lets a FreeRTOS task's stack itself live in PSRAM, needed once internal
  SRAM is under enough pressure (WiFi + espeak + the engine together) that
  a normal internal-SRAM task stack won't fit.

None of this is Arduino-sketch-controllable in a portable way; if you need
it, build against ESP-IDF directly using `mcu/ports/esp32s3/firmware/` as
the reference, not this Arduino library.

## Compile verification

```bash
cd arduino/extras
./host_check.sh
```

Three phases, all passing as of this packaging:

1. Every `src/*.c` file compiles standalone with a plain `cc -std=c99`
   (proves no hidden Arduino/ESP-IDF header dependency).
2. `SanoTTS.cpp` compiles standalone with a plain C++ compiler.
3. The primary pipeline, linked against the golden fixture
   (`mcu/test/fixtures/en_us_r7`): the raw C API reproduces the upstream
   `mcu/` correlation gate exactly (**corr 0.989148**, identical to `cd mcu
   && make test`'s own figure — confirming the sibilant-injection patch is
   a true no-op when disabled), and the sibilant-injection addition is
   confirmed reachable and non-trivial (RMS diff 0.017 against the same
   input with it off). The `SanoTTS` C++ class is then exercised the same
   way end to end (`begin()` → `synthesize()` → `enableSibilantInjection()`).

**`arduino-cli` / full ESP32 toolchain compile**: `arduino-cli` was
installed and the ESP32 core install was started, but it ran out of disk
space in this environment partway through (the combined esp32/esp32c3/
esp32c5/esp32c6/esp32h2 toolchain set is several GB) and was not completed.
This is the explicitly-sanctioned fallback path for that situation: the
host-level self-containment + correctness check above is the verification
this packaging relies on. If you have a few GB free, `arduino-cli core
install esp32:esp32` followed by `arduino-cli compile --fqbn
esp32:esp32:esp32s3 examples/SpeakGolden` (after providing the model blobs
per "Model blobs" above, since the sketch will otherwise fail at runtime,
not compile time, without them) is the natural next check to run.

## Distribution channels

| Channel | Status today | How to get the library |
| --- | --- | --- |
| Arduino IDE (.zip) | works now | zip the `arduino/` directory, Sketch → Include Library → Add .ZIP Library (see "Install" above) |
| PlatformIO (`lib_deps` git URL) | works now | `lib_deps = https://github.com/Ampixa/saanotts.git#master` (see "Install" above) |
| Arduino Library Manager | not yet submitted | none — no registry listing exists yet |
| PlatformIO Registry (`pio pkg install`) | not yet published | none — no registry listing exists yet |
| ESP-IDF Component Registry | not yet published | none — `idf_component.yml` manifest is prepared (`arduino/idf_component.yml`) but nothing has been uploaded |

### Publishing to registries (maintainer note)

None of the three package-manager registries below have had anything
submitted or uploaded yet — this is a plan, not a status report.

- **Arduino Library Manager** — tag a GitHub release of this repo, then
  submit a one-time pull request adding the repo URL
  (`https://github.com/Ampixa/saanotts`) to
  [arduino/library-registry](https://github.com/arduino/library-registry).
  Arduino's indexer re-scans the repo on every subsequent tag, so only the
  first submission needs a PR. Run `arduino-lint` against `arduino/` before
  submitting (see "Compile verification" above for why the full
  `arduino-cli` toolchain wasn't installed in this environment).
- **PlatformIO Registry** — `pio pkg publish` run from inside `arduino/`
  (needs a PlatformIO account; `library.json` in this directory is already
  in the required format).
- **ESP-IDF Component Registry** — `compote component upload --name
  sanotts --namespace ampixa` (needs an Espressif account and an
  `IDF_COMPONENT_API_TOKEN`); manifest is `arduino/idf_component.yml`. Note
  its `license: "GPL-3.0"` matches this directory's own
  `library.properties`/`library.json`, but that bare SPDX identifier is
  deprecated in favor of `GPL-3.0-only` or `GPL-3.0-or-later` — resolve
  which one this project actually intends (the `sanotts` PyPI package's
  `pypkg/pyproject.toml` already says `GPL-3.0-or-later`, a different
  choice than this directory's own manifests) before the first real
  upload, since the component registry may validate against current SPDX
  identifiers strictly.

## License

GPL-3.0 (see `LICENSE`).
