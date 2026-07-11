# saanotts-mcu — portable neural TTS runtime for microcontrollers

One C99 core, one small port API, any MCU. The full saanoTTS stack
(phoneme IDs -> duration -> acoustic -> iSTFT decoder -> PCM) in int8,
validated bit-exact against the PyTorch reference on every platform via
embedded golden vectors.

To our knowledge this is the first **real-time** full neural TTS stack
(text -> PCM) to run on a general-purpose microcontroller with **no
dedicated neural accelerator**: real-time on the ESP32-S3 using only its
int8 SIMD (0.22x real-time, 4.5x faster than playback). Prior on-device
neural TTS either required an NPU (TinyTTS: Cortex-M55 + Ethos-U55) or ran
vocoder-only (TinyVocos: ARM Cortex-M, mel->waveform). The same core runs
the complete stack offline / near-real-time on RISC-V (ESP32-C3); the C3 is
not a real-time target. We make no "smallest model" or unqualified "first"
claim -- the interesting results are the measured per-tier numbers below and
the honest evaluation behind them.

## Measured workload (the numbers that define feasibility)

| Requirement | whole-utterance | streaming (planned) |
| --- | --- | --- |
| compute | ~45 MMAC/s of int8 per second of audio | same |
| weights (int8, R7 en_US stack) | 640 KB flash, SRAM-resident copies staged | same |
| RAM working set | ~300 KB | ~130-160 KB |
| float ops | glue only (scales, norm, iSTFT); FPU strongly recommended | same |

Reference point: ESP32-S3 (dual LX7 @ 240 MHz, PIE int8 SIMD) runs this
at **0.22x real-time** (4.5x faster than playback), output correlation
0.985 vs the float reference.

## MCU classes

**Tier V — vector int8 (real-time, margin).**
128-bit int8 SIMD + >=400 KB SRAM. The three kernels map to native
vector MACs (16+ MAC/cycle).
- ESP32-S3 (Xtensa PIE): **reference port, measured 0.22x RT**
- Cortex-M55/M85 (Helium/MVE): Renesas RA8, Alif Ensemble — projected
  <0.1x RT at 400-480 MHz via CMSIS-NN or hand MVE
- ESP32-P4 (RV32 + vendor SIMD): pending esp-nn support check

**Tier D — dual-MAC DSP (real-time, tight).**
SMLAD-class int16/int8 dual MACs, >=512 KB SRAM.
- Cortex-M7 @ 480-600 MHz (STM32H7, Teensy 4.x, i.MX RT): projected
  0.2-0.5x RT with SXTB16+SMLAD int8 kernels
- Cortex-M4 @ >=168 MHz: offline / near-RT for short utterances

**Tier S — scalar (near-RT to offline, correctness-identical).**
Any 32-bit MCU with the RAM floor. Reference C kernels, no assembly.
- ESP32-C3 class (RV32IMC 160 MHz, no FPU): projected 1.5-2x RT with
  the int16-activation refactor; today's float glue makes it slower
- Anything else that can hold the working set

**Tier N — NPU offload (quality scaling, future).**
Ethos-U55 (Alif, Renesas RA8P1, Himax WE2), ST Neural-ART (STM32N6).
256 MAC/cycle class: run a *bigger, better* model in real-time instead
of a faster small one. Needs static activation scales + vendor graph
compiler; the iSTFT and orchestration stay on this runtime.

## Library design

```
mcu/
  include/snt_port.h   <- THE port API: 3 kernels + 4 shims. Port = this.
  include/snt_tts.h    <- public API
  src/snt_tts.c        <- the entire pipeline, platform-free C99
  src/snt_kernels_ref.c<- scalar reference kernels (Tier S default)
  ports/host/          <- POSIX port (CI + golden gate)
  ports/esp32s3/       <- PIE asm kernels + FreeRTOS worker + IDF glue
  ports/wasm/          <- WebAssembly port: full stack in the browser, no server
  test/golden_main.c   <- bit-exactness gate vs PyTorch golden vectors
  test/fixtures/       <- small versioned model/golden contract used by CI
```

Host verification is self-contained:

```bash
make -C mcu test
```

The default fixture is `test/fixtures/en_us_r7`. The historical
`test/golden_c3` path is a compatibility link to the same bytes.

Runs in a browser too: `ports/wasm/` compiles the same core to WebAssembly
(`bash mcu/ports/wasm/build.sh`), and `web/index.html` synthesizes a full
utterance client-side and shows the golden correlation computed live in-page.
The WASM golden gate (`node mcu/ports/wasm/verify_node.mjs`) reproduces the
host result (corr 0.987 vs the PyTorch reference). Browser speed is a desktop-
CPU figure (~50-60x real time), not an MCU measurement.

Port API (complete):
- `int32_t snt_dot_s8(const int8_t *a, const int8_t *b, int len)`
- `void snt_matvec_s8(const int8_t *act, const int8_t *w, int32_t *out,
   int rows, int len)` — weights row-contiguous, n16-padded, 16B tail pad
- `int snt_weights_resident(const void *p)` — may SIMD read p? (SRAM test)
- `void snt_par_run(snt_par_fn f, int n, void *ctx)` — optional 2nd core;
   default runs serial. All parallel sections are column-disjoint with
   barriers; a port never needs to know the model.
- `SNT_NOW_US()` — profiling only
- memory: the CALLER hands the library one arena buffer; the library
  never allocates. Deterministic bump layout, documented peak per model.

Model format: per-output-channel symmetric int8 weight blobs + generated
offset headers (zero parse, zero copy from flash-mapped storage).
Activations: per-frame symmetric dynamic int8 (no calibration shipping),
frozen calibrated GroupNorm statistics baked at export.

Correctness contract: every port must pass `test/golden_main.c` with
correlation >= 0.98 against the shipped golden audio; the scalar
reference kernels define the exact integer semantics SIMD must match.

## Hard-won portability rules (measured, not theoretical)

1. SIMD reads require resident operands — flash-XIP vector loads return
   garbage silently on ESP32-S3 (corr 0.011 at full speed).
2. Residency is about WHERE bytes live, not model size: a 36k-param
   stage cost 3x a 74 KB matrix because its weights sat in flash.
3. libm is not free: lroundf ~50 cycles, float div ~40, doubles are
   software-emulated on FPUs without double support (~300 ms of pure
   instrumentation cost). The core ships division-free fast math.
4. Idle busy-wait workers poison the memory bus (~10% on everything);
   parallel workers must block, not spin.
5. malloc at 95% utilization is bin-packing roulette; the arena is the
   only allocation strategy that survives.
