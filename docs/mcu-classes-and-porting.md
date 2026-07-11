# saanoTTS-MCU — MCU classes and portable-library design

Authoritative definition of (1) which microcontrollers can run the saanoTTS
runtime and how well, and (2) the library architecture that lets one C99
codebase span all of them by wrapping whatever kernel library each platform
ships. This document is locked; changes are deliberate revisions, not proposals.

Target model: the deployable ~745k-param saanoTTS stack (duration → acoustic →
iSTFT decoder), int8, ~680 KB shipped weights. (The larger Fork-B quality model
is a Tier-N / desktop target, not the general-MCU target.)

---

## 1. Workload envelope — the fixed numbers that gate everything

| Requirement | whole-utterance | streaming |
| --- | --- | --- |
| compute | **~45 MMAC/s of int8** per second of audio | same |
| weights (int8, R7 en_US stack) | **~680 KB** flash; SRAM-resident copies staged | same |
| RAM working set | **~300 KB** | **~130–160 KB** |
| float ops | glue only (scales, GroupNorm, iSTFT); FPU strongly preferred | same |

These are measured, not projected. Every MCU decision reduces to them.

---

## 2. The feasibility gate — one rule that classifies any MCU

A chip's class is a function of exactly two quantities:

```
verdict = f( effective int8 MAC/s ,  usable SRAM )
```

**Effective int8 throughput** — sustained after memory-bandwidth and float-glue
overhead, NOT datasheet peak (peak overstates by 10–50×):

| effective int8 MAC/s | real-time verdict |
| --- | --- |
| ≥ ~180 MMAC/s | real-time with margin |
| ~90–180 | real-time, tight |
| ~25–90 | near-real-time / short utterances |
| < 25 | offline only |

**SRAM floor** — below this the model cannot run regardless of speed:
- ~300 KB whole-utterance, ~130–160 KB streaming.

Real-time = "synthesizes 1 s of audio in ≤1 s" (RTF ≤ 1.0). The workload needs
~45 MMAC/s sustained, so real-time margin means the chip clears it with headroom
for the float glue and the iSTFT.

---

## 3. MCU classes (defined)

### Tier V — Vector int8 — *the product tier, real-time with margin*
128-bit int8 SIMD + ≥400 KB SRAM. The matvec/dot kernels map to native vector
MACs (16+ MAC/cycle). This is where real-time lives.
- **ESP32-S3** (dual Xtensa LX7 @ 240 MHz, PIE SIMD) — **reference port, MEASURED 0.22× RT** (4.5× faster than playback), output corr 0.985 vs float.
- **Cortex-M55 / M85** (Helium/MVE) — Alif Ensemble, Renesas RA8 — projected <0.1× RT at 400–480 MHz via CMSIS-NN or hand MVE.
- **ESP32-P4** (RV32 + vendor SIMD) — pending esp-nn support.

### Tier D — Dual-MAC DSP — *real-time, tight → near-RT*
SMLAD-class int16/int8 dual MACs, ≥512 KB SRAM.
- **Cortex-M7 @ 480–600 MHz** (STM32H7, Teensy 4.x, i.MX RT) — projected 0.2–0.5× RT with SXTB16+SMLAD int8 kernels (CMSIS-NN).
- **Cortex-M4 @ ≥168 MHz** — offline / near-RT for short utterances.

### Tier S — Scalar — *runs anywhere, correctness-identical, near-RT → offline*
Any 32-bit MCU meeting the RAM floor. Reference C kernels, zero assembly, zero
external deps. This is the "it will always run and always be bit-correct" tier.
- **ESP32-C3** (RV32IMC @ 160 MHz, no FPU) — **MEASURED 5.72× RT** (offline); the float glue, not the MACs, dominates.
- **RP2040** (dual Cortex-M0+ @ 133 MHz) — **port present**, unmeasured.
- Most Cortex-M3/M4-without-DSP, other RV32IMC — inherit the scalar kernels unchanged.

### Tier N — NPU offload — *quality scaling, the growth path*
≥256 MAC/cycle neural accelerator. The move here is NOT to run the tiny model
faster — it's to run a **bigger, better** model (the Fork-B quality stack) in
real time. The acoustic predictor goes on the NPU via the vendor graph compiler;
the iSTFT and orchestration stay on this runtime.
- Ethos-**U55** (Alif, Himax WE2, Renesas RA8P1), ST **Neural-ART** (STM32N6).

**Honest cut:** V is the shippable product (S3 proven). S is the universal
baseline (C3 measured, RP2040 ported). D is the strong projected middle. N is
the future quality path.

---

## 4. Library architecture — one core, many platforms, many kernel libraries

Two senses of "library", both true by design:
1. **saanoTTS-MCU IS a library** firmware links into its build — one static
   core, a caller-owned arena, a PCM callback. It never allocates, never owns a
   thread, never assumes an OS.
2. **Each port WRAPS whatever kernel library its platform ships** — esp-nn,
   CMSIS-NN, TFLM kernels, a vendor NPU driver, or nothing (scalar). The core
   never changes; only the port's kernel backing does.

```
src/snt_tts.c          the ENTIRE pipeline, platform-free C99 (no per-chip #ifdef)
include/snt_tts.h      public API: snt_synthesize(cfg, ids, n, pcm_cb, user, stats)
include/snt_port.h     the COMPLETE porting surface — 8 functions, nothing else
src/snt_kernels_ref.c  scalar reference kernels = the EXACT integer semantics
ports/<platform>/      per-platform impl: hand kernels OR delegate to a vendor lib
test/golden_main.c     bit-exactness gate: the correctness oracle for every port
```

### The port contract — 8 functions, the whole boundary

**4 kernels** (exact int32 accumulation, no saturation/rounding; len multiple of
16, 16-byte aligned, weights row-contiguous with ≥16 B tail pad):
```c
int32_t snt_dot_s8   (const int8_t  *a, const int8_t *b, int len);
void    snt_matvec_s8(const int8_t  *act, const int8_t *w, int32_t *out, int rows, int len);
int32_t snt_dot_s16s8(const int16_t *a, const int8_t *b, int len);   /* residual chains */
void    snt_matvec_s16s8(const int16_t *act, const int8_t *w, int32_t *out, int rows, int len);
```
**4 shims:**
```c
int     snt_weights_resident(const void *p);  /* may SIMD read here? (SRAM vs flash-XIP) */
void    snt_par_run(snt_par_fn f, int n, void *ctx);  /* optional 2nd core; default serial */
int64_t snt_now_us(void);                     /* profiling only; may return 0 */
int     snt_scratch_id(void);                 /* which core's scratch bank (0 main, 1 worker) */
```

### Kernel-library delegation — why it "works for multiple libraries"

The kernels are pure int8/int16 dot/matvec with exact-int32 semantics — a
contract nearly every embedded-ML kernel library already exposes. So a port
chooses a backing instead of hand-writing everything:

| Class | Port's kernel backing (pick one) | Proven here |
| --- | --- | --- |
| V (ESP32-S3) | esp-nn PIE, or hand PIE asm | ✅ both |
| V (Cortex-M55) | CMSIS-NN `arm_nn_*` (Helium), or hand MVE | projected |
| D (Cortex-M7) | CMSIS-NN SMLAD | projected |
| S (C3/RP2040/any) | the scalar reference (zero deps) | ✅ |
| — (host/WASM) | scalar reference | ✅ |
| N (Ethos-U55) | vendor graph compiler (acoustic); runtime keeps iSTFT+orchestration | future |

### The correctness oracle makes delegation safe

Whatever library a port wraps, `test/golden_main.c` requires the port to
reproduce the shipped golden audio at **corr ≥ 0.98 vs the PyTorch float model**,
and the scalar reference defines the exact integer semantics any SIMD/vendor
kernel must match. Wrapping esp-nn *or* CMSIS-NN *or* TFLM *or* hand asm is
therefore one contract with a CI-enforced oracle, not a maintenance sprawl.

### Cross-class invariants (hold on every platform)
- **Caller-owned arena** — the library is handed one buffer and bump-allocates
  within it; deterministic peak per model; never `malloc`s. Survives 95%-full
  MCU heaps where `malloc` is bin-packing roulette.
- **Residency awareness** — `snt_weights_resident` lets flash-XIP platforms
  stage weights into SRAM before any SIMD load (the bug that silently returned
  garbage — corr 0.011 — on the S3 at full speed).
- **Blocking parallelism** — `snt_par_run` workers must block while idle;
  busy-wait poisons the memory bus ~10% on everything.

---

## 5. Porting recipe — add a new MCU or wrap a new kernel library

1. Create `ports/<platform>/snt_port_<platform>.c`.
2. Implement the 4 kernels — either delegate to the platform's kernel library
   (esp-nn / CMSIS-NN / …) or start from the scalar reference and optimize.
   Implement the 4 shims (residency test, par_run, clock, scratch id).
3. Build core + reference + your port + `test/golden_main.c`.
4. Pass the golden gate (corr ≥ 0.98). If the SIMD path fails, the scalar
   reference is the ground truth to diff against.
5. Measure RTF; slot the chip into a class by the §2 gate.

A port is one file plus a build glue; the model, the pipeline, and the numerics
are never touched.

---

## 6. Status matrix

| Port | Class | Kernel backing | State |
| --- | --- | --- | --- |
| host (POSIX) | — | scalar ref | ✅ CI golden gate |
| **esp32s3** | V | PIE asm + esp-nn | ✅ **measured 0.22× RT** |
| **esp32c3** | S | scalar (int16 activation refactor) | ✅ **measured 5.72× RT** |
| **rp2040** | S | scalar | ✅ ported, unmeasured |
| **riscv-rvv** | V | RVV 1.0 kernels | ✅ QEMU-validated |
| **wasm** | S | scalar | ✅ golden gate (browser) |
| cortex-m7 | D | CMSIS-NN SMLAD | ☐ projected (highest-value next port) |
| cortex-m55 | V | CMSIS-NN Helium | ☐ projected |
| ethos-u55 | N | vendor graph compiler | ☐ future (quality model) |

**Highest-value next port:** Cortex-M7 via CMSIS-NN — it fills the one projected
class with no measured port, and it is the second concrete proof of the
"wrap a vendor kernel library" property (beyond esp-nn on the S3).
