/* SanoTTS.h -- Arduino/PlatformIO C++ wrapper around the saanoTTS mcu/
 * portable C99 runtime (int8 iSTFT engine, ~745k params, ~680 KB weights).
 *
 * ------------------------------------------------------------------------
 * Where phoneme ids come from
 * ------------------------------------------------------------------------
 * synthesize() consumes Piper phoneme ids (int32), NOT text. This library
 * does not do grapheme-to-phoneme (G2P) conversion. Get ids one of two
 * ways:
 *   1. Host tool: run a phonemizer offline (e.g. piper-phonemize, or
 *      espeak-ng on your dev machine) and bake/stream the resulting id
 *      array into the sketch. examples/SpeakGolden does exactly this with
 *      a pre-computed "golden" id array.
 *   2. On-chip espeak-ng G2P: a real, working ESP32-S3 port exists but is
 *      NOT part of this library (v1) -- it needs ~2500 files of vendored
 *      espeak-ng source, a SPIFFS data partition, and three ESP-IDF-
 *      specific patches (gnu11, a data-path patch, stack sizing). See
 *      mcu/ports/esp32s3/firmware/components/espeak-ng/README.md in the
 *      saanoTTS repo if you want to wire that in yourself; the id-producing
 *      half (esp_g2p.c) is a small, separable piece of that firmware.
 *
 * ------------------------------------------------------------------------
 * What this library ships vs. what it only documents
 * ------------------------------------------------------------------------
 *  SHIPPED, compiled by default (every src/ .c file):
 *   - snt_tts.c + snt_kernels_ref.c   the 745k int8 fsd pipeline (SanoTTS
 *                                     class below) -- the product path.
 *   - snt_front_f32.c + snt_piperlite.c / snt_piperlite_q8.c
 *                                     the bigger-voice path (fp32 front +
 *                                     a choice of fp32 or int8 decoder).
 *                                     Not wrapped in a polished class --
 *                                     see "Bigger voices" below.
 *   - sibilant fricative-noise injection, ported into snt_tts.c from the
 *     ESP32-S3 firmware reference (see snt_sibilant_configure below).
 *     Off by default.
 *   - snt_port_default.c              default serial port; optional ESP32
 *                                     second-core worker behind a build
 *                                     flag (see that file's header).
 *  DOCUMENTED POINTER ONLY (not in this library):
 *   - esp-nn PIE SIMD kernels + hand Xtensa asm (the 0.22x-RT ESP32-S3
 *     measurement) -- ESP-IDF-component-only, see mcu/ports/esp32s3/.
 *   - on-chip espeak-ng G2P -- see above.
 *   - PSRAM weight-bank placement / internal-SRAM arena reservation /
 *     PSRAM-stack worker tasks -- firmware-level build configuration, see
 *     arduino/README.md's ESP32-S3 section.
 *
 * ------------------------------------------------------------------------
 * Compile-time performance/precision flags (all optional; none are set by
 * this library's default build -- add them yourself via build_flags /
 * platformio.ini / boards custom build properties if you want them)
 * ------------------------------------------------------------------------
 *   FSD_FAST_MATH   RECOMMENDED. Polynomial exp/tanh/rsqrt + Newton
 *                   reciprocal instead of libm. This is the ONLY flag
 *                   combination the host golden gate (mcu/test/golden_main.c,
 *                   corr >= 0.98) is proven under -- ship with it on.
 *   FSD_FROZEN_NORM Skip per-utterance GroupNorm stats, use calibrated
 *                   frozen constants (model/frozen_norm.h). Validated on
 *                   the ESP32-S3 firmware (its CMakeLists sets this), not
 *                   by the portable host golden test. Small speed win,
 *                   small unmeasured-on-host accuracy trade.
 *   SNT_ACT_LUT     512-entry gelu/silu lookup tables instead of exact
 *                   poly. Useful on FPU-less cores. Error <= 0.02, below
 *                   the int8 noise floor -- but not separately golden-
 *                   gated from FSD_FAST_MATH; treat as experimental.
 *   SNT_INT_FFT / FSD_FULL_FFT
 *                   Alternate integer-FFT / full-N-FFT code paths.
 *                   Present, uncommonly exercised; not part of the proven
 *                   default build. Measure before shipping with these on.
 *   SNT_INT_CHAIN   DO NOT DEFINE. Present in the source (int16 residual-
 *                   chain activations) but the upstream implementation
 *                   comment states it is "DISABLED pending precision
 *                   redesign" -- 11-instance int8 silu grids measurably
 *                   crater correlation. It compiles, it is not correct.
 *                   Kept only because removing dead code from a copied
 *                   file risked silently drifting from upstream mcu/.
 *   SNT_PROF        printf() a per-substage timing line every
 *                   synthesize() call. Debug only.
 *
 * ------------------------------------------------------------------------
 * Memory
 * ------------------------------------------------------------------------
 * Weights: ~680 KB (front_q8.bin ~280 KB + model_q8.bin ~400 KB for the
 * shipped Kristin en_US voice) -- flash/PROGMEM-resident, never copied
 * whole; the runtime stages small resident working copies out of the
 * arena as it goes. Arena: ~300 KB working RAM for a whole utterance
 * (measured: the golden host test runs in a 320 KB arena). See
 * arduino/README.md for which boards actually have that much free RAM.
 */
#ifndef SANOTTS_H
#define SANOTTS_H

#include <stddef.h>
#include <stdint.h>

extern "C" {
#include "snt_tts.h"
#include "snt_front_f32.h"
#include "snt_piperlite.h"
#include "snt_piperlite_q8.h"

/* Declared in snt_port_default.c. Always safe to call: it is a real
 * FreeRTOS-worker start on ESP32 when built with -DSANOTTS_ESP32_DUALCORE,
 * and a harmless no-op everywhere else (including plain ESP32 without that
 * flag, RP2040, AVR, SAMD, ...). */
void snt_port_dualcore_start(void);
}

/* ------------------------------------------------------------------------
 * SanoTTS -- the 745k int8 fsd pipeline (the product path: en_US Kristin
 * and compatible voices sharing the same architecture dims baked into
 * the src/model header files at compile time -- dimensions are fixed by this build,
 * weights are swappable as long as they match those dims).
 * ---------------------------------------------------------------------- */
class SanoTTS {
public:
    SanoTTS();

    /* model_blob:  model_q8.bin contents (decoder weights) -- flash/
     *              PROGMEM/LittleFS-resident pointer, must outlive every
     *              synthesize() call.
     * front_blob:  front_q8.bin contents (duration + acoustic weights),
     *              same lifetime requirement.
     * arena:       caller-owned scratch RAM, 16-byte aligned, >= the
     *              value returned by recommendedArenaBytes(). Pass
     *              nullptr to let SanoTTS allocate+own one internally
     *              (falls back to malloc; on a heap-constrained board
     *              prefer supplying your own, e.g. a static/PSRAM buffer).
     * arena_bytes: size of `arena` if you supplied one; ignored if arena
     *              is nullptr.
     * Returns false if an internally-allocated arena could not be
     * obtained; never validates the blobs themselves (there is no magic/
     * checksum in this format -- mismatched blobs corrupt silently, same
     * as upstream mcu/).
     */
    bool begin(const void *model_blob, const void *front_blob,
              void *arena = nullptr, size_t arena_bytes = 0);

    /* Recommended arena size for this pipeline: the same 320 KB the host
     * golden gate (mcu/test/golden_main.c) runs in. A generous margin over
     * the ~300 KB measured whole-utterance peak documented in
     * docs/mcu-classes-and-porting.md. */
    static constexpr size_t recommendedArenaBytes() { return 320 * 1024; }

    /* Synthesize one utterance of Piper phoneme ids into a caller-owned
     * float32 PCM buffer (22.05 kHz, mono, [-1,1]-ish range matching the
     * decoder's tanh output). Returns the number of samples written, or
     * a negative value on failure (not begin()'d, or synthesis internally
     * aborted). `cap` bounds how many samples pcm_out can hold; longer
     * utterances are truncated, not overflowed.
     * `stats_out` is optional (pass nullptr to skip). */
    int synthesize(const int32_t *ids, int n_ids, float *pcm_out, int cap,
                  snt_stats *stats_out = nullptr);

    /* Sibilant fricative-noise injection (off by default). See the block
     * comment in snt_tts.h / snt_tts.c for the mechanism. Both arrays must
     * outlive subsequent synthesize() calls (no copy is made) and must be
     * calibrated for the SAME voice as model_blob/front_blob -- they are
     * not portable across voices. beta ~0.9 matches the host-verified
     * Kristin/r7 calibration; treat it as a per-voice dial. */
    void enableSibilantInjection(const float *tea_std,
                                 const int32_t *sib_id_set, int n_sib_ids,
                                 float beta);
    /* Equivalent to enableSibilantInjection(..., beta=0.0f). */
    void disableSibilantInjection();

    /* Start the optional ESP32 second-core worker (see
     * snt_port_default.c). No-op unless built for ARDUINO_ARCH_ESP32 with
     * -DSANOTTS_ESP32_DUALCORE; safe to call unconditionally otherwise.
     * Call once from setup(), before the first synthesize(). */
    void enableDualCore() { snt_port_dualcore_start(); }

    ~SanoTTS();

private:
    const void *model_blob_;
    const void *front_blob_;
    void *arena_;
    size_t arena_bytes_;
    bool owns_arena_;
    bool ready_;

    SanoTTS(const SanoTTS &) = delete;
    SanoTTS &operator=(const SanoTTS &) = delete;
};

/* ------------------------------------------------------------------------
 * SanoTTSPiperLiteQ8 -- the "bigger voices" path: fp32 duration+acoustic
 * front end (snt_front_f32.c, same student architecture family as above
 * but runtime-shaped from meta.bin instead of compiled-in dims) feeding
 * the int8-weight piperlite decoder (snt_piperlite_q8.c, W8/A12, gate-
 * passed >=0.9995 corr on amy/vi/kristin goldens). This is the memory-
 * realistic way to run a full Piper-class voice on MCU-class hardware:
 *
 *   decoder weights   piperlite int8  ~1.0 MB   (weights_q8.bin)
 *                     piperlite fp32  ~4.0 MB   (weights_f32.bin, PSRAM-
 *                                                class boards only --
 *                                                call snt_piperlite_*
 *                                                directly for that path,
 *                                                not wrapped here)
 *
 * This class is a thin, direct pass-through over the 3-stage C API
 * (front init -> durations -> latent -> decoder init -> decode); it does
 * NOT manage arenas for you -- every stage takes a caller-owned buffer,
 * exactly like the rest of this runtime, because "how much RAM do I have
 * left" is a decision only the sketch author can make on a given board.
 * Compute each arena size with the snt_*_arena_*() sizing functions
 * declared in snt_front_f32.h / snt_piperlite_q8.h before calling begin().
 * ---------------------------------------------------------------------- */
class SanoTTSPiperLiteQ8 {
public:
    SanoTTSPiperLiteQ8();

    /* Parse both meta blobs and bind weight pointers. Every pointer must
     * outlive this object. Returns false on a malformed/out-of-range meta
     * blob (see snt_front_init / snt_piperlite_q8_init). */
    bool begin(const void *front_meta, size_t front_meta_bytes,
              const float *front_weights, size_t front_weight_floats,
              const void *dec_meta_q8, size_t dec_meta_bytes,
              const int8_t *dec_weights_q8, size_t dec_weight_bytes);

    /* Stage 1: ids -> per-token durations. dur_out must hold n_tokens
     * int32s; arena sizing via snt_front_duration_arena_floats(). Returns
     * total frame count (sum of dur_out) or negative on error. */
    long durations(const int32_t *ids, int n_tokens, float length_scale,
                  int32_t *dur_out, float *arena, size_t arena_floats);

    /* Stage 2: ids+durations -> latent [a_out, frames] channel-major.
     * arena sizing via snt_front_latent_arena_floats(). */
    int latent(const int32_t *ids, const int32_t *durations, int n_tokens,
              long frames, float *latent_out, float *arena,
              size_t arena_floats);

    /* Stage 3: latent -> PCM. audio_out must hold frames*hop samples
     * (hop == SNT_PIPERLITE_Q8_HOP == 256). arena sizing via
     * snt_piperlite_q8_arena_bytes(). */
    int decode(const float *latent, int frames, float *audio_out,
              void *arena, size_t arena_bytes);

    const snt_front_model &frontModel() const { return front_; }
    const snt_piperlite_q8_model &decoderModel() const { return dec_; }

private:
    snt_front_model front_;
    snt_piperlite_q8_model dec_;
    bool ready_;
};

#endif /* SANOTTS_H */
