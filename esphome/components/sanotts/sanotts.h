/* sanotts.h -- ESPHome external component: on-device neural text-to-speech.
 *
 * WHAT THIS IS FOR
 * ----------------
 * Home Assistant voice satellites cannot speak unless a server synthesizes the
 * audio and streams it back. This component runs the whole TTS stack on the
 * ESP32-S3 itself, so Home Assistant can send text instead of audio and the
 * device still talks when the server is gone.
 *
 * The single question it exists to answer is whether that fits in memory
 * ALONGSIDE ESPHome, so the instrumentation is not a debugging aid here -- it
 * is the product. Every synthesis records free internal heap and the largest
 * free internal block before, during and after, next to the utterance length
 * in frames, so the arena cost model can be checked against silicon rather
 * than trusted.
 *
 * VOICE LINEAGE -- READ THIS BEFORE QUOTING ANY NUMBER FROM IT
 * ------------------------------------------------------------
 * The weights are `en_us_e12nano` (294,642 parameters, 24 kHz, hop 256,
 * 93.75 frames/second). That lineage is a SIBLING of the `heart-nano` voice,
 * not the same weights. There is a standing project rule that chip numbers are
 * never attached to heart-nano's name; keep it.
 *
 * MEMORY MODEL
 * ------------
 * The runtime never allocates. The caller hands it ONE arena and it
 * bump-allocates inside it, reporting the high-water mark as
 * snt_nano_stats::arena_peak. The arena must be ONE CONTIGUOUS BLOCK, which is
 * why this component reports heap_caps_get_largest_free_block() and not just
 * total free: a classic ESP32 with 250,040 B free and a 110,580 B largest
 * block cannot run a row that needs 128,944 B, and total-free would have said
 * it could.
 *
 * The arena must also live in INTERNAL SRAM, not PSRAM, and that is a measured
 * requirement rather than a preference. The PIE vector kernels return garbage
 * silently when they read flash-XIP (correlation 0.011, a plausible-looking
 * waveform rather than a crash), so the runtime stages weights into the arena
 * and dispatches SIMD only for operands snt_weights_resident() accepts. Back
 * the arena with PSRAM and nothing is staged: the output is still correct, it
 * just runs about 5.7x slower. g_snt_macs_simd / g_snt_macs_scalar make that
 * split visible instead of leaving it to be inferred from a timing number.
 *
 * LICENCE
 * -------
 * The inference runtime (snt_nano.c and friends) is MIT. The text frontend is
 * the espeak-free lexicon port in nano_lex_g2p.c, whose dictionary data is
 * misaki's us_gold.json (Apache-2.0) -- deliberately NOT espeak-ng, which is
 * GPL-3.0 and costs 62 KB of internal SRAM this budget cannot spare.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "esphome/core/automation.h"
#include "esphome/core/component.h"

extern "C" {
#include "snt_nano.h"
}

namespace esphome {
namespace sanotts {

/* One heap reading. Both figures matter and for different reasons: `free` says
 * how much is left in total, `largest` says how much can be handed out in one
 * piece, and the arena needs the second. */
struct HeapReading {
  size_t internal_free{0};
  size_t internal_largest{0};
  size_t spiram_free{0};
  size_t spiram_largest{0};
};

/* Everything one synthesis produced. Populated even on failure so a failed run
 * still reports the heap it failed at. */
struct SynthReport {
  int rc{0};                  /* snt_nano_synthesize return code, 0 == OK    */
  int tokens{0};
  int frames{0};
  int samples{0};
  int64_t elapsed_us{0};
  size_t arena_bytes{0};      /* what we handed the runtime                  */
  size_t arena_peak{0};       /* what it actually used (high-water)          */
  bool arena_internal{false}; /* was the arena SIMD-readable internal SRAM?  */
  HeapReading before{};       /* immediately before the arena allocation     */
  HeapReading at_peak{};      /* with the arena held: the worst moment       */
  HeapReading after{};        /* after the arena is released                 */
  int64_t macs_simd{0};
  int64_t macs_scalar{0};
  double rms{0.0};
  double audio_seconds{0.0};
  double rtf{0.0};
};

/* Result of the golden-fixture correctness gate. BOARDS.md is explicit that a
 * speed number without a passing gate is not a result, so this runs before any
 * timing is reported. */
struct GateReport {
  bool ran{false};
  bool pass{false};
  double corr{0.0};
  double rms_ratio{0.0};
  SynthReport synth{};
};

class SanoTTS : public Component {
 public:
  void setup() override;
  void loop() override;
  void dump_config() override;
  /* LATE so the heap reading at setup() is taken with the rest of ESPHome --
   * WiFi, the API server, the logger -- already constructed. A number taken
   * before them would flatter the result. */
  float get_setup_priority() const override { return setup_priority::LATE; }

  void set_i2s_pins(int bclk, int lrclk, int dout) {
    this->i2s_bclk_ = bclk;
    this->i2s_lrclk_ = lrclk;
    this->i2s_dout_ = dout;
  }
  void set_arena_reserve(size_t bytes) { this->arena_reserve_ = bytes; }
  void set_gate_on_boot(bool on) { this->gate_on_boot_ = on; }
  void set_ladder_on_boot(bool on) { this->ladder_on_boot_ = on; }
  void set_volume(float v) { this->volume_ = v; }

  /* Speak arbitrary text. Runs the on-device lexicon G2P, then synthesis. */
  void say(const std::string &text);

  /* The correctness gate: the 415-frame golden row, frozen durations, compared
   * against the float PyTorch reference waveform embedded in this image. */
  void run_gate();

  /* The memory ladder: roughly 1, 2, 3, 4.4, 5.3 and 6.7 seconds of speech,
   * each measured twice -- once with a generous arena to learn arena_peak, and
   * once with an arena of exactly that many bytes to prove the figure is a
   * real minimum and not an artefact of having had room to spare. */
  void run_ladder();

  /* Dump the most recent utterance's PCM over the serial log as base64 int16,
   * for hosts with no DAC wired. Bounded: refuses over `max_samples`. */
  void dump_last_pcm(int max_samples);

  static HeapReading read_heap();
  void log_heap(const char *label);

 protected:
  /* The one place synthesis happens. `durs` may be nullptr, in which case the
   * int8 duration student predicts its own timing. */
  bool synthesize_(const int32_t *ids, int n_ids, const int32_t *durs, uint64_t seed,
                   size_t arena_bytes, bool play, SynthReport *out);

  bool i2s_start_(uint32_t sample_rate);
  void i2s_stop_();
  static void log_report_(const char *tag, const SynthReport &r);

  int i2s_bclk_{-1};
  int i2s_lrclk_{-1};
  int i2s_dout_{-1};
  bool i2s_up_{false};
  void *i2s_tx_{nullptr}; /* i2s_chan_handle_t, opaque to keep IDF out of this header */

  size_t arena_reserve_{24 * 1024};
  bool gate_on_boot_{true};
  bool ladder_on_boot_{false};
  float volume_{0.8f};

  bool g2p_ready_{false};
  bool boot_work_done_{false};
  uint32_t boot_work_at_{0};

  HeapReading boot_heap_{};
  GateReport gate_{};

  /* Retained so dump_last_pcm() can work; NOT allocated unless a dump is
   * requested, and freed straight afterwards. */
  int16_t *capture_{nullptr};
  int capture_cap_{0};
  int capture_n_{0};
  bool capture_on_{false};

  friend int sanotts_pcm_trampoline(const float *pcm, int n, void *user);
};

template<typename... Ts> class SayAction final : public Action<Ts...> {
 public:
  explicit SayAction(SanoTTS *parent) : parent_(parent) {}
  TEMPLATABLE_VALUE(std::string, text)

  void play(const Ts &...x) override { this->parent_->say(this->text_.value(x...)); }

 protected:
  SanoTTS *parent_;
};

template<typename... Ts> class GateAction final : public Action<Ts...>, public Parented<SanoTTS> {
 public:
  void play(const Ts &...x) override { this->parent_->run_gate(); }
};

template<typename... Ts> class LadderAction final : public Action<Ts...>, public Parented<SanoTTS> {
 public:
  void play(const Ts &...x) override { this->parent_->run_ladder(); }
};

template<typename... Ts> class HeapAction final : public Action<Ts...>, public Parented<SanoTTS> {
 public:
  TEMPLATABLE_VALUE(std::string, label)
  void play(const Ts &...x) override {
    this->parent_->log_heap(this->label_.value(x...).c_str());
  }
};

}  // namespace sanotts
}  // namespace esphome
