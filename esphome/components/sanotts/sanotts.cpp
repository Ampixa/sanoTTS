/* sanotts.cpp -- see sanotts.h for what this component is and why the
 * instrumentation is the point rather than a debugging aid.
 *
 * MEASUREMENT CONVENTION (mcu/ports/esp32s3/measurements/README.md)
 * Every number this file logs that was produced by the chip is prefixed
 * `DEVICE:`. Constants compiled in from a host run are prefixed `HOST-REF:`
 * and are printed on their own lines, never beside a measured value. That
 * separation exists because a reference number in parentheses next to a
 * measured one is exactly how a host constant gets misread as a silicon
 * result.
 */
#include "sanotts.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "esphome/core/application.h"
#include "esphome/core/hal.h"
#include "esphome/core/log.h"

#include "driver/i2s_std.h"
#include "esp_heap_caps.h"

extern "C" {
#include "nano_lex_g2p.h"
#include "sanotts_golden.h"
#include "sanotts_model_data.h"
#include "snt_port.h"

/* Residency accounting added to snt_kernels_esp32s3.c for this port. On a
 * build without the SIMD kernels these are not defined, so the scalar
 * fallback declares them here as zero. */
#include "snt_arch.h"
#if SANOTTS_S3_SIMD
extern int64_t g_snt_macs_simd, g_snt_macs_scalar;
extern int64_t g_snt_calls_simd, g_snt_calls_scalar;
void snt_res_reset(void);
#endif
}

namespace esphome {
namespace sanotts {

static const char *const TAG = "sanotts";

/* The runtime's own token ceiling (MAX_TOKENS_RT in snt_nano.c). Sizing the
 * G2P output buffer to it means a long sentence is refused by the G2P with a
 * named error rather than overrunning anything downstream. */
static constexpr int MAX_IDS = 1024;

/* ---------------------------------------------------------------------------
 * PCM sink
 *
 * snt_nano_synthesize() calls back once per finished frame. One sink handles
 * every mode so that only one code path exists: counting (the timed runs),
 * correlating against the embedded reference (the gate), capturing to a buffer
 * (the serial dump) and writing to I2S (actually speaking).
 *
 * Nothing here allocates, and nothing here prints: instrumentation that moves
 * the number it measures is not instrumentation.
 * ------------------------------------------------------------------------ */
struct PcmSink {
  SanoTTS *self{nullptr};
  long count{0};

  /* Kahan-free double accumulation is fine at these magnitudes: 160k samples
   * in [-1,1] cannot exhaust a double's 53-bit mantissa. */
  const float *ref{nullptr};
  size_t ref_n{0};
  size_t ref_pos{0};
  double sa{0}, sb{0}, saa{0}, sbb{0}, sab{0};

  int16_t *capture{nullptr};
  int capture_cap{0};
  int capture_n{0};

  void *i2s{nullptr};
  float volume{1.0f};

  /* Feeding the task watchdog matters: synthesis holds the ESPHome main loop
   * for over a second, and the loop task is a watchdog subscriber. Every 32
   * frames is ~0.34 s of audio and far under any sane timeout. */
  int since_wdt{0};
};

int sanotts_pcm_trampoline(const float *pcm, int n, void *user) {
  auto *s = static_cast<PcmSink *>(user);
  s->count += n;

  if (s->ref != nullptr) {
    for (int i = 0; i < n && s->ref_pos < s->ref_n; i++, s->ref_pos++) {
      const double a = pcm[i];
      const double b = s->ref[s->ref_pos];
      s->sa += a;
      s->sb += b;
      s->saa += a * a;
      s->sbb += b * b;
      s->sab += a * b;
    }
  } else {
    /* RMS still needs the sum of squares even when there is no reference. */
    for (int i = 0; i < n; i++) {
      const double a = pcm[i];
      s->saa += a * a;
    }
  }

  if (s->capture != nullptr && s->capture_n < s->capture_cap) {
    const int room = s->capture_cap - s->capture_n;
    const int take = n < room ? n : room;
    for (int i = 0; i < take; i++) {
      float v = pcm[i] * s->volume;
      if (v > 1.0f)
        v = 1.0f;
      else if (v < -1.0f)
        v = -1.0f;
      s->capture[s->capture_n + i] = static_cast<int16_t>(v * 32767.0f);
    }
    s->capture_n += take;
  }

  if (s->i2s != nullptr) {
    /* One frame is 256 samples; convert on the stack and hand it straight to
     * the DMA ring. No whole-utterance PCM buffer is needed anywhere, which is
     * the point: a 5-second buffer would be 240 KB and would dwarf the arena
     * this component exists to measure. */
    int16_t block[SANOTTS_HOP];
    int off = 0;
    while (off < n) {
      const int chunk = (n - off) < SANOTTS_HOP ? (n - off) : SANOTTS_HOP;
      for (int i = 0; i < chunk; i++) {
        float v = pcm[off + i] * s->volume;
        if (v > 1.0f)
          v = 1.0f;
        else if (v < -1.0f)
          v = -1.0f;
        block[i] = static_cast<int16_t>(v * 32767.0f);
      }
      size_t written = 0;
      const esp_err_t err = i2s_channel_write(static_cast<i2s_chan_handle_t>(s->i2s), block,
                                              static_cast<size_t>(chunk) * sizeof(int16_t),
                                              &written, 1000 / portTICK_PERIOD_MS);
      if (err != ESP_OK) {
        /* Abort rather than emit a silently truncated utterance. The non-zero
         * return propagates out of snt_nano_synthesize() as ERR_ABORT (1). */
        ESP_LOGE(TAG, "i2s_channel_write failed: %s", esp_err_to_name(err));
        return 1;
      }
      if (written != static_cast<size_t>(chunk) * sizeof(int16_t)) {
        ESP_LOGE(TAG, "i2s_channel_write short: %u of %u bytes", (unsigned) written,
                 (unsigned) (chunk * sizeof(int16_t)));
        return 1;
      }
      off += chunk;
    }
  }

  if (++s->since_wdt >= 32) {
    s->since_wdt = 0;
    App.feed_wdt();
  }
  return 0;
}

/* ------------------------------------------------------------------------ */

HeapReading SanoTTS::read_heap() {
  HeapReading h;
  h.internal_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  h.internal_largest = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  h.spiram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  h.spiram_largest = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
  return h;
}

void SanoTTS::log_heap(const char *label) {
  const HeapReading h = read_heap();
  ESP_LOGI(TAG, "DEVICE: heap[%s] internal_free=%u internal_largest=%u psram_free=%u psram_largest=%u",
           label, (unsigned) h.internal_free, (unsigned) h.internal_largest,
           (unsigned) h.spiram_free, (unsigned) h.spiram_largest);
}

void SanoTTS::setup() {
  this->boot_heap_ = read_heap();
  ESP_LOGI(TAG, "sanoTTS on-device TTS -- voice en_us_e12nano (294,642 params, 24 kHz)");
  ESP_LOGI(TAG, "  (a SIBLING of the heart-nano voice, NOT the same weights)");
  this->log_heap("setup");

  /* The G2P has no init beyond a self-check of its own packed tables; doing it
   * here rather than lazily means a corrupt table is a boot-time error, not a
   * surprise the first time somebody speaks. */
  int32_t probe[MAX_IDS];
  const int rc = nano_lex_g2p_text_to_ids("test", probe, MAX_IDS);
  if (rc < 0) {
    ESP_LOGE(TAG, "G2P self-check failed: %s (%d)", nano_lex_g2p_strerror(rc), rc);
    this->mark_failed();
    return;
  }
  this->g2p_ready_ = true;
  ESP_LOGI(TAG, "DEVICE: g2p workspace=%u B, self-check produced %d ids",
           (unsigned) nano_lex_g2p_workspace_bytes(), rc);

  /* Deferred rather than done here: setup() runs before WiFi has associated,
   * and the whole question is what is left once ESPHome is actually up and
   * doing its job. Running the gate at setup() would measure the wrong moment. */
  this->boot_work_at_ = millis() + 15000;
}

void SanoTTS::loop() {
  if (this->boot_work_done_)
    return;
  if (!this->g2p_ready_)
    return;
  if (millis() < this->boot_work_at_)
    return;
  this->boot_work_done_ = true;

  this->log_heap("esphome-up");
  if (this->gate_on_boot_)
    this->run_gate();
  if (this->ladder_on_boot_)
    this->run_ladder();
}

void SanoTTS::dump_config() {
  ESP_LOGCONFIG(TAG, "sanoTTS:");
  ESP_LOGCONFIG(TAG, "  Voice: en_us_e12nano, 294,642 params, %d Hz, hop %d", SANOTTS_SAMPLE_RATE,
                SANOTTS_HOP);
  ESP_LOGCONFIG(TAG, "  Weights in flash: front %u B + decoder %u B", SANOTTS_FRONT_BLOB_BYTES,
                SANOTTS_DEC_BLOB_BYTES);
  ESP_LOGCONFIG(TAG, "  SIMD kernels: %s", SANOTTS_S3_SIMD ? "Xtensa LX7 PIE" : "scalar C");
  ESP_LOGCONFIG(TAG, "  Arena reserve: %u B", (unsigned) this->arena_reserve_);
  if (this->i2s_dout_ >= 0) {
    ESP_LOGCONFIG(TAG, "  I2S: bclk=%d lrclk=%d dout=%d volume=%.2f", this->i2s_bclk_,
                  this->i2s_lrclk_, this->i2s_dout_, this->volume_);
  } else {
    ESP_LOGCONFIG(TAG, "  I2S: not configured (synthesis proven by the golden gate instead)");
  }
  ESP_LOGCONFIG(TAG, "  HOST-REF: golden row 000001_i200, 415 frames, corr 0.987887,"
                     " rms_ratio 0.995239, arena_peak 84208 B");
  ESP_LOGCONFIG(TAG, "  HOST-REF: arena is now CONSTANT in utterance length:"
                     " 84208 B on all eight host fixture rows, 415 to 629 frames");
}

/* ------------------------------------------------------------------------ */

bool SanoTTS::i2s_start_(uint32_t sample_rate) {
  if (this->i2s_up_)
    return true;
  if (this->i2s_dout_ < 0)
    return false;

  /* Built field by field rather than through I2S_CHANNEL_DEFAULT_CONFIG so a
   * field the macro gains or renames in a future IDF is a compile error here
   * instead of a silently different DMA setup. Mirrors what ESPHome's own
   * i2s_audio speaker does. */
  i2s_chan_handle_t tx = nullptr;
  i2s_chan_config_t chan_cfg = {
      .id = I2S_NUM_AUTO,
      .role = I2S_ROLE_MASTER,
      .dma_desc_num = 6,
      .dma_frame_num = 240,
      .auto_clear = true,
      .intr_priority = 3,
  };
  esp_err_t err = i2s_new_channel(&chan_cfg, &tx, nullptr);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "i2s_new_channel failed: %s", esp_err_to_name(err));
    return false;
  }

  i2s_std_clk_config_t clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(sample_rate);
  i2s_std_slot_config_t slot_cfg =
      I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
  i2s_std_gpio_config_t gpio_cfg = {};
  gpio_cfg.mclk = I2S_GPIO_UNUSED;
  gpio_cfg.bclk = static_cast<gpio_num_t>(this->i2s_bclk_);
  gpio_cfg.ws = static_cast<gpio_num_t>(this->i2s_lrclk_);
  gpio_cfg.dout = static_cast<gpio_num_t>(this->i2s_dout_);
  gpio_cfg.din = I2S_GPIO_UNUSED;

  i2s_std_config_t std_cfg = {
      .clk_cfg = clk_cfg,
      .slot_cfg = slot_cfg,
      .gpio_cfg = gpio_cfg,
  };

  err = i2s_channel_init_std_mode(tx, &std_cfg);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "i2s_channel_init_std_mode failed: %s", esp_err_to_name(err));
    i2s_del_channel(tx);
    return false;
  }
  err = i2s_channel_enable(tx);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "i2s_channel_enable failed: %s", esp_err_to_name(err));
    i2s_del_channel(tx);
    return false;
  }
  this->i2s_tx_ = tx;
  this->i2s_up_ = true;
  return true;
}

void SanoTTS::i2s_stop_() {
  if (!this->i2s_up_)
    return;
  auto tx = static_cast<i2s_chan_handle_t>(this->i2s_tx_);
  const esp_err_t err = i2s_channel_disable(tx);
  if (err != ESP_OK)
    ESP_LOGW(TAG, "i2s_channel_disable failed: %s", esp_err_to_name(err));
  i2s_del_channel(tx);
  this->i2s_tx_ = nullptr;
  this->i2s_up_ = false;
}

/* ------------------------------------------------------------------------ */

void SanoTTS::log_report_(const char *tag, const SynthReport &r) {
  ESP_LOGI(TAG,
           "DEVICE: %s rc=%d tokens=%d frames=%d samples=%d audio_s=%.4f "
           "us=%lld rtf=%.4f arena_bytes=%u arena_peak=%u arena_internal=%d",
           tag, r.rc, r.tokens, r.frames, r.samples, r.audio_seconds, (long long) r.elapsed_us,
           r.rtf, (unsigned) r.arena_bytes, (unsigned) r.arena_peak, r.arena_internal ? 1 : 0);
  ESP_LOGI(TAG,
           "DEVICE: %s heap_before_free=%u heap_before_largest=%u "
           "heap_at_peak_free=%u heap_at_peak_largest=%u heap_after_free=%u heap_after_largest=%u",
           tag, (unsigned) r.before.internal_free, (unsigned) r.before.internal_largest,
           (unsigned) r.at_peak.internal_free, (unsigned) r.at_peak.internal_largest,
           (unsigned) r.after.internal_free, (unsigned) r.after.internal_largest);
  ESP_LOGI(TAG, "DEVICE: %s macs_simd=%lld macs_scalar=%lld rms=%.6f psram_free=%u", tag,
           (long long) r.macs_simd, (long long) r.macs_scalar, r.rms,
           (unsigned) r.at_peak.spiram_free);

  /* An arena smaller than the row wants does not fail -- it silently stops
   * staging weights into SRAM, hands the kernels flash pointers, and every
   * unstaged MAC takes the scalar loop. The audio stays correct and the speed
   * collapses, which is the worst shape a regression can have. Say it out
   * loud, with the arena the row would have needed. */
  if (r.rc == 0 && r.macs_scalar > 0) {
    const long long total = r.macs_simd + r.macs_scalar;
    /* SANOTTS_ARENA_PER_FRAME is 0 since the runtime went to fixed windows;
     * the term is kept so a lineage whose arena does scale can restore it. */
    const size_t wanted = SANOTTS_ARENA_FIXED + (size_t) SANOTTS_ARENA_PER_FRAME * r.frames;
    ESP_LOGW(TAG,
             "DEVICE: %s ARENA-STARVED -- %lld%% of MACs fell off the SIMD path. Arena was "
             "%u B; full staging for %d frames wants %u B. Output is still correct; the "
             "speed number is not the part's speed.",
             tag, total > 0 ? (100LL * r.macs_scalar / total) : 0,
             (unsigned) r.arena_bytes, r.frames, (unsigned) wanted);
  }
}

bool SanoTTS::synthesize_(const int32_t *ids, int n_ids, const int32_t *durs, uint64_t seed,
                          size_t arena_bytes, bool play, SynthReport *out) {
  SynthReport r;
  r.tokens = n_ids;
  r.before = read_heap();

  if (arena_bytes == 0) {
    /* Take the largest block this part will actually hand over in one piece,
     * minus a reserve so WiFi, lwIP and the API server can still allocate
     * while synthesis holds it. An arena that starves them turns a clean
     * number into a hang. */
    if (r.before.internal_largest <= this->arena_reserve_) {
      ESP_LOGE(TAG,
               "DEVICE: arena refused -- largest free internal block is %u B, "
               "reserve alone is %u B",
               (unsigned) r.before.internal_largest, (unsigned) this->arena_reserve_);
      r.rc = -100;
      r.after = read_heap();
      if (out != nullptr)
        *out = r;
      return false;
    }
    arena_bytes = r.before.internal_largest - this->arena_reserve_;
  }
  /* The runtime aligns the base pointer up by as much as 16 bytes and charges
   * that to arena_size, so hand it the slack rather than lose it. */
  arena_bytes += 16;

  void *arena = heap_caps_aligned_alloc(16, arena_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (arena == nullptr) {
    ESP_LOGE(TAG,
             "DEVICE: arena allocation of %u B FAILED -- largest free internal block was %u B",
             (unsigned) arena_bytes, (unsigned) r.before.internal_largest);
    r.rc = -101;
    r.arena_bytes = arena_bytes;
    r.after = read_heap();
    if (out != nullptr)
      *out = r;
    return false;
  }
  r.arena_bytes = arena_bytes;
  r.arena_internal = snt_weights_resident(arena) != 0;
  if (!r.arena_internal) {
    /* Not fatal, but it changes the result completely and must never be a
     * silent difference: with a non-resident arena nothing is staged and every
     * MAC takes the scalar path. */
    ESP_LOGW(TAG,
             "DEVICE: arena at %p is NOT SIMD-readable -- weights will not be staged and "
             "synthesis will run on the scalar path",
             arena);
  }
  r.at_peak = read_heap();

  PcmSink sink;
  sink.self = this;
  sink.volume = this->volume_;
  if (this->capture_on_ && this->capture_ != nullptr) {
    sink.capture = this->capture_;
    sink.capture_cap = this->capture_cap_;
  }
  if (play && this->i2s_start_(SANOTTS_SAMPLE_RATE))
    sink.i2s = this->i2s_tx_;

  snt_nano_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.front_blob = SANOTTS_FRONT_BLOB;
  cfg.dec_blob = SANOTTS_DEC_BLOB;
  cfg.arena = arena;
  cfg.arena_size = arena_bytes;
  cfg.dur_override = durs;
  cfg.noise_seed = seed;

#if SANOTTS_S3_SIMD
  snt_res_reset();
#endif

  snt_nano_stats st;
  memset(&st, 0, sizeof(st));
  const int rc = snt_nano_synthesize(&cfg, ids, n_ids, sanotts_pcm_trampoline, &sink, &st);

  r.rc = rc;
  r.frames = st.frames;
  r.samples = st.samples;
  r.elapsed_us = st.elapsed_us;
  r.arena_peak = st.arena_peak;
#if SANOTTS_S3_SIMD
  r.macs_simd = g_snt_macs_simd;
  r.macs_scalar = g_snt_macs_scalar;
#endif
  if (st.samples > 0) {
    r.audio_seconds = static_cast<double>(st.samples) / static_cast<double>(SANOTTS_SAMPLE_RATE);
    r.rtf = (static_cast<double>(st.elapsed_us) / 1e6) / r.audio_seconds;
    r.rms = sqrt(sink.saa / static_cast<double>(sink.count > 0 ? sink.count : 1));
  }
  this->capture_n_ = sink.capture_n;

  /* Carry the correlation accumulators out via the caller's sink is not
   * possible here, so the gate re-runs its own comparison. Keep this function
   * about memory and time only. */
  heap_caps_free(arena);
  r.after = read_heap();

  if (rc != 0) {
    const char *why = "unknown";
    switch (rc) {
      case -1: why = "null config or arena"; break;
      case -2: why = "arena exhausted (ERR_OOM)"; break;
      case -3: why = "token count out of range (ERR_TOKENS)"; break;
      case -4: why = "phoneme id outside the 62-symbol vocabulary"; break;
      case -5: why = "duration model produced fewer than 2 frames"; break;
      case -6: why = "noise generator failure"; break;
      case 1: why = "aborted by the PCM sink (I2S write failed)"; break;
      default: break;
    }
    ESP_LOGE(TAG, "DEVICE: synthesis FAILED rc=%d (%s)", rc, why);
  }

  if (out != nullptr)
    *out = r;
  return rc == 0;
}

/* ------------------------------------------------------------------------ */

void SanoTTS::say(const std::string &text) {
  if (!this->g2p_ready_) {
    ESP_LOGE(TAG, "say() refused: the G2P failed its self-check at boot");
    return;
  }
  if (text.empty()) {
    ESP_LOGW(TAG, "say() called with empty text");
    return;
  }

  static int32_t ids[MAX_IDS];
  const int n = nano_lex_g2p_text_to_ids(text.c_str(), ids, MAX_IDS);
  if (n < 0) {
    ESP_LOGE(TAG, "G2P failed for \"%s\": %s (%d)", text.c_str(), nano_lex_g2p_strerror(n), n);
    return;
  }
  ESP_LOGI(TAG, "DEVICE: say \"%s\" -> %d phoneme ids", text.c_str(), n);

  /* Real text gets NO duration override: the int8 duration student predicts
   * its own timing, which is the path a product actually uses. The frozen
   * durations exist only so the correlation gate is meaningful. */
  const bool playing = this->i2s_dout_ >= 0;
  SynthReport r;
  this->synthesize_(ids, n, nullptr, SANOTTS_R00_SEED, 0, playing, &r);
  log_report_("say", r);
  if (playing) {
    /* The PCM sink writes straight into the I2S DMA ring, so once the ring is
     * full the synthesis loop blocks waiting for it to drain. elapsed_us for a
     * played utterance is therefore playback time, not compute time, and its
     * RTF floors at ~1.0 no matter how fast the part is. The ladder numbers,
     * which never touch I2S, are the compute figures. */
    ESP_LOGW(TAG, "DEVICE: say us/rtf above INCLUDE I2S back-pressure -- not a compute figure");
  }
}

void SanoTTS::run_gate() {
  ESP_LOGI(TAG, "---- correctness gate: golden row %s, %d frames ----", SANOTTS_R00_ID,
           SANOTTS_R00_FRAMES);
  ESP_LOGI(TAG, "HOST-REF: gates are corr > 0.98 and 0.80 < rms_ratio < 1.25 (BOARDS.md)");
  ESP_LOGI(TAG, "HOST-REF: the same fixture on the host, same fast-math build:"
                " corr 0.987887, rms_ratio 0.995239, arena_peak 84208");

  this->gate_.ran = true;

  SynthReport r;
  r.tokens = SANOTTS_R00_TOKENS;
  r.before = read_heap();
  size_t arena_bytes = r.before.internal_largest;
  if (arena_bytes <= this->arena_reserve_) {
    ESP_LOGE(TAG, "DEVICE: gate cannot run -- largest free internal block %u B",
             (unsigned) arena_bytes);
    this->gate_.pass = false;
    return;
  }
  arena_bytes = arena_bytes - this->arena_reserve_ + 16;

  void *arena = heap_caps_aligned_alloc(16, arena_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (arena == nullptr) {
    ESP_LOGE(TAG, "DEVICE: gate arena allocation of %u B FAILED", (unsigned) arena_bytes);
    this->gate_.pass = false;
    return;
  }
  r.arena_bytes = arena_bytes;
  r.arena_internal = snt_weights_resident(arena) != 0;
  r.at_peak = read_heap();

  PcmSink sink;
  sink.self = this;
  sink.ref = SANOTTS_R00_AUDIO;
  sink.ref_n = SANOTTS_R00_SAMPLES;

  snt_nano_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.front_blob = SANOTTS_FRONT_BLOB;
  cfg.dec_blob = SANOTTS_DEC_BLOB;
  cfg.arena = arena;
  cfg.arena_size = arena_bytes;
  cfg.dur_override = SANOTTS_R00_DURS;
  cfg.noise_seed = SANOTTS_R00_SEED;

#if SANOTTS_S3_SIMD
  snt_res_reset();
#endif
  snt_nano_stats st;
  memset(&st, 0, sizeof(st));
  const int rc = snt_nano_synthesize(&cfg, SANOTTS_R00_IDS, SANOTTS_R00_TOKENS,
                                     sanotts_pcm_trampoline, &sink, &st);
  r.rc = rc;
  r.frames = st.frames;
  r.samples = st.samples;
  r.elapsed_us = st.elapsed_us;
  r.arena_peak = st.arena_peak;
#if SANOTTS_S3_SIMD
  r.macs_simd = g_snt_macs_simd;
  r.macs_scalar = g_snt_macs_scalar;
#endif

  heap_caps_free(arena);
  r.after = read_heap();

  if (rc != 0) {
    ESP_LOGE(TAG, "DEVICE: gate synthesis FAILED rc=%d -- NO SPEED NUMBER IS VALID", rc);
    this->gate_.pass = false;
    log_report_("gate", r);
    return;
  }

  const double n = static_cast<double>(sink.ref_pos);
  if (n < 2.0) {
    ESP_LOGE(TAG, "DEVICE: gate compared %d samples -- too few to correlate", (int) n);
    this->gate_.pass = false;
    return;
  }
  const double num = sink.sab - sink.sa * sink.sb / n;
  const double da = sink.saa - sink.sa * sink.sa / n;
  const double db = sink.sbb - sink.sb * sink.sb / n;
  const double corr = (da > 0.0 && db > 0.0) ? num / sqrt(da * db) : NAN;
  const double rms_a = sqrt(sink.saa / n);
  const double rms_b = sqrt(sink.sbb / n);
  const double rms_ratio = rms_b > 0.0 ? rms_a / rms_b : NAN;

  r.rms = rms_a;
  r.audio_seconds = static_cast<double>(st.samples) / static_cast<double>(SANOTTS_SAMPLE_RATE);
  r.rtf = (static_cast<double>(st.elapsed_us) / 1e6) / r.audio_seconds;

  /* Non-finite is a HARD failure, not a pass. A build with a defective integer
   * iFFT once scored corr 0.989 while emitting samples ~500x hot, which is why
   * rms_ratio is gated too and why NaN is never allowed through. */
  const bool ok = std::isfinite(corr) && std::isfinite(rms_ratio) && corr > 0.98 &&
                  rms_ratio > 0.80 && rms_ratio < 1.25;
  this->gate_.pass = ok;
  this->gate_.corr = corr;
  this->gate_.rms_ratio = rms_ratio;
  this->gate_.synth = r;

  log_report_("gate", r);
  ESP_LOGI(TAG, "DEVICE: gate samples_compared=%d corr=%.6f rms_ratio=%.6f ==> %s", (int) n, corr,
           rms_ratio, ok ? "PASS" : "FAIL");
  if (!ok)
    ESP_LOGE(TAG, "DEVICE: CORRECTNESS GATE FAILED -- every speed number below is void");
}

/* ------------------------------------------------------------------------ */

void SanoTTS::run_ladder() {
  if (!this->gate_.ran)
    ESP_LOGW(TAG, "ladder is running without a correctness gate result");
  else if (!this->gate_.pass)
    ESP_LOGE(TAG, "ladder is running after a FAILED gate -- these numbers are void");

  /* A shorter row is a plain PREFIX of ids AND durs taken together. That is
   * exactly how mcu/test/fixtures/en_us_e12nano_short was cut, and it is
   * verified: that fixture's 51 ids and 51 durs are byte-identical to r00's
   * first 51 and sum to its stated 255 frames.
   *
   * The token counts below were chosen on the host to land near 1, 2, 3, 4.4,
   * 5.3 and 6.7 seconds at 93.75 frames/s. The frame count is NOT taken from
   * that host arithmetic -- it is read back out of snt_nano_stats, so every
   * frames figure in the log is a device measurement. */
  /* Rows PAST the old ceiling. Until snt_nano.c went to fixed-width windows
   * the arena grew at 192 B/frame, so 4.4 s starved the weight staging (3x
   * slower, still correct) and 5.3 s failed outright with rc=-2. A ladder
   * that stops at 6.7 s cannot show that the slope is gone, so L7..L10 run
   * out to about 31 seconds.
   *
   * A long row is the r00 token sequence REPEATED k times, ids and durs
   * repeated together. The frame count is then exactly k * 415, the sequence
   * is a legal one (the same sentence spoken k times), and the durations stay
   * frozen so the row is comparable with the six above it. */
  struct Row {
    const char *name;
    const int32_t *ids;
    const int32_t *durs;
    int tokens;
    uint64_t seed;
    int repeat;                 /* 1 == use ids/durs exactly as they are */
  };
  static const Row rows[] = {
      {"L1", SANOTTS_R00_IDS, SANOTTS_R00_DURS, 15, SANOTTS_R00_SEED, 1},
      {"L2", SANOTTS_R00_IDS, SANOTTS_R00_DURS, 39, SANOTTS_R00_SEED, 1},
      {"L3", SANOTTS_R00_IDS, SANOTTS_R00_DURS, 56, SANOTTS_R00_SEED, 1},
      {"L4", SANOTTS_R00_IDS, SANOTTS_R00_DURS, SANOTTS_R00_TOKENS, SANOTTS_R00_SEED, 1},
      {"L5", SANOTTS_R06_IDS, SANOTTS_R06_DURS, 91, SANOTTS_R06_SEED, 1},
      {"L6", SANOTTS_R06_IDS, SANOTTS_R06_DURS, SANOTTS_R06_TOKENS, SANOTTS_R06_SEED, 1},
      {"L7", SANOTTS_R00_IDS, SANOTTS_R00_DURS, SANOTTS_R00_TOKENS, SANOTTS_R00_SEED, 2},
      {"L8", SANOTTS_R00_IDS, SANOTTS_R00_DURS, SANOTTS_R00_TOKENS, SANOTTS_R00_SEED, 3},
      {"L9", SANOTTS_R00_IDS, SANOTTS_R00_DURS, SANOTTS_R00_TOKENS, SANOTTS_R00_SEED, 5},
      {"L10", SANOTTS_R00_IDS, SANOTTS_R00_DURS, SANOTTS_R00_TOKENS, SANOTTS_R00_SEED, 7},
  };
  /* 7 * 73. Sized for the longest row above and checked against it below, so
   * adding a longer row without raising this is a logged refusal, never an
   * overrun. */
  static constexpr int LADDER_MAX_TOKENS = 7 * SANOTTS_R00_TOKENS;
  static int32_t long_ids[LADDER_MAX_TOKENS];
  static int32_t long_durs[LADDER_MAX_TOKENS];

  ESP_LOGI(TAG, "---- memory ladder ----");
  this->log_heap("ladder-start");

  size_t peak_min = 0, peak_max = 0;
  int peak_rows = 0;

  for (const Row &row : rows) {
    const int32_t *ids = row.ids;
    const int32_t *durs = row.durs;
    int tokens = row.tokens;
    if (row.repeat > 1) {
      tokens = row.tokens * row.repeat;
      if (tokens > LADDER_MAX_TOKENS) {
        ESP_LOGE(TAG, "DEVICE: %s needs %d tokens, ladder buffer holds %d -- skipped", row.name,
                 tokens, LADDER_MAX_TOKENS);
        continue;
      }
      for (int rep = 0; rep < row.repeat; rep++) {
        for (int t = 0; t < row.tokens; t++) {
          long_ids[rep * row.tokens + t] = row.ids[t];
          long_durs[rep * row.tokens + t] = row.durs[t];
        }
      }
      ids = long_ids;
      durs = long_durs;
    }

    /* Pass 1: a generous arena, to learn what the row actually needs. */
    SynthReport big;
    char tag[32];
    snprintf(tag, sizeof(tag), "%s/auto", row.name);
    if (!this->synthesize_(ids, tokens, durs, row.seed, 0, false, &big)) {
      log_report_(tag, big);
      ESP_LOGE(TAG, "DEVICE: %s did not complete -- skipping its exact-fit pass", row.name);
      continue;
    }
    log_report_(tag, big);

    /* Pass 2: an arena of exactly arena_peak bytes. This is the claim being
     * tested -- that arena_peak is a real minimum. Two things could go wrong
     * and both are checked rather than assumed: the run could fail outright,
     * or (worse, because it is silent) stage_buf() could fail to place the
     * resident weight copies, hand back flash pointers, and quietly drop every
     * MAC onto the scalar path while still producing correct audio. The SIMD
     * MAC count catches the second. */
    SynthReport tight;
    snprintf(tag, sizeof(tag), "%s/exact", row.name);
    const bool ok = this->synthesize_(ids, tokens, durs, row.seed, big.arena_peak,
                                      false, &tight);
    log_report_(tag, tight);
    if (!ok) {
      ESP_LOGE(TAG, "DEVICE: %s FAILED at an arena of exactly arena_peak=%u B", row.name,
               (unsigned) big.arena_peak);
    } else if (tight.arena_peak != big.arena_peak || tight.macs_simd != big.macs_simd) {
      ESP_LOGE(TAG,
               "DEVICE: %s degraded at the exact arena: peak %u->%u, simd MACs %lld->%lld "
               "(the exact-fit arena is NOT sufficient)",
               row.name, (unsigned) big.arena_peak, (unsigned) tight.arena_peak,
               (long long) big.macs_simd, (long long) tight.macs_simd);
    } else {
      ESP_LOGI(TAG, "DEVICE: %s exact-fit CONFIRMED at %u B (frames=%d)", row.name,
               (unsigned) big.arena_peak, big.frames);
    }

    if (peak_rows == 0 || big.arena_peak < peak_min)
      peak_min = big.arena_peak;
    if (peak_rows == 0 || big.arena_peak > peak_max)
      peak_max = big.arena_peak;
    peak_rows++;
  }

  /* The whole point of the ladder, stated as one line the log can be grepped
   * for: across every length it ran, did the arena move at all? */
  if (peak_rows > 0) {
    ESP_LOGI(TAG, "DEVICE: arena_peak over %d ladder rows: min=%u max=%u spread=%u B", peak_rows,
             (unsigned) peak_min, (unsigned) peak_max, (unsigned) (peak_max - peak_min));
  }

  this->log_heap("ladder-end");
  ESP_LOGI(TAG, "---- memory ladder done ----");
}

/* ------------------------------------------------------------------------ */

static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

void SanoTTS::dump_last_pcm(int max_samples) {
  if (max_samples <= 0) {
    ESP_LOGW(TAG, "dump_last_pcm: nothing requested");
    return;
  }
  /* Capture into PSRAM: an int16 buffer big enough to be interesting is bigger
   * than the arena we are trying to prove fits, and putting it in internal SRAM
   * would corrupt the very measurement this component exists for. */
  this->capture_cap_ = max_samples;
  this->capture_ =
      static_cast<int16_t *>(heap_caps_malloc(static_cast<size_t>(max_samples) * sizeof(int16_t),
                                              MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (this->capture_ == nullptr) {
    ESP_LOGE(TAG, "dump_last_pcm: could not allocate %d samples in PSRAM", max_samples);
    this->capture_cap_ = 0;
    return;
  }
  this->capture_on_ = true;
  this->capture_n_ = 0;

  SynthReport r;
  const bool ok = this->synthesize_(SANOTTS_R00_IDS, SANOTTS_R00_TOKENS, SANOTTS_R00_DURS,
                                    SANOTTS_R00_SEED, 0, false, &r);
  this->capture_on_ = false;
  log_report_("dump", r);

  if (ok && this->capture_n_ > 0) {
    ESP_LOGI(TAG, "DEVICE: pcm_dump begin samples=%d rate=%d format=int16le_base64",
             this->capture_n_, SANOTTS_SAMPLE_RATE);
    const auto *bytes = reinterpret_cast<const uint8_t *>(this->capture_);
    const int nbytes = this->capture_n_ * 2;
    char line[81];
    int li = 0;
    for (int i = 0; i < nbytes; i += 3) {
      const uint32_t b0 = bytes[i];
      const uint32_t b1 = (i + 1 < nbytes) ? bytes[i + 1] : 0;
      const uint32_t b2 = (i + 2 < nbytes) ? bytes[i + 2] : 0;
      const uint32_t v = (b0 << 16) | (b1 << 8) | b2;
      line[li++] = B64[(v >> 18) & 0x3F];
      line[li++] = B64[(v >> 12) & 0x3F];
      line[li++] = (i + 1 < nbytes) ? B64[(v >> 6) & 0x3F] : '=';
      line[li++] = (i + 2 < nbytes) ? B64[v & 0x3F] : '=';
      if (li >= 76) {
        line[li] = '\0';
        ESP_LOGI(TAG, "PCM %s", line);
        li = 0;
        App.feed_wdt();
      }
    }
    if (li > 0) {
      line[li] = '\0';
      ESP_LOGI(TAG, "PCM %s", line);
    }
    ESP_LOGI(TAG, "DEVICE: pcm_dump end");
  }

  heap_caps_free(this->capture_);
  this->capture_ = nullptr;
  this->capture_cap_ = 0;
}

}  // namespace sanotts
}  // namespace esphome
