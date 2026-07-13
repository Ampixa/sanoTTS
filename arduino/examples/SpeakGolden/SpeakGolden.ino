/* SpeakGolden.ino -- minimal SanoTTS example.
 *
 * What this sketch does:
 *   1. Mounts LittleFS and loads the two saanoTTS model blobs
 *      (front_q8.bin + model_q8.bin, ~680 KB total) that you flashed
 *      there separately -- see extras/README.md. They are NOT embedded
 *      in this sketch; a 680 KB const array would blow past the flash
 *      budget of most boards and is not how this library expects
 *      weights to be delivered.
 *   2. Synthesizes one pre-baked utterance from golden_ids.h -- an array
 *      of Piper phoneme ids, NOT text. This library does no text-to-
 *      phoneme conversion; see SanoTTS.h's header comment for where ids
 *      come from in a real product (a host phonemizer, or the separate,
 *      NOT-included-here on-chip espeak-ng port).
 *   3. Plays the result:
 *        - ESP32 / ESP32-S3: over I2S to an external DAC (e.g. a
 *          MAX98357A breakout). Set I2S_BCLK/I2S_LRCLK/I2S_DOUT below to
 *          your wiring.
 *        - every other board (including boards this library nominally
 *          supports, like RP2040, that don't have a wired I2S DAC in
 *          this example): prints the PCM's RMS level instead, so you can
 *          confirm synthesis produced real, non-degenerate audio without
 *          needing a speaker attached.
 *
 * Memory: the 320 KB synthesis arena below is sized for the shipped
 * ~745k-param voice (SanoTTS::recommendedArenaBytes()). See
 * arduino/README.md for which boards actually have that much free RAM
 * (plain ESP32 and ESP32-S3: yes; RP2040: marginal -- read that section
 * before trying this sketch there).
 */
#include <SanoTTS.h>

#include "golden_ids.h"

#if defined(ARDUINO_ARCH_ESP32)
#include <FS.h>
#include <LittleFS.h>
#include <ESP_I2S.h>

/* ---- I2S wiring -- edit to match your DAC breakout ---- */
static const int I2S_BCLK = 26;
static const int I2S_LRCLK = 25;
static const int I2S_DOUT = 22;
static I2SClass i2s;
#endif

/* PCM output buffer: ~4 s at 22.05 kHz mono, comfortably longer than the
 * ~1.5 s golden utterance. Static (.bss), not on the stack. */
#define PCM_CAP (4 * 22050)
static float g_pcm[PCM_CAP];

/* Synthesis working arena -- caller-owned, never malloc'd internally by
 * the runtime. On a PSRAM-equipped board you can point this at a PSRAM
 * buffer instead (e.g. heap_caps_malloc(n, MALLOC_CAP_SPIRAM)); on plain
 * internal SRAM this static buffer is fine for the shipped 745k voice. */
static uint8_t g_arena[SanoTTS::recommendedArenaBytes()] __attribute__((aligned(16)));

static SanoTTS tts;

/* Reads an entire LittleFS file into a freshly malloc'd buffer (freed:
 * never -- it must outlive every synthesize() call, so this sketch just
 * leaks it for its one-shot lifetime). Returns nullptr on any failure;
 * a missing file is the most common first-run mistake -- see
 * extras/README.md to build+flash the filesystem image. */
static uint8_t *load_blob(const char *path, size_t *out_size) {
#if defined(ARDUINO_ARCH_ESP32)
    File f = LittleFS.open(path, "r");
    if (!f) {
        Serial.printf("missing %s on LittleFS -- see extras/README.md\n", path);
        return nullptr;
    }
    size_t sz = f.size();
    uint8_t *buf = (uint8_t *)malloc(sz);
    if (!buf) {
        Serial.println("out of memory loading blob");
        f.close();
        return nullptr;
    }
    f.read(buf, sz);
    f.close();
    if (out_size) *out_size = sz;
    return buf;
#else
    (void)path;
    (void)out_size;
    Serial.println("this example's load_blob() only implements LittleFS on ESP32 -- "
                   "port it to your board's filesystem/flash API");
    return nullptr;
#endif
}

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("SanoTTS SpeakGolden example");

#if defined(ARDUINO_ARCH_ESP32)
    if (!LittleFS.begin(false)) {
        Serial.println("LittleFS mount failed -- flash the filesystem image first "
                       "(see extras/README.md), then reset the board");
        return;
    }
#endif

    size_t front_sz = 0, model_sz = 0;
    uint8_t *front_blob = load_blob("/front_q8.bin", &front_sz);
    uint8_t *model_blob = load_blob("/model_q8.bin", &model_sz);
    if (!front_blob || !model_blob) return;
    Serial.printf("loaded front_q8.bin (%u B) + model_q8.bin (%u B), arena %u B\n",
                 (unsigned)front_sz, (unsigned)model_sz, (unsigned)sizeof(g_arena));

    if (!tts.begin(model_blob, front_blob, g_arena, sizeof g_arena)) {
        Serial.println("SanoTTS::begin() failed (out of memory allocating internally? "
                       "shouldn't happen -- an arena was supplied)");
        return;
    }

    /* Optional extras, both off by default:
     *   tts.enableDualCore();  // ESP32 second-core worker; needs
     *                          // -DSANOTTS_ESP32_DUALCORE as a build flag
     *   tts.enableSibilantInjection(tea_std, sib_ids, n, 0.9f);
     *                          // fixes whistly /s z sh zh/ -- needs a
     *                          // per-voice calibration this example
     *                          // doesn't ship (see SanoTTS.h)
     */

    Serial.printf("synthesizing %d phoneme ids...\n", SANOTTS_GOLDEN_IDS_N);
    snt_stats st;
    int n = tts.synthesize(SANOTTS_GOLDEN_IDS, SANOTTS_GOLDEN_IDS_N, g_pcm, PCM_CAP, &st);
    if (n <= 0) {
        Serial.printf("synthesize() failed (rc=%d)\n", n);
        return;
    }
    Serial.printf("synthesized %d frames, %d samples, %lld us\n",
                 st.frames, st.samples, (long long)st.elapsed_us);

#if defined(ARDUINO_ARCH_ESP32)
    i2s.setPins(I2S_BCLK, I2S_LRCLK, I2S_DOUT);
    if (!i2s.begin(I2S_MODE_STD, 22050, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
        Serial.println("I2S init failed -- check wiring/pins above");
        return;
    }
    static int16_t pcm16[PCM_CAP];
    for (int i = 0; i < n; i++) {
        float v = g_pcm[i];
        if (v > 1.0f) v = 1.0f;
        else if (v < -1.0f) v = -1.0f;
        pcm16[i] = (int16_t)(v * 32767.0f);
    }
    i2s.write((const uint8_t *)pcm16, (size_t)n * sizeof(int16_t));
    Serial.println("played over I2S");
#else
    double sumsq = 0.0;
    for (int i = 0; i < n; i++) sumsq += (double)g_pcm[i] * (double)g_pcm[i];
    double rms = sqrt(sumsq / (n > 0 ? n : 1));
    Serial.print("no I2S output wired for this board in this example -- PCM RMS = ");
    Serial.println(rms, 4);
    Serial.println("(a healthy non-silent utterance is roughly 0.02-0.3; 0.0 means "
                   "something upstream failed silently)");
#endif
}

void loop() {}
