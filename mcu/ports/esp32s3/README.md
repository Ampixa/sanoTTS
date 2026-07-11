# ESP32-S3 port

This directory holds the ESP32-S3 work: the **portable-runtime port** (Tier V
reference kernels) and a full **standalone talking-device application** built on
the `fsd` engine.

## `firmware/` — standalone on-device neural TTS (the application)

A self-contained talking device: type text into a WiFi dashboard and the board
turns it into speech entirely on-chip — **no cloud, no network phonemizer, no
companion app**. Real espeak-ng grapheme-to-phoneme + the int8 `fsd` r7 TTS
engine + LEDC-PWM audio out a GPIO into an LM386 and speaker.

```
browser text ──http──► ESP32-S3 ─► espeak-ng G2P ─► fsd int8 TTS ─► LEDC-PWM ─► LM386 ─► speaker
                       (on-chip)    (phoneme ids)    (22 kHz PCM)    GPIO17
```

### Results (measured on-device)

| metric | value | notes |
| --- | --- | --- |
| intelligibility (Whisper WER) | **18.5%** | vs **77%** for the naive on-chip letter-to-sound frontend; desktop-espeak reference 15.8% |
| G2P parity vs desktop espeak | 2.74% phoneme-id error | pure espeak-version drift (board 1.52.0 vs piper 1.52.0.1); logic is bit-identical |
| synthesis speed | ~1.1× real time | whole-utterance PSRAM buffer + 2 s lead cover it |
| max utterance | ~7 s (~640 frames) | longer is rejected gracefully, not a crash |
| voice | en_US Kristin, ~745k-param int8 | intelligible, a bit synthetic (small-model ceiling) |

Numbers come from the host eval loop (`tools/eval_g2p_parity.py`), not vibes —
methodology in [`docs/s3-audio-handoff.md`](../../../docs/s3-audio-handoff.md).

### Layout

| Path | What |
| --- | --- |
| `firmware/main/` | App: `fsd_e2e.c` (dashboard + engine), `esp_g2p.*` (espeak wrapper), `cp_id_table.h`, `sn_matvec_s8_esp32s3.S` (SIMD kernel), `golden/` (staged model). |
| `firmware/components/espeak-ng/` | The espeak-ng ESP32 port — `config.h`, `CMakeLists.txt`, `patch_speech.py`, and `README.md` for populating upstream source. |
| `firmware/espeak-ng-data/` | 275 KB minimal en-US espeak data, flashed to a SPIFFS partition. |
| `firmware/calib/` | `sibilant_fsd40.npz` — sibilant noise-injection calibration. |
| `host/` | Host validation + operator tools (parity reference, phoneme server, serial capture). |
| `bringup-archive/` | Dead-end bring-up experiments (PDM/PWM tests, earlier u600 dashboard). Kept for history, not built. |

### Build & flash

From `firmware/`, on an ESP-IDF host:

```bash
# one-time: populate espeak-ng source     (components/espeak-ng/README.md)
# per build tree: stage the golden model
main/golden/stage.sh

idf.py set-target esp32s3
idf.py build
idf.py -p <PORT> flash monitor
```

Flash is 8 MB (16 MB physical); the custom `partitions.csv` adds an `espeak`
SPIFFS partition for the voice data.

### Hardware

`GPIO17` → RC low-pass → AC-couple → LM386 → 8 Ω speaker. The RC filter must cut
**below** the 11 kHz Nyquist. PWM (156 kHz carrier) is used instead of I2S-PDM
because PDM's shaped noise sits in-band and the gain-200 LM386 amplifies it to a
buzz. A MAX98357 I2S DAC is a clean drop-in upgrade.

### Where the deep detail lives

The full engineering ledger — PWM-vs-PDM, the espeak port fixes (gnu11, the
`check_data_path` SPIFFS patch, task stacks), the internal-SRAM memory fight
(PSRAM-stack worker, weight-bank placement, the fixed-gain silence bug), and the
sibilant diagnosis — is in
[`docs/s3-audio-handoff.md`](../../../docs/s3-audio-handoff.md). Read it before
touching the memory layout; internal SRAM is the tight budget (~33 KB free at
runtime).

## Portable-runtime port (Tier V reference, measured 0.22× RT)

Separately from the application above, this port supplies the mcu/ runtime's
Tier-V (vector int8) kernels. Register the runtime sources + `sn_matvec_s8_esp32s3.S`
as an IDF component alongside the vendored `espressif__esp-nn` (for the aligned
dot). Call `snt_port_esp32s3_start_worker()` once at boot for dual-core; without
it everything runs single-core (correct, ~1.4× slower). Arena: grab
`heap_caps_get_largest_free_block(MALLOC_CAP_8BIT) - 4096` at boot and hand it to
`snt_config.arena`; weight blobs flash-map via `EMBED_FILES` and the core stages
what SIMD needs into the arena.
