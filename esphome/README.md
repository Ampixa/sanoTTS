# sanoTTS as an ESPHome external component

**Milestone 1: does sanoTTS fit in memory alongside ESPHome on an ESP32-S3?**

Yes, and — since the runtime went to fixed-width windows — for utterances of any
length, at a cost that does not depend on the length. On a bare ESP32-S3 running
a full ESPHome build — WiFi, native API, OTA, logger — with **PSRAM switched off
in software**, the largest free contiguous internal SRAM block is **122,880 B**,
and the synthesis arena is a constant **84,208 B**:

| what you want to say | frames | arena needed | fits? | RTF |
|---|---:|---:|---|---:|
| ~1 s | 93 | 67,120 B | yes | **0.336** |
| ~2 s | 192 | 84,208 B | yes | **0.357** |
| ~3 s | 281 | 84,208 B | yes | **0.363** |
| ~4.4 s | 415 | 84,208 B | yes | **0.359** |
| ~6.7 s | 629 | 84,208 B | yes | **0.353** |
| ~13.3 s | 1,245 | 84,208 B | yes | **0.357** |
| ~22.1 s | 2,075 | 84,208 B | yes | **0.360** |
| ~31.0 s | 2,905 | 84,208 B | yes | **0.360** |

RTF is seconds of compute per second of audio; 0.357 means a two-second
announcement is synthesized in 0.73 s. Every figure in that table was measured
on silicon in [`measurements/05-nopsram-windowed-arena-ladder-COM5.log`](measurements/05-nopsram-windowed-arena-ladder-COM5.log),
with every row fully staged (`macs_scalar == 0`) and each one re-run at an arena
of exactly `arena_peak` to prove the figure is a real minimum.

The 1 s row is lower because at 93 frames the whole utterance is shorter than
the window, so the window is clamped to the utterance. Above the window the
arena does not move at all: 192 frames and 2,905 frames both report 84,208 B.

**This used to be a ceiling, and the ceiling is what changed.** The runtime
allocated its activation planes for the whole utterance, so the arena was a line
in the frame count — DEVICE-measured at 49,264 B + 192.0 B/frame — and that line
ran out of contiguous block:

| what you want to say | frames | old arena | old outcome | old RTF |
|---|---:|---:|---|---:|
| ~1 s | 93 | 67,120 B | fine | 0.329 |
| ~3 s | 281 | 103,216 B | fine, 20 KB spare | 0.315 |
| ~4.4 s | 415 | 128,944 B | **staging starved**, 69% of MACs scalar | 1.174 |
| ~5.3 s | 494 | 144,112 B | **`rc=-2`**, arena exhausted | — |

The two "old arena" figures for the rows that did not run properly are the
49,264 + 192.0 B/frame model, not readings: a starved run reports a *lower*
`arena_peak` than it needed (the 415-frame row reported 100,992 B on device
precisely because the staging it failed to place is what would have raised the
high-water mark), and a failed one reports zero. The model is DEVICE-measured
from the three rows that did run fully staged, and reproduces the host gate's
128,944 B for the 415-frame row exactly.

`mcu/src/snt_nano.c` now runs every stage on a fixed-width window
(`NANO_FRAME_CHUNK` output frames plus the pipeline's receptive field on each
side), recomputing the receptive-field skirt at each chunk boundary instead of
keeping a whole-utterance plane. The emitted PCM is **byte-identical** to the
old runtime — verified on the host over three lineages and four build
configurations, not by correlation — and the on-chip gate reproduces the same
`corr 0.987887`, `rms_ratio 0.995239` it always did.

The skirt is not free: the 281-frame row now issues 73,089,120 int8 MACs where
it used to issue 62,599,776 (+16.8%) and its RTF moves 0.315 -> 0.363 (+15.2%).
That is the price on rows that already fitted, against 3.3x faster on the row
that used to starve and possible-at-all on everything above it.

Home automation says "the front door is unlocked", not paragraphs — but it also
reads out a shopping list, and that used to be the thing this could not do.

The correctness gate passes: `corr 0.987887`, `rms_ratio 0.995239` against the
float PyTorch reference, identical to six decimals to the host build, in four
independent flashes. `BOARDS.md` is explicit that a speed number without a
passing gate is not a result, so the gate runs first, every boot, before
anything is timed.

**This is a bare ESP32-S3, not a Home Assistant Voice PE.** We do not have one
and no number here is a Voice PE number. It is also ESPHome *alone*: a real
voice satellite adds a microphone, wake word and media player, and every byte
those take comes out of the same contiguous block. See "What this does not
prove" below.

## What is here

```
esphome/
  components/sanotts/     the external component (flat, see the layout note)
  sanotts-s3.yaml         the config, PSRAM enabled
  sanotts-s3-nopsram.yaml the same config with PSRAM off -- GENERATED
  secrets.yaml            placeholders; replace before quoting a radio number
  tools/                  the two generators
  g2p/                    the text-to-phoneme work (see g2p/lex/README.md)
  measurements/           raw device captures + the measurement notes
```

Build and flash:

```bash
cd esphome
python3 tools/gen_component_data.py         # regenerate the embedded model data
IDF_CCACHE_ENABLE=0 esphome compile sanotts-s3.yaml
# then flash .esphome/build/sanotts-s3/build/firmware.factory.bin at 0x0
```

`IDF_CCACHE_ENABLE=0` is only needed on this machine: its Homebrew `ccache`
is linked against a `libfmt` that is no longer installed and aborts on every
invocation. `brew reinstall ccache` fixes it properly; the env var avoids
touching the user's system.

## The voice

`en_us_e12nano` — 294,642 parameters, 24 kHz, `n_fft` 1024, hop 256, so 93.75
frames per second of audio. Weights are 339,680 B of int8 embedded in flash
(`front_q8.bin` 175,456 + `model_q8.bin` 164,224), byte-identical to
`mcu/test/fixtures/en_us_e12nano/`.

That lineage is a **sibling of the `heart-nano` voice, not the same weights.**
There is a standing project rule that chip numbers are never attached to
heart-nano's name, and nothing in this directory does.

## How it works

Four pieces, all on-chip:

1. **Text to phonemes** — `nano_lex_g2p.c`, a C99 port of the dictionary-first
   path in `pypkg/sanotts/nano_g2p.py`. misaki's `us_gold` pronunciation
   dictionary packed into 1.5 MB of flash with binary search over a
   length-prefixed blob, plus number spelling, stress rules and the 62-symbol
   tokenisation. Apache-2.0 data, no espeak, no GPL. 18,648 B of static
   workspace, no `malloc`, and 1,257 B of stack (HOST-REF: measured on a host
   build, see `g2p/lex/README.md`; not measured on the chip).
2. **Synthesis** — `snt_nano.c` unchanged from `mcu/src/`: duration student,
   acoustic student, mel-100 interface, ConvNeXt1D decoder, iSTFT, DC blocker.
   The runtime never allocates; the caller hands it one arena.
3. **The vector kernels** — the Xtensa LX7 PIE int8 matvec assembly, carried as
   file-scope inline assembly (see the layout note).
4. **Audio out** — I2S standard mode to a MAX98357A on GPIO5/6/7, written
   straight from the per-frame PCM callback into the DMA ring. No
   whole-utterance PCM buffer exists anywhere: a 5-second one would be 240 KB
   and would dwarf the arena this whole exercise is about.

## The layout note (the thing that shapes this component)

ESPHome only copies external-component sources whose extension is in
`esphome.const.SOURCE_FILE_EXTENSIONS` — `{.cpp, .hpp, .h, .c, .tcc, .ino}` —
and `ComponentManifest.resources` only descends into subdirectories for
manifests that set `recursive_sources`, which external components do not. The
generated `src/CMakeLists.txt` then globs C and C++ only.

Two consequences, both load-bearing:

- **Every file must sit flat in `components/sanotts/`.** A `model/`
  subdirectory would be silently dropped.
- **A `.S` file cannot ship.** It is not copied, and would not be assembled if
  it were. The PIE kernels therefore live inside
  `snt_matvec_esp32s3_asm.c` as a file-scope `__asm__` block — a verbatim
  transcription of `arduino/src/snt_matvec_esp32s3.S`. Verified in the link map:
  `sn_matvec_s8_c3`, `_c5` and `_g` are all present and the residency counters
  confirm they execute.

The alternative was `esp32.add_idf_component(path=...)`, which writes a local
`path:` dependency into the generated `idf_component.yml` and would give a real
IDF component with per-file flags and `EMBED_FILES`. That is the right route for
anything needing custom compile options (an espeak-ng port would need it); it
was not needed here, and not needing it keeps this a plain external component
with no ESP-IDF plumbing.

## The memory ledger

From the link map of the flashed no-PSRAM image:

| | internal SRAM (`.bss`) | flash |
|---|---:|---:|
| `snt_nano.c` — iSTFT rings, mel ring, scratch banks, LUTs | 85,548 | 11,222 |
| `nano_lex_g2p.c` — G2P workspace | 18,648 | 25,087 |
| `sanotts.cpp` — static phoneme-id buffer | 4,096 | 9,695 |
| `snt_kernels_esp32s3.c` + asm + port | 32 | 778 |
| `nano_lex_tables.c` — us_gold dictionary | 0 | 1,500,446 |
| `sanotts_golden.c` — gate fixture (**test only**) | 0 | 425,376 |
| `sanotts_model_data.c` — the voice | 0 | 339,680 |
| **component total** | **108,324** | **2,312,284** |
| whole image | 209,871 of 341,760 | 3,160,495 of 8,126,464 |

Two things worth pulling out of that table.

**The arena is not the whole memory cost.** `BOARDS.md` characterises the
requirement as the arena, and the arena is what varies with utterance length —
but `snt_nano.c` also carries **85,548 B of static `.bss`** that is present from
boot whether or not anything is ever spoken: the 1024-point iSTFT window and
overlap-add rings, the paired FFT buffers, the 7-column mel ring, two scratch
banks and the fast-math lookup tables. Peak internal SRAM for a 2-second
announcement is therefore 108,324 (static) + 86,128 (arena) = **194,452 B**, not
86,128 B. Anyone budgeting from the arena formula alone will be 108 KB short.

That table is from the flash whose captures are `03`/`04`. The windowed runtime
(capture `05`) changes two rows and neither is `snt_nano.c`: its `.bss` is still
**85,548 B** to the byte, because the change removes arena, not statics. What
moved is `sanotts.cpp`, 4,096 -> **8,184 B**, which is the extended ladder's
static id/duration buffers for the repeated long rows (7 x 73 tokens x 2 arrays
x 4 B) -- test scaffolding, not runtime cost. Component `.bss` total 108,324 ->
**112,412 B**; whole image 209,871 -> **213,951 B** of 341,760 SRAM and
3,160,495 -> **3,164,763 B** of flash. And the arena that sits on top of it is
now 84,208 B for a two-second announcement AND for a thirty-second one.

**425,376 B of that flash is the correctness gate**, not the product. It is the
float PyTorch reference waveform for the 415-frame row, embedded so the gate can
run on the chip with no host in the loop. A production image drops it, and the
1.5 MB dictionary can be halved again by dropping `us_silver` — already the
default here, measured byte-identical on 52 Home Assistant announcements.

## Free heap, measured

All `DEVICE:` values, WiFi/API/OTA/logger all up, 15 s after `setup()`:

| build | free internal | largest free internal block |
|---|---:|---:|
| PSRAM on, WiFi/lwIP internal | 160,896 | 114,688 |
| PSRAM on, WiFi/lwIP in PSRAM | 170,744 | 126,976 |
| **PSRAM off (512 KB-only part)** | **165,072** | **122,880** |
| PSRAM off, after I2S is initialised | 160,936 | 106,496 |

Total free badly overstates what the part will hand out at once — the arena must
be **one contiguous block**, which is why `heap_caps_get_largest_free_block()`
is reported everywhere beside the total. That is not theoretical: `BOARDS.md`
records a classic ESP32 with 250,040 B free and a 110,580 B largest block
failing a row that needed 128,944 B.

Note the last row. **Initialising I2S costs 16,384 B of the contiguous
block** (122,880 -> 106,496). Under the old runtime that dropped the utterance
ceiling from 383 frames (4.09 s) to 298 frames (3.18 s); under the windowed
runtime there is no length ceiling on either side of it, because 84,208 B fits
in both. What the 16 KB still costs is the `arena_reserve` headroom: with I2S up
and a 16 kB reserve the component hands the runtime about 82,320 B, which is
1,888 B short of full staging, and capture `05` shows the three spoken sentences
running with 19-20% of their MACs on the scalar path as a result. Lowering
`arena_reserve` or tearing the I2S channel down between announcements both fix
it; the numbers to choose between them are in the capture.

## What this does not prove

- **WiFi is up but not associated in every capture.** `secrets.yaml` has
  placeholders, so the radio scans, fails and retries; the driver, lwIP and the
  API server are all allocated but no station is associated and no Home
  Assistant client is connected. Real association plus a live API client will
  consume more. **Every free-heap figure here is an upper bound.**
- **ESPHome alone is not a voice satellite.** No microphone, no wake word, no
  media player, no `voice_assistant`. Those are the components that make a
  satellite a satellite and they are not in this budget.
- **No Voice PE was tested.** The 80–140 KB free-internal-heap figure in the
  original brief belongs to hardware we do not have. Nothing here is a claim
  about it.
- **The PSRAM-off build still runs on a part that physically has PSRAM.** It is
  disabled in software, which reproduces the internal-SRAM budget but not, for
  example, the different cache configuration a genuinely PSRAM-less module might
  ship with.

## Milestone 2 — what wiring this to Home Assistant still needs

1. **Real credentials and an associated-radio measurement.** Re-run captures
   `03` and `04` against a live network with a connected API client, and see how
   much of the 122,880 B block survives. This is the one number that decides the
   whole thing and it is cheap to get.
2. **Coexistence with the voice-satellite stack.** Add `microphone`,
   `micro_wake_word` and `media_player` to the YAML and measure the same ladder.
   If the contiguous block drops below ~86 KB, two-second announcements stop
   fitting and the answer changes.
3. ~~**Streaming synthesis.**~~ **Done, capture `05`.** The arena is decoupled
   from utterance length: `mcu/src/snt_nano.c` runs on fixed-width windows and
   reports 84,208 B for every length from 192 to 2,905 frames, byte-identically.
   Latency came with it, because the head now emits a chunk at a time instead
   of after the whole trunk: HOST-REF, on a 2,905-frame (31.0 s) utterance,
   time to the first PCM sample is 3.2 ms against 39.1 ms before, and the
   arena that run needs is 84,208 B against 675,904 B. What is still
   whole-utterance is the duration student, which runs over every token before
   the first frame is projected -- cheap, but it is the remaining serial head.
4. **A `tts` platform rather than a bare action.** Right now Home Assistant
   calls the `say` API action. Registering as an ESPHome `speaker`/`tts` platform
   would let the existing `voice_assistant` pipeline target it directly.
5. **Arena admission control.** No longer about utterance length -- the arena
   is constant now -- but still about the block. If the contiguous block is
   under 84,208 B the runtime shrinks its own chunk to fit rather than failing,
   which trades speed for the ability to speak at all and is silent unless the
   `ARENA-STARVED` warning fires. The component should say which regime it is
   in; the policy is not written yet.
6. **Decide the flash budget.** 3.2 MB of a 16 MB part is fine; on an 8 MB
   module with dual OTA slots it is not. Dropping the gate fixture (425 KB) is
   free; dropping the dictionary for a smaller lexicon is the real lever.
