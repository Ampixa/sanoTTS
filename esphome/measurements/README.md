# ESP32-S3 serial captures: sanoTTS inside ESPHome

Raw, unedited `cap.py` output from COM5, one file per flashed firmware, in the
order they were taken. Each begins with the ROM boot banner (which is UART
noise at this baud until the app takes the port) and ends where the capture
script's stop marker fired.

Same convention as `mcu/ports/esp32s3/measurements/README.md`, and for the same
reason: **every value the chip produced is prefixed `DEVICE:`; constants
compiled into the binary from a host run are prefixed `HOST-REF:` and are
printed on their own lines, never beside a measured value.** A reference number
in parentheses next to a measured one is exactly how a host constant gets
misread as a silicon result.

## The board

`esptool chip-id` / `flash-id`, 2026-09-09, COM5 (`USB-Enhanced-SERIAL CH343`,
VID:PID 1A86:55D3):

```
Chip type:          ESP32-S3 (QFN56) (revision v0.2)
Features:           Wi-Fi, BT 5 (LE), Dual Core + LP Core, 240MHz,
                    Embedded PSRAM 8MB (AP_3v3)
Crystal frequency:  40MHz
MAC:                e0:72:a1:f3:e7:84
Flash:              16MB, manufacturer 0x68 device 0x4018,
                    eFuse flash type quad, voltage 3.3V
```

The second board on that host, COM8 (`USB-SERIAL CH340`, 1A86:7523), is an
**ESP32-D0WD-V3 rev v3.1, 4 MB flash, no PSRAM** — a classic ESP32, not an S3.
Nothing here was flashed to it.

The board's pre-existing 8 MB of flash was read back to
`C:\esp\sanotts_esphome\com5_backup_pre_esphome.bin` on the Windows host before
anything was written, because the ESPHome app partition covers the espeak
SPIFFS at 0x390000 that the earlier firmware used.

## Protocol

One flash, one hard reset, one capture. The component runs its work 15 s after
`setup()` finishes, so every heap figure labelled `esphome-up` or later is taken
with WiFi, the native API server, OTA and the logger all constructed and
running.

**WiFi is UP but NOT ASSOCIATED in every capture here.** `secrets.yaml` carries
placeholder credentials, so the radio scans, fails and retries; the driver, lwIP
and the API server are all allocated, but there is no associated station and no
connected Home Assistant client. An associated radio with a live API client will
consume more internal heap than these numbers show. Treat every figure below as
an **upper bound** on what is free.

| file | firmware under test |
|---|---|
| `01-psram-wifi-lwip-internal-COM5.log` | PSRAM on, WiFi/lwIP buffers left in internal SRAM. The 415-frame gate row could not allocate its arena (`rc=-2`) — largest free internal block 114,688 B against 128,944 B needed. Kept because the negative is the reason for the next build. |
| `02-psram-wifi-lwip-in-psram-COM5.log` | + `CONFIG_SPIRAM_TRY_ALLOCATE_WIFI_LWIP=y`. Gate PASSES: `corr 0.987887`, `rms_ratio 0.995239`. Full ladder. |
| `03-psram-with-say-COM5.log` | Same, plus the arena-starvation warning and three real Home Assistant sentences through the on-device lexicon G2P. |
| `04-nopsram-with-say-COM5.log` | **PSRAM disabled in software** (`CONFIG_SPIRAM=n`), so the part behaves as a 512 KB-internal-only ESP32-S3. Same component, same gate, same ladder, same sentences. This is the capture that answers the feasibility question. |
| `05-nopsram-windowed-arena-ladder-COM5.log` | Same no-PSRAM config, after `mcu/src/snt_nano.c` was changed to run the pipeline on fixed-width windows. The arena stops growing with the utterance, so the ladder runs out to a 2,905-frame (31.0 s) row instead of stopping at 6.7 s. See "Capture 05" below. |

`01` and `02` differ in one sdkconfig line. `03` and `04` differ in one
sdkconfig line (`sanotts-s3-nopsram.yaml` is generated from `sanotts-s3.yaml`
by `esphome/tools/gen_nopsram_variant.py` precisely so it cannot drift into
differing in two).

## Correctness gate

`BOARDS.md`: gates are `corr > 0.98` and `0.80 < rms_ratio < 1.25` against the
shipped golden fixture, and **a speed number without a passing correctness gate
is not a result.**

The gate row is `000001_i200` from `mcu/test/fixtures/en_us_e12nano/`: 73
tokens, 415 frames, 105,984 samples, frozen durations, compared against
`r00_audio.bin` — the **float PyTorch stack's** waveform, so this is the model
gate, not merely a port gate.

```
HOST-REF: same fixture, same fast-math build, mcu/Makefile test-nano-fastmath:
          corr 0.987887, rms_ratio 0.995239, arena_peak 128944

DEVICE:   gate samples_compared=105984 corr=0.987887 rms_ratio=0.995239 ==> PASS
```

Identical to six decimal places, in captures `02`, `03` and `04` — two firmware
builds and three independent flashes. The chip reproduces the host bit-for-bit.

It passes in `04` too, in a run where 69% of the MACs had fallen off the SIMD
path: correctness is independent of whether weights got staged, which is exactly
what the runtime promises and why speed is gated separately.

## The arena model, checked against silicon

The runtime never allocates. It is handed one contiguous arena and
bump-allocates inside it, reporting the high-water mark as
`snt_nano_stats::arena_peak`. From the three ladder rows that ran **fully
staged** (`macs_scalar == 0`):

| frames | audio (s) | `arena_peak` (B) |
|---:|---:|---:|
| 93 | 0.9813 | 67,120 |
| 192 | 2.0373 | 86,128 |
| 281 | 2.9867 | 103,216 |

Both pairwise slopes are **exactly 192.0 B/frame** and the intercept is
**49,264 B**, with zero residual:

```
arena_peak = 49,264 B + 192.0 B x frames
```

Extrapolated to the 415-frame gate row that gives **128,944 B**, which is
precisely the `arena_peak` the host gate reports for that row. Host and silicon
agree exactly, so this is a model, not a fit.

`BOARDS.md` states `46.5 KB fixed + 195.7 B/frame, r2 0.9998`. That is the
least-squares line through the eight host fixture rows (415 to 629 frames), and
it is reproduced exactly by that regression — but those rows straddle a step
where the slope changes from 192 to 208 B/frame above roughly 500 frames, so the
line is a good summary and a poor predictor at either end. Announcement-length
utterances live entirely on the 192 B/frame segment.

**A correction that matters to the brief.** `BOARDS.md` quotes the 415-frame row
as 4.81 s and the 255-frame row as 2.95 s; those are 22,050 Hz durations, and
the `en_us_e12nano` stack is **24 kHz** (`mcu/test/fixtures/en_us_e12nano/README.md`,
`nano_q8_meta.h`: hop 256, so 93.75 frames/s). The device harness
`mcu/ports/esp32s3/nano_app/main/nano_dev_main.c` uses 24000 and so does this
component. At 24 kHz the same rows are 4.42 s and 2.72 s, and the
announcement-length arena figures are:

| announcement | frames | arena needed |
|---|---:|---:|
| 1 s | 94 | 67,312 B |
| 2 s | 188 | 85,360 B |
| 3 s | 281 | 103,216 B |
| 5 s | 469 | 139,312 B |

The "about 78 KB for two seconds" in the original brief came from the 22,050 Hz
frame rate; the measured figure is **85,360 B**.

## Capture 05: the arena stops growing

Captures `01`-`04` measured a runtime whose activation planes were allocated for
the WHOLE utterance, so the arena was a line in the frame count and the line was
the ceiling. `mcu/src/snt_nano.c` now runs every stage on a fixed-width window
(`NANO_FRAME_CHUNK` output frames plus the pipeline's receptive field on each
side), recomputing the receptive-field skirt at each chunk boundary rather than
keeping a whole-utterance plane. Capture `05` is the same board, the same
no-PSRAM config and the same component with that runtime under it, plus four
longer ladder rows built by repeating the r00 token sequence 2, 3, 5 and 7 times
(ids and durs together, so the frame count is exactly k x 415).

The gate still passes, on the same numbers:

```
HOST-REF: same fixture, same fast-math build, mcu/Makefile test-nano-fastmath:
          corr 0.987887, rms_ratio 0.995239, arena_peak 84208

DEVICE:   gate samples_compared=105984 corr=0.987887 rms_ratio=0.995239 ==> PASS
DEVICE:   gate rc=0 frames=415 arena_peak=84208 macs_simd=107093120 macs_scalar=0
```

The correlation and the RMS ratio are unchanged to six decimals because the PCM
is unchanged: the change is verified on the host as BYTE-IDENTICAL output over
three lineages and four build configurations, not as a correlation.

The ladder, every row fully staged (`macs_scalar == 0`):

| row | frames | audio (s) | `arena_peak` (B) | RTF | `macs_scalar` |
|---|---:|---:|---:|---:|---:|
| L1 | 93 | 0.9813 | 67,120 | 0.3359 | 0 |
| L2 | 192 | 2.0373 | 84,208 | 0.3573 | 0 |
| L3 | 281 | 2.9867 | 84,208 | 0.3629 | 0 |
| L4 | 415 | 4.4160 | 84,208 | 0.3587 | 0 |
| L5 | 494 | 5.2587 | 84,208 | 0.3530 | 0 |
| L6 | 629 | 6.6987 | 84,208 | 0.3525 | 0 |
| L7 | 830 | 8.8427 | 84,208 | 0.3574 | 0 |
| L8 | 1,245 | 13.2693 | 84,208 | 0.3570 | 0 |
| L9 | 2,075 | 22.1227 | 84,208 | 0.3598 | 0 |
| L10 | 2,905 | 30.9760 | 84,208 | 0.3599 | 0 |

```
DEVICE: arena_peak over 10 ladder rows: min=67120 max=84208 spread=17088 B
```

Read the spread carefully: it is not a residual slope. From 192 frames upward
every row reports the SAME 84,208 B, over a 15x range in length. L1 is lower
because at 93 frames the whole utterance is shorter than the window, so the
window is clamped to the utterance and the plane is smaller -- the arena is
`min(utterance, window)`, and below the window the old behaviour is what you
get. There is no row above L1 where it moves.

Each row is run twice, once with the whole free block and once with an arena of
exactly `arena_peak`; all ten confirmed the exact fit with identical
`macs_simd`, so 84,208 B is a real minimum and not a starved reading.

Against capture `04` -- the same no-PSRAM config, the same board, the old
runtime -- row for row:

| frames | old `arena_peak` | new `arena_peak` | old RTF | new RTF | old `macs_scalar` |
|---:|---:|---:|---:|---:|---:|
| 93 | 67,120 | 67,120 | 0.3287 | 0.3359 | 0 |
| 192 | 86,128 | 84,208 | 0.3189 | 0.3573 | 0 |
| 281 | 103,216 | 84,208 | 0.3149 | 0.3629 | 0 |
| 415 | 100,992 (starved) | 84,208 | 1.1740 | 0.3587 | 63,319,968 of 91,630,848 |
| 494 | `rc=-2` | 84,208 | -- | 0.3530 | -- |
| 629 | `rc=-2` | 84,208 | -- | 0.3525 | -- |
| 830 .. 2,905 | not attempted | 84,208 | -- | 0.3574 .. 0.3599 | -- |

The old ceiling is visible in one column: 415 frames ran with 69% of its MACs
on the scalar path (3.7x the time, same audio), and 494 frames -- 5.3 s -- was
`rc=-2`, arena exhausted. The rows past that were never attempted on the old
runtime because they could not have run.

The compute cost of the change is the recomputed skirt, and it is visible
directly in the MAC counters rather than only in the clock: the 281-frame row
issues 73,089,120 int8 MACs where it used to issue 62,599,776, +16.8%, and its
RTF moves 0.3149 -> 0.3629, +15.2%. That is the price on rows that already
fitted, against 3.3x faster on the row that used to starve and possible-at-all
on everything above it. The skirt shrinks as `NANO_FRAME_CHUNK` grows; 128 was
chosen because it keeps both the 48-wide and the 62-wide lineages under this
part's contiguous block.

One byte-level caveat on the heap numbers below: `internal_largest` reads
118,784 B in this capture against 122,880 B in `04`, because the extended ladder
carries 4,088 B of static id/duration buffers for the repeated rows (7 x 73
tokens x 2 arrays x 4 B). That is ladder scaffolding, not runtime cost.

## Silent SIMD degradation

The single most useful thing the instrumentation caught. An arena smaller than
the row wants does **not** fail. `stage_buf()` cannot place its resident weight
copies, `res_copy()` hands back the flash pointer, `snt_weights_resident()`
answers false for it, and every unstaged MAC quietly takes the scalar loop. The
audio stays bit-identical and the speed collapses by roughly 3x.

`g_snt_macs_simd` / `g_snt_macs_scalar` (added to `snt_kernels_esp32s3.c` for
this port, mirroring `mcu/ports/esp32s3/snt_port_esp32s3.c`) make it a
measurement. Capture `01` row L3:

```
DEVICE: L3/auto rc=0 frames=281 rtf=0.8579 arena_bytes=90704 arena_peak=89888
DEVICE: L3/auto macs_simd=38553792 macs_scalar=24045984
```

against the same row with enough arena in capture `02`:

```
DEVICE: L3/auto rc=0 frames=281 rtf=0.3147 arena_bytes=110992 arena_peak=103216
DEVICE: L3/auto macs_simd=62599776 macs_scalar=0
```

Same audio, 2.7x the time, and the only visible difference without the counters
would have been the clock. The component now logs an explicit `ARENA-STARVED`
warning naming the arena it had and the arena the row wanted.

Note the second-order trap this creates: a degraded run also reports a **lower**
`arena_peak` (89,888 instead of 103,216), because the staging buffers it failed
to allocate are the ones that would have raised the high-water mark. An
`arena_peak` from a starved run is not the row's real requirement.
