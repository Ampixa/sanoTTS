# `espeak_ng` — on-device text→phoneme-ids for the `en_us_e12nano` voice

An ESP-IDF component that compiles the espeak-ng 1.52.0 **translator** (no audio
synthesis) for the ESP32-S3 and turns its IPA output into the 62-symbol token
ids the 294,642-parameter nano stack was trained on.

Drop-in for an ESPHome 2026.8.x build via
`esphome.components.esp32.add_idf_component()`.

---

## Licence — read this first

**espeak-ng is GPL-3.0-or-later.** `vendor_espeak.sh` copies 33 espeak-ng
translation units and ~31 headers into this directory, and `CMakeLists.txt`
links them into your firmware. The upstream licence text travels with them as
`COPYING.espeak-ng`.

What that means in practice:

- A firmware image built with this component is a work that links GPL-3.0 code.
  If you **distribute that image** — sell a device, publish a `.bin`, hand a
  built image to someone else — the GPL-3.0 obligations attach to the whole
  combined work: complete corresponding source, under GPL-3.0-compatible terms,
  offered to every recipient.
- GPL-3.0 also carries the **anti-tivoisation** requirement: if the device is a
  "User Product", you must supply the installation information needed to
  install a modified version. Secure boot with a locked key is in direct tension
  with that.
- Building it for **yourself** and never distributing the image triggers none of
  this. Private use is unrestricted.
- `nano_g2p.c`, `nano_g2p.h`, `nano_g2p_table.h` and the scripts in this
  directory are this repository's own code, but they are useless without
  espeak-ng, and once linked the combined work is GPL-3.0.

If any of that is unacceptable for how you ship, do not use this component. The
espeak-free alternative already exists in this repo: `pypkg/sanotts/nano_g2p.py`
is a vendored-lexicon front end that scores **better** against misaki than the
espeak path does (token error rate 0.30% vs 6.09%, see
`artifacts/nano-g2p-espeak-free-20260904/results.json`). It has not been ported
to C, but it would be a far smaller and licence-clean thing to port than this.

---

## What it does

```
text
  → phonemizer Punctuation.preserve            (split punctuation out)
  → espeak_TextToPhonemes, IPA with U+0361 ties
  → phonemizer EspeakBackend._postprocess_line
  → phonemizer Punctuation.restore
  → nano_frontend.postprocess_line             (rewrites the tie to '^')
  → misaki EspeakFallback E2M rewrites
  → 62-symbol vocabulary filter, <bos> … <eos>
```

This is a C transcription of `pypkg/sanotts/nano_frontend.py`, the front end the
`sanotts` package ships. It is **not** the Piper/Kristin path in
`mcu/ports/esp32s3/firmware/main/esp_g2p.c`: that one uses a 157-entry id table
from a different voice lineage and interleaves pad tokens. Output here is
**dense** — `<bos> t0 t1 … <eos>`, no pad interleaving — matching the golden
fixture `mcu/test/fixtures/en_us_e12nano/r00_ids.bin`.

The tie is load-bearing. Every diphthong rule matches on `a^ɪ`, `e^ɪ`, `o^ʊ`,
`a^ʊ`, `ɔ^ɪ`, `d^ʒ`, `t^ʃ`. Ask espeak for plain IPA and not one of them fires,
and the model gets `ɪ`/`ʊ`/`ʃ`/`ʒ` everywhere it was trained on `I`/`A`/`O`/`W`/
`Y`/`ʤ`/`ʧ`.

## Measured parity

`esphome/g2p/host/verify_parity.py`, 330 sentences from
`artifacts/nano-g2p-espeak-free-20260904/evalset.jsonl`, run through the
**vendored** espeak sources and the **exact 275 KB data image that gets
flashed**. Full output in `esphome/g2p/host/parity-report.json`.

| reference | exact id sequences | token error rate |
|---|---|---|
| `shipped-espeak.jsonl` — what `pypkg/sanotts/nano_frontend.py` produces | **100.00%** (330/330) | **0.0000%** (0 / 30,783 ids) |
| `misaki-reference.jsonl` — what real misaki/Kokoro produces | 1.82% (6/330) | 6.0936% (1,895 / 31,098 ids) |

The first row is the port-correctness number: this C is byte-identical to the
Python front end, so nothing was lost in translation. The second is the
phonetic-fidelity number, and it is **not** a defect of this port — it is the
espeak front end's own distance from misaki, recorded independently on k2 as
`exact_match 0.01818…, token_error_rate 0.06093639…` in
`artifacts/nano-g2p-espeak-free-20260904/results.json`. This port reproduces
both digits.

## Known defect inherited from the shipped front end: contractions

`pypkg/sanotts/frontend.py`'s `PUNCTUATION_MARKS` includes the apostrophe, so
phonemizer splits every contraction and espeak reads the orphaned tail as a
letter name:

| text | phonemes | reads as |
|---|---|---|
| `It's unlocked.` | `ɪt'ˈɛs ʌnlˈɑkt.` | "it **ess** unlocked" |
| `Don't forget the door.` | `dˈɑn'tˈi fəɹɡˈɛt ðə dˈɔɹ.` | "don **tee** forget the door" |
| `You're home.` | `ju'ɹˌi hˈOm.` | "you **ar** home" |

That is the shipped behaviour, faithfully reproduced — the Python and this C
agree on it byte for byte. It matters here because Home Assistant announcements
are contraction-heavy.

Compile with `-DNANO_G2P_SPLIT_ON_APOSTROPHE=0` (a commented-out line in
`CMakeLists.txt`) to keep contractions whole: `ɪts`, `dˈOnt`, `jʊɹ`.

HOST-REF: measured over the same 330 rows against misaki, where only 35 rows
contain an apostrophe at all:

| setting | token error rate vs misaki | exact |
|---|---|---|
| `1` (default, bit-exact with the shipped front end) | 6.0936% (1,895 edits) | 1.82% |
| `0` (contractions kept whole) | **5.6756%** (1,765 edits) | 2.12% |

So `0` is measurably closer to what the model was trained on. It is not the
default only because `1` is what makes the parity harness report an exact match.
Removing the hyphen as well was tried and is *worse* (5.7013%), so the hyphen
stays.

---

## Setup

```bash
# 1. Vendor the espeak-ng 1.52.0 translator subset (33 .c files + headers).
./vendor_espeak.sh                    # clones upstream into a temp dir
./vendor_espeak.sh /path/to/espeak-ng # or reuse a checkout

# 2. Prove the source set compiles and still matches the reference, on the host,
#    before involving any cross toolchain.
../../host/build_espeak_host.sh
python3 ../../host/verify_parity.py \
    --binary ../../host/nano_ids_vendored \
    --espeak-data ../../../../mcu/ports/esp32s3/firmware/espeak-ng-data

# 3. Build the SPIFFS data image (does not flash).
./mkspiffs_espeak.sh /tmp/espeak.bin
```

`vendor_espeak.sh` applies `patch_speech.py`, which is required: espeak's
`check_data_path()` proves a directory holds the data by requiring `stat()` to
report `-EISDIR`, and SPIFFS is flat and cannot. The patch accepts the path when
`<path>/phontab` opens for reading — true on SPIFFS and on a real filesystem, so
the host build is unaffected.

(The equivalent script in `mcu/ports/esp32s3/firmware/components/espeak-ng/`
hardcodes an absolute `C:\esp\…` path and cannot run anywhere but the one
Windows build host. This one takes the file as an argument.)

## Wiring it into ESPHome

ESPHome 2026.8.2 builds ESP-IDF directly, and its `src/CMakeLists.txt` globs
only `*.c/*.cpp/*.cc/*.cxx/*.c++` with no per-file flags — so espeak-ng cannot
live in `src/`. It needs `-std=gnu11` (IDF defaults to gnu23, which turns
espeak's implicit int→pointer conversions into hard errors) and its own include
paths. Both are only expressible in a real IDF component.

From your component's `to_code()`:

```python
import os
from esphome.components import esp32

G2P = os.path.join(os.path.dirname(__file__), "..", "..", "g2p", "idf", "espeak_ng")

async def to_code(config):
    # Writes a `path:` dependency into
    # .esphome/build/<name>/src/idf_component.yml; the component manager then
    # resolves it as a normal IDF component with its own CMakeLists.txt.
    esp32.add_idf_component(name="espeak_ng", path=os.path.abspath(G2P))

    # 768 KB SPIFFS for the espeak data, matching
    # mcu/ports/esp32s3/firmware/partitions.csv.
    esp32.add_partition("espeak", "data", "spiffs", 0xC0000)
```

Then, in C++:

```cpp
extern "C" {
#include "nano_g2p.h"
}

int rc = nano_g2p_init();               // mounts /espeak, starts espeak en-us
if (rc != NANO_G2P_OK) {
    ESP_LOGE(TAG, "g2p init failed: %s", nano_g2p_strerror(rc));
    return;
}

int32_t ids[NANO_MAX_TOKENS];
int n = nano_g2p_text_to_ids("The front door is unlocked.", ids, NANO_MAX_TOKENS);
if (n < 0) {
    ESP_LOGE(TAG, "g2p failed: %s", nano_g2p_strerror(n));
    return;
}
// ids[0..n) is <bos> … <eos>, dense, ready for the duration student.
```

`nano_g2p_init()` must run on a task with a deep stack — espeak's clause
translator recurses. See the SRAM notes below.

### Flashing the data image

The offset comes from **your build's** partition table, never from this README.
Read it back:

```bash
esptool.py --chip esp32s3 read_flash 0x8000 0xc00 pt.bin
python3 $IDF_PATH/components/partition_table/parttool.py \
    --partition-table-file pt.bin get_partition_info \
    --partition-name espeak --info offset size
esptool.py --chip esp32s3 write_flash <offset> /tmp/espeak.bin
```

DEVICE: on the existing `mcu/ports/esp32s3` firmware that partition sits at
`0x390000`, size `0xC0000` — `mcu/ports/esp32s3/measurements/06-FINAL-labelled-COM5.log:24`,
`espeak Unknown data 01 82 00390000 000c0000`. An ESPHome build defines its own
layout, so confirm before writing.

---

## What it costs

### Flash

BUILD: from the ESP-IDF link map `C:\esp\fsd_audio\build\fsd_timing.map` on the
`windows` build host — the shipped ESP32-S3 dashboard firmware, which links this
same espeak source set. `libespeak-ng.a` is the single archive holding all 33
translation units.

| section | bytes |
|---|---:|
| `.text` | 117,062 |
| `.literal` | 8,884 |
| `.rodata` | 96,594 |
| `.data` (image in flash) | 204 |
| **total flash** | **222,744 B (217.5 KiB)** |

Cross-checked against an independent build, `C:\esp\espeaktest\build\espeaktest.map`
(espeak-only test app): `.rodata` and `.data` identical to the byte, `.text`
6,871 B smaller, total 215,613 B.

Plus the data partition: **281,248 B** of payload (`en_dict` 168,204,
`phontab` 59,520, `phonindex` 44,904, a 4 KB `phondata` stub, `intonations`
2,312, nine `lang/gmw/en*` files totalling 2,212) inside a **768 KB** partition.

`nano_g2p.c` itself: HOST-REF: arm64 `-Os` gives `__text` 6,552 B + 1,838 B of
constants and strings. Xtensa will differ; it is small either way.

### Internal SRAM — this is the problem

BUILD: both link maps agree exactly. `libespeak-ng.a` places **61,886 B of
`.bss` + 204 B of `.data` = 62,090 B in internal DRAM**, unconditionally, from
boot, whether or not you ever phonemize anything.

| symbol | bytes | file |
|---|---:|---|
| `phoneme_list` | 32,032 | `synthesize.c` |
| `ph_list2` | 8,000 | `translate.c` |
| `phoneme_tab_list` | 6,600 | `synthdata.c` |
| `wcmdq` | 2,720 | `wavegen.c` |
| `ssml_stack` | 1,520 | `ssml.c` |
| … | | |

More than half of that belongs to espeak's **formant synthesiser**, which this
port never calls. `synthesize.c` and `wavegen.c` are in the compile list only to
satisfy link references. That is the largest available saving if internal SRAM is
the binding constraint, and it is unexploited.

`nano_g2p.c` adds **7,096 B** of static workspace (HOST-REF: `sizeof` of its
workspace struct, reported at runtime by `nano_g2p_workspace_bytes()` and printed
by `nano_g2p_init()`). No dynamic allocation.

**Total static internal SRAM: ~69.2 KB before a single byte of heap or stack.**

### Stack

espeak's clause translator is deep. Recorded evidence, all from the existing
firmware, none of it an actual measurement:

- SRC: `C:\esp\espeaktest\sdkconfig.defaults:14` — `CONFIG_ESP_MAIN_TASK_STACK_SIZE=40960`.
  This is the origin of the "40 KB stack" figure. It is a chosen value; no
  high-water mark justifies it.
- SRC: `main/fsd_e2e.c:1747` — the shipped firmware runs espeak on a **48 KB
  stack allocated in PSRAM** (`MALLOC_CAP_SPIRAM`, enabled by
  `CONFIG_SPIRAM_ALLOW_STACK_EXTERNAL_MEMORY=y`), precisely to keep it out of
  internal SRAM.
- SRC: `main/fsd_e2e.c:1656-1659` — an earlier attempt ran espeak on the httpd
  task, which forced a 28 KB **internal** stack and "starved wifi TX → lwip OOM
  crash after send".

`uxTaskGetStackHighWaterMark()` is never called anywhere in that firmware, so the
real requirement is **not recorded**. If you have PSRAM, put the G2P task's stack
there.

### Coexistence with ESPHome — the honest answer

**On a 512 KB-internal-SRAM ESP32-S3 with no PSRAM: no.** The arithmetic does not
work. ~62 KB of espeak `.bss` + ~7 KB of workspace, plus WiFi's own internal
allocations (~50 KB and it wants contiguity), plus lwip buffers, the ESPHome API
task, the logger, and a ~100 KB **contiguous** synthesis arena, out of an
internal pool that is already ~180 KB smaller than the nominal 512 KB once IDF's
static allocations and the 32 KB DMA reservation are taken. The contiguous 100 KB
is the part that fails first: espeak's `.bss` is linked into the middle of DRAM
and fragments the largest free block. The existing firmware measured
`free heap after wifi+http: 55295` (DEVICE: `C:\esp\crashcap.txt:149`) **without**
espeak linked in at all.

**With the 8 MB octal PSRAM the existing board has: yes, and it is already
proven** — `mcu/ports/esp32s3/firmware/` is exactly that firmware: espeak-ng +
WiFi + HTTP server + the synthesis engine, all in one image. But it only fits
because of deliberate work:

- the synthesis arena and all float activation matrices live in a 4 MB **PSRAM**
  arena, not internal SRAM;
- the espeak task's 48 KB stack is in PSRAM;
- `CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL=2048` sends every allocation over 2 KB to
  PSRAM, which is what keeps the 168 KB `en_dict` out of internal RAM;
- the WiFi reservation was cut from 60 KB to 42 KB "once espeak forced the budget
  tight" (SRC: `main/fsd_e2e.c:377-380`);
- the boot golden self-test had to be **deleted** because "with espeak sharing
  internal RAM the arena can't fit the 391-frame golden" (SRC:
  `main/fsd_e2e.c:1752-1757`).

So: PSRAM is not an optimisation here, it is the precondition. Budget espeak at
**~62 KB of internal SRAM you cannot move** and put everything else you can in
PSRAM.

Two caveats on numbers you may have seen:

- The "~33 KB free at runtime" figure in `mcu/ports/esp32s3/README.md:74-75` is
  **unsourced** — it appears in no log, doc, or code in this repo or on the build
  host. Do not budget against it.
- `docs/s3-audio-handoff.md` **predates the espeak port** and contains no espeak
  SRAM ledger, despite `mcu/ports/esp32s3/README.md` pointing at it for exactly
  that. Its espeak section is about the *host* phoneme server.
- The free-heap number for a firmware with espeak **and** WiFi both linked is
  **not recorded anywhere**. `nano_g2p_init()` prints its workspace size; adding
  a `heap_caps_get_free_size(MALLOC_CAP_INTERNAL)` line next to it would close
  that gap on the first boot.

---

## Files

| file | what |
|---|---|
| `nano_g2p.c` / `include/nano_g2p.h` | the front end; the only file you call |
| `nano_g2p_table.h` | GENERATED — 59 codepoint→id entries + the E2M rewrite tables. Regenerate with `esphome/g2p/gen_nano_g2p_table.py`, verify with `--check` |
| `CMakeLists.txt` | IDF component registration, `-std=gnu11`, the 33-file list |
| `idf_component.yml` | manifest read by the IDF component manager |
| `config.h` | `USE_*=0`, `PATH_ESPEAK_DATA`, `PACKAGE_VERSION` |
| `vendor_espeak.sh` | clone upstream 1.52.0, copy the subset, apply the patch |
| `patch_speech.py` | `check_data_path()` → accept a flat SPIFFS mount |
| `mkspiffs_espeak.sh` | build the data image, print the offset (never flashes) |
| `COPYING.espeak-ng` | upstream GPL-3.0 text |
| `libespeak-ng/`, `ucd-tools/`, `include/espeak-ng/` | vendored, GPL-3.0 |
