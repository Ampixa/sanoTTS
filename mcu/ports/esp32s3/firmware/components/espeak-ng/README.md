# espeak-ng ESP32-S3 component

Real espeak-ng grapheme-to-phoneme, compiled for the ESP32-S3 (translator only —
no audio synthesizer). This directory holds only the **port-specific** files; the
~2500-file upstream source is not vendored. Populate it once:

## Setup

```bash
# 1. Clone the matching upstream tag next to this project
git clone --branch 1.52.0 --depth 1 https://github.com/espeak-ng/espeak-ng.git

# 2. Copy the translator subset into this component
cp -R espeak-ng/src/libespeak-ng      ./libespeak-ng
cp -R espeak-ng/src/ucd-tools/src     ./ucd-tools
cp -R espeak-ng/src/include           ./include

# 3. Apply the SPIFFS data-path patch (flat FS has no dir stat)
python3 patch_speech.py               # edits libespeak-ng/speech.c in place
```

`CMakeLists.txt` lists the 27 translator `.c` files + 6 ucd-tools files and
compiles them `-std=gnu11` (IDF defaults to gnu23, which turns espeak's old-C
implicit int→pointer into hard errors). `config.h` sets all `USE_*` to 0 and
defines `PATH_ESPEAK_DATA` (the upstream fallback macro is a broken comma
expression under C23).

## The three port fixes (all required)

1. **`-std=gnu11`** — see above.
2. **`config.h`** — `USE_*=0`, `PATH_ESPEAK_DATA` defined, `PACKAGE_VERSION`.
3. **`patch_speech.py`** — `check_data_path()` requires `stat()` to report the
   data dir as a directory (`-EISDIR`); SPIFFS is flat and can't, so the patch
   accepts the path when `<path>/phontab` is readable. Data is mounted at
   `/espeak` and loaded with `espeak_ng_InitializePath("/espeak")`, voice `"en"`.

The wrapper that turns espeak output into Kristin phoneme IDs is in
`../../main/esp_g2p.c` (init: SPIFFS mount + espeak init; `esp_g2p_text_to_ids`).

## Version parity

Piper bundles espeak-ng **1.52.0.1**; this port builds **1.52.0**, a ~2.7%
phoneme-id drift (identical G2P logic, minor dictionary/rule differences). Use
1.52.0.1's source if you need exact parity with the training-time phonemizer.
