# extras/

Dev-time tooling for the SanoTTS Arduino library. Nothing here is compiled
into a sketch.

## Files

| File | What |
| --- | --- |
| `gen_golden_ids.py` | Turns an `e2e_ids.bin` (int32 LE Piper phoneme ids) into a small C header. Used to produce `examples/SpeakGolden/golden_ids.h`. |
| `host_check.sh` | Compiles every `src/*.c` + `src/SanoTTS.cpp` file standalone with plain `cc`/`c++` (proves the source set has no hidden Arduino/ESP-IDF dependency), then links the primary pipeline against the golden fixture and checks it still clears the upstream `mcu/` correctness gate. Run it from a full saanoTTS checkout: `./host_check.sh`. |
| `host_check_main.c` / `host_check_cpp_main.cpp` | The two test harnesses `host_check.sh` builds and runs (raw C API, then the `SanoTTS` C++ class). |

## Flashing the model blobs (LittleFS / SPIFFS)

The example sketch (`examples/SpeakGolden`) loads `front_q8.bin` (~280 KB)
and `model_q8.bin` (~400 KB) from a LittleFS filesystem image at runtime --
they are **not** embedded in the sketch. Get them from a saanoTTS release
(see the top-level `arduino/README.md`'s "Model blobs" section for the
download pointer), or from `mcu/test/fixtures/en_us_r7/` in a full saanoTTS
checkout if you just want to run the golden-fixture demo end to end.

Three ways to build+flash the filesystem image, pick whichever matches
your toolchain:

### PlatformIO (recommended -- built-in filesystem upload)

1. Put `front_q8.bin` and `model_q8.bin` in your PlatformIO project's
  `data/` directory (create it next to `platformio.ini` if it doesn't
  exist).
2. Set the filesystem in `platformio.ini`: `board_build.filesystem = littlefs`.
3. `pio run --target uploadfs` (builds the LittleFS image from `data/`
  and flashes it). Then `pio run --target upload` for the sketch itself.

### Arduino IDE 2.x

The IDE 2.x line dropped the old "Sketch Data Upload" menu item from
IDE 1.x; use the community `arduino-littlefs-upload` extension
(install via the IDE's boards/library-adjacent extension mechanism, or
see its README for the manual VS-Code-extension-style install), point it
at a `data/` folder next to your `.ino`, and it adds an upload-filesystem
command back to the IDE.

### Manual (arduino-cli + mklittlefs + esptool, no plugin)

Works anywhere `arduino-cli` and its bundled `mklittlefs`/`esptool_py`
tools are installed (`arduino-cli core install esp32:esp32` pulls both
into `~/Library/Arduino15/packages/esp32/tools/` or your platform's
equivalent):

```bash
mkdir -p /tmp/sanotts_fs && cp front_q8.bin model_q8.bin /tmp/sanotts_fs/
mklittlefs -c /tmp/sanotts_fs -s 0x170000 /tmp/littlefs.bin   # size = your board's FS partition
esptool.py --chip esp32s3 --port /dev/tty.YOUR_PORT write_flash 0x670000 /tmp/littlefs.bin
```

The two numbers (`-s` size and the flash offset `0x670000`) come from your
board's partition table -- check `Tools > Partition Scheme` in the IDE, or
your board's default `partitions.csv`; they are not universal constants.
Get them wrong and you'll either truncate the image or overwrite something
else in flash.

## Verifying self-containment / correctness without any board

```bash
cd arduino/extras
./host_check.sh
```

This is the check described in the top-level task and in
`arduino/README.md`'s "Compile verification" section -- it's the thing to
run after touching anything in `src/` before you trust a board flash.
