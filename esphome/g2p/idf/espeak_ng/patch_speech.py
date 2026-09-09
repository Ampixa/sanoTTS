#!/usr/bin/env python3
"""Make espeak-ng's `check_data_path()` accept a SPIFFS mount point.

`check_data_path()` decides a directory holds the espeak data by asking
`GetFileLength()` for it and requiring the answer `-EISDIR`. SPIFFS is a flat
filesystem with no directories to stat, so on the ESP32 that check always fails,
`espeak_ng_InitializePath()` silently keeps the compiled-in fallback path, and
the first phonemize call dies looking for `phontab`.

The patch keeps the original test and adds a fallback: if `<path>/phontab` can
be opened for reading, the path holds the data. That is true on SPIFFS and
equally true on a normal filesystem, so the patched source still behaves
correctly on the host build.

This is a rewrite of
`mcu/ports/esp32s3/firmware/components/espeak-ng/patch_speech.py`, which
hardcodes `C:\\esp\\espeaktest\\components\\espeak-ng\\libespeak-ng\\speech.c`
and therefore cannot run on any machine but the one Windows build host. This
version takes the file as an argument.

    python3 patch_speech.py path/to/libespeak-ng/speech.c

Idempotent: a second run reports "already patched" and exits 0.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

OLD = (
    '\tsnprintf(path_home, sizeof(path_home), "%s", path);\n'
    "\treturn GetFileLength(path_home) == -EISDIR;\n"
)

NEW = (
    '\tsnprintf(path_home, sizeof(path_home), "%s", path);\n'
    "\tif (GetFileLength(path_home) == -EISDIR) return 1;\n"
    "\t/* ESP/SPIFFS: flat fs, no dir stat -- accept if a data file is readable here */\n"
    "\t{ char pb[sizeof(path_home)+16]; snprintf(pb, sizeof(pb), \"%s/phontab\", path_home);\n"
    "\t  FILE *pf = fopen(pb, \"rb\"); if (pf) { fclose(pf); return 1; } }\n"
    "\treturn 0;\n"
)

MARKER = "ESP/SPIFFS: flat fs, no dir stat"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("speech_c", type=Path, help="path to libespeak-ng/speech.c")
    args = parser.parse_args()

    if not args.speech_c.is_file():
        print(f"error: {args.speech_c} does not exist", file=sys.stderr)
        return 2

    try:
        source = io.open(args.speech_c, encoding="utf-8", errors="strict").read()
    except (OSError, UnicodeDecodeError) as exc:
        print(f"error: cannot read {args.speech_c}: {exc}", file=sys.stderr)
        return 2

    if MARKER in source:
        print(f"already patched: {args.speech_c}")
        return 0

    if OLD not in source:
        print(
            f"error: the check_data_path() pattern was not found in {args.speech_c}.\n"
            "       This patch is written against espeak-ng 1.52.0; upstream changed.",
            file=sys.stderr,
        )
        return 1

    patched = source.replace(OLD, NEW, 1)
    try:
        io.open(args.speech_c, "w", encoding="utf-8").write(patched)
    except OSError as exc:
        print(f"error: cannot write {args.speech_c}: {exc}", file=sys.stderr)
        return 2

    print(f"patched check_data_path() in {args.speech_c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
