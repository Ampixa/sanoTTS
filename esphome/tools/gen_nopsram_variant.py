#!/usr/bin/env python3
"""Derive sanotts-s3-nopsram.yaml from sanotts-s3.yaml.

The two configs must differ in EXACTLY ONE THING -- whether PSRAM is enabled --
or the comparison between them says nothing. Hand-maintaining two copies is how
that invariant quietly breaks, so the variant is generated and this script is
the definition of the difference.

Run:  python3 esphome/tools/gen_nopsram_variant.py
      python3 esphome/tools/gen_nopsram_variant.py --check   (CI: no rewrite)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "sanotts-s3.yaml"
DST = HERE / "sanotts-s3-nopsram.yaml"

PSRAM_BLOCK = """      # PSRAM is present on this part and ESPHome will use it for its own
      # allocations, which is exactly the situation a real satellite is in.
      # The synthesis arena is NOT allowed to land there -- see sanotts.h: a
      # PSRAM-backed arena is correct but ~5.7x slower because the PIE vector
      # kernels cannot read it, and the component logs a warning and reports
      # arena_internal=0 if it ever happens.
      CONFIG_SPIRAM: y
      CONFIG_SPIRAM_MODE_OCT: y
      CONFIG_SPIRAM_SPEED_80M: y
      CONFIG_SPIRAM_USE_CAPS_ALLOC: y
      # Measured necessity, not tuning. Without this the largest free INTERNAL
      # block after ESPHome is up is 114,688 B, and the 415-frame golden gate
      # row needs 128,944 B in one piece -- so the correctness gate cannot even
      # run. WiFi and lwIP are the biggest internal-heap consumers on the part
      # and their buffers do not need to be SIMD-readable; the synthesis arena
      # does. Moving them out is what buys the arena its contiguous block.
      CONFIG_SPIRAM_TRY_ALLOCATE_WIFI_LWIP: y"""

NO_PSRAM_BLOCK = """      # PSRAM DELIBERATELY OFF. This board physically has 8 MB of octal PSRAM;
      # this variant switches it off in software so the part behaves as a
      # 512 KB-internal-SRAM-only ESP32-S3. That is the configuration the
      # feasibility question is really about -- the Home Assistant Voice PE
      # class of hardware runs with only 80-140 KB of free internal heap on a
      # 512 KB part -- and it is the one case a PSRAM-equipped devkit would
      # otherwise flatter. Nothing else differs from sanotts-s3.yaml.
      CONFIG_SPIRAM: n"""

PSRAM_COMPONENT = """psram:
  mode: octal
  speed: 80MHz

"""

HEADER = (
    "# sanoTTS on an ESP32-S3 with PSRAM DISABLED -- the 512 KB-internal-only case.\n"
    "# Generated from sanotts-s3.yaml by esphome/tools/gen_nopsram_variant.py;\n"
    "# edit that script, not this file.\n"
    "#\n"
    "# sanoTTS on an ESP32-S3, inside ESPHome."
)


def render() -> str:
    if not SRC.is_file():
        raise SystemExit(f"missing source config: {SRC}")
    text = SRC.read_text()
    for needle in ("  name: sanotts-s3", PSRAM_BLOCK, PSRAM_COMPONENT,
                   "# sanoTTS on an ESP32-S3, inside ESPHome."):
        if needle not in text:
            raise SystemExit(
                "sanotts-s3.yaml no longer contains a block this generator rewrites:\n"
                f"---\n{needle}\n---\n"
                "Update gen_nopsram_variant.py rather than editing the variant by hand."
            )
    text = text.replace("  name: sanotts-s3", "  name: sanotts-s3-nopsram", 1)
    text = text.replace(PSRAM_BLOCK, NO_PSRAM_BLOCK, 1)
    text = text.replace(PSRAM_COMPONENT, "", 1)
    text = text.replace("# sanoTTS on an ESP32-S3, inside ESPHome.", HEADER, 1)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail if the checked-in variant is not what this would write")
    args = ap.parse_args()

    want = render()
    if args.check:
        if not DST.is_file():
            print(f"{DST} does not exist", file=sys.stderr)
            return 1
        if DST.read_text() != want:
            print(f"{DST} is stale -- rerun without --check", file=sys.stderr)
            return 1
        print(f"{DST.name} is up to date")
        return 0
    DST.write_text(want)
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
