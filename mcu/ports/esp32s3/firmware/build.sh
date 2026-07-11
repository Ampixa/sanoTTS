#!/usr/bin/env bash
# Build the standalone TTS firmware. Assumes ESP-IDF v6.x is installed and
# `. $IDF_PATH/export.sh` has been sourced (or run it here). The espeak-ng
# source must be populated first (components/espeak-ng/README.md).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
main/golden/stage.sh                       # stage embedded model from test fixtures
idf.py set-target esp32s3
idf.py build
echo "built. flash with: idf.py -p <PORT> flash monitor"
