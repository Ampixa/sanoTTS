#!/usr/bin/env bash
# Stage the embedded golden model into main/golden/ from the canonical test
# fixtures. Run before `idf.py build` (build.sh does this automatically).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="$here/../../../../../test/fixtures/en_us_r7"
for f in model_q8.bin front_q8.bin e2e_ids.bin e2e_durs.bin e2e_audio.bin; do
  cp "$src/$f" "$here/$f"
done
echo "staged golden model from $src"
