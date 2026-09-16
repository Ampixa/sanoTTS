#!/usr/bin/env bash
# Build + run the E13b stage tracer against the golden fixtures.
#
#   bash tools/run_traces.sh [freeze]
#
#   freeze=1  validate: fixture-frozen durations, PCM must correlate >= 0.98
#             with the golden reference audio (the repo's own golden gate).
#   freeze=0  full-student traces (default): the inference path the paper
#             reports; this is what the site ships in public/traces/.
#
# Output: public/traces/<row_id>/{*.f32,*.i32,manifest.json}
set -euo pipefail
cd "$(dirname "$0")/.."

freeze="${1:-0}"
cc="${CC:-gcc}"

"$cc" -O2 -std=c99 -DSNT_NANO_TRACE \
  -Iengine/include -Iengine/src -Iengine/models/en_us_e13b -Iengine/trace \
  engine/src/snt_nano.c \
  engine/src/snt_kernels_ref.c \
  engine/ports/host/snt_port_host.c \
  engine/trace/nano_trace_main.c \
  -lm -o engine/trace/nano_trace

./engine/trace/nano_trace engine/fixtures/en_us_e13b public/traces "$freeze"
