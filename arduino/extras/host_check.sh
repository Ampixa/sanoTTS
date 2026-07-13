#!/usr/bin/env bash
# extras/host_check.sh -- proves the SanoTTS Arduino library's C sources
# are self-contained (compile standalone, no Arduino/PlatformIO toolchain
# needed) and that the primary 745k-model pipeline is still bit-correct
# after being copied out of mcu/ and patched with the sibilant-injection
# addition.
#
# Two phases:
#   1. Compile every src/*.c file to an object file with a plain
#      cc -std=c99 -c -- proves no file secretly depends on Arduino.h or
#      an ESP-IDF header to even compile.
#   2. Link the primary pipeline (snt_tts.c + snt_kernels_ref.c +
#      snt_port_default.c) against the shipped golden fixture
#      (mcu/test/fixtures/en_us_r7 in the parent saanoTTS checkout) and
#      run host_check_main.c's two-pass gate (default-off bit-exactness,
#      then a reachability+non-triviality check of the sibilant fix).
#
# Usage: ./extras/host_check.sh   (run from anywhere; paths are relative
# to this script's own location)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
lib="$here/.."
src="$lib/src"
CC=${CC:-cc}
CFLAGS="-O2 -std=c99 -Wall -Wextra -I$src"

echo "== phase 1: compile every src/*.c file standalone =="
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for f in "$src"/*.c; do
  echo "  cc -c $(basename "$f")"
  "$CC" $CFLAGS -DFSD_FAST_MATH -c "$f" -o "$tmp/$(basename "$f").o"
done
echo "OK: every src/*.c file compiles standalone (no Arduino/ESP-IDF headers required)"
echo

echo "== phase 1b: compile the C++ wrapper standalone (no Arduino.h needed) =="
CXX=${CXX:-c++}
"$CXX" -O2 -std=c++11 -Wall -Wextra -I"$src" -DFSD_FAST_MATH \
  -c "$src/SanoTTS.cpp" -o "$tmp/SanoTTS.cpp.o"
echo "OK: SanoTTS.cpp compiles standalone"
echo

echo "== phase 2: link + golden correctness gate =="
golden="$lib/../mcu/test/fixtures/en_us_r7"
if [ ! -d "$golden" ]; then
  echo "golden fixture not found at $golden -- this phase only runs from a"
  echo "full saanoTTS checkout (arduino/ is a subdirectory of it); skipping."
  exit 0
fi
"$CC" $CFLAGS -DFSD_FAST_MATH \
  "$src/snt_tts.c" "$src/snt_kernels_ref.c" "$src/snt_port_default.c" \
  "$here/host_check_main.c" -lm -o "$tmp/sanotts_host_check"
"$tmp/sanotts_host_check" "$golden"
echo

echo "== phase 3: same gate through the actual SanoTTS C++ class =="
# Mirrors how Arduino/PlatformIO actually build a library: .c files go
# through the C compiler, .cpp files through the C++ compiler, and the
# extern "C" declarations in SanoTTS.h/snt_tts.h/etc. make the two link
# together. (Handing snt_tts.c to $CXX directly, as if it were C++, is
# NOT how the real toolchains build it -- and fails, since C++'s stricter
# narrowing-conversion rules reject one of the C99 aggregate initializers
# in snt_tts.c. That is expected and not a library bug.)
"$CC"  $CFLAGS -DFSD_FAST_MATH -c "$src/snt_tts.c"          -o "$tmp/p3_snt_tts.o"
"$CC"  $CFLAGS -DFSD_FAST_MATH -c "$src/snt_kernels_ref.c"  -o "$tmp/p3_kernels.o"
"$CC"  $CFLAGS -DFSD_FAST_MATH -c "$src/snt_port_default.c" -o "$tmp/p3_port.o"
"$CC"  $CFLAGS -DFSD_FAST_MATH -c "$src/snt_front_f32.c"     -o "$tmp/p3_front_f32.o"
"$CC"  $CFLAGS -DFSD_FAST_MATH -c "$src/snt_piperlite_q8.c"  -o "$tmp/p3_piperlite_q8.o"
"$CXX" -O2 -std=c++11 -Wall -Wextra -I"$src" -DFSD_FAST_MATH -c "$src/SanoTTS.cpp" -o "$tmp/p3_wrapper.o"
"$CXX" -O2 -std=c++11 -Wall -Wextra -I"$src" -DFSD_FAST_MATH -c "$here/host_check_cpp_main.cpp" -o "$tmp/p3_main.o"
"$CXX" "$tmp/p3_snt_tts.o" "$tmp/p3_kernels.o" "$tmp/p3_port.o" "$tmp/p3_front_f32.o" \
  "$tmp/p3_piperlite_q8.o" "$tmp/p3_wrapper.o" "$tmp/p3_main.o" \
  -lm -o "$tmp/sanotts_cpp_check"
"$tmp/sanotts_cpp_check" "$golden"
