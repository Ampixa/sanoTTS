#!/usr/bin/env bash
# Build the espeak SPIFFS image and print the offset to flash it at.
#
# The espeak data is NOT embedded in the firmware: espeak reads it with fopen(),
# so it has to live on a filesystem. This packs the repo's 275 KB minimal en-US
# data set into a SPIFFS image sized to match the `espeak` partition.
#
#   ./mkspiffs_espeak.sh <out.bin> [data-dir] [partition-size-bytes]
#
# Defaults: data-dir  mcu/ports/esp32s3/firmware/espeak-ng-data
#           size      0xC0000 (768 KB), matching
#                     mcu/ports/esp32s3/firmware/partitions.csv
#
# Needs `mkspiffs` (https://github.com/igrr/mkspiffs), or ESP-IDF's bundled
# spiffsgen.py, which this prefers when IDF_PATH is set.
#
# DOES NOT FLASH ANYTHING. It prints the esptool command; run it yourself.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${here}/../../../.." && pwd)"

out="${1:-}"
data_dir="${2:-${repo_root}/mcu/ports/esp32s3/firmware/espeak-ng-data}"
size="${3:-786432}"   # 0xC0000

if [ -z "${out}" ]; then
    echo "usage: $0 <out.bin> [data-dir] [partition-size-bytes]" >&2
    exit 2
fi
if [ ! -d "${data_dir}" ]; then
    echo "error: data dir ${data_dir} does not exist" >&2
    exit 1
fi
if [ ! -f "${data_dir}/phontab" ]; then
    echo "error: ${data_dir} has no phontab; that is not an espeak data dir" >&2
    exit 1
fi

# README.md in the data dir documents the set; it must not be flashed, since
# every byte of the partition is budget.
staging="$(mktemp -d)"
trap 'rm -rf "${staging}"' EXIT
( cd "${data_dir}" && tar cf - --exclude README.md . ) | ( cd "${staging}" && tar xf - )

payload=$(find "${staging}" -type f -exec stat -f %z {} + 2>/dev/null | awk '{s+=$1} END{print s+0}' \
    || find "${staging}" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
echo "payload: ${payload} bytes from ${data_dir}"
if [ "${payload}" -ge "${size}" ]; then
    echo "error: payload ${payload} does not fit a ${size}-byte partition" >&2
    exit 1
fi

if [ -n "${IDF_PATH:-}" ] && [ -f "${IDF_PATH}/components/spiffs/spiffsgen.py" ]; then
    echo "using spiffsgen.py from ${IDF_PATH}"
    python3 "${IDF_PATH}/components/spiffs/spiffsgen.py" "${size}" "${staging}" "${out}"
elif command -v mkspiffs >/dev/null 2>&1; then
    echo "using mkspiffs"
    # Page/block sizes must match CONFIG_SPIFFS_* in sdkconfig; these are the
    # IDF defaults that the existing firmware uses.
    mkspiffs -c "${staging}" -b 4096 -p 256 -s "${size}" "${out}"
else
    echo "error: neither IDF_PATH/spiffsgen.py nor mkspiffs is available" >&2
    exit 1
fi

echo
echo "wrote ${out} ($(stat -f %z "${out}" 2>/dev/null || stat -c %s "${out}") bytes)"
echo
echo "The offset comes from the partition table, never from this script."
echo "Read it back with:"
echo "    esptool.py --chip esp32s3 read_flash 0x8000 0xc00 pt.bin && \\"
echo "        python3 \$IDF_PATH/components/partition_table/parttool.py \\"
echo "            --partition-table-file pt.bin get_partition_info \\"
echo "            --partition-name espeak --info offset size"
echo
echo "On the firmware in mcu/ports/esp32s3 that partition sits at 0x390000"
echo "(DEVICE: mcu/ports/esp32s3/measurements/06-FINAL-labelled-COM5.log:24,"
echo "  'espeak Unknown data 01 82 00390000 000c0000'), giving:"
echo "    esptool.py --chip esp32s3 write_flash 0x390000 ${out}"
echo
echo "An ESPHome build defines its own partition layout, so CONFIRM the offset"
echo "against your build's partition table before writing anything."
