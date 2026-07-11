# en_US R7 Golden Fixture

This immutable fixture makes the portable host correctness gate independent of
the legacy `esp32c3/` tree. It contains the quantized front half, R7 decoder,
phoneme IDs, frozen durations, and expected float32 audio used by
`mcu/test/golden_main.c`.

The bytes match the corrected MCU package in the local Kristin 2026-07-08
release and the verified `tts-preservation-20260710` draft archive. Regenerate
or replace the complete fixture as one versioned contract; never edit an
individual binary in place.

Verify with:

```bash
cd mcu/test/fixtures/en_us_r7
shasum -a 256 -c SHA256SUMS
cd ../../../..
make -C mcu test
```
