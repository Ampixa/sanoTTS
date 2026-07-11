# RP2040 port (Tier S, dual-core) — UNMEASURED

Pico-SDK port: scalar kernels (M0+ single-cycle multiplier), core-1
worker via the inter-core FIFO, XIP-flash residency dispatch. Build as
a pico-sdk app with `src/snt_tts.c + src/snt_kernels_ref.c` excluded
(this file provides the kernels) and the golden app pattern from
`ports/esp32c3/app`. RAM plan: ~200 KB arena fits the 264 KB SRAM for
short utterances; long utterances want the streaming build.

Projection (honest, unmeasured): 5-7x RT on the bundled voice —
C3-class per core, recovered by the genuine second core. Flash a board
and run the golden app to replace this estimate with a measurement.
