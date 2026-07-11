# Host-side tools for the ESP32-S3 TTS port

Validation and operator scripts that run on a desktop, not the board.

| file | purpose |
| --- | --- |
| `espk_ids_host.c` + `cp_id_table.h` | The exact on-device G2P logic (espeak IPA → codepoint→id → BOS/pad/EOS), linked against the system `libespeak-ng`. Used to validate parity before flashing: `cc -O2 -I/opt/homebrew/include espk_ids_host.c -lespeak-ng -o espk_ids && ./espk_ids "your sentence"`. Compare its ids to the board's via `tools/eval_g2p_parity.py`. |
| `phoneme_server.py` | Optional desktop phonemizer (`GET /ids?text=...` → CSV ids) using piper's espeak. Pre-dates the on-chip port; kept for A/B testing the board's G2P against desktop espeak. |
| `speak_via_espeak.py` | Drives the board's `/api/speak` with desktop-espeak ids (bypasses the on-chip G2P) for isolating engine vs frontend. |
| `serial_capture.py` | Reads the board's UART without resetting it (RTS/DTR held), ascii-safe — for watching runtime logs. |

The full G2P + WER eval loop lives in the repo `tools/`:
`eval_g2p_parity.py`, `g2p_parity_probe.py`, `render_crude_frontend.py`,
`render_eval_dirs.py`, `calibrate_sibilant_noise_fsd.py`.
