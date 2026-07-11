# sanoTTS — distill a tiny neural voice that runs anywhere

***sano*** — "sound" / "healthy" (Latin · Spanish · Italian · Esperanto). A lean,
*sound* neural TTS.

Distill a Piper/VITS teacher voice into a **sub-1M-parameter** neural TTS, then
run it with **no cloud and no NPU** — real-time on a ~$3 ESP32-S3 (out a GPIO into
an LM386 and a speaker), or in the browser via WASM.

The reference voice (en_US "Kristin") is **~745k int8 params**, runs **faster than
real time on the ESP32-S3**, and is intelligible on-device (Whisper WER ~18% with
on-chip espeak G2P). It's a full stack — duration + acoustic + iSTFT decoder —
distilled from the teacher, not a lookup of pre-rendered clips.

## Distill your own voice

The end-to-end recipe is [`docs/distillation-recipe.md`](docs/distillation-recipe.md):
build a probe pack from a Piper teacher → train the duration, acoustic-latent, and
decoder students → joint finetune → export int8. Porting to a new language is
[`docs/roota-language-porting-recipe.md`](docs/roota-language-porting-recipe.md).

```bash
pip install -e .
# then follow docs/distillation-recipe.md against any en_US Piper voice
```

## Deploy

- **ESP32-S3 talking device** — a standalone WiFi dashboard: type text, the board
  phonemizes (on-chip espeak-ng) and speaks. See
  [`mcu/ports/esp32s3/`](mcu/ports/esp32s3/).
- **Browser** — the full stack in WASM, no server. See [`web/`](web/).
- **Other MCUs** — which chips can run it and how well:
  [`docs/mcu-classes-and-porting.md`](docs/mcu-classes-and-porting.md).

## Verify your result

The eval loop measures what actually matters — intelligibility (Whisper WER),
phoneme-class fidelity, and G2P parity — not just a gameable MOS score:
`tools/eval_scorecard.py`, `tools/eval_phoneme_class_fidelity.py`,
`tools/eval_g2p_parity.py`.

## Layout

[`docs/repository-layout.md`](docs/repository-layout.md). In short: `src/saanotts/`
(package), `tools/` (pipeline + eval commands), `mcu/` (portable C runtime + device
ports), `web/` (browser demo), `configs/` + `data/textsets/` (contracts).

## License

GPLv3 — see [`LICENSE`](LICENSE). The distillation + G2P path builds on
[piper](https://github.com/OHF-Voice/piper1-gpl) and
[espeak-ng](https://github.com/espeak-ng/espeak-ng), both GPLv3, so the project
as a whole is GPLv3.

Copyright (C) 2026 Ampixa.
