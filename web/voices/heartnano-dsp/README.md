# heart-nano (sanotts 0.5.7), browser bundle

This is the **sanotts 0.5.7 `heart-nano` voice**: 104,215 parameters (duration
14,746 + acoustic 42,235 + harmonic-plus-noise decoder 47,234), English,
24 kHz, distilled from Kokoro-82M `af_heart`. `web/index.html` plays it as
`heart-nano` through `web/heartnano_dsp.js`, a plain-JavaScript port of the
Python package (`sanotts/nano.py` local-feature front end and
`sanotts/nano_dsp.py` decoder).

The 294k heart-nano that this replaces in the picker is still in
`web/voices/heartnano/` and `web/snt_nano_heartnano.{js,wasm}`, unchanged, so
deployed older copies of the page keep working.

## Files

| File | Bytes | sha256 | What it is |
|---|---|---|---|
| `front_q8.bin` | 75,504 | `a8681189df2e8299800c56179b66d8c274cc6756cee2114492ef9b6898354acb` | int8 front end (durations + acoustic model), **byte-identical** to the package's `front_q8.bin` |
| `dsp_decoder_f32.bin` | 188,936 | `c453281fb2b408ec93af079e12838fd343857fb41b54397a10e77d1947b2ba15` | the 34 float tensors of the package's `dsp_decoder.npz` (47,234 floats), little-endian float32, concatenated; lossless (the npz stores float32) |
| `meta.json` | 8,723 | `57384535034491910feb3f1cf30faec5b345a9f50e0bf7a1f38560d868a790d1` | byte offsets and shapes from `nano_q8_meta.h` (`offsets`), the tensor table (`dsp_tensors`: name, shape, float offset), the decoder's integer fields (`dsp_config`), blob sizes and sha256s, the synthesis seed, and the source hashes (`source`) |

The page checks both blobs' size and sha256 against `meta.json` when it
fetches them (`fetchVoiceFile`), and `SaanoNanoDsp.createVoice` re-checks sizes,
offsets and tensor shapes and throws on any mismatch.

The package's `model_q8.bin` (the 0.5.6 neural decoder, kept in the package
only for sanotts <= 0.5.6) is not used by this voice and is not copied.

## Source

Hugging Face [`ampixa/sanoTTS`](https://huggingface.co/ampixa/sanoTTS), folder
`heartnano-dsp/`, revision `091814b27c17febea3b5951f0a836e5c30506636`
(the sanotts 0.5.7 voice package). Downloaded 2026-11-26 and verified against
that package's `meta.json`:

| Original file | sha256 |
|---|---|
| `meta.json` | `171729d2f5a830547d988bea0980dc02c3947359c70435cb152a430de0d4a04b` |
| `front_q8.bin` | `a8681189df2e8299800c56179b66d8c274cc6756cee2114492ef9b6898354acb` |
| `nano_q8_meta.h` | `fb6298b5d2cb595139151796374e32d87532f05f6b7340d942f2d963e68d5692` |
| `dsp_decoder.npz` | `caca0d0ab83ff2b774e61d28bd5dd409e053ba9f8632b48a3f1c8a003fc44029` |

Lineage (from the package's `meta.json`): heartnanos front end (k3 local
features) + harmonic-plus-noise decoder hn-s48 (ControlNet dim 48, 80
harmonics with minimum-phase phases, 512-point noise), 65k steps, EMA;
exported 2026-11-24.

This folder is not yet mirrored under `web/voices/heartnano-dsp/` on Hugging
Face. The page tries Hugging Face first and falls back to GitHub, then to this
site's copy.

## How it was made

```bash
REV=091814b27c17febea3b5951f0a836e5c30506636
mkdir -p /tmp/hn && for f in dsp_decoder.npz front_q8.bin nano_q8_meta.h meta.json; do
  curl -sSfL -o /tmp/hn/$f https://huggingface.co/ampixa/sanoTTS/resolve/$REV/heartnano-dsp/$f; done
python3 -I tools/convert_heartnano_dsp_web.py /tmp/hn web/voices/heartnano-dsp --revision $REV
```

The converter checks every input's sha256 against the package `meta.json`,
reads each tensor back from the written `.bin` bit for bit, and is
deterministic: rerunning it gives byte-identical files.

## Verification

`tools/dump_heartnano_dsp_reference.py` (sanotts 0.5.7) writes the package's
durations, mel frames and waveforms for 12 rows (the 8 `en_us_e13b` golden
fixture id rows and 4 sentences through the package's own G2P), and
`tools/verify_heartnano_dsp_node.mjs` compares this runtime against them. On
2026-11-26 (node v26.4.0): durations and frame counts were identical on every
row; the decoder alone (JS on the package's mel) was within 8.2e-8 of
`nano_dsp.render`; end to end (ids -> audio) the minimum correlation with
`Synthesizer.synthesize_ids` was 0.99999991 and the largest sample difference
4.5e-4 (peak 0.47). Synthesis ran at 16 ms per second of audio (62x real time).

## Licence

GPL-3.0-or-later, as the rest of `ampixa/sanoTTS` (see the repository
`LICENSE`). The teacher, Kokoro-82M, is Apache-2.0.
