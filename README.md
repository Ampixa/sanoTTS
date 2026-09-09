# sanoTTS — a tiny neural voice that runs anywhere

***sano*** (सानो) — Nepali for **"small."** A family of tiny neural text-to-speech
voices — **294k to 2.3M parameters** — that run with **no cloud and no NPU**:
real-time on a ~$3 ESP32-S3 (out a GPIO into an LM386 and a speaker), or live
in the browser via WASM.

![sanoTTS — thirty tiny voices, sixteen languages, browser + $3 chip](docs/assets/saanotts-hero-v2.png)

- smallest neural TTS family known — **294k to 2.3M parameters**
- runs **real-time on a $3 microcontroller** — measured **RTF 0.383 on an
  ESP32-S3**, 2.6x faster than real time, with the Xtensa LX7 SIMD kernels the
  Arduino library assembles by default ([BOARDS.md](BOARDS.md), 2026-09-04).
  Scalar C on the same board is 1.58, i.e. slower than real time
- runs right in your browser — **WebAssembly**, no server
- **0.3 to 8.7 MB** per voice depending on precision — 337 KB int8 (heart-nano),
  ~3 MB fp16, 5.5-8.7 MB fp32 — zero dependencies (espeak-ng phonemizer included)
- **30 voices** across **16 languages** in the browser demo — English, German,
  French, Spanish, Italian, Portuguese, Russian, Czech, Romanian, Turkish,
  Arabic, Nepali (नेपाली), Hindi (हिन्दी), Vietnamese (Tiếng Việt),
  Indonesian (Bahasa), Chinese (中文). The pip package ships 27 of them across
  13 languages; Nepali, Hindi and Chinese are browser-only for now
- **new (2026-09-08):** **ten more languages** — German, Turkish, Russian, French,
  Spanish, Italian, Portuguese, Romanian, Czech and Arabic — each ~1.56M
  parameters, and eight of them also in a ~511k variant. Whisper word error rate
  on out-of-domain text runs 0.038 (Portuguese) to 0.392 (Romanian); that is
  intelligibility over 16 sentences per language, not naturalness, and no one has
  run a listening test. Numbers and method in
  [`experiments/evidence/ood-tatoeba-20260908.json`](experiments/evidence/ood-tatoeba-20260908.json)
- **heart**, our best-sounding voice at **2.27M parameters** (24 kHz), and
  **heart-nano**, the same voice in **294k parameters** — a complete text-to-speech
  stack, int8, in 337 KB. Both synthesize live in the browser demo
- open source, **GPL-3.0**

## Live demo

**[ampixa.github.io/sanoTTS](https://ampixa.github.io/sanoTTS/)** — every voice
synthesizes your text live in the browser. No server, no upload: text goes
through an espeak-ng-in-WASM phonemizer and that voice's own neural stack, all
client-side.

## Install & use

| Platform | Install | Then |
| --- | --- | --- |
| Python | `pip install sanotts` | `sanotts say "Hello" --voice amy -o hello.wav` |
| Web (npm) | `npm install sanotts-web` | `const tts = await SanoTTS.load(); await tts.synthesize('Hello', {voice:'amy'})` |
| Web (no build) | copy `dist/` + `voices/` | see [Deploy on your own site](#deploy-on-your-own-site) |
| Arduino / PlatformIO | zip-install or `lib_deps = https://github.com/Ampixa/sanoTTS.git` | [`arduino/README.md`](arduino/README.md) |
| Hugging Face | [`huggingface.co/ampixa/sanoTTS`](https://huggingface.co/ampixa/sanoTTS) | voice packages + samples |
| Browser | nothing | [ampixa.github.io/sanoTTS](https://ampixa.github.io/sanoTTS/) |

### Python

CLI + library; voices download on first use:

```bash
pip install sanotts

sanotts say "Hello from a two megabyte voice." --voice amy -o hello.wav
sanotts say "Xin chào!" --voice vi -o xinchao.wav
```

```python
import sanotts
result = sanotts.synthesize("Hello world", voice="amy")   # numpy audio @ 22.05 kHz
```

Voices, largest to smallest: `heart` (2.27M), `hfc` and `amy-1p8m` (1.8M),
`amy`, `kristin`, `vi` and `id` (1.46M), `amy-1p1m` (1.1M), and `heart-nano`
(294k, 337 KB of weights).

They are fetched from
[huggingface.co/ampixa/sanoTTS](https://huggingface.co/ampixa/sanoTTS) into
`~/.cache/sanotts/`, and fall back to the GitHub
[voices-v1](https://github.com/Ampixa/sanoTTS/releases/tag/voices-v1) and
[voices-v2](https://github.com/Ampixa/sanoTTS/releases/tag/voices-v2) releases
if Hugging Face cannot be reached. Set `SANOTTS_VOICE_SOURCE=hf` or `=github`
to pin one host. Pure numpy inference, no torch, no onnxruntime.

### Arduino / PlatformIO

The `SanoTTS` library lives in [`arduino/`](arduino/): add it to the Arduino
IDE as a .zip library, or in `platformio.ini`:

```ini
lib_deps = https://github.com/Ampixa/sanoTTS.git
```

See [`arduino/README.md`](arduino/README.md) for board support (ESP32-S3 ✓),
memory guidance, and flashing the model blobs
([mcu-kristin-745k-q8.tar.gz](https://github.com/Ampixa/sanoTTS/releases/tag/voices-v1)).

### Hugging Face

Every voice package, plus audio samples, lives at
[huggingface.co/ampixa/sanoTTS](https://huggingface.co/ampixa/sanoTTS). That is
where `pip install sanotts` downloads from by default; the GitHub releases are
the fallback.

### Browser

Nothing to install: [tts.ampixa.com/sanoTTS](https://tts.ampixa.com/sanoTTS).

### Deploy on your own site

sanoTTS's browser demo runs entirely client-side — WebAssembly, no server —
so "deploying" it just means hosting a handful of static files. Two ways to
do it:

**Option A — `npm install sanotts-web`** ([on npm](https://www.npmjs.com/package/sanotts-web))

```js
import { SanoTTS, playAudio } from 'sanotts-web';

const tts = await SanoTTS.load({
  assetBase: 'https://your-cdn.example.com/sanotts/',   // where you copied dist/
});
const result = await tts.synthesize('Hello from my own server.', {
  voice: 'amy',
  voiceBase: 'https://your-cdn.example.com/sanotts/',   // where you copied voices/
});
playAudio(result);
```

Copy the package's `dist/` (the wasm runtime) and this repo's `web/voices/`
directory to your own static host, then point `assetBase`/`voiceBase` at it.
Everything else — phonemization, synthesis, playback — happens in the
visitor's browser.

**Option B — no build, no npm**

Copy `web/snt_g2p.js`, `web/snt_g2p.wasm`, `web/snt_g2p.data`,
`web/snt_voice.js`, `web/snt_voice.wasm`, and `web/voices/` from this repo
(or scrape them straight from `ampixa.github.io/sanoTTS`) onto your static
host, and load them the same way `web/index.html` does:

```html
<script src="/sanotts/snt_g2p.js"></script>
<script src="/sanotts/snt_voice.js"></script>
<script type="module">
  const [G2P, Voice] = await Promise.all([SaanoG2P(), SaanoVoice()]);
  G2P._snt_g2p_init();
  // ...set voice, phonemize, synthesize — see web/index.html for the full sequence.
</script>
```

**Sizes to plan around:**

- wasm runtime: ~700 KB total gzipped over the wire (espeak-ng G2P
  ships ~2.5 MB uncompressed including its phoneme-table `.data`, ~700 KB
  gzipped; the acoustic/decoder wasm adds another ~40 KB)
- per-voice weights, fetched lazily on first use of that voice and never
  bundled with the runtime:
  - **int8** — heart-nano at **337 KB**, the on-device stack unchanged
  - **fp16** — the ten voices added 2026-09-08 at **~3.0 MB** each; fp16 is the
    reference for these, not an approximation, since every number they have was
    measured on the fp16 package. The browser widens them to fp32 after
    download and checks a sha256 of the widened bytes
  - **fp32** — the twelve older voices, **5.5 to 8.7 MB** (amy 5.6, Vietnamese
    6.0, heart 8.7). Converting these to fp16 would halve them, and is not done
    only because their published output is the fp32 output

**CSP note:** the wasm runtime needs `'wasm-unsafe-eval'` (or
`'unsafe-eval'` on older browsers) in your `script-src` Content-Security-Policy,
for `WebAssembly.instantiate`/`instantiateStreaming`. Nothing else needs
relaxing — the runtime never `eval()`s JavaScript. Most default/modern CSPs
(including having no explicit `script-src`) already allow this.

## How it stacks up

Open small-scale TTS on an honest gate — a diverse 24-sentence set scored with the
**same** no-reference suite (SCOREQ / UTMOS are naturalness predictors, DNSMOS-SIG
is signal quality; higher is better). Parameter counts are inference-time and
exclude the shared external G2P.

![Size comparison: sanoTTS 0.75M-1.8M params vs TinyTTS 1.62M vs Inflect Nano 4.63M vs Kokoro 82M, linear axis](docs/assets/chart-size-comparison.svg)

Kokoro is 36x larger than our largest voice (heart, 2.27M), and 279x larger than
our smallest (heart-nano, 294k).
Shipped-file sizes: sanoTTS amy 2.8 MB fp16 and TinyTTS 3.5 MB fp16, both
verified from the released files; Kokoro's ~330 MB fp32 is its widely cited
public figure.

| System | Params | SCOREQ | UTMOS | DNS-SIG |
| --- | ---: | :---: | :---: | :---: |
| **sanoTTS (amy)** | **1.46 M** | **4.13** | **4.10** | 3.61 |
| sanoTTS (heart) | 2.27 M | 3.48 | 3.38 | 3.51 |
| sanoTTS (heart-nano) | 0.29 M | 2.29 | 2.45 | 3.35 |
| TinyTTS | 1.62 M | 3.94 | 3.65 | **3.62** |
| Inflect Nano | 4.63 M | 3.81 | 3.65 | 3.58 |
| Kitten TTS nano | 15 M | 3.02 | 3.58 | 3.43 |
| _reference (~15 M)_ | _~15 M_ | _4.71_ | _4.47_ | _3.65_ |
| _Kokoro_ | _82 M_ | _4.89_ | _4.52_ | _3.69_ |

sanoTTS is the **smallest** model here and the **best on naturalness (SCOREQ and
UTMOS) among everything up to 15M params** — beating TinyTTS while being smaller.
On DNSMOS-SIG, TinyTTS edges us by 0.01 — no single metric tells the whole story.
It's the only one that runs a full neural stack on a $3 MCU. Parameter count
isn't destiny at this scale: Kitten TTS at 10x the size scores a full SCOREQ
point lower. The frontier only pulls ahead at ~15M-class models and Kokoro (82M,
60x larger) — a gap we don't claim to close. Reproduce it with
`tools/eval_mos_all.py` + `tools/eval_scorecard.py`.

![The size-quality frontier: SCOREQ rising from 3.70 to 4.16 as decoder size grows from 1.09M to 1.84M params, same voice](docs/assets/chart-frontier.svg)

Same voice (amy), same duration/acoustic recipe — only decoder size changes.
Quality lives in the decoder: doubling it from 1.09M to 1.84M params moves
SCOREQ from 3.70 to 4.16.

## The voices

| Language | Voice | Params | SCOREQ | WER |
| --- | --- | ---: | :---: | :---: |
| English 🇺🇸 | amy | 1.45 M | 4.13 | — |
| | kristin | 1.40 M | 4.09 | — |
| | hfc | 1.83 M | 3.94 | — |
| | amy-small | 1.08 M | 3.70 | — |
| | heart (24 kHz) | 2.27 M | 3.48 | — |
| | heart-nano (int8, 24 kHz) | 294 k | 2.29 | — |
| | robot (on-device, int8) | 567 k | — | — |
| German 🇩🇪 | German | 1.57 M | — | 0.099 |
| | German small | 512 k | — | 0.145 |
| Turkish 🇹🇷 | Turkish | 1.56 M | — | 0.303 |
| | Turkish small | 510 k | — | — |
| Russian 🇷🇺 | Russian | 1.57 M | — | 0.120 |
| | Russian small | 512 k | — | — |
| French 🇫🇷 | French | 1.57 M | — | 0.221 |
| Spanish 🇪🇸 | Spanish | 1.56 M | — | 0.147 |
| | Spanish small | 510 k | — | — |
| Italian 🇮🇹 | Italian | 1.57 M | — | 0.080 |
| | Italian small | 512 k | — | — |
| Portuguese 🇧🇷 | Portuguese | 1.57 M | — | 0.038 |
| | Portuguese small | 512 k | — | — |
| Romanian 🇷🇴 | Romanian | 1.57 M | — | 0.392 |
| | Romanian small | 512 k | — | — |
| Czech 🇨🇿 | Czech | 1.57 M | — | 0.351 |
| | Czech small | 513 k | — | — |
| Arabic 🇯🇴 | Arabic | 1.57 M | — | 0.274 |
| Nepali नेपाली | Nepali | 1.47 M | — | — |
| Hindi हिन्दी | Hindi | 1.50 M | — | — |
| Vietnamese Tiếng Việt | Vietnamese | 1.57 M | 1.53 | 0.468 |
| Indonesian Bahasa | Indonesian | 1.56 M | 1.71 | 0.256 |
| Chinese 中文 | Chinese | 1.50 M | — | — |

**Two different questions, and neither column answers the other.** SCOREQ is a
no-reference quality predictor — how it sounds. WER is Whisper word error rate —
whether the words arrive. A voice can score well on one and badly on the other,
and on the two voices where we have both, the second number is much less
flattering: Indonesian reads 0.256 word error, so the words do arrive, and
still scores 1.71 against amy's 4.13.

An empty cell means unmeasured, not zero. The English SCOREQ figures share one
eval set; the Indonesian and Vietnamese ones come from a different set (24
held-out Tatoeba sentences each), so the column ranks within a language rather
than across them — SCOREQ is a predictor trained largely on English and
comparing it across languages is not an established use of it. WER for the
ten languages added on 2026-09-08 comes from 16 held-out Tatoeba sentences each,
out of domain from the corpus they were distilled on — directional, not precise.
Nepali, Hindi and Chinese predate that harness and have neither number.
**No listening test has been run on any non-English voice.**

The "robot" row is the same 567,008-parameter model that runs on the ESP32-S3 —
bit-exact with the chip's own output. (Some older packaging, including the
`mcu-kristin-745k-q8.tar.gz` filename, carries a "745k" label; that was a
directory name, never a parameter count for these binaries.)

`heart` and `heart-nano` are a second recipe: a 100-band mel interface between
the acoustic model and a noise-shaping ConvNeXt + iSTFT decoder, at 24 kHz.
`heart-nano` is the smallest complete neural TTS stack we have built —
duration 22,858 + acoustic 65,299 + decoder 206,122 = 294,279 parameters, shipped
as 337 KB of int8 blobs (`web/voices/heartnano/`) and run in the browser with the
int8 arithmetic of the microcontroller build unchanged (`mcu/src/snt_nano.c`,
golden fixture `mcu/test/fixtures/en_us_e13b`). `heart` ships float32 weights
(`web/voices/heart/`, 9.1 MB): its int8 form fails the 0.98 golden gate at 0.951
minimum correlation, the float build reproduces the training-side output at
1.000000 (`mcu/test/fixtures/en_us_r227f32`). Rebuild both with
`mcu/ports/wasm/build_nano.sh`, gate with `mcu/ports/wasm/verify_nano_node.mjs`.

## How it works

![text → duration → acoustic → decoder → audio](docs/assets/saanotts-signal-path.png)

espeak-ng provides phoneme IDs; a duration model predicts timing; an acoustic
model predicts generator latents; a decoder renders 22 kHz audio.
The web voices (amy, kristin, hfc, and the other languages) use a compact
time-domain decoder running in fp32 WASM; the on-device model instead uses
a quantized int8 iSTFT decoder, sized to fit and run in real time on the
ESP32-S3. `heart` / `heart-nano` predict a 100-band mel spectrogram and render it
with a noise-fed ConvNeXt + iSTFT decoder at 24 kHz (`mcu/src/snt_nano.c`).

## Train your own voice

The end-to-end recipe is [in the docs](docs/distillation-recipe.md):
build a probe pack → train the duration, acoustic-latent, and decoder models →
joint finetune → export int8. New-language porting is
[`docs/roota-language-porting-recipe.md`](docs/roota-language-porting-recipe.md).

```bash
pip install -e .
# then follow the training recipe in docs/ to make a new voice
```

## Deploy

- **ESP32-S3 talking device** — a standalone WiFi dashboard: type text, the board
  phonemizes (on-chip espeak-ng) and speaks. See
  [`mcu/ports/esp32s3/`](mcu/ports/esp32s3/).
- **Browser** — the full stack in WASM, no server. **[▶ Hear and synthesize all 9
  voices live](https://ampixa.github.io/sanoTTS/)** (GitHub Pages); source in
  [`web/`](web/).
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

**The inference runtime is MIT. The project as a whole is GPLv3.**

| | Licence |
|---|---|
| Runtime + language bindings (`mcu/src/snt_*.c`, `mcu/include/`, `mobile/`) | **MIT** — see [`LICENSE.MIT`](LICENSE.MIT) |
| Everything else, including the espeak-ng G2P ports and the training tooling | **GPL-3.0-or-later** — see [`LICENSE`](LICENSE) |

The copyleft comes from [espeak-ng](https://github.com/espeak-ng/espeak-ng)
alone, which is used for grapheme-to-phoneme. An earlier version of this note
said piper was GPLv3 too. That needs splitting: the original
[rhasspy/piper](https://github.com/rhasspy/piper) and piper-phonemize are MIT,
and the teacher voices we distil from are MIT data — but `piper-tts` on PyPI,
which our training tools import, is
[OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) and is
**GPL-3.0-or-later**. It is a training-time tool that the shipped package never
imports, so it does not reach the runtime, but "piper is MIT" is too loose to
leave standing.

The runtime files were audited against that boundary: none of them reference
espeak, and the espeak-ng code lives entirely in the G2P and port layers, which
stay GPLv3. `LICENSE.MIT` lists every covered file and shows the reasoning,
including the upstream licences it was checked against. So you can embed the
runtime in a permissively-licensed project; you cannot embed the espeak-ng G2P
without taking GPLv3 with it.

Copyright (C) 2026 Ampixa.
