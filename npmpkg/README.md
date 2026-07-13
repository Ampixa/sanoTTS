# sanotts-web

Browser-native neural text-to-speech — WebAssembly, no server, no API key.
This package wraps the same runtime that powers the live demo at
[ampixa.github.io/sanoTTS](https://ampixa.github.io/sanoTTS): an espeak-ng
phonemizer compiled to wasm, feeding a tiny (0.75M–1.8M parameter)
duration/acoustic/decoder stack, also compiled to wasm.

Zero runtime dependencies. `dist/` ships the wasm runtime (~2.5 MB); voice
weights are fetched lazily at runtime (see [Voices](#voices) below).

## Install

```
npm install sanotts-web
```

> Not yet published to the npm registry — see [Deploy on your own site](#deploy-on-your-own-site)
> for a no-install way to use it today.

## Usage

```js
import { SanoTTS, playAudio } from 'sanotts-web';

const tts = await SanoTTS.load();               // defaults to the hosted demo's assets
const result = await tts.synthesize('Hello! I am a tiny voice living in your browser.', {
  voice: 'amy',
});

console.log(result.samples.length, result.sampleRate); // Float32Array, 22050
playAudio(result);
```

`synthesize()` resolves to:

```ts
{
  samples: Float32Array,   // mono PCM, range [-1, 1]
  sampleRate: number,      // 22050 for every current voice
  phonemeCount: number,    // length of the id sequence the phonemizer produced
  elapsedMs: number,       // wall-clock time spent in the wasm synth call
}
```

### API

- **`SanoTTS.load({ assetBase })`** — loads the G2P + acoustic/decoder wasm
  modules. Call once. `assetBase` defaults to the hosted demo
  (`https://ampixa.github.io/sanoTTS/`); point it at your own host to
  self-host the wasm runtime (see below).
- **`tts.synthesize(text, { voice, voiceBase, lengthScale, maxSeconds })`** —
  phonemizes `text` and renders audio with the given voice. `voiceBase`
  defaults to the hosted demo; `lengthScale` overrides the voice's default
  speaking rate; `maxSeconds` caps the output buffer (default 20s).
- **`tts.loadVoice(key, { voiceBase })`** — fetch + cache a voice's weight
  bundle ahead of time (e.g. while the user is still choosing a voice), so
  the first `synthesize()` call for that voice doesn't pay the network
  round-trip.
- **`playAudio(result, { audioContext })`** — plays a `synthesize()` result
  through WebAudio. Pass an existing `AudioContext` to reuse it instead of
  creating (and leaking) a new one per call.
- **`KNOWN_VOICES`** — a small static array (`{ key, label, language, flag }`)
  for building a voice picker UI. Not authoritative for synthesis — the
  actual espeak voice code, phoneme-table slot, and length scale used at
  synthesis time always come from that voice's own `meta.json`, fetched
  lazily by `loadVoice()`/`synthesize()`.
- **`DEFAULT_ASSET_BASE`** — the hosted-demo URL used as the default for
  both `assetBase` and `voiceBase`.

This package is browser-only: it loads the Emscripten wasm glue by injecting
`<script>` tags (the glue files are UMD builds, not ES modules, so
`import`-ing them directly would trap their exports in module-private
scope). It will throw if `document` is unavailable (e.g. under plain Node).

## Voices

| Key | Language | espeak voice | notes |
|---|---|---|---|
| `amy` | English | `en-us` | 1.46M params, SCOREQ 4.13 |
| `kristin` | English | `en` | 1.40M params, SCOREQ 4.09 |
| `hfc` | English | `en-us` | 1.83M params, SCOREQ 3.94 |
| `vietnamese` | Vietnamese | `vi` | 1.46M params |
| `indonesian` | Indonesian | `id` | 1.46M params |
| `nepali` | Nepali | `ne` | 1.47M params |
| `hindi` | Hindi | `hi` | 1.50M params |
| `chinese` | Mandarin | `cmn` | 1.50M params |

Each voice is `meta.json` (params, ~1KB) + `front_f32.bin` (duration +
acoustic, ~1.6–3.4 MB) + `dec_f32.bin` (decoder, ~2.5–4 MB) — 4–7 MB per
voice in fp32. These are **not** bundled in this package; they are fetched
from `voiceBase` (default: the hosted demo) the first time a voice is used,
then cached in memory for the life of the page. An int8-quantized voice
format (roughly 4x smaller) is planned but not yet shipped.

## Deploy on your own site

### Option A — npm install (pending publish)

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

Copy this package's `dist/` (the wasm runtime, ~2.5 MB total including the
espeak-ng phoneme data) and the `voices/` directory from
[the sanoTTS repo](https://github.com/Ampixa/sanoTTS) (or scrape it from the
live Pages site) to your own static host, and point `assetBase`/`voiceBase`
at it.

### Option B — no build step, no npm

Skip the package entirely: copy `snt_g2p.js`, `snt_g2p.wasm`, `snt_g2p.data`,
`snt_voice.js`, `snt_voice.wasm`, and the `voices/` directory from
`web/` in the sanoTTS repo (or from `ampixa.github.io/sanoTTS`) onto your
static host, then load them exactly as `web/index.html` does:

```html
<script src="/sanotts/snt_g2p.js"></script>
<script src="/sanotts/snt_voice.js"></script>
<script type="module">
  const [G2P, Voice] = await Promise.all([SaanoG2P(), SaanoVoice()]);
  G2P._snt_g2p_init();
  // ...set voice, phonemize, synthesize — see web/index.html for the full sequence.
</script>
```

Either way, sizes to plan around:

- wasm runtime: ~700 KB total (espeak-ng G2P: ~2.5 MB uncompressed / ~700 KB
  gzipped including its phoneme-table data; acoustic/decoder wasm: ~40 KB)
- per-voice weights: 4–7 MB, fp32 (int8 quantized voices are planned, not
  yet shipped)

### CSP note

The wasm runtime needs `'wasm-unsafe-eval'` (or `'unsafe-eval'` on older
browsers that don't support the narrower directive) in your `script-src`
Content-Security-Policy — required for `WebAssembly.instantiate`/
`instantiateStreaming`. No other relaxations are needed; the runtime does
not `eval()` JavaScript.

## License

GPL-3.0. Builds on [piper](https://github.com/OHF-Voice/piper1-gpl) and
[espeak-ng](https://github.com/espeak-ng/espeak-ng) (both GPLv3). Full
pipeline: [github.com/Ampixa/sanoTTS](https://github.com/Ampixa/sanoTTS).
