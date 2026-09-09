// sanotts-web — browser wrapper around the saanoTTS WebAssembly runtime.
//
// This mirrors the exact call sequence verified in web/index.html:
//   1. load the espeak-ng G2P wasm module (SaanoG2P) and the acoustic+decoder
//      wasm module (SaanoVoice) — both are Emscripten MODULARIZE builds. They
//      are UMD, not ESM (a bare `var SaanoG2P = (()=>{...})()` with a
//      CommonJS/AMD fallback, no `export`), so importing them as ES modules
//      would trap the binding in that module's private scope. We inject them
//      as classic <script> tags instead — exactly what index.html does — and
//      read the resulting global off `window`.
//   2. per voice: snt_g2p_set_voice(espeak_voice, g2p_voice_slot) switches the
//      phonemizer's espeak voice/table, snt_g2p_text_to_ids() phonemizes text
//      into the [BOS,PAD,(id,PAD)*,EOS] id sequence the acoustic model expects.
//   3. snt_voice_synthesize(front_blob, dec_blob, ids, n_ids, length_scale,
//      out_ptr, out_cap) renders the waveform.
//
// Voice weights (the blobs meta.json names, plus meta.json itself) are NOT
// bundled — 0.3 MB for the int8 heart-nano, ~3 MB for the f16 voices added
// 2026-09-08, 5.5-8.7 MB for the older f32 ones — and are fetched and cached
// lazily on first use. See README.md for self-hosting and exact sizes.

// Where the wasm modules live. These are small (~2.5 MB all told) and are
// injected as <script> tags, which is picky about MIME types, so they stay on
// our own Pages host.
const DEFAULT_ASSET_BASE = 'https://ampixa.github.io/sanoTTS/';

// Where the voice WEIGHTS live, which is a different question: they are
// 0.3-9 MB each and are plain fetch() calls, so they come from Hugging Face,
// which is built to serve model weights, sends `access-control-allow-origin`,
// and counts the download against the model. `voices/<key>/` resolves under
// this base exactly as it does under the Pages host, because the layout there
// mirrors `web/` in the repo.
const DEFAULT_VOICE_BASE = 'https://huggingface.co/ampixa/sanoTTS/resolve/main/web/';

const G2P_SCRIPT = 'snt_g2p.js';
const VOICE_SCRIPT = 'snt_voice.js';
const G2P_GLOBAL = 'SaanoG2P';
const VOICE_GLOBAL = 'SaanoVoice';

const TRELLIS_SCRIPT = 'trellis_frontend.js';
const TRELLIS_GLOBAL = 'SaanoTrellisFrontend';

const MAX_PHONEME_IDS = 1024;
const DEFAULT_MAX_SECONDS = 20; // output-buffer cap passed to snt_voice_synthesize

// snt_nano working memory. The arena is one contiguous allocation inside the
// wasm heap; 16 MB is far above what any shipped lineage needs (the 294k
// voice peaks near 100 KB for a 3 s utterance) and costs nothing on a
// browser heap. The output cap bounds a single synthesize() call.
const NANO_ARENA = 16 * 1024 * 1024;
const NANO_OUT_CAP = 24000 * 30;

// url -> Promise<factoryFn>. Keyed by exact script URL so loading the same
// assetBase twice never injects a duplicate <script> tag, while loading a
// *different* assetBase (e.g. two SanoTTS.load() calls pointed at different
// self-hosted mirrors) always actually fetches that url.
const scriptLoadPromises = new Map();

/**
 * Widen a float16 weight blob to the float32 the runtime reads.
 *
 * The ten languages added 2026-09-08 ship f16 to halve the download; float16
 * is their reference rather than an approximation, since every quality number
 * they have was measured on the fp16 package. The wasm runtime has always been
 * handed f32, so the conversion happens here rather than in C.
 *
 * A blob is `meta_bytes` of header followed by `weight_floats` values, and the
 * header is copied through untouched.
 */
function widenF16(bytes, dims) {
  const head = Number(dims && dims.meta_bytes);
  const n = Number(dims && dims.weight_floats);
  if (!Number.isInteger(head) || !Number.isInteger(n) || head < 0 || n <= 0) {
    throw new Error('sanotts-web: f16 blob has no usable meta_bytes/weight_floats');
  }
  if (bytes.length !== head + 2 * n) {
    throw new Error(
      `sanotts-web: f16 blob is ${bytes.length} bytes, meta.json implies ${head + 2 * n}`);
  }
  const out = new Uint8Array(head + 4 * n);
  out.set(bytes.subarray(0, head), 0);
  const src = new DataView(bytes.buffer, bytes.byteOffset + head, 2 * n);
  const dst = new DataView(out.buffer, head, 4 * n);
  for (let i = 0; i < n; i++) {
    const b = src.getUint16(i * 2, true);
    const exp = (b >> 10) & 0x1f;
    const frac = b & 0x3ff;
    const sign = (b & 0x8000) ? -1 : 1;
    let v;
    if (exp === 0) v = sign * frac * 5.960464477539063e-8;
    else if (exp === 31) v = frac ? NaN : sign * Infinity;
    else v = sign * Math.pow(2, exp - 25) * (1024 + frac);
    dst.setFloat32(i * 4, v, true);
  }
  return out;
}

function joinUrl(base, path) {
  return (base.endsWith('/') ? base : base + '/') + path;
}

/**
 * Inject a classic <script> and resolve with the global it defines.
 *
 * `expect` matters: the Emscripten modules define a factory *function*, while
 * trellis_frontend.js defines an *object* with createFrontend(). Asserting
 * "function" for both rejected a perfectly good frontend, so the check is now
 * told what shape to require.
 */
function loadGlobalScript(url, globalName, expect = 'function') {
  if (typeof document === 'undefined') {
    return Promise.reject(new Error(
      `sanotts-web: no DOM available to load "${url}" — this package is browser-only ` +
      `(it injects the Emscripten runtime as a <script> tag).`
    ));
  }
  let pending = scriptLoadPromises.get(url);
  if (!pending) {
    pending = new Promise((resolve, reject) => {
      const el = document.createElement('script');
      el.src = url;
      el.async = true;
      el.onload = () => {
        const value = globalThis[globalName];
        const ok = expect === 'function'
          ? typeof value === 'function'
          : (value !== null && typeof value === 'object');
        if (!ok) {
          reject(new Error(
            `sanotts-web: loaded "${url}" but window.${globalName} was not ` +
            `${expect === 'function' ? 'a function' : 'an object'} afterward ` +
            `(got ${typeof value})`));
          return;
        }
        resolve(value);
      };
      el.onerror = () => reject(new Error(`sanotts-web: failed to load script "${url}"`));
      document.head.appendChild(el);
    });
    // A script that failed to load should not poison future load() calls
    // (e.g. transient network error) — let a retry actually try again.
    pending.catch(() => scriptLoadPromises.delete(url));
    scriptLoadPromises.set(url, pending);
  }
  return pending;
}

function toHeap(mod, bytes) {
  const ptr = mod._malloc(bytes.length);
  mod.HEAPU8.set(bytes, ptr);
  return ptr;
}

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`sanotts-web: fetch ${url} failed: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`sanotts-web: fetch ${url} failed: HTTP ${r.status}`);
  return r.json();
}

/**
 * Find a voice by trying each base in order, returning the first that answers
 * along with the directory it answered from — so the blobs are fetched from
 * the same host as the meta.json rather than being split across two.
 *
 * Every failure is kept and reported together. A message naming only the last
 * host would send someone debugging a Hugging Face outage off to look at their
 * own mirror instead.
 */
async function fetchVoiceMeta(key, bases) {
  const failures = [];
  for (const base of bases) {
    const dir = joinUrl(base, `voices/${key}/`);
    try {
      return { dir, meta: await fetchJson(dir + 'meta.json') };
    } catch (err) {
      failures.push(`${dir}meta.json: ${err && err.message ? err.message : err}`);
    }
  }
  throw new Error(`sanotts-web: could not load voice "${key}" from any host — ${failures.join('; ')}`);
}

/**
 * A slim, known-voice registry for building a UI (voice picker labels,
 * flags, language). This is NOT authoritative for synthesis — the actual
 * espeak_voice / g2p_voice_slot / length_scale used at synth time always
 * come from that voice's own meta.json, fetched lazily by loadVoice(). Kept
 * here only as a convenience; it never needs to match the server exactly.
 */
export const KNOWN_VOICES = Object.freeze([
  { key: 'heart', label: 'heart', language: 'English', flag: '🇺🇸' },
  { key: 'heartnano', label: 'heart-nano', language: 'English', flag: '🇺🇸' },
  { key: 'amy', label: 'amy', language: 'English', flag: '🇺🇸' },
  { key: 'kristin', label: 'kristin', language: 'English', flag: '🇺🇸' },
  { key: 'hfc', label: 'hfc', language: 'English', flag: '🇺🇸' },
  { key: 'vietnamese', label: 'Vietnamese', language: 'Vietnamese', flag: '🇻🇳' },
  { key: 'indonesian', label: 'Indonesian', language: 'Indonesian', flag: '🇮🇩' },
  { key: 'nepali', label: 'Nepali', language: 'Nepali', flag: '🇳🇵' },
  { key: 'hindi', label: 'Hindi', language: 'Hindi', flag: '🇮🇳' },
  { key: 'chinese', label: 'Chinese', language: 'Chinese', flag: '🇨🇳' },
]);

export class SanoTTS {
  /** @private — use SanoTTS.load() */
  constructor({ G2P, Voice, assetBase }) {
    this._G2P = G2P;
    this._Voice = Voice;
    this._assetBase = assetBase;
    this._setVoiceFn = null;
    this._voiceBundles = new Map(); // "voiceBase\0key" -> Promise<{meta,front,dec}>
    this._nanoModules = new Map();  // export name -> Promise<Emscripten module>
    this._trellis = null;           // lazily-built 62-symbol frontend
  }

  /**
   * Load the G2P (espeak-ng) and acoustic/decoder WebAssembly modules. Must
   * be called (and awaited) once before synthesize().
   *
   * @param {object} [opts]
   * @param {string} [opts.assetBase] - directory holding snt_g2p.{js,wasm,data}
   *   and snt_voice.{js,wasm}. Defaults to the live sanoTTS Pages demo; point
   *   this at your own host to self-host (see README "Deploy on your own site").
   * @returns {Promise<SanoTTS>}
   */
  static async load({ assetBase = DEFAULT_ASSET_BASE } = {}) {
    if (!assetBase.endsWith('/')) assetBase += '/';

    const [g2pFactory, voiceFactory] = await Promise.all([
      loadGlobalScript(joinUrl(assetBase, G2P_SCRIPT), G2P_GLOBAL),
      loadGlobalScript(joinUrl(assetBase, VOICE_SCRIPT), VOICE_GLOBAL),
    ]);

    const locateFile = (path) => joinUrl(assetBase, path);
    const [G2P, Voice] = await Promise.all([
      g2pFactory({ locateFile }),
      voiceFactory({ locateFile }),
    ]);

    if (typeof G2P._snt_g2p_init !== 'function') {
      throw new Error('sanotts-web: snt_g2p.wasm did not export _snt_g2p_init — wrong/stale build at assetBase?');
    }
    const rc = G2P._snt_g2p_init();
    if (rc !== 0) {
      throw new Error(`sanotts-web: snt_g2p_init() failed, rc=${rc}`);
    }

    return new SanoTTS({ G2P, Voice, assetBase });
  }

  _setG2PVoice(espeakVoice, slot) {
    if (!this._setVoiceFn) {
      this._setVoiceFn = this._G2P.cwrap('snt_g2p_set_voice', 'number', ['string', 'number']);
    }
    const rc = this._setVoiceFn(espeakVoice, slot);
    if (rc !== 0) {
      throw new Error(`sanotts-web: snt_g2p_set_voice("${espeakVoice}", ${slot}) failed, rc=${rc}`);
    }
  }

  _g2pIds(text) {
    const G2P = this._G2P;
    const nBytes = G2P.lengthBytesUTF8(text) + 1;
    const textPtr = G2P._malloc(nBytes);
    const idsPtr = G2P._malloc(MAX_PHONEME_IDS * 4);
    try {
      G2P.stringToUTF8(text, textPtr, nBytes);
      const n = G2P._snt_g2p_text_to_ids(textPtr, idsPtr, MAX_PHONEME_IDS);
      if (n <= 0) {
        throw new Error(`sanotts-web: phonemizer returned ${n} for text ${JSON.stringify(text)}`);
      }
      return new Int32Array(G2P.HEAP32.buffer, idsPtr, n).slice();
    } finally {
      G2P._free(textPtr);
      G2P._free(idsPtr);
    }
  }

  /**
   * Fetch (and cache) a voice's weight bundle: meta.json + front_f32.bin +
   * dec_f32.bin. Safe to call ahead of synthesize() to prefetch while the
   * user is still typing or picking a voice (this is what the reference
   * demo does on every mascot click).
   *
   * @param {string} key - voice key, e.g. "amy"
   * @param {object} [opts]
   * @param {string} [opts.voiceBase] - directory holding voices/<key>/.
   *   Defaults to Hugging Face, with the Pages host as a fallback. Passing
   *   this explicitly disables that fallback: a self-hosted deployment that
   *   quietly reached out to our servers whenever its own mirror was missing a
   *   file would be a nasty surprise, so an explicit base is the only base.
   * @returns {Promise<{meta: object, front: Uint8Array, dec: Uint8Array}>}
   */
  loadVoice(key, opts = {}) {
    const explicitBase = typeof opts.voiceBase === 'string';
    let voiceBase = explicitBase ? opts.voiceBase : DEFAULT_VOICE_BASE;
    if (!voiceBase.endsWith('/')) voiceBase += '/';
    const cacheKey = voiceBase + ' ' + key;
    let pending = this._voiceBundles.get(cacheKey);
    if (!pending) {
      const bases = explicitBase ? [voiceBase] : [voiceBase, DEFAULT_ASSET_BASE];
      pending = (async () => {
        const { dir, meta } = await fetchVoiceMeta(key, bases);
        // Blob filenames come from meta.json rather than being assumed: the
        // nano voices ship front_q8/model_q8 (or *_f32 for the 2.27M one),
        // not front_f32/dec_f32, and hardcoding either would break the other.
        const frontName = meta.front || 'front_f32.bin';
        const decName = meta.dec || 'dec_f32.bin';
        const [front, dec] = await Promise.all([
          fetchBytes(dir + frontName),
          fetchBytes(dir + decName),
        ]);
        if (meta.front_bytes !== undefined && front.length !== meta.front_bytes) {
          throw new Error(`sanotts-web: ${key}/${frontName} is ${front.length} bytes, meta.json says ${meta.front_bytes}`);
        }
        if (meta.dec_bytes !== undefined && dec.length !== meta.dec_bytes) {
          throw new Error(`sanotts-web: ${key}/${decName} is ${dec.length} bytes, meta.json says ${meta.dec_bytes}`);
        }
        // meta.weights === 'f16' means the blobs were narrowed for transfer.
        // Without this the raw halves reach a runtime expecting f32 and it
        // emits confident noise rather than failing.
        if (meta.weights === 'f16') {
          return {
            meta,
            front: widenF16(front, meta.front_dims),
            dec: widenF16(dec, meta.dec_dims),
          };
        }
        return { meta, front, dec };
      })();
      // A transient network failure shouldn't permanently poison this voice
      // for the rest of the session — let the next call retry the fetch.
      pending.catch(() => this._voiceBundles.delete(cacheKey));
      this._voiceBundles.set(cacheKey, pending);
    }
    return pending;
  }

  /**
   * Lazily load one snt_nano voice module.
   *
   * These are per-lineage builds: a module compiled for int8 weights cannot
   * read f32 blobs and vice versa, which is why meta.json names both the
   * module file and its global. Loading is lazy because most consumers never
   * touch a nano voice, and eagerly pulling two more wasm modules would slow
   * every load() that does not need them.
   *
   * @private
   */
  _loadNanoModule(meta, assetBase) {
    const globalName = meta.export_name;
    const script = meta.module;
    if (!globalName || !script) {
      throw new Error('sanotts-web: nano meta.json is missing "module"/"export_name"');
    }
    const cacheKey = assetBase + ' ' + script;
    let pending = this._nanoModules.get(cacheKey);
    if (!pending) {
      pending = (async () => {
        const factory = await loadGlobalScript(joinUrl(assetBase, script), globalName);
        const mod = await factory({ locateFile: (path) => joinUrl(assetBase, path) });
        if (typeof mod._snt_nano_wasm_synthesize !== 'function') {
          throw new Error(`sanotts-web: ${script} did not export _snt_nano_wasm_synthesize — stale build?`);
        }
        // A module built for the other weight format would read the blobs as
        // the wrong element type and emit confident noise rather than fail,
        // so the mismatch is caught here instead of being heard later.
        const fmt = mod._snt_nano_wasm_weight_format();
        const wantF32 = meta.weights === 'f32';
        if ((fmt === 1) !== wantF32) {
          throw new Error(
            `sanotts-web: ${script} weight format ${fmt} does not match meta.json "${meta.weights}"`);
        }
        return mod;
      })();
      pending.catch(() => this._nanoModules.delete(cacheKey));
      this._nanoModules.set(cacheKey, pending);
    }
    return pending;
  }

  /**
   * The 62-symbol Misaki/espeak-IPA frontend the nano voices use. Different
   * from the piper phoneme table the other voices use: same espeak-ng
   * underneath, different symbol set and tokenisation.
   *
   * @private
   */
  async _nanoFrontend(assetBase) {
    if (!this._trellis) {
      const frontendNs = await loadGlobalScript(
        joinUrl(assetBase, TRELLIS_SCRIPT), TRELLIS_GLOBAL, 'object');
      if (typeof frontendNs.createFrontend !== 'function') {
        throw new Error('sanotts-web: trellis_frontend.js did not expose createFrontend()');
      }
      this._trellis = frontendNs.createFrontend({ module: this._G2P });
    }
    return this._trellis;
  }

  /** @private — the snt_nano synthesis path. */
  async _synthesizeNano(text, bundle, { assetBase, maxSeconds }) {
    const N = await this._loadNanoModule(bundle.meta, assetBase);
    const frontend = await this._nanoFrontend(assetBase);
    const { ids } = frontend.textToIds(text);
    const ids32 = Int32Array.from(ids);

    // The decoder is noise-fed, so the seed decides which of many valid
    // renderings you get. Deriving it from the text (sha256(text)[:8]) is the
    // renderer's own convention and makes a given string reproducible.
    const textBytes = N.lengthBytesUTF8(text) + 1;
    const textPtr = N._malloc(textBytes);
    const seedPtr = N._malloc(8);
    const frontPtr = toHeap(N, bundle.front);
    const decPtr = toHeap(N, bundle.dec);
    const idsPtr = toHeap(N, new Uint8Array(ids32.buffer, 0, ids32.length * 4));
    const outCap = Math.min(NANO_OUT_CAP, Math.round((bundle.meta.sample_rate || 24000) * maxSeconds));
    const arenaPtr = N._malloc(NANO_ARENA);
    const outPtr = N._malloc(outCap * 4);
    try {
      N.stringToUTF8(text, textPtr, textBytes);
      if (N._snt_nano_wasm_seed_from_text(textPtr, seedPtr) !== 0) {
        throw new Error('sanotts-web: nano seed derivation failed');
      }
      const lo = N.HEAPU32[seedPtr >> 2];
      const hi = N.HEAPU32[(seedPtr >> 2) + 1];
      const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
      const n = N._snt_nano_wasm_synthesize(
        frontPtr, decPtr, idsPtr, ids32.length, 0, lo, hi, arenaPtr, NANO_ARENA, outPtr, outCap);
      const elapsedMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
      if (n < 0) {
        const rc = typeof N._snt_nano_wasm_last_rc === 'function' ? N._snt_nano_wasm_last_rc() : 'n/a';
        throw new Error(`sanotts-web: nano synthesis failed, code ${n} (core ${rc})`);
      }
      const samples = new Float32Array(n);
      samples.set(N.HEAPF32.subarray(outPtr >> 2, (outPtr >> 2) + n));
      const sampleRate = (typeof N._snt_nano_wasm_sample_rate === 'function'
        && N._snt_nano_wasm_sample_rate()) || bundle.meta.sample_rate || 24000;
      return { samples, sampleRate, phonemeCount: ids32.length, elapsedMs };
    } finally {
      [textPtr, seedPtr, frontPtr, decPtr, idsPtr, arenaPtr, outPtr].forEach((ptr) => N._free(ptr));
    }
  }

  /**
   * Synthesize speech.
   *
   * @param {string} text
   * @param {object} [opts]
   * @param {string} [opts.voice] - voice key, e.g. "amy" or "heartnano"
   *   (default "amy"). Voices whose meta.json says runtime "snt_nano" are
   *   routed to the nano runtime automatically.
   * @param {string} [opts.voiceBase] - see loadVoice()
   * @param {number} [opts.lengthScale] - override the voice's default
   *   length_scale from meta.json (speaking rate; larger = slower)
   * @param {number} [opts.maxSeconds] - output buffer cap in seconds (default 20)
   * @returns {Promise<{samples: Float32Array, sampleRate: number, phonemeCount: number, elapsedMs: number}>}
   */
  async synthesize(text, { voice = 'amy', voiceBase, lengthScale, maxSeconds = DEFAULT_MAX_SECONDS } = {}) {
    if (typeof text !== 'string' || text.length === 0) {
      throw new Error('sanotts-web: synthesize() requires a non-empty string');
    }
    // Forwarded only when the caller actually named a host. Passing
    // `{ voiceBase: undefined }` through would look explicit to loadVoice and
    // defeat the Hugging Face default it is supposed to fall through to.
    const bundle = await this.loadVoice(
      voice, voiceBase === undefined ? {} : { voiceBase });

    // Which runtime a voice needs is a property of the voice, recorded in its
    // own meta.json, so adding a lineage needs no change here.
    if (bundle.meta.runtime === 'snt_nano') {
      return this._synthesizeNano(text, bundle, { assetBase: this._assetBase, maxSeconds });
    }

    const sampleRate = bundle.meta.sample_rate || 22050;
    const scale = lengthScale !== undefined ? lengthScale : bundle.meta.length_scale;

    this._setG2PVoice(bundle.meta.espeak_voice, bundle.meta.g2p_voice_slot);
    const ids = this._g2pIds(text);

    const Voice = this._Voice;
    const outCap = Math.round(sampleRate * maxSeconds);
    const frontPtr = toHeap(Voice, bundle.front);
    const decPtr = toHeap(Voice, bundle.dec);
    const idsPtr = toHeap(Voice, new Uint8Array(ids.buffer, 0, ids.length * 4));
    const outPtr = Voice._malloc(outCap * 4);
    const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
    try {
      const n = Voice._snt_voice_synthesize(frontPtr, decPtr, idsPtr, ids.length, scale, outPtr, outCap);
      if (n < 0) {
        throw new Error(`sanotts-web: snt_voice_synthesize() returned ${n} for voice "${voice}"`);
      }
      const samples = new Float32Array(n);
      samples.set(Voice.HEAPF32.subarray(outPtr >> 2, (outPtr >> 2) + n));
      const elapsedMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
      return { samples, sampleRate, phonemeCount: ids.length, elapsedMs };
    } finally {
      Voice._free(frontPtr);
      Voice._free(decPtr);
      Voice._free(idsPtr);
      Voice._free(outPtr);
    }
  }
}

/**
 * Play a synthesize() result via WebAudio.
 *
 * @param {{samples: Float32Array, sampleRate: number}} result
 * @param {object} [opts]
 * @param {AudioContext} [opts.audioContext] - reuse an existing context
 *   instead of creating (and leaking) a new one per call
 * @returns {AudioBufferSourceNode} already-started source node
 */
export function playAudio({ samples, sampleRate }, { audioContext } = {}) {
  const AudioCtor = typeof window !== 'undefined' && (window.AudioContext || window.webkitAudioContext);
  if (!AudioCtor) {
    throw new Error('sanotts-web: playAudio() requires a browser WebAudio API (window.AudioContext)');
  }
  const ctx = audioContext || new AudioCtor();
  const buffer = ctx.createBuffer(1, samples.length, sampleRate);
  buffer.getChannelData(0).set(samples);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start();
  return source;
}

export { DEFAULT_ASSET_BASE };
