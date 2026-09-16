/**
 * traces.ts — loads the web-packed engine traces from public/traces/.
 * Every tensor is real engine output (int8 device math, full-student path);
 * the packet scales from manifest.json are applied on load so the app only
 * ever sees dequantized floats.
 */

export interface RowInfo {
  row_id: string;
  text: string;
  n_tokens: number;
  frames: number;
  samples: number;
  duration_s: number;
}

export interface TraceIndex {
  model: string;
  params: number;
  sample_rate: number;
  hop: number;
  n_fft: number;
  vocab: string[];
  source: string;
  rows: RowInfo[];
}

interface TensorMeta { scale?: number; lo?: number; hi?: number }

export interface RowManifest {
  row_id: string;
  n_tokens: number;
  frames: number;
  got_frames: number;
  got_samples: number;
  seed: number;
  seed_lo: number;
  seed_hi: number;
  text: string;
  phonemes: string;
  duration_s: number;
  pcm_peak: number;
  arena_peak: number;
  scales: Record<string, TensorMeta>;
}

export class RowTrace {
  constructor(
    public info: RowInfo,
    public manifest: RowManifest,
    public ids: Uint8Array,
    public durs: Uint8Array,
    public durHidden: Float32Array, // [26, N]
    public tokHidden: Float32Array, // [31, N]
    public frameHidden: Float32Array, // [31, T]
    public mel: Float32Array, // [100, T]
    public noise: Float32Array, // [4, T]
    public stem: Float32Array, // [62, T]
    public trunk: Float32Array[], // 4 × [62, T]
    public specLogmag: Float32Array, // [513, T]
    public specPhase: Float32Array, // [513, T]
    public pcm: Float32Array, // [(T-1)*256]
  ) {}

  get T() { return this.info.frames; }
  get N() { return this.info.n_tokens; }

  /** cumulative frame boundary where token t starts */
  tokenFrameBounds(): Int32Array {
    const out = new Int32Array(this.N + 1);
    for (let t = 0; t < this.N; t++) out[t + 1] = out[t] + this.durs[t];
    return out;
  }
}

async function fetchBuf(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.arrayBuffer();
}

function deqI8(buf: ArrayBuffer, scale = 1): Float32Array {
  const src = new Int8Array(buf);
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = src[i] * scale;
  return out;
}

function deqU8(buf: ArrayBuffer, lo = 0, hi = 1): Float32Array {
  const src = new Uint8Array(buf);
  const out = new Float32Array(src.length);
  const span = hi - lo;
  for (let i = 0; i < src.length; i++) out[i] = lo + (src[i] / 255) * span;
  return out;
}

function deqPcm(buf: ArrayBuffer): Float32Array {
  const src = new Int16Array(buf);
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = src[i] / 32767;
  return out;
}

let indexCache: Promise<TraceIndex> | null = null;
export function loadIndex(): Promise<TraceIndex> {
  if (!indexCache)
    indexCache = fetch("traces/index.json").then((r) => {
      if (!r.ok) throw new Error(`traces/index.json: ${r.status}`);
      return r.json();
    });
  return indexCache;
}

const rowCache = new Map<string, Promise<RowTrace>>();
export function loadRow(rowId: string, info: RowInfo): Promise<RowTrace> {
  const hit = rowCache.get(rowId);
  if (hit) return hit;
  const base = `traces/${rowId}`;
  const p = (async () => {
    const man: RowManifest = await fetch(`${base}/manifest.json`).then((r) => {
      if (!r.ok) throw new Error(`${base}/manifest.json: ${r.status}`);
      return r.json();
    });
    const S = man.scales;
    const [ids, durs, durH, tokH, frameH, mel, noise, stem, b0, b1, b2, b3, lm, ph, pcm] =
      await Promise.all([
        fetchBuf(`${base}/ids.u8`).then((b) => new Uint8Array(b)),
        fetchBuf(`${base}/durs.u8`).then((b) => new Uint8Array(b)),
        fetchBuf(`${base}/dur_hidden.i8`).then((b) => deqI8(b, S.dur_hidden?.scale)),
        fetchBuf(`${base}/tok_hidden.i8`).then((b) => deqI8(b, S.tok_hidden?.scale)),
        fetchBuf(`${base}/frame_hidden.i8`).then((b) => deqI8(b, S.frame_hidden?.scale)),
        fetchBuf(`${base}/mel.u8`).then((b) => deqU8(b, S.mel?.lo, S.mel?.hi)),
        fetchBuf(`${base}/noise.i8`).then((b) => deqI8(b, S.noise?.scale)),
        fetchBuf(`${base}/stem.i8`).then((b) => deqI8(b, S.stem?.scale)),
        fetchBuf(`${base}/trunk_b0.i8`).then((b) => deqI8(b, S.trunk_b0?.scale)),
        fetchBuf(`${base}/trunk_b1.i8`).then((b) => deqI8(b, S.trunk_b1?.scale)),
        fetchBuf(`${base}/trunk_b2.i8`).then((b) => deqI8(b, S.trunk_b2?.scale)),
        fetchBuf(`${base}/trunk_b3.i8`).then((b) => deqI8(b, S.trunk_b3?.scale)),
        fetchBuf(`${base}/spec_logmag.u8`).then((b) => deqU8(b, S.spec_logmag?.lo, S.spec_logmag?.hi)),
        fetchBuf(`${base}/spec_phase.u8`).then((b) => deqU8(b, S.spec_phase?.lo, S.spec_phase?.hi)),
        fetchBuf(`${base}/pcm.i16`).then(deqPcm),
      ]);
    return new RowTrace(
      info, man, ids, durs, durH, tokH, frameH, mel, noise, stem,
      [b0, b1, b2, b3], lm, ph, pcm,
    );
  })();
  rowCache.set(rowId, p);
  return p;
}
