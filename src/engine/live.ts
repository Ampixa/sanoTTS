/**
 * live.ts — runs the prebuilt int8 E13b engine (the same WASM module the
 * product site ships, built from mcu/ports/wasm/snt_nano_wasm.c) inside this
 * page: fetch the voice blobs, hand it the selected sentence's real ids and
 * seed, and receive the waveform the ESP32 would emit.
 */

export interface LiveResult {
  pcm: Float32Array;
  frames: number;
  samples: number;
  arenaPeak: number;
  elapsedMs: number;
  corrVsTrace: number | null;
}

interface NanoModule {
  _malloc(n: number): number;
  _free(p: number): void;
  /* snt_nano_wasm.c: synthesize(front*, dec*, ids*, n_ids, durs*|0,
   * seed_lo, seed_hi, arena*, arena_size, out*, out_cap) — byte offsets
   * into linear memory; blob sizes are implicit in the compiled header. */
  _snt_nano_wasm_synthesize(
    front: number, dec: number,
    ids: number, nIds: number, durs: number,
    seedLo: number, seedHi: number,
    arena: number, arenaSize: number, out: number, outCap: number,
  ): number;
  _snt_nano_wasm_last_frames(): number;
  _snt_nano_wasm_last_arena_peak(): number;
  HEAPU8: Uint8Array;
  HEAPF32: Float32Array;
}

declare global {
  interface Window {
    SaanoNanoHeartNano?: (arg?: object) => Promise<NanoModule>;
  }
}

let modPromise: Promise<NanoModule> | null = null;

function loadModule(): Promise<NanoModule> {
  if (modPromise) return modPromise;
  modPromise = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "engine/snt_nano_heartnano.js";
    s.onload = () => {
      const f = window.SaanoNanoHeartNano;
      if (!f) return reject(new Error("SaanoNanoHeartNano factory missing"));
      f({}).then(resolve, reject);
    };
    s.onerror = () => reject(new Error("failed to load engine/snt_nano_heartnano.js"));
    document.head.appendChild(s);
  });
  return modPromise;
}

let blobsPromise: Promise<{ front: Uint8Array; dec: Uint8Array }> | null = null;
function loadBlobs() {
  if (!blobsPromise) {
    blobsPromise = Promise.all([
      fetch("engine/front_q8.bin").then((r) => r.arrayBuffer()),
      fetch("engine/model_q8.bin").then((r) => r.arrayBuffer()),
    ]).then(([f, d]) => ({ front: new Uint8Array(f), dec: new Uint8Array(d) }));
  }
  return blobsPromise;
}

export interface LiveInput {
  ids: Uint8Array;
  seedLo: number;
  seedHi: number;
  /** optional reference for the correlation badge (the shipped trace) */
  refPcm?: Float32Array;
}

export async function synthesizeLive(input: LiveInput): Promise<LiveResult> {
  const [M, blobs] = await Promise.all([loadModule(), loadBlobs()]);

  const nIds = input.ids.length;
  // floats; per-token durations clamp to NANO_DUR_MAX_DURATION (80) frames,
  // each frame emits NANO_HOP (256) samples → nIds × 80 × 256 covers any input
  const outCap = nIds * 80 * 256;
  const arenaSize = 768 * 1024;

  const pFront = M._malloc(blobs.front.length);
  const pDec = M._malloc(blobs.dec.length);
  const pIds = M._malloc(nIds * 4);
  const pArena = M._malloc(arenaSize);
  const pOut = M._malloc(outCap * 4);

  try {
    M.HEAPU8.set(blobs.front, pFront);
    M.HEAPU8.set(blobs.dec, pDec);
    const ids32 = new Int32Array(nIds);
    for (let i = 0; i < nIds; i++) ids32[i] = input.ids[i];
    new Int32Array(M.HEAPU8.buffer, pIds, nIds).set(ids32);

    const t0 = performance.now();
    const rc = M._snt_nano_wasm_synthesize(
      pFront, pDec,
      pIds, nIds, 0 /* durs=NULL: run the duration student */,
      input.seedLo >>> 0, input.seedHi >>> 0,
      pArena, arenaSize, pOut, outCap,
    );
    const elapsedMs = performance.now() - t0;
    if (rc <= 0) throw new Error(`synthesize returned ${rc}`);

    const pcm = M.HEAPF32.slice(pOut / 4, pOut / 4 + rc);

    let corr: number | null = null;
    const ref = input.refPcm;
    if (ref && ref.length > 0) {
      const n = Math.min(ref.length, pcm.length);
      let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
      for (let i = 0; i < n; i++) {
        const a = pcm[i], b = ref[i];
        sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
      }
      const cov = sab - (sa * sb) / n;
      corr = cov / (Math.sqrt((saa - (sa * sa) / n) * (sbb - (sb * sb) / n)) + 1e-30);
    }

    return {
      pcm,
      frames: M._snt_nano_wasm_last_frames(),
      samples: rc,
      arenaPeak: M._snt_nano_wasm_last_arena_peak(),
      elapsedMs,
      corrVsTrace: corr,
    };
  } finally {
    M._free(pFront); M._free(pDec); M._free(pIds); M._free(pArena); M._free(pOut);
  }
}
