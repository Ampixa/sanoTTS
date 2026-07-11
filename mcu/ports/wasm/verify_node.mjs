// verify_node.mjs -- the WASM golden gate, run headless under Node.
//
// Drives the exact same exported entry the browser page calls, on the same
// golden vectors, and asserts correlation > 0.98 against the PyTorch
// reference audio. This is the WASM port's equivalent of test/golden_main.c:
// if this passes, the browser produces bit-identical PCM (same core, same
// scalar kernels, same inputs).
//
//   node mcu/ports/wasm/verify_node.mjs
//
// Exit 0 on PASS, 1 on FAIL. Reads the staged assets from web/assets/.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../../..");
const web = resolve(repo, "web");

const SaanoTTS = (await import(resolve(web, "snt_tts.js"))).default;
const A = f => new Uint8Array(readFileSync(resolve(web, "assets", f)));

const front = A("front_q8.bin");
const dec = A("model_q8.bin");
const ids = A("e2e_ids.bin");
const durs = A("e2e_durs.bin");
const goldBytes = A("e2e_audio.bin");
const gold = new Float32Array(goldBytes.buffer, goldBytes.byteOffset,
                             goldBytes.byteLength / 4);

const ARENA = 320 * 1024;
const OUT_CAP = 400000;

const Mod = await SaanoTTS();
const put = b => { const p = Mod._malloc(b.length); Mod.HEAPU8.set(b, p); return p; };

const frontP = put(front), decP = put(dec), idsP = put(ids), dursP = put(durs);
const nIds = ids.byteLength / 4;
const arenaP = Mod._malloc(ARENA);
const outP = Mod._malloc(OUT_CAP * 4);

const t0 = performance.now();
const n = Mod._snt_web_synthesize(frontP, decP, idsP, nIds, dursP,
                                  arenaP, ARENA, outP, OUT_CAP);
const wall = (performance.now() - t0) / 1000;

if (n < 0) { console.error(`synth failed: rc=${n}`); process.exit(1); }

const pcm = Mod.HEAPF32.subarray(outP >> 2, (outP >> 2) + n);
let sa=0, sb=0, saa=0, sbb=0, sab=0;
const m = Math.min(n, gold.length);
for (let i=0;i<m;i++){ const x=pcm[i], y=gold[i];
  sa+=x; sb+=y; saa+=x*x; sbb+=y*y; sab+=x*y; }
const cov = sab - sa*sb/m;
const corr = cov / Math.sqrt((saa - sa*sa/m)*(sbb - sb*sb/m) + 1e-30);

const secs = n / 22050;
console.log(`samples ${n} (${secs.toFixed(2)}s audio)  wall ${wall.toFixed(3)}s  ` +
            `${(secs/wall).toFixed(1)}x RT on this CPU`);
console.log(`golden corr ${corr.toFixed(6)} vs PyTorch reference`);
const pass = corr > 0.98;
console.log(pass ? "PASS" : "FAIL");
process.exit(pass ? 0 : 1);
