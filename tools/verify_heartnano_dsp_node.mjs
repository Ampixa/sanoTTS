// verify_heartnano_dsp_node.mjs -- check web/heartnano_dsp.js (the browser runtime of the sanotts 0.5.7
// heart-nano) against the Python package's own output, headless under Node.
//
//   python3 tools/dump_heartnano_dsp_reference.py --pkg <sanotts pypkg> --voice-dir <heartnano-dsp> --out REF
//   node tools/verify_heartnano_dsp_node.mjs REF
//
// Loads the browser bundle web/voices/heartnano-dsp/ (sha256-checked against its meta.json), then per row:
//   durations / frame count      must equal sanotts nano.duration_forward exactly
//   mel                          max |JS - Python| of the acoustic model's mel-100 frames
//   dsp                          JS decoder on the PYTHON mel vs nano_dsp.render (decoder port alone)
//   e2e                          JS ids -> audio vs Synthesizer.synthesize_ids (the package default path)
// and the speed of the JS synthesis per second of audio. Exit 0 when every row has matching durations and
// e2e correlation > 0.9999, 1 otherwise.
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { performance } from "node:perf_hooks";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..");
const voiceDir = resolve(repo, "web/voices/heartnano-dsp");
const GATE = 0.9999;

const ref = process.argv[2];
if (!ref) { console.error("usage: node tools/verify_heartnano_dsp_node.mjs <reference dir>"); process.exit(2); }

function bytes(path) { return new Uint8Array(readFileSync(path)); }
function typed(path, T) { const b = readFileSync(path); return new T(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
function sha256(b) { return createHash("sha256").update(b).digest("hex"); }

const meta = JSON.parse(readFileSync(resolve(voiceDir, "meta.json"), "utf8"));
const front = bytes(resolve(voiceDir, meta.front)), dsp = bytes(resolve(voiceDir, meta.dsp));
for (const [name, b, want] of [[meta.front, front, meta.front_sha256], [meta.dsp, dsp, meta.dsp_sha256]]) {
  const got = sha256(b);
  if (got !== want) throw new Error(`${name}: sha256 ${got} != meta.json ${want}`);
}
createRequire(import.meta.url)(resolve(repo, "web/heartnano_dsp.js"));
const voice = globalThis.SaanoNanoDsp.createVoice({ meta, front, dsp });

function stats(a, b) {
  if (a.length !== b.length) throw new Error(`length ${a.length} != ${b.length}`);
  let maxd = 0, peak = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  const n = a.length;
  for (let i = 0; i < n; i++) {
    const x = a[i], y = b[i], d = Math.abs(x - y);
    if (d > maxd) maxd = d;
    if (Math.abs(y) > peak) peak = Math.abs(y);
    sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
  }
  const cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
  return { maxd, peak, corr: cov / Math.sqrt(va * vb) };
}
function maxAbs(a, b) { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; }

// ---- unit checks: band matrices, noise, controls
const tb = voice.tables();
const mpm = typed(resolve(ref, "unit_mpm32.f32"), Float32Array), nim = typed(resolve(ref, "unit_nim32.f32"), Float32Array);
if (mpm.length !== tb.mpm32.length || nim.length !== tb.nim32.length) throw new Error("band matrix shapes differ");
const white = typed(resolve(ref, "unit_white.f32"), Float32Array);   // [257, 8, 2]
let whiteMax = 0;
for (let f = 0; f < 8; f++) {
  const w = voice.white(f);
  for (let j = 0; j < w.re.length; j++) {
    whiteMax = Math.max(whiteMax, Math.abs(w.re[j] - white[(j * 8 + f) * 2]), Math.abs(w.im[j] - white[(j * 8 + f) * 2 + 1]));
  }
}
const rows = JSON.parse(readFileSync(resolve(ref, "rows.json"), "utf8"));
if (rows.seed !== meta.seed) throw new Error(`reference seed ${rows.seed} != meta.json seed ${meta.seed}`);
const r0 = rows.rows[0];
const ctlRef = typed(resolve(ref, "unit_ctl_r00.f64"), Float64Array);
const ctl = voice.controls(typed(resolve(ref, `${r0.tag}_mel.f32`), Float32Array), r0.frames);
let ctlMax = 0, ctlPeak = 0;
for (let i = 0; i < ctl.length; i++) { ctlMax = Math.max(ctlMax, Math.abs(ctl[i] - ctlRef[i])); ctlPeak = Math.max(ctlPeak, Math.abs(ctlRef[i])); }
console.log(`unit: minphase matrix max|d| ${maxAbs(tb.mpm32, mpm).toExponential(2)}, noise matrix ${maxAbs(tb.nim32, nim).toExponential(2)}, ` +
            `white noise (8 frames) ${whiteMax.toExponential(2)}, controls r00 ${ctlMax.toExponential(2)} (peak ${ctlPeak.toFixed(2)})`);

// ---- per row
voice.synthesize(Int32Array.from(typed(resolve(ref, `${r0.tag}_ids.i32`), Int32Array)));   // warm-up (erf table, FFT plans)
let pass = true, wallSum = 0, audioSum = 0;
console.log("row  tokens frames  durs  mel max|d|  dsp max|d|  dsp corr      e2e max|d|  e2e corr      peak   py stream-vs-render  JS x realtime");
for (const row of rows.rows) {
  const ids = typed(resolve(ref, `${row.tag}_ids.i32`), Int32Array);
  const pyDurs = typed(resolve(ref, `${row.tag}_durs.i32`), Int32Array);
  const pyMel = typed(resolve(ref, `${row.tag}_mel.f32`), Float32Array);
  const pyAudio = typed(resolve(ref, `${row.tag}_audio.f32`), Float32Array);
  const pyRender = typed(resolve(ref, `${row.tag}_render.f32`), Float32Array);

  const durs = voice.durations(ids);
  const dursOk = durs.length === pyDurs.length && durs.every((v, i) => v === pyDurs[i]);
  let melD = NaN, framesOk = false;
  if (dursOk) {
    const am = voice.mel(ids, durs);
    framesOk = am.frames === row.frames;
    if (framesOk) melD = maxAbs(am.mel, pyMel);
  }
  const dspOnly = voice.render(pyMel, row.frames);
  const dspS = stats(Float32Array.from(dspOnly), pyRender);
  const t0 = performance.now();
  const out = voice.synthesize(ids);
  const wall = (performance.now() - t0) / 1000;
  const secs = out.pcm.length / out.sampleRate;
  wallSum += wall; audioSum += secs;
  let e2e = { maxd: NaN, corr: NaN, peak: NaN };
  if (out.pcm.length === pyAudio.length) e2e = stats(out.pcm, pyAudio);
  const pyClip = Float32Array.from(pyRender, v => Math.max(-1, Math.min(1, v)));
  const pySelf = stats(pyClip, pyAudio);
  const ok = dursOk && framesOk && e2e.corr > GATE;
  pass = pass && ok;
  console.log(`${row.tag}  ${String(ids.length).padStart(5)} ${String(row.frames).padStart(6)}  ${dursOk && framesOk ? " ok " : "DIFF"}  ` +
              `${melD.toExponential(2).padStart(10)}  ${dspS.maxd.toExponential(2).padStart(10)}  ${dspS.corr.toFixed(9)}  ` +
              `${e2e.maxd.toExponential(2).padStart(10)}  ${e2e.corr.toFixed(9)}  ${e2e.peak.toFixed(3)}  ` +
              `${pySelf.maxd.toExponential(2).padStart(10)}           ${(secs / wall).toFixed(1).padStart(6)}x  ${ok ? "" : "FAIL"}`);
}
console.log(`speed: ${audioSum.toFixed(2)} s of audio in ${wallSum.toFixed(3)} s = ${(wallSum / audioSum * 1000).toFixed(1)} ms per second of audio ` +
            `(${(audioSum / wallSum).toFixed(1)}x real time), node ${process.version}`);
console.log(pass ? `PASS (every row: durations exact, e2e correlation > ${GATE})` : "FAIL");
process.exit(pass ? 0 : 1);
