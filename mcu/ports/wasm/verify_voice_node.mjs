// verify_voice_node.mjs -- end-to-end gate for the all-voices live browser
// synth chain, run headless under Node.
//
// For each voice under test: loads web/voices/<key>/meta.json, drives
// web/snt_g2p.js (SaanoG2P) with that voice's espeak_voice + g2p_voice_slot
// to turn a sentence into phoneme ids, feeds those ids plus
// web/voices/<key>/{front_f32.bin,dec_f32.bin} into web/snt_voice.js
// (SaanoVoice)'s snt_voice_synthesize, and asserts:
//   - synth returns a positive sample count
//   - the resulting audio is longer than 1 second at 22.05kHz
//   - every sample is finite (no NaN/Inf from a mis-parsed blob or a
//     shape mismatch that silently walked off a weight tensor)
//   - the audio is not silence (some samples exceed a tiny amplitude floor)
//
//   node mcu/ports/wasm/verify_voice_node.mjs             # amy + hindi
//   node mcu/ports/wasm/verify_voice_node.mjs amy kristin hfc ...  # subset
//
// Exit 0 when all requested voices PASS, 1 otherwise.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../../..");
const web = resolve(repo, "web");

const SR = 22050;
const MAX_IDS = 1024;
const OUT_CAP = 22050 * 20; // 20s ceiling, generous for a short test sentence

const SENTENCES = {
  amy: "Hello! I am a tiny voice living entirely in your browser.",
  kristin: "Hello! I am a tiny voice living entirely in your browser.",
  hfc: "Hello! I am a tiny voice living entirely in your browser.",
  vietnamese: "Xin chào! Rất vui được gặp bạn hôm nay.",
  indonesian: "Halo! Senang bertemu dengan Anda hari ini.",
  nepali: "नमस्ते! तपाईंलाई भेटेर खुशी लाग्यो।",
  hindi: "नमस्ते! आपसे मिलकर बहुत खुशी हुई।",
  chinese: "你好！很高兴今天见到你。",
};

const args = process.argv.slice(2);
const voices = args.length ? args : ["amy", "hindi"];

const SaanoVoice = (await import(resolve(web, "snt_voice.js"))).default;
const SaanoG2P = (await import(resolve(web, "snt_g2p.js"))).default;

const V = await SaanoVoice();
const G = await SaanoG2P({ locateFile: (f) => resolve(web, f) });

const setVoice = G.cwrap("snt_g2p_set_voice", "number", ["string", "number"]);
const g2pIds = G.cwrap("snt_g2p_text_to_ids", "number", ["number", "number", "number"]);
const synth = V.cwrap("snt_voice_synthesize", "number",
  ["number", "number", "number", "number", "number", "number", "number"]);

function loadBlob(mod, path) {
  const bytes = new Uint8Array(readFileSync(path));
  const p = mod._malloc(bytes.length);
  mod.HEAPU8.set(bytes, p);
  return { p, len: bytes.length };
}

function textToIds(text) {
  const nBytes = G.lengthBytesUTF8(text) + 1;
  const textP = G._malloc(nBytes);
  G.stringToUTF8(text, textP, nBytes);
  const outP = G._malloc(MAX_IDS * 4);
  const n = g2pIds(textP, outP, MAX_IDS);
  G._free(textP);
  if (n <= 0) { G._free(outP); throw new Error(`g2p failed: rc=${n}`); }
  const ids = Int32Array.from(G.HEAP32.subarray(outP >> 2, (outP >> 2) + n));
  G._free(outP);
  return ids;
}

let allOk = true;
for (const key of voices) {
  const dir = resolve(web, "voices", key);
  const meta = JSON.parse(readFileSync(resolve(dir, "meta.json"), "utf8"));
  const text = SENTENCES[key];
  if (!text) throw new Error(`no test sentence for voice ${key}`);

  const rc = setVoice(meta.espeak_voice, meta.g2p_voice_slot);
  if (rc !== 0) {
    console.log(`${key}: FAIL (snt_g2p_set_voice("${meta.espeak_voice}", ${meta.g2p_voice_slot}) rc=${rc})`);
    allOk = false;
    continue;
  }
  const ids = textToIds(text);

  const front = loadBlob(V, resolve(dir, "front_f32.bin"));
  const dec = loadBlob(V, resolve(dir, "dec_f32.bin"));
  const idsBytes = new Uint8Array(ids.buffer, ids.byteOffset, ids.byteLength);
  const idsBlob = { p: V._malloc(idsBytes.length), len: idsBytes.length };
  V.HEAPU8.set(idsBytes, idsBlob.p);
  const outP = V._malloc(OUT_CAP * 4);

  const t0 = performance.now();
  const n = synth(front.p, dec.p, idsBlob.p, ids.length, meta.length_scale, outP, OUT_CAP);
  const wall = (performance.now() - t0) / 1000;

  let ok = n > 0;
  let secs = 0, finite = true, peak = 0;
  if (ok) {
    secs = n / SR;
    const pcm = V.HEAPF32.subarray(outP >> 2, (outP >> 2) + n);
    for (let i = 0; i < n; i++) {
      const v = pcm[i];
      if (!Number.isFinite(v)) { finite = false; break; }
      const a = Math.abs(v);
      if (a > peak) peak = a;
    }
    ok = finite && secs > 1.0 && peak > 1e-4;
  }

  [front.p, dec.p, idsBlob.p, outP].forEach((p) => V._free(p));

  console.log(
    `${key}: ${ok ? "PASS" : "FAIL"}  n=${n} ids=${ids.length} ` +
    `secs=${secs.toFixed(2)} finite=${finite} peak=${peak.toFixed(4)} ` +
    `wall=${wall.toFixed(3)}s${wall > 0 ? ` (${(secs / wall).toFixed(1)}x RT)` : ""}`
  );
  if (!ok) allOk = false;
}

console.log(allOk ? "ALL PASS" : "FAIL");
process.exit(allOk ? 0 : 1);
