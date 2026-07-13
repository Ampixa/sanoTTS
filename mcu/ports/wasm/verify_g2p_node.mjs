// verify_g2p_node.mjs -- parity gate for the espeak-ng G2P WASM module.
//
// For each of the six release languages this loads web/snt_g2p.js, selects
// the voice with snt_g2p_set_voice(espeak_voice, voice_slot), runs
// snt_g2p_text_to_ids() on a short native sentence, and compares the ids
// against ground truth from python piper (PiperVoice.phonemize +
// phonemes_to_ids on the SAME voice's onnx config). The two must match
// EXACTLY -- both sides frame as BOS,PAD, (id,PAD)*, EOS with the ids taken
// from that voice's phoneme_id_map, so no normalization is applied.
//
// Also re-runs the original English smoke test through the default init path
// (no snt_g2p_set_voice call) to prove the legacy en-us + kristin behavior
// is unchanged.
//
//   node mcu/ports/wasm/verify_g2p_node.mjs
//
// Exit 0 when all languages PASS, 1 otherwise.
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../../..");
const web = resolve(repo, "web");

// slot numbers are the SNT_VOICE_TABS order in cp_id_tables_multi.h.
// espeakVoice must equal the "espeak.voice" field of the onnx config so the
// python ground truth and the wasm module run the same G2P rules.
const G2P_VOICES = "/Users/cdjk/github/llm/g2p/data/external/piper_voices";
const CASES = [
  { lang: "en", text: "Hello world.", slot: 1, espeakVoice: "en-us",
    onnx: resolve(repo, "models/teachers/en_US-amy-medium/en_US-amy-medium.onnx") },
  { lang: "vi", text: "Xin chào bạn.", slot: 3, espeakVoice: "vi",
    onnx: resolve(repo, "models/teachers/vi_VN-vais1000-medium/vi_VN-vais1000-medium.onnx") },
  { lang: "id", text: "Selamat pagi.", slot: 4, espeakVoice: "id",
    onnx: resolve(repo, "models/teachers/id_ID-news_tts-medium/id_ID-news_tts-medium.onnx") },
  { lang: "ne", text: "नमस्ते संसार।", slot: 5, espeakVoice: "ne",
    onnx: resolve(G2P_VOICES, "ne_NP/chitwan-medium.onnx") },
  { lang: "hi", text: "नमस्ते दुनिया।", slot: 6, espeakVoice: "hi",
    onnx: resolve(repo, "models/teachers/hi_IN-pratham-medium/hi_IN-pratham-medium.onnx") },
  { lang: "zh", text: "你好世界。", slot: 7, espeakVoice: "cmn",
    onnx: resolve(repo, "models/teachers/zh_CN-huayan-medium/zh_CN-huayan-medium.onnx") },
];

// ---- ground truth: python PiperVoice, one process for all languages ------
const pyScript = `
import json, sys
from piper.voice import PiperVoice
cases = json.load(sys.stdin)
out = {}
for c in cases:
    v = PiperVoice.load(c["onnx"])
    sents = v.phonemize(c["text"])
    assert len(sents) == 1, f"{c['lang']}: expected 1 sentence, got {len(sents)}"
    out[c["lang"]] = v.phonemes_to_ids(sents[0])
json.dump(out, sys.stdout)
`;
const python = resolve(repo, ".venv/bin/python");
const truth = JSON.parse(execFileSync(python, ["-c", pyScript], {
  input: JSON.stringify(CASES.map(({ lang, text, onnx }) => ({ lang, text, onnx }))),
  maxBuffer: 1 << 20,
}));

// ---- wasm module ----------------------------------------------------------
const SaanoG2P = (await import(resolve(web, "snt_g2p.js"))).default;
const Mod = await SaanoG2P({ locateFile: (f) => resolve(web, f) });

const g2p = Mod.cwrap("snt_g2p_text_to_ids", "number", ["number", "number", "number"]);
const setVoice = Mod.cwrap("snt_g2p_set_voice", "number", ["string", "number"]);

const MAX_IDS = 512;
const outP = Mod._malloc(MAX_IDS * 4);

function wasmIds(text) {
  const nBytes = Mod.lengthBytesUTF8(text) + 1;
  const textP = Mod._malloc(nBytes);
  Mod.stringToUTF8(text, textP, nBytes);
  const n = g2p(textP, outP, MAX_IDS);
  Mod._free(textP);
  if (n < 0) throw new Error(`snt_g2p_text_to_ids failed: rc=${n}`);
  return Array.from(Mod.HEAP32.subarray(outP >> 2, (outP >> 2) + n));
}

// ---- legacy smoke test: default init path, no set_voice call --------------
const legacy = wasmIds("Hello world, this is a test.");
const legacyOk = legacy.length > 10 && legacy.every((id) => id >= 0 && id < 256);
console.log(`legacy en-us+kristin default: n=${legacy.length} ${legacyOk ? "PASS" : "FAIL"}`);

// ---- per-language parity gate ---------------------------------------------
let allOk = legacyOk;
for (const c of CASES) {
  const rc = setVoice(c.espeakVoice, c.slot);
  if (rc !== 0) {
    console.log(`${c.lang}: FAIL (snt_g2p_set_voice("${c.espeakVoice}", ${c.slot}) rc=${rc})`);
    allOk = false;
    continue;
  }
  const got = wasmIds(c.text);
  const want = truth[c.lang];
  const match = got.length === want.length && got.every((v, i) => v === want[i]);
  console.log(`${c.lang}: "${c.text}" ${match ? "PASS" : "FAIL"} (${got.length} ids)`);
  if (!match) {
    console.log(`  wasm  : [${got.join(", ")}]`);
    console.log(`  python: [${want.join(", ")}]`);
    allOk = false;
  }
}

console.log(allOk ? "ALL PASS" : "FAIL");
process.exit(allOk ? 0 : 1);
