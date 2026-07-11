#!/usr/bin/env python3
"""G2P parity + WER eval loop for the on-device espeak port.

Reference = piper/espeak in-context phonemization -> Kristin ids (the 15.3%-WER
path). Candidate = a jsonl of {"text":..., "ids":[...]} produced by the port
(host build or board dump).

Reports, per sentence and overall:
  * PER  — phoneme-id error rate (Levenshtein(candidate, reference)/len(ref)).
           0% == bit-exact parity with espeak (the goal for a correct port).
  * WER  — optional: render candidate ids through the r7 student, Whisper WER,
           vs the espeak reference (needs --render + checkpoints).

Usage (parity only):
  python3 tools/eval_g2p_parity.py --candidate board_ids.jsonl
Usage (parity + WER):
  python3 tools/eval_g2p_parity.py --candidate board_ids.jsonl --render \
     --acoustic ... --decoder-student ... --duration ... \
     --piper-model ... --piper-config ... --out-dir artifacts/g2p-wer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from piper.phonemize_espeak import EspeakPhonemizer
from piper.phoneme_ids import phonemes_to_ids

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "models" / "teachers" / "kristin-medium" / "en_US-kristin-medium.onnx.json"
ID_MAP = json.load(open(CFG, encoding="utf-8"))["phoneme_id_map"]
PH = EspeakPhonemizer()


def lev(a, b):
    m, n = len(a), len(b)
    if m == 0:
        return n
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def espeak_ids(text: str) -> list[int]:
    return phonemes_to_ids([p for s in PH.phonemize("en-us", text) for p in s], ID_MAP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="jsonl of {text, ids}")
    ap.add_argument("--render", action="store_true", help="also render + WER")
    ap.add_argument("--acoustic"); ap.add_argument("--decoder-student"); ap.add_argument("--duration")
    ap.add_argument("--piper-model"); ap.add_argument("--piper-config")
    ap.add_argument("--out-dir", default="artifacts/g2p-wer")
    ap.add_argument("--length-scale", type=float, default=1.08)
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.candidate).read_text().splitlines() if l.strip()]
    tot_err = tot_len = 0
    exact = 0
    for r in rows:
        ref = espeak_ids(r["text"])
        cand = list(r["ids"])
        e = lev(cand, ref)
        tot_err += e; tot_len += len(ref)
        if e == 0:
            exact += 1
        print(f"PER {e/max(1,len(ref)):5.1%}  ref={len(ref):3d} cand={len(cand):3d}  {r['text'][:44]!r}")
    print(f"\nPARITY: PER {tot_err/max(1,tot_len):.2%}  ({tot_err}/{tot_len} id edits)  "
          f"exact sentences {exact}/{len(rows)}")

    if a.render:
        sys.path.insert(0, str(ROOT / "tools"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("rc", ROOT / "tools" / "render_crude_frontend.py")
        rc = importlib.util.module_from_spec(spec); sys.modules["rc"] = spec; spec.loader.exec_module(rc)  # type: ignore
        serve = rc.load_serve()
        st = rc.build_state(serve, a, Path(a.out_dir) / "_work")
        lm = serve.latent_mod
        d = Path(a.out_dir) / "candidate"; d.mkdir(parents=True, exist_ok=True)
        man = []
        for i, r in enumerate(rows):
            try:
                audio = rc.render_ids(serve, st, list(r["ids"]), r["text"])
                lm.write_wav(d / f"{i:05d}.wav", audio, st.sample_rate)
                man.append({"index": i, "wav": f"{i:05d}.wav", "text": r["text"], "ok": True})
            except Exception as exc:
                man.append({"index": i, "wav": f"{i:05d}.wav", "text": r["text"], "ok": False})
                print(f"[{i}] render fail: {exc}")
        (d / "manifest.jsonl").write_text("\n".join(json.dumps(x) for x in man) + "\n")
        print(f"rendered candidate wavs -> {d}  (run tools/eval_scorecard.py candidate:{d} ... for WER)")


if __name__ == "__main__":
    main()
