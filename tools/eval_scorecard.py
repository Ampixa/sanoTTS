#!/usr/bin/env python3
"""Multi-axis scorecard for a TTS render dir vs the reference texts.

SCOREQ is one axis and a gameable one. This adds the axes that actually decide
whether a TTS is good: intelligibility (WER via Whisper), pace (speaking rate),
and prosodic range (F0 mean/std/range). Run on the same NNNNN.wav + manifest
dirs the SCOREQ comparison uses, so every system is judged on identical text.

  python3 tools/eval_scorecard.py \
     tinytts:artifacts/tinytts-compare-20260706/tinytts_diverse24 \
     dec503:artifacts/tinytts-compare-20260706/dec503_diverse24 \
     --out artifacts/tinytts-compare-20260706/scorecard.json

WER is a RELATIVE comparison: both systems' Whisper transcripts are normalised
the same way against the same reference, so ASR/number-format quirks cancel.
"""
import argparse
import json
import re
import statistics
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio
import whisper

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def norm_text(s: str) -> list[str]:
    s = s.lower().replace("-", " ")
    s = _PUNCT.sub("", s)
    return _WS.sub(" ", s).strip().split()


def wer(ref: list[str], hyp: list[str]) -> float:
    # word-level Levenshtein / len(ref)
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev = d[0]
        d[0] = i
        for j in range(1, m + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return d[m] / n


def wav_info(path: Path):
    with wave.open(str(path), "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        a = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    return a, sr, n / sr


def f0_stats(a: np.ndarray, sr: int):
    # voiced F0 via torchaudio; keep speech-range frames only
    t = torch.from_numpy(a).unsqueeze(0)
    try:
        pitch = torchaudio.functional.detect_pitch_frequency(t, sr).squeeze(0).numpy()
    except Exception:
        return None
    v = pitch[(pitch > 70) & (pitch < 400)]
    if v.size < 5:
        return None
    return {"f0_mean": float(np.mean(v)), "f0_std": float(np.std(v)),
            "f0_p10": float(np.percentile(v, 10)), "f0_p90": float(np.percentile(v, 90))}


def load_manifest(d: Path) -> list[dict]:
    rows = []
    for line in (d / "manifest.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def score_dir(model, name: str, d: Path) -> dict:
    rows = load_manifest(d)
    per = []
    for r in rows:
        if not r.get("ok", False):
            continue
        wav = d / r["wav"]
        a, sr, dur = wav_info(wav)
        ref = norm_text(r.get("text", ""))
        res = model.transcribe(str(wav), language="en", fp16=False)
        hyp = norm_text(res["text"])
        w = wer(ref, hyp)
        rate = len(ref) / dur if dur > 0 else 0.0
        f0 = f0_stats(a, sr) or {}
        per.append({"index": r["index"], "dur_s": dur, "ref_words": len(ref),
                    "wer": w, "rate_wps": rate, **f0,
                    "hyp": " ".join(hyp)})
        print(f"  [{name}] idx {r['index']:02d}  WER {w:5.1%}  {rate:4.2f} w/s  "
              f"dur {dur:5.2f}s  F0 {f0.get('f0_mean',0):5.1f}±{f0.get('f0_std',0):4.1f}")
    def mean(k):
        vals = [x[k] for x in per if k in x and x[k] is not None]
        return statistics.mean(vals) if vals else None
    return {"name": name, "dir": str(d), "n": len(per),
            "wer_mean": mean("wer"),
            "wer_median": statistics.median([x["wer"] for x in per]),
            "rate_wps_mean": mean("rate_wps"), "dur_s_mean": mean("dur_s"),
            "f0_mean": mean("f0_mean"), "f0_std_mean": mean("f0_std"),
            "f0_range_mean": (mean("f0_p90") - mean("f0_p10")) if mean("f0_p90") else None,
            "rows": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+", help="name:dir pairs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="small.en")
    a = ap.parse_args()

    print(f"loading Whisper {a.model}…")
    model = whisper.load_model(a.model)

    results = {}
    for spec in a.systems:
        name, _, path = spec.partition(":")
        print(f"scoring {name} ({path})…")
        results[name] = score_dir(model, name, Path(path))

    a.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n==== SCORECARD (extra axes) ====")
    hdr = f"{'system':10s} {'WER':>7s} {'rate w/s':>9s} {'dur s':>7s} {'F0 mean':>8s} {'F0 std':>7s} {'F0 rng':>7s}"
    print(hdr)
    for name, r in results.items():
        print(f"{name:10s} {r['wer_mean']:6.1%} {r['rate_wps_mean']:9.2f} "
              f"{r['dur_s_mean']:7.2f} {r['f0_mean']:8.1f} {r['f0_std_mean']:7.1f} "
              f"{(r['f0_range_mean'] or 0):7.1f}")
    print("\nWER lower=better (intelligibility). rate/dur: pace. "
          "F0 std/range higher=more expressive (less monotone).")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
