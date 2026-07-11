#!/usr/bin/env python3
"""Multi-MOS eval: SCOREQ + UTMOS + DNSMOS on a render dir, in one report.

Built after SCOREQ alone misled us (it rated a metallic-tail model above TinyTTS
while UTMOS/DNSMOS + human ears disagreed). Never trust one predictor.

  python3 tools/eval_mos_all.py \
     ours:artifacts/.../forkbfinal_fj1_diverse24 \
     tinytts:artifacts/.../tinytts_diverse24

The pip `speechmos` (DNSMOS) and torch.hub UTMOS both claim the module name
`speechmos`, so UTMOS runs in an isolated subprocess; SCOREQ+DNSMOS run together.
All metrics higher=better; DNSMOS-SIG is the one that catches metallic distortion.
"""
import argparse
import json
import statistics
import subprocess
import sys
import warnings
from math import gcd
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def to16k(p):
    a, sr = sf.read(str(p))
    if a.ndim > 1:
        a = a.mean(1)
    a = a.astype(np.float32)
    if sr != 16000:
        g = gcd(sr, 16000)
        a = resample_poly(a, 16000 // g, sr // g).astype(np.float32)
    return a


def ok_wavs(d: Path):
    for r in (json.loads(x) for x in (d / "manifest.jsonl").read_text().splitlines() if x.strip()):
        if r.get("ok", True):
            yield d / r["wav"]


def run_sqdns(d: Path):
    """SCOREQ + DNSMOS (coexist fine)."""
    from diagnose_roota_sourcefilter_codebook import load_scoreq_class
    from speechmos import dnsmos
    scoreq = load_scoreq_class()(data_domain="synthetic", mode="nr", use_onnx=True)
    sq, ov, sg = [], [], []
    for wav in ok_wavs(d):
        try:
            sq.append(float(scoreq.predict(str(wav))))
        except Exception:
            pass
        m = dnsmos.run(to16k(wav), 16000)
        ov.append(float(m["ovrl_mos"]))
        sg.append(float(m["sig_mos"]))
    mean = lambda x: statistics.mean(x) if x else float("nan")
    return {"n": len(ov), "scoreq": mean(sq), "dnsmos_ovrl": mean(ov), "dnsmos_sig": mean(sg)}


def run_utmos(d: Path):
    """UTMOS only (torch.hub; isolated process)."""
    import torch
    m = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    m.eval()
    sc = []
    for wav in ok_wavs(d):
        with torch.no_grad():
            sc.append(float(m(torch.from_numpy(to16k(wav)).unsqueeze(0), 16000)))
    return {"utmos": statistics.mean(sc) if sc else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--_mode", choices=["sqdns", "utmos"], default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_dir", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()

    # worker modes: score one dir with one metric group, emit JSON
    if a._mode:
        fn = run_sqdns if a._mode == "sqdns" else run_utmos
        print("__RESULT__" + json.dumps(fn(Path(a._dir))))
        return

    res = {}
    for spec in a.systems:
        name, _, path = spec.partition(":")
        print(f"scoring {name}…", flush=True)
        row = {}
        for mode in ("sqdns", "utmos"):
            out = subprocess.run([sys.executable, str(HERE / "eval_mos_all.py"),
                                  "--_mode", mode, "--_dir", path, "x:y"],
                                 capture_output=True, text=True)
            line = next((ln for ln in out.stdout.splitlines() if ln.startswith("__RESULT__")), None)
            if line:
                row.update(json.loads(line[len("__RESULT__"):]))
            else:
                print(f"  [{mode}] failed:\n{out.stderr[-400:]}")
        res[name] = row

    print(f"\n{'system':22s} {'SCOREQ':>7s} {'UTMOS':>7s} {'DNS-OVRL':>8s} {'DNS-SIG':>8s}  n")
    for name, r in res.items():
        print(f"{name:22s} {r.get('scoreq', float('nan')):7.3f} {r.get('utmos', float('nan')):7.3f} "
              f"{r.get('dnsmos_ovrl', float('nan')):8.3f} {r.get('dnsmos_sig', float('nan')):8.3f}  {r.get('n', 0)}")
    print("\nHigher=better. Disagreement = a metric being gamed (metallic artifact = "
          "high SCOREQ/UTMOS, low DNSMOS-SIG). Trust the consensus + ears.")
    if a.out:
        a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
