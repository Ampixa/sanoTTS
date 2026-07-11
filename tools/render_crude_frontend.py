#!/usr/bin/env python3
"""Render the CRUDE on-device letter-to-sound frontend (the one the dashboard
text box uses when you don't supply espeak IDs) through the r7 student stack,
into an eval_scorecard dir, so we can WER it against the espeak path.

Ports text_to_ids/emit_word_approx from mcu/.../fsd_e2e_dash.c verbatim.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

# --- crude frontend, ported from fsd_e2e_dash.c ---------------------------
ID = dict(PAD=0, BOS=1, EOS=2, SPACE=3, DOT=10, A=14, B=15, D=17, E=18, F=19,
          H=20, I=21, J=22, K=23, L=24, M=25, N=26, O=27, P=28, R=30, S=31, T=32,
          U=33, V=34, W=35, Z=38, AE=39, DH=41, NG=44, AH0=50, AA=51, AO=54,
          ER=60, EH=61, G=66, IH=74, RR=88, SH=96, UH=100, AH=102, ZH=108,
          STRESS=120, LEN=122, TH=126)


def normalize(t: str) -> str:
    out = []
    last_space = False
    for c in t:
        if c.isalnum():
            out.append(c.lower()); last_space = False
        elif c.isspace() or c in "-_.!?":
            if not last_space and out:
                out.append(" ")
            last_space = True
    return "".join(out).strip()


def emit_word(w: str, out: list[int]) -> None:
    def add(x): out.extend([x, ID["PAD"]])
    i, n = 0, len(w)
    while i < n:
        c = w[i]; c2 = w[i+1] if i+1 < n else ""; c3 = w[i+2] if i+2 < n else ""
        if c == "t" and c2 == "h": add(ID["DH"] if (i == 0 and n <= 5) else ID["TH"]); i += 2; continue
        if c == "s" and c2 == "h": add(ID["SH"]); i += 2; continue
        if c == "c" and c2 == "h": add(ID["T"]); add(ID["SH"]); i += 2; continue
        if c == "n" and c2 == "g": add(ID["NG"]); i += 2; continue
        if c == "p" and c2 == "h": add(ID["F"]); i += 2; continue
        if c == "q" and c2 == "u": add(ID["K"]); add(ID["W"]); i += 2; continue
        if c == "c" and c2 == "k": add(ID["K"]); i += 2; continue
        if (c == "e" and c2 in "ea") or (c == "i" and c2 == "e"): add(ID["I"]); add(ID["LEN"]); i += 2; continue
        if c == "o" and c2 == "o": add(ID["UH"]); i += 2; continue
        if (c == "o" and c2 == "u") or (c == "o" and c2 == "w"): add(ID["A"]); add(ID["UH"]); i += 2; continue
        if (c == "a" and c2 in "iy") or (c == "e" and c2 == "y"): add(ID["E"]); i += 2; continue
        if c == "i" and c2 == "g" and c3 == "h": add(ID["A"]); add(ID["IH"]); i += 3; continue
        if c in "eiu" and c2 == "r": add(ID["ER"]); i += 2; continue
        if c == "a" and c2 == "r": add(ID["AA"]); add(ID["RR"]); i += 2; continue
        if c == "o" and c2 == "r": add(ID["AO"]); add(ID["RR"]); i += 2; continue
        m = {"a": ID["AE"], "b": ID["B"], "d": ID["D"], "e": (ID["EH"] if i != n-1 else None),
             "f": ID["F"], "g": ID["G"], "h": ID["H"], "i": ID["IH"], "j": ID["J"], "k": ID["K"],
             "l": ID["L"], "m": ID["M"], "n": ID["N"], "o": ID["AO"], "p": ID["P"], "r": ID["RR"],
             "s": ID["S"], "t": ID["T"], "u": ID["AH"], "v": ID["V"], "w": ID["W"], "y": ID["J"], "z": ID["Z"]}
        if c == "c":
            add(ID["S"] if c2 in "eiy" else ID["K"])
        elif c == "x":
            add(ID["K"]); add(ID["S"])
        elif c in m:
            if m[c] is not None:
                add(m[c])
        i += 1


def crude_text_to_ids(text: str) -> list[int]:
    norm = normalize(text)
    out: list[int] = [ID["BOS"], ID["PAD"]]
    for word in norm.split():
        if len(out) > 2 and out[-2] != ID["SPACE"]:
            out.extend([ID["SPACE"], ID["PAD"]])
        emit_word(word, out)
    out.extend([ID["DOT"], ID["PAD"], ID["EOS"]])
    return out


# --- serve harness --------------------------------------------------------
def load_serve():
    spec = importlib.util.spec_from_file_location("serve_mod", ROOT / "tools" / "serve_roota_arbitrary_tts_dashboard.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["serve_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_state(serve, a, work):
    import argparse as ap
    args = ap.Namespace(host="127.0.0.1", port=0, acoustic_checkpoint=Path(a.acoustic), decoder_backend="student",
        decoder=ROOT / "unused.onnx", decoder_student_checkpoint=Path(a.decoder_student), duration_source="student",
        duration_checkpoint=Path(a.duration), duration_length_scale=a.length_scale, audio_enhancer_checkpoint=None,
        postprocess_gain=1.0, postprocess_filter="none", piper_model=Path(a.piper_model), piper_config=Path(a.piper_config),
        out_dir=work, device="cpu", noise_scale=0.0, length_scale=1.0, noise_w=0.0, sentence_silence=0.12,
        text_chunking="none", dashboard_title="x", dashboard_subtitle="", default_text="x")
    return serve.DashboardState(args)


def render_ids(serve, st, ids: list[int], text: str) -> np.ndarray:
    lm, dm = serve.latent_mod, serve.duration_mod
    ids_t = torch.as_tensor([ids], dtype=torch.long)
    mask = torch.ones_like(ids_t, dtype=torch.bool)
    md = int(st.duration_config.get("max_duration", 80)) if st.duration_config else 80
    durs = dm.predict_durations(st.duration_model, ids_t, mask, max_duration=md, length_scale=1.08)
    durs = durs.squeeze(0).cpu().numpy().astype(np.int64)
    F = int(durs.sum())
    oc = int(st.acoustic_config.get("out_channels") or 40)
    sample = lm.ChunkSample(row_id="x", row_index=1, text=text, chunk_index=0,
                            phoneme_ids=np.asarray(ids, dtype=np.int64), durations=durs,
                            target=np.zeros((F, oc), dtype=np.float32), tensor_path=Path("live"), audio_samples=F * 256)
    lat = lm.predict_chunk(st.acoustic_model, sample, st.device)
    return st.decode_latent(lat)


def main() -> None:
    ap = argparse.ArgumentParser()
    for x in ("acoustic", "decoder-student", "duration", "piper-model", "piper-config", "textset", "out-dir"):
        ap.add_argument("--" + x, required=True)
    ap.add_argument("--length-scale", type=float, default=1.08)
    a = ap.parse_args()
    serve = load_serve()
    st = build_state(serve, a, Path(a.out_dir) / "_work")
    lm = serve.latent_mod
    texts = [json.loads(l)["text"] for l in Path(a.textset).read_text().splitlines() if l.strip()]
    d = Path(a.out_dir) / "crude"; d.mkdir(parents=True, exist_ok=True)
    man = []
    for i, text in enumerate(texts):
        try:
            ids = crude_text_to_ids(text)
            audio = render_ids(serve, st, ids, text)
            lm.write_wav(d / f"{i:05d}.wav", audio, st.sample_rate)
            man.append({"index": i, "wav": f"{i:05d}.wav", "text": text, "ok": True})
            print(f"[{i:02d}] ok {len(ids)} ids  {text[:44]!r}")
        except Exception as exc:
            man.append({"index": i, "wav": f"{i:05d}.wav", "text": text, "ok": False})
            print(f"[{i:02d}] FAIL {exc}")
    (d / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in man) + "\n")
    print("wrote", d)


if __name__ == "__main__":
    main()
