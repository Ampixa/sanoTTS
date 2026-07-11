#!/usr/bin/env python3
"""Calibrate per-channel teacher-latent std at sibilant frames in the FSD 40-dim
code space (the space the MCU decoder consumes), for on-device sibilant noise
injection.

The shipped releases/.../sibilant-injection/calib.npz is for the 192-dim piper
generator-latent path. The MCU runs the fsd stack whose acoustic outputs 40-dim
`c` (FSD_CODE_DIM). This tool recalibrates in that space by capturing the teacher
latent (piper generator_input, projected to 40-dim for this fsd teacher) at
sibilant frames via the serve_roota harness the fidelity eval uses.

Emits calib_fsd.npz:
  tea_std : float32[40]  per-channel teacher std at sibilant frames
  sib_ids : int64[k]     piper phoneme-ids for /s z ʃ ʒ/
  n_frames: int          how many sibilant frames it averaged over

Usage:
  python3 tools/calibrate_sibilant_noise_fsd.py \
    --acoustic <latent-student.pt> --decoder-student <decoder-student.pt> \
    --duration <duration-student.pt> \
    --piper-model <teacher.onnx> --piper-config <teacher.onnx.json> \
    --textset <jsonl> --out <calib_fsd.npz>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIBILANTS = set("szʃʒ")


def load_serve():
    spec = importlib.util.spec_from_file_location(
        "serve_mod", ROOT / "tools" / "serve_roota_arbitrary_tts_dashboard.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load serve_roota_arbitrary_tts_dashboard.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["serve_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_state(serve, a, work):
    args = argparse.Namespace(
        host="127.0.0.1", port=0,
        acoustic_checkpoint=Path(a.acoustic), decoder_backend="student", decoder=ROOT / "unused.onnx",
        decoder_student_checkpoint=Path(a.decoder_student),
        duration_source="oracle", duration_checkpoint=Path(a.duration), duration_length_scale=1.0,
        audio_enhancer_checkpoint=None, postprocess_gain=1.0, postprocess_filter="none",
        piper_model=Path(a.piper_model), piper_config=Path(a.piper_config),
        out_dir=work, device="cpu", noise_scale=0.0, length_scale=1.0, noise_w=0.0,
        sentence_silence=0.12, text_chunking="none", dashboard_title="x", dashboard_subtitle="", default_text="x")
    return serve.DashboardState(args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acoustic", required=True)
    ap.add_argument("--decoder-student", required=True)
    ap.add_argument("--duration", required=True)
    ap.add_argument("--piper-model", required=True)
    ap.add_argument("--piper-config", required=True)
    ap.add_argument("--textset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    serve = load_serve()
    st = build_state(serve, a, ROOT / "artifacts" / "_sibcal_work")

    # capture, per chunk: (phonemes, phoneme_ids), oracle durations, teacher latent
    cap_ph: list[tuple[list, list]] = []
    cap_du: list[np.ndarray] = []
    cap_lat: list[np.ndarray] = []

    orig_ids = st.voice.phonemes_to_ids
    st.voice.phonemes_to_ids = lambda ph: (cap_ph.append((list(ph), list(orig_ids(ph)))) or orig_ids(ph))  # type: ignore

    orig_du = st.predict_student_durations
    def wrap_du(ids, oracle, src, ls):
        d, s = orig_du(ids, oracle, src, ls)
        cap_du.append(np.asarray(oracle).reshape(-1))  # oracle durations align to teacher_latent
        return d, s
    st.predict_student_durations = wrap_du  # type: ignore

    orig_dec = st.decode_latent
    def wrap_dec(latent):
        cap_lat.append(np.asarray(latent))  # pairs: [teacher, student] per chunk
        return orig_dec(latent)
    st.decode_latent = wrap_dec  # type: ignore

    texts = [json.loads(l)["text"] for l in Path(a.textset).read_text().splitlines() if l.strip()]

    sib_cols: list[np.ndarray] = []  # each [40, nsib_frames]
    sib_ids: set[int] = set()
    for text in texts:
        cap_ph.clear(); cap_du.clear(); cap_lat.clear()
        try:
            st.synthesize(text, duration_source="oracle", duration_length_scale=1.0)
        except Exception as exc:
            print(f"skip: {text[:40]!r}: {exc}")
            continue
        # teacher latents are the even-indexed decode calls (oracle before student)
        teacher_lats = cap_lat[0::2]
        for (phon, ids), dur, lat in zip(cap_ph, cap_du, teacher_lats):
            lat = np.squeeze(lat)          # [40, F]
            if lat.ndim != 2:
                continue
            if lat.shape[0] != 40 and lat.shape[1] == 40:
                lat = lat.T
            w = np.rint(dur).astype(int)
            cum = np.concatenate([[0], np.cumsum(w)])
            ids_arr = np.asarray(ids).reshape(-1)
            P = len(phon)
            for i, p in enumerate(phon):
                if p not in SIBILANTS:
                    continue
                j = 2 + 2 * i if len(ids_arr) >= 2 * P + 1 else i  # piper [BOS,pad,p,pad,...]
                if j + 1 >= len(cum) or j >= len(ids_arr):
                    continue
                x, y = cum[j], cum[j + 1]
                if y <= x or y > lat.shape[1]:
                    continue
                sib_cols.append(lat[:, x:y])
                sib_ids.add(int(ids_arr[j]))

    if not sib_cols:
        raise SystemExit("no sibilant frames collected")
    allcols = np.concatenate(sib_cols, axis=1)   # [40, N]
    tea_std = allcols.std(axis=1).astype(np.float32)
    sib_ids_arr = np.asarray(sorted(sib_ids), dtype=np.int64)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, tea_std=tea_std, sib_ids=sib_ids_arr, n_frames=np.int64(allcols.shape[1]))
    print(f"tea_std[40] mean={tea_std.mean():.4f} min={tea_std.min():.4f} max={tea_std.max():.4f}")
    print(f"sib_ids={sib_ids_arr.tolist()}  n_sibilant_frames={allcols.shape[1]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
