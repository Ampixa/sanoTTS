#!/usr/bin/env python3
"""Render teacher + student (r7 fsd stack) wavs for a textset into eval_scorecard
dirs (NNNNN.wav + manifest.jsonl), so we can run Whisper WER / SCOREQ on the same
sentences. Mirrors the fidelity-eval harness setup.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_serve():
    spec = importlib.util.spec_from_file_location("serve_mod", ROOT / "tools" / "serve_roota_arbitrary_tts_dashboard.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["serve_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_state(serve, a, work):
    import argparse as ap
    args = ap.Namespace(
        host="127.0.0.1", port=0, acoustic_checkpoint=Path(a.acoustic), decoder_backend="student",
        decoder=ROOT / "unused.onnx", decoder_student_checkpoint=Path(a.decoder_student),
        duration_source="student", duration_checkpoint=Path(a.duration), duration_length_scale=a.length_scale,
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
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--length-scale", type=float, default=1.08)
    a = ap.parse_args()

    serve = load_serve()
    st = build_state(serve, a, Path(a.out_dir) / "_work")
    texts = [json.loads(l)["text"] for l in Path(a.textset).read_text().splitlines() if l.strip()]

    dirs = {lane: Path(a.out_dir) / lane for lane in ("teacher", "student")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    man = {lane: [] for lane in dirs}
    for i, text in enumerate(texts):
        try:
            meta = st.synthesize(text, duration_source="student", duration_length_scale=a.length_scale)
        except Exception as exc:
            print(f"[{i:02d}] FAIL {text[:40]!r}: {exc}")
            for lane in dirs:
                man[lane].append({"index": i, "wav": f"{i:05d}.wav", "text": text, "ok": False})
            continue
        rid = meta["id"]
        for lane, suffix in (("teacher", "teacher"), ("student", "student")):
            src = st.audio_dir / f"{rid}-{suffix}.wav"
            shutil.copy(src, dirs[lane] / f"{i:05d}.wav")
            man[lane].append({"index": i, "wav": f"{i:05d}.wav", "text": text, "ok": True})
        print(f"[{i:02d}] ok  {text[:50]!r}")
    for lane, rows in man.items():
        (dirs[lane] / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print("wrote", ", ".join(str(d) for d in dirs.values()))


if __name__ == "__main__":
    main()
