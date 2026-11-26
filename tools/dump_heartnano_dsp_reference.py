#!/usr/bin/env python3
"""Dump the sanotts package's heart-nano (0.5.7, harmonic-plus-noise decoder) outputs for the browser port's
numerical check (tools/verify_heartnano_dsp_node.mjs).

For each row it writes, into OUT/:
  rNN_ids.i32     phoneme ids (int32 LE)
  rNN_durs.i32    sanotts.nano.duration_forward durations
  rNN_mel.f32     sanotts.nano.acoustic_forward mel [100, T] (C order), the whole-utterance pass
  rNN_audio.f32   Synthesizer.synthesize_ids(ids).audio: the package's default path (streamed local
                  front end + nano_dsp.DspStream, seed engine.DEFAULT_NANO_SEED, clipped to [-1, 1])
  rNN_render.f32  nano_dsp.render(dsp_weights, mel, seed=DEFAULT_NANO_SEED), unclipped
plus unit_mpm32.f32 [40, 513], unit_nim32.f32 [24, 257], unit_white.f32 (frames 0..7 of the noise,
[257, 8] complex64 as interleaved re/im), unit_ctl_r00.f64 (controls of row 0, [T, 66]) and rows.json.

Rows: the en_us_e13b golden fixture ids (same 62-symbol vocabulary) and a few sentences through the
package's own default G2P (nano_g2p "lexicon").

    python3 tools/dump_heartnano_dsp_reference.py --pkg ../sanoTTS-ttfa/pypkg \
        --voice-dir /path/to/heartnano-dsp --out /tmp/hn-ref
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SENTENCES = [
    "Hello! I am a tiny voice living in your browser.",
    "The quick brown fox jumps over the lazy dog, and then it runs away.",
    "At 7:45 on Tuesday, the train to Boston was already twenty minutes late.",
    "Can you hear the difference between a whisper and a shout?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=Path, required=True, help="directory containing the sanotts package (pypkg)")
    ap.add_argument("--voice-dir", type=Path, required=True)
    ap.add_argument("--fixture", type=Path, default=Path(__file__).resolve().parents[1] / "mcu/test/fixtures/en_us_e13b")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.pkg.resolve()))
    import sanotts  # noqa: PLC0415
    from sanotts import engine, nano, nano_dsp, nano_frontend  # noqa: PLC0415

    if sanotts.__version__ != "0.5.7":
        raise SystemExit(f"error: expected sanotts 0.5.7, found {sanotts.__version__}")
    synth = engine.Synthesizer(voice_dir=args.voice_dir)
    if synth.pack.dsp is None or not nano.is_local(synth.pack.offsets):
        raise SystemExit("error: the voice is not the local-feature heart-nano with a DSP decoder")
    w = synth._nano_weights()
    dsp = synth._dsp_weights()
    seed = engine.DEFAULT_NANO_SEED
    if dict(synth._vocabulary) != dict(nano_frontend.DEFAULT_VOCABULARY):
        raise SystemExit("error: the voice vocabulary is not the default 62-symbol table")

    rows = []
    fixture_rows = sorted(args.fixture.glob("r*_ids.bin"))
    if not fixture_rows:
        raise SystemExit(f"error: no fixture ids under {args.fixture}")
    for p in fixture_rows:
        ids = np.fromfile(p, dtype="<i4")
        if ids.min() < 0 or ids.max() >= len(nano_frontend.DEFAULT_VOCABULARY):
            raise SystemExit(f"error: {p} has ids outside the vocabulary")
        rows.append({"source": f"fixture {args.fixture.name}/{p.name}", "ids": ids.astype(np.int64)})
    for text in SENTENCES:
        rows.append({"source": "g2p: " + text, "ids": np.asarray(synth._nano_ids(text), dtype=np.int64)})

    args.out.mkdir(parents=True, exist_ok=True)
    listing = []
    for i, row in enumerate(rows):
        ids = row["ids"]
        durs = nano.duration_forward(w, ids).astype(np.int64)
        mel = nano.acoustic_forward(w, ids, durs).astype(np.float32)
        audio = synth.synthesize_ids(ids).audio.astype(np.float32)
        ren = nano_dsp.render(dsp, mel, seed=seed).astype(np.float32)
        if audio.size != (mel.shape[1] - 1) * nano_dsp.HOP:
            raise SystemExit(f"error: row {i}: synthesize_ids gave {audio.size} samples for {mel.shape[1]} frames")
        tag = f"r{i:02d}"
        ids.astype("<i4").tofile(args.out / f"{tag}_ids.i32")
        durs.astype("<i4").tofile(args.out / f"{tag}_durs.i32")
        np.ascontiguousarray(mel, dtype="<f4").tofile(args.out / f"{tag}_mel.f32")
        audio.astype("<f4").tofile(args.out / f"{tag}_audio.f32")
        ren.astype("<f4").tofile(args.out / f"{tag}_render.f32")
        listing.append({"tag": tag, "source": row["source"], "tokens": int(ids.size), "frames": int(mel.shape[1]),
                        "samples": int(audio.size)})
        if i == 0:
            nano_dsp.controls(dsp, mel, True, True).astype("<f8").tofile(args.out / "unit_ctl_r00.f64")
    syn = nano_dsp._Synth(dsp, seed)
    syn.mpm32.astype("<f4").tofile(args.out / "unit_mpm32.f32")
    syn.nim32.astype("<f4").tofile(args.out / "unit_nim32.f32")
    white = syn.white_cache[:, :8]
    np.stack([white.real, white.imag], -1).astype("<f4").tofile(args.out / "unit_white.f32")
    (args.out / "rows.json").write_text(json.dumps({"seed": str(seed), "sanotts": sanotts.__version__,
                                                    "rows": listing}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(listing)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
