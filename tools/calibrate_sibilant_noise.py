#!/usr/bin/env python3
"""Calibrate per-channel teacher latent std at sibilant frames, for inference-time
sibilant fricative-noise injection (see serve_roota_arbitrary_tts_dashboard.py
--sibilant-inject-beta). A deterministic acoustic regresses the *mean* latent, so
sibilants (/s ʃ z ʒ/, broadband noise) collapse to a whistly tone. At inference we
add beta * tea_std Gaussian noise into the predicted latent ONLY at sibilant frames,
restoring the variance the decoder needs to render proper hiss (the decoder already
renders high-variance teacher fricative latents cleanly).

Emits an .npz with:
  tea_std : float32[latent_channels]  per-channel teacher std at sibilant frames
  sib_ids : int64[k]                  Piper phoneme-ids for the sibilant symbols

Usage:
  python3 tools/calibrate_sibilant_noise.py \
    --pack   <a *-decoder-piper-native-* pack dir with generator_input + phoneme_ids + w_ceil> \
    --out    releases/<voice>/sibilant-injection/calib.npz
The pack must be the SAME teacher voice you serve; sib_ids are voice-specific.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SIBILANTS = set("szʃʒ")  # s z ʃ ʒ — the broadband-noise fricatives that collapse to tones


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, help="decoder-piper-native pack dir (rows.json + tensors/*.npz)")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--latent-key", default="generator_input")
    a = ap.parse_args()

    pack = Path(a.pack)
    rows = json.load(open(pack / "rows.json"))
    cols = []
    sib_ids: set[int] = set()
    n_sib = 0
    for r in rows:
        for ch in r["chunks"]:
            npz = np.load(pack / "tensors" / Path(ch["tensor_npz"]).name)
            phon = ch["phonemes"]
            ids = np.asarray(npz["phoneme_ids"]).reshape(-1)
            w = np.rint(np.asarray(npz["w_ceil"]).reshape(-1)).astype(int)
            lat = np.asarray(npz[a.latent_key])
            if lat.ndim == 3:
                lat = lat[0]  # [C, F]
            cum = np.concatenate([[0], np.cumsum(w)])
            P = len(phon)
            for i, p in enumerate(phon):
                if p not in SIBILANTS:
                    continue
                # Piper id layout [BOS, pad, p0, pad, ...] -> phoneme i at index 2+2i
                j = 2 + 2 * i if len(ids) >= 2 * P + 1 else i
                if j + 1 >= len(cum) or j >= len(ids):
                    continue
                x, y = int(cum[j]), int(cum[j + 1])
                if 0 <= x < y <= lat.shape[1]:
                    cols.append(lat[:, x:y])
                    sib_ids.add(int(ids[j]))
                    n_sib += 1
    if not cols:
        raise SystemExit("no sibilant frames found — is this the right voice's pack?")
    allcols = np.concatenate(cols, axis=1)  # [C, N]
    tea_std = allcols.std(axis=1).astype(np.float32)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, tea_std=tea_std, sib_ids=np.asarray(sorted(sib_ids), dtype=np.int64))
    print(f"calibrated from {n_sib} sibilant instances ({allcols.shape[1]} frames), "
          f"{len(sib_ids)} sibilant ids {sorted(sib_ids)}, "
          f"latent_channels={tea_std.shape[0]}, tea_std mean {tea_std.mean():.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
