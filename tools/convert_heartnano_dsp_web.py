#!/usr/bin/env python3
"""Convert the sanotts 0.5.7 heart-nano voice package into the browser bundle under web/voices/heartnano-dsp/.

The browser runtime (web/heartnano_dsp.js) must not parse C headers or .npz archives, so this tool:

  * verifies every input file's sha256 against the package's meta.json (front_q8.bin, nano_q8_meta.h,
    dsp_decoder.npz) and refuses to write anything on a mismatch;
  * copies front_q8.bin unchanged (int8 rows + float32 scales/biases, read with the header's offsets);
  * parses nano_q8_meta.h (`#define NAME integer` lines, as sanotts.nano.parse_meta_header) into the
    "offsets" object of the output meta.json;
  * flattens dsp_decoder.npz into dsp_decoder_f32.bin: every float tensor as little-endian float32, in a
    fixed order, with its name, shape and float offset listed in meta.json "dsp_tensors"; the integer
    fields (pads, minphase, bands, noise_fft, mp_noclamp) go to meta.json "dsp_config". The npz stores
    float32 already, so the conversion is lossless (checked below by reading the .bin back).

The neural decoder blob model_q8.bin is not used by this voice (meta.json "dsp_decoder" replaces it) and
is not copied.

Rerun (inputs downloaded from Hugging Face ampixa/sanoTTS, folder heartnano-dsp/, at the revision below):

    REV=091814b27c17febea3b5951f0a836e5c30506636
    mkdir -p /tmp/hn && for f in dsp_decoder.npz front_q8.bin nano_q8_meta.h meta.json; do
      curl -sSfL -o /tmp/hn/$f https://huggingface.co/ampixa/sanoTTS/resolve/$REV/heartnano-dsp/$f; done
    python3 -I tools/convert_heartnano_dsp_web.py /tmp/hn web/voices/heartnano-dsp --revision $REV

The output is deterministic: rerunning on the same inputs gives byte-identical files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HF_REPO = "ampixa/sanoTTS"
HF_FOLDER = "heartnano-dsp"
# sanotts.engine.DEFAULT_NANO_SEED: the seed every default synthesize() call uses. Stored as a string
# because it does not fit in a JavaScript number.
DEFAULT_NANO_SEED = 2236265385529901705
DSP_INT_FIELDS = ("pads", "minphase", "bands", "noise_fft", "mp_noclamp")
DSP_BLOCK_FIELDS = ("dw_w", "dw_b", "n_w", "n_b", "p1_w", "p1_b", "p2_w", "p2_b", "gamma")
DSP_TOP_FIELDS = ("inp_w", "inp_b", "norm_w", "norm_b", "head_w", "head_b", "theta")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(path: Path, want: str | None, what: str) -> str:
    if not path.is_file():
        raise SystemExit(f"error: {path} is missing")
    if not want:
        raise SystemExit(f"error: the package meta.json states no sha256 for {what}")
    got = sha256(path)
    if got != want:
        raise SystemExit(f"error: {path}: sha256 {got} != meta.json {want}")
    return got


def parse_header(path: Path) -> dict[str, int]:
    pattern = re.compile(r"^#define\s+([A-Z0-9_]+)\s+(-?\d+)\s*$")
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if m:
            if m.group(1) in out:
                raise SystemExit(f"error: {path} defines {m.group(1)} twice")
            out[m.group(1)] = int(m.group(2))
    if "NANO_FRONT_BYTES" not in out:
        raise SystemExit(f"error: {path} does not look like a nano meta header")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path, help="directory holding the downloaded heartnano-dsp package files")
    ap.add_argument("out", type=Path, help="output directory (web/voices/heartnano-dsp)")
    ap.add_argument("--revision", required=True, help="Hugging Face revision the inputs were downloaded at")
    args = ap.parse_args()

    meta = json.loads((args.src / "meta.json").read_text(encoding="utf-8"))
    if meta.get("key") != "heart-nano" or not meta.get("local_features") or meta.get("dsp_decoder") != "dsp_decoder.npz":
        raise SystemExit("error: this is not the heart-nano local-feature voice with a dsp_decoder")
    src_meta_sha = sha256(args.src / "meta.json")
    front_sha = check(args.src / meta["front"], meta.get("front_sha256"), "front")
    header_sha = check(args.src / meta["offsets_header"], meta.get("offsets_header_sha256"), "offsets header")
    dsp_src_sha = check(args.src / meta["dsp_decoder"], meta.get("dsp_decoder_sha256"), "dsp_decoder")

    offsets = parse_header(args.src / meta["offsets_header"])
    front_bytes = (args.src / meta["front"]).stat().st_size
    if offsets["NANO_FRONT_BYTES"] != front_bytes or meta["front_bytes"] != front_bytes:
        raise SystemExit(f"error: front_q8.bin is {front_bytes} bytes, header says {offsets['NANO_FRONT_BYTES']}")
    if offsets.get("NANO_LOCAL_FEATURES") != 1:
        raise SystemExit("error: the header does not declare NANO_LOCAL_FEATURES 1")
    if offsets.get("NANO_WEIGHT_FORMAT", 0) != 0:
        raise SystemExit("error: expected int8 front weights (NANO_WEIGHT_FORMAT 0)")
    # The decoder offsets (DOFF_*) describe model_q8.bin, which this voice does not use.
    offsets = {k: v for k, v in offsets.items() if not k.startswith("DOFF_")}

    with np.load(args.src / meta["dsp_decoder"], allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    n_blocks = (arrays["pads"].size - 2) // 2
    names = list(DSP_TOP_FIELDS) + [f"b{i}_{k}" for i in range(n_blocks) for k in DSP_BLOCK_FIELDS]
    expected = set(names) | set(DSP_INT_FIELDS)
    if set(arrays) != expected:
        raise SystemExit(f"error: unexpected npz members: extra {sorted(set(arrays) - expected)}, "
                         f"missing {sorted(expected - set(arrays))}")
    tensors, chunks, at = [], [], 0
    for name in names:
        a = arrays[name]
        if a.dtype != np.float32:
            raise SystemExit(f"error: {name} is {a.dtype}, expected float32")
        flat = np.ascontiguousarray(a, dtype="<f4").reshape(-1)
        tensors.append({"name": name, "shape": list(a.shape), "offset": at})
        chunks.append(flat)
        at += flat.size
    blob = np.concatenate(chunks).astype("<f4").tobytes()
    for name in DSP_INT_FIELDS:
        if arrays[name].dtype.kind not in "iu":
            raise SystemExit(f"error: {name} is {arrays[name].dtype}, expected an integer")
    config = {
        "pads": [int(v) for v in arrays["pads"].tolist()],
        "minphase": bool(int(arrays["minphase"])),
        "harm_bands": int(arrays["bands"][0]),
        "noise_bands": int(arrays["bands"][1]),
        "noise_fft": int(arrays["noise_fft"]),
        "mp_noclamp": bool(int(arrays["mp_noclamp"])),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.src / meta["front"], args.out / "front_q8.bin")
    (args.out / "dsp_decoder_f32.bin").write_bytes(blob)
    # read back: every tensor must round-trip bit for bit
    back = np.frombuffer((args.out / "dsp_decoder_f32.bin").read_bytes(), dtype="<f4")
    for t in tensors:
        n = int(np.prod(t["shape"])) if t["shape"] else 1
        if not np.array_equal(back[t["offset"]:t["offset"] + n].reshape(t["shape"]), arrays[t["name"]]):
            raise SystemExit(f"error: {t['name']} did not round-trip")

    out_meta = {
        "key": "heart-nano",
        "runtime": "nano_dsp_js",
        "description": "sanotts 0.5.7 heart-nano (104,215 params): int8 local-feature front end + "
                       "harmonic-plus-noise decoder, run by web/heartnano_dsp.js",
        "sanotts_version": "0.5.7",
        "params": meta["params"],
        "param_breakdown": meta["param_breakdown"],
        "sample_rate": meta["sample_rate"],
        "hop": meta["hop"],
        "vocab_size": meta["vocab_size"],
        "frontend": meta["frontend"],
        "seed": str(DEFAULT_NANO_SEED),
        "front": "front_q8.bin",
        "front_bytes": front_bytes,
        "front_sha256": front_sha,
        "dsp": "dsp_decoder_f32.bin",
        "dsp_bytes": len(blob),
        "dsp_sha256": hashlib.sha256(blob).hexdigest(),
        "dsp_config": config,
        "dsp_tensors": tensors,
        "offsets": offsets,
        "source": {
            "hf_repo": HF_REPO,
            "revision": args.revision,
            "folder": HF_FOLDER,
            "meta.json": src_meta_sha,
            "front_q8.bin": front_sha,
            "nano_q8_meta.h": header_sha,
            "dsp_decoder.npz": dsp_src_sha,
            "lineage": meta["lineage"],
            "exported": meta["exported"],
        },
        "converted_by": "tools/convert_heartnano_dsp_web.py",
    }
    (args.out / "meta.json").write_text(json.dumps(out_meta, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: front_q8.bin {front_bytes} B, dsp_decoder_f32.bin {len(blob)} B "
          f"({len(tensors)} tensors, {at} floats), meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
