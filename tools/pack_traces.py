#!/usr/bin/env python3
"""pack_traces.py -- compress raw engine traces into web-ready tensors.

Reads  public/traces/<row_id>/{*.f32,*.i32}  produced by tools/run_traces.sh
(the vendored int8 E13b engine, full-student path) and writes web packets:

  public/traces/<row_id>/{ids,durs}.u8
  public/traces/<row_id>/{dur_hidden,tok_hidden,frame_hidden,stem,
                          trunk_b0..b3,noise}.i8      symmetric int8 + absmax
  public/traces/<row_id>/mel.u8                        per-tensor lo/hi
  public/traces/<row_id>/{spec_logmag,spec_phase}.u8   derived from re/im
  public/traces/<row_id>/pcm.i16                       24 kHz waveform
  public/traces/<row_id>/manifest.json                 shapes, scales, text,
                                                       phoneme string, stats
  public/traces/index.json                             all rows + vocab

Every number stays faithful: int8/uint8 scales are stored in the manifest and
the site dequantizes before display. pcm int16 keeps > 90 dB SNR vs the f32
trace, far above the int8 engine's own quantization noise.

Pure stdlib. No numpy.
"""
import json
import math
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "public" / "traces"

# The model's frozen 62-symbol vocabulary (ids 0..61), shared by the golden
# fixtures and the engine's duration/acoustic embedding tables.
VOCAB = ["<pad>", "<bos>", "<eos>"] + [
    " ", "!", '"', "(", ")", ",", ".", ":", ";", "?", "A", "I", "O", "T", "W",
    "Y", "b", "d", "f", "h", "i", "j", "k", "l", "m", "n", "p", "s", "t", "u",
    "v", "w", "z", "æ", "ð", "ŋ", "ɐ", "ɑ", "ɔ", "ə", "ɛ", "ɜ", "ɡ", "ɪ",
    "ɹ", "ʃ", "ʊ", "ʌ", "ʒ", "ʤ", "ʧ", "ˈ", "ˌ", "θ", "ᵊ", "ᵻ", "—", "“",
    "”",
]
assert len(VOCAB) == 62

# Sentence texts for the eight golden rows, from the internal evaluation
# record (rows_meta.rows). Override with SANOTTS_EVIDENCE if it lives elsewhere.
EVIDENCE = Path(
    os.environ.get(
        "SANOTTS_EVIDENCE",
        str(Path(__file__).resolve().parent / "evidence" / "rows.json"),
    )
)


def read_f32(path: Path) -> list[float]:
    data = path.read_bytes()
    return list(struct.unpack("<%df" % (len(data) // 4), data))


def read_i32(path: Path) -> list[int]:
    data = path.read_bytes()
    return list(struct.unpack("<%di" % (len(data) // 4), data))


def pack_i8(vals: list[float]) -> tuple[bytes, float]:
    amax = max((abs(v) for v in vals), default=0.0) or 1.0
    q = bytes(max(-127, min(127, round(v / amax * 127.0))) & 0xFF for v in vals)
    return q, amax / 127.0


def pack_u8_range(vals: list[float]) -> tuple[bytes, float, float]:
    lo = min(vals, default=0.0)
    hi = max(vals, default=1.0)
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    q = bytes(max(0, min(255, round((v - lo) / (hi - lo) * 255.0))) for v in vals)
    return q, lo, hi


def pack_pcm_i16(vals: list[float]) -> bytes:
    out = bytearray()
    for v in vals:
        v = max(-1.0, min(1.0, v))
        out += struct.pack("<h", round(v * 32767.0))
    return bytes(out)


def main() -> int:
    texts: dict[str, str] = {}
    if EVIDENCE.is_file():
        ev = json.loads(EVIDENCE.read_text())
        for row in ev.get("rows_meta", {}).get("rows", []):
            texts[row["row_id"]] = row["text"]
    else:
        print(f"warn: {EVIDENCE} not found; texts will be empty", file=sys.stderr)

    index_rows = []
    for row_dir in sorted(TRACES.iterdir()):
        if not row_dir.is_dir():
            continue
        man = json.loads((row_dir / "manifest.json").read_text())
        row_id = man["row_id"]
        n_tok, T = man["n_tokens"], man["got_frames"]

        ids = read_i32(row_dir / "ids.i32")
        durs = read_i32(row_dir / "durs.i32")
        bad = [i for i in ids if not (0 <= i < 62)]
        if bad:
            raise SystemExit(f"{row_id}: id(s) out of vocab: {bad[:5]}")
        phones = "".join(VOCAB[i] for i in ids[1:-1])  # strip <bos>/<eos>

        (row_dir / "ids.u8").write_bytes(bytes(ids))
        (row_dir / "durs.u8").write_bytes(bytes(durs))

        scales: dict[str, object] = {}
        for name in ("dur_hidden", "tok_hidden", "frame_hidden", "stem",
                     "trunk_b0", "trunk_b1", "trunk_b2", "trunk_b3", "noise"):
            vals = read_f32(row_dir / f"{name}.f32")
            q, scale = pack_i8(vals)
            (row_dir / f"{name}.i8").write_bytes(q)
            scales[name] = {"scale": scale}

        mel = read_f32(row_dir / "mel.f32")
        q, lo, hi = pack_u8_range(mel)
        (row_dir / "mel.u8").write_bytes(q)
        scales["mel"] = {"lo": lo, "hi": hi}

        re = read_f32(row_dir / "spec_re.f32")
        im = read_f32(row_dir / "spec_im.f32")
        logmag, phase = [], []
        for a, b in zip(re, im):
            logmag.append(0.5 * math.log(a * a + b * b + 1e-20))
            phase.append(math.atan2(b, a))
        q, lm_lo, lm_hi = pack_u8_range(logmag)
        (row_dir / "spec_logmag.u8").write_bytes(q)
        q, ph_lo, ph_hi = pack_u8_range(phase)
        (row_dir / "spec_phase.u8").write_bytes(q)
        scales["spec_logmag"] = {"lo": lm_lo, "hi": lm_hi}
        scales["spec_phase"] = {"lo": ph_lo, "hi": ph_hi}

        pcm = read_f32(row_dir / "pcm.f32")
        (row_dir / "pcm.i16").write_bytes(pack_pcm_i16(pcm))
        peak = max((abs(v) for v in pcm), default=0.0)

        man.update({
            "text": texts.get(row_id, ""),
            "phonemes": phones,
            "duration_s": man["got_samples"] / 24000.0,
            "pcm_peak": peak,
            "seed_lo": man["seed"] & 0xFFFFFFFF,
            "seed_hi": (man["seed"] >> 32) & 0xFFFFFFFF,
            "packs": {
                "ids": "u8", "durs": "u8", "pcm": "i16",
                "mel": "u8", "spec_logmag": "u8", "spec_phase": "u8",
                "hidden": "i8",
            },
            "scales": scales,
        })
        (row_dir / "manifest.json").write_text(json.dumps(man, indent=1))

        # drop the heavy float intermediates once packed
        for f in row_dir.glob("*.f32"):
            f.unlink()
        (row_dir / "ids.i32").unlink()
        (row_dir / "durs.i32").unlink()

        index_rows.append({
            "row_id": row_id,
            "text": man["text"],
            "n_tokens": n_tok,
            "frames": T,
            "samples": man["got_samples"],
            "duration_s": round(man["duration_s"], 3),
        })
        print(f"[pack] {row_id}: N={n_tok} T={T} "
              f"mel[{lo:.2f},{hi:.2f}] pcm_peak={peak:.3f} "
              f"phones={phones[:44]!r}...")

    (TRACES / "index.json").write_text(json.dumps({
        "model": "en_us_e13b",
        "params": 294279,
        "sample_rate": 24000,
        "hop": 256,
        "n_fft": 1024,
        "vocab": VOCAB,
        "source": ("int8 device-math trace, full-student path, vendored "
                   "snt_nano.c -DSNT_NANO_TRACE; validated against golden "
                   "fixtures at min corr 0.981 (frozen-duration mode)"),
        "rows": index_rows,
    }, indent=1, ensure_ascii=False))
    print(f"[pack] wrote index.json with {len(index_rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
