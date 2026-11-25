"""Gate the numpy nano runtime against the same PyTorch reference the C is held to.

`make -C mcu test-nano` holds the C runtime to corr >= 0.98 over eight rows.
This holds the numpy runtime to the same reference at the same threshold, so
the two implementations are compared against PyTorch rather than against each
other -- neither can drift the gate by agreeing with the other's mistake.

Run:  python3 -m pytest pypkg/tests/test_nano_golden.py -v
  or: python3 pypkg/tests/test_nano_golden.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pypkg"))

from sanotts.nano import parse_meta_header, synthesize_ids  # noqa: E402
from sanotts.nano_rng import seeded_noise, uniform_stream  # noqa: E402

GATE = 0.98

# Every nano lineage that ships, and the weight format each uses. e12nano is
# the original int8 export; e13b is what the first heart-nano package (heartnano/, sanotts <= 0.5.5) contains; and
# r227f32 is the 2.27M heart voice, whose float rows the int8 path cannot
# read. All three go through the same code, so all three are gated.
LINEAGES = (
    ("en_us_e12nano", "q8"),
    ("en_us_e13b", "q8"),
    ("en_us_r227f32", "f32"),
)
FIXTURE = ROOT / "mcu/test/fixtures/en_us_e12nano"
META = ROOT / "mcu/models/en_us_e12nano/nano_q8_meta.h"


def _rows():
    return [line.split() for line in (FIXTURE / "rows.txt").read_text().splitlines() if line.strip()]


def test_uniform_stream_is_bit_exact():
    """The integer half of the generator has no tolerance: it either
    reproduces torch.rand's float32 stream exactly or the port is wrong."""
    seed = int(_rows()[0][4])
    ref = np.fromfile(FIXTURE / "e2e_uniform.bin", dtype=np.float32)
    assert np.array_equal(uniform_stream(seed, ref.size), ref)


def test_seeded_noise_within_libm_drift():
    """log/cos/sin are not bit-identical across libm builds, so the Gaussian
    half gets a tolerance. The C runtime measures 1.91e-06 on this data."""
    seed = int(_rows()[0][4])
    ref = np.fromfile(FIXTURE / "e2e_noise.bin", dtype=np.float32)
    got = seeded_noise(seed, 4, ref.size // 4).reshape(-1)
    assert np.abs(got - ref).max() < 1e-5


def _gate_lineage(lineage: str, ext: str) -> float:
    fixture = ROOT / "mcu/test/fixtures" / lineage
    meta = parse_meta_header(ROOT / "mcu/models" / lineage / "nano_q8_meta.h")
    front = (fixture / f"front_{ext}.bin").read_bytes()
    dec = (fixture / f"model_{ext}.bin").read_bytes()
    rows = [ln.split() for ln in (fixture / "rows.txt").read_text().splitlines() if ln.strip()]
    worst, worst_row = 1.0, ""
    for i, row in enumerate(rows):
        ids = np.fromfile(fixture / f"r{i:02d}_ids.bin", dtype=np.int32)
        # The fixture's frozen durations, so the comparison isolates the
        # graph. Letting the duration model predict would change the timing
        # and make a correlation against this reference meaningless.
        durs = np.fromfile(fixture / f"r{i:02d}_durs.bin", dtype=np.int32)
        ref = np.fromfile(fixture / f"r{i:02d}_audio.bin", dtype=np.float32)
        wav = synthesize_ids(front, dec, meta, ids, seed=int(row[4]), durations=durs)
        assert wav.size == ref.size, f"{row[0]}: {wav.size} samples, expected {ref.size}"
        corr = float(np.corrcoef(wav.astype(np.float64), ref.astype(np.float64))[0, 1])
        if corr < worst:
            worst, worst_row = corr, row[0]
    print(f"  {lineage:16s} DIM {meta['NANO_DIM']:3d}  MIN corr {worst:.6f} ({worst_row})")
    return worst


def test_waveform_matches_pytorch_reference():
    for lineage, ext in LINEAGES:
        worst = _gate_lineage(lineage, ext)
        assert worst > GATE, f"{lineage}: corr {worst:.6f} below {GATE}"


if __name__ == "__main__":
    test_uniform_stream_is_bit_exact()
    print("uniform : BIT-EXACT vs torch.rand")
    test_seeded_noise_within_libm_drift()
    print("noise   : within 1e-5 of the reference")
    test_waveform_matches_pytorch_reference()
    print("PASS")
