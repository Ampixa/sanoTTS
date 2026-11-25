"""Local-feature voices (heart-nano since 0.5.6, heartnanos/): streaming before the text is converted.

Checked here:

* frame_major_noise gives the same values for a frame range whatever range
  it is drawn in.
* With local features a token's duration depends only on nearby tokens: the
  durations of a prefix's trusted tokens equal the whole text's.
* nano.stream_local concatenates to the whole-utterance pass (durations,
  acoustic plan, frame-major noise, decoder, iSTFT, DC block), with no prefix
  and with prefixes of several lengths, for several chunk schedules.
* stream_local refuses a voice without NANO_LOCAL_FEATURES.
* StreamingPhonemes' prefix ids are a prefix of its full ids, and full()
  equals phonemize's ids.

The weights are the first heart-nano (en_us_e13b) fixture with the local flag set.
That network was not trained with local features, so its audio is not
meaningful, but every property tested is structural (receptive fields, noise
layout, stream bookkeeping) and holds for any weights.

Run:  python3 -m pytest pypkg/tests/test_nano_local.py -v
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pypkg"))

from sanotts import nano, nano_g2p  # noqa: E402

# Chunked passes differ from the whole pass by BLAS association only (the
# unchunked schedule matches bit for bit). Measured on OpenBLAS (i5-10400):
# up to 1.4e-5 here, and 8e-6 for the global-feature stream_ids on the same
# box; a structural error (wrong halo, noise or duration) is >1e-3.
STREAM_TOL = 5e-5
SCHEDULES = ((1, 1), (4, 128), (3, 5), (10_000, 10_000))
SEED = 11


def _local_lineage():
    fixture = ROOT / "mcu/test/fixtures/en_us_e13b"
    meta = dict(nano.parse_meta_header(ROOT / "mcu/models/en_us_e13b/nano_q8_meta.h"))
    meta["NANO_LOCAL_FEATURES"] = 1
    front = (fixture / "front_q8.bin").read_bytes()
    dec = (fixture / "model_q8.bin").read_bytes()
    return fixture, meta, nano.NanoWeights(front, dec, meta)


def _whole_pass(w: nano.NanoWeights, ids: np.ndarray) -> np.ndarray:
    m = w.m
    durations = nano.duration_forward(w, ids).astype(np.int64)
    plan = nano.acoustic_plan(w, ids, durations)
    mel = nano.acoustic_frames(w, plan, 0, plan.frames)
    noise = nano.frame_major_noise(SEED, m["NANO_NOISE_CH"], 0, plan.frames)
    spec = nano.decoder_forward(w, mel, noise)
    return nano.dc_block(nano.istft(spec, m["NANO_N_FFT"], m["NANO_HOP"]))


def test_frame_major_noise_is_range_independent():
    whole = nano.frame_major_noise(SEED, 8, 0, 300)
    for lo, hi in ((0, 1), (5, 9), (100, 300), (299, 300)):
        assert np.array_equal(nano.frame_major_noise(SEED, 8, lo, hi), whole[:, lo:hi]), (lo, hi)
    with pytest.raises(ValueError):
        nano.frame_major_noise(SEED, 8, 5, 4)


def test_prefix_durations_are_final():
    fixture, meta, w = _local_lineage()
    halo = nano.duration_halo_tokens(meta)
    for row in range(3):
        ids = np.fromfile(fixture / f"r{row:02d}_ids.bin", dtype=np.int32).astype(np.int64)
        full = nano.duration_forward(w, ids)
        for k in (halo + 2, halo + 9, ids.size - 1):
            if k <= halo or k > ids.size:
                continue
            part = nano.duration_forward(w, ids[:k])
            assert np.array_equal(part[: k - halo], full[: k - halo]), (row, k)


def test_stream_local_matches_whole_pass():
    fixture, meta, w = _local_lineage()
    for row in range(3):
        ids = np.fromfile(fixture / f"r{row:02d}_ids.bin", dtype=np.int32).astype(np.int64)
        ref = _whole_pass(w, ids)
        for k in (None, 8, 20, 40):
            prefix = None if k is None else ids[:k].tolist()
            for first, chunk in SCHEDULES:
                pieces = list(nano.stream_local(w, prefix, lambda: ids, seed=SEED,
                                                first_chunk_frames=first, chunk_frames=chunk))
                assert pieces and all(p.size for p in pieces), (row, k, first, chunk)
                got = np.concatenate(pieces)
                assert got.shape == ref.shape, (row, k, first, chunk, got.shape, ref.shape)
                assert np.abs(got - ref).max() <= STREAM_TOL, (row, k, first, chunk)


def test_stream_local_starts_before_the_full_ids():
    fixture, meta, w = _local_lineage()
    ids = np.fromfile(fixture / "r00_ids.bin", dtype=np.int32).astype(np.int64)
    calls = []

    def full_ids():
        calls.append(1)
        return ids

    stream = nano.stream_local(w, ids[:40].tolist(), full_ids, seed=SEED)
    first = next(stream)
    assert first.size and not calls, "a long enough prefix must yield audio before full_ids() runs"
    list(stream)
    assert calls == [1]


def test_stream_local_rejects_global_voice():
    fixture = ROOT / "mcu/test/fixtures/en_us_e13b"
    meta = nano.parse_meta_header(ROOT / "mcu/models/en_us_e13b/nano_q8_meta.h")
    w = nano.NanoWeights((fixture / "front_q8.bin").read_bytes(), (fixture / "model_q8.bin").read_bytes(), meta)
    assert not nano.is_local(meta)
    with pytest.raises(ValueError):
        list(nano.stream_local(w, None, lambda: [1, 2, 3], seed=SEED))


TEXTS = (
    "The quick brown fox jumps over the lazy dog, then naps.",
    "Dr. Okonkwo (born 1962) moved to Kathmandu in 1999 and wrote \"Rivers\".",
    "It costs $4.50, or about 3.2 percent more than last year's price.",
    "Szczepanski and Nguyen met Thorvaldsdottir at the Wexford festival.",
    "Hi.",
)


@pytest.mark.parametrize("text", TEXTS)
def test_streaming_phonemes_prefix_is_exact(text):
    for items in (3, 6, 10):
        sp = nano_g2p.StreamingPhonemes(text, prefix_items=items, lookahead=2)
        ids, _dropped = sp.full()
        if sp.prefix_ids is not None:
            assert ids[: len(sp.prefix_ids)] == sp.prefix_ids, (text, items)
        assert ids == nano_g2p.phonemize(text)[0], (text, items)
