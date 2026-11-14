"""The 0.5.2 OOV decoder (cached keys/values, batched prefetch, memo) and
NanoG2P's two-pass driver agree with the original decoder word for word.

The original full-prefix decoder is kept as OOVPhonemizer.greedy_reference.
Run:  python3 -m pytest pypkg/tests/test_nano_g2p_oov_fast.py -v
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pypkg"))

from sanotts import nano_g2p, nano_g2p_oov as oov  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return oov.OOVPhonemizer()


@pytest.fixture(scope="module")
def words():
    found = set()
    for name in ("us_gold.json", "us_silver.json"):
        with open(oov.DATA_DIR / name, encoding="utf-8") as fh:
            found.update(k for k in json.load(fh) if re.fullmatch(r"[A-Za-z'\-.]+", k))
    ordered = sorted(found)
    rng = np.random.default_rng(0)
    return [ordered[i] for i in rng.choice(len(ordered), 600, replace=False)]


def test_layer_norm_and_gelu_bit_identical():
    rng = np.random.default_rng(1)
    for shape in ((1, 128), (7, 128), (3, 1024), (2, 5, 128)):
        x = (rng.standard_normal(shape) * rng.uniform(0.01, 30)).astype(np.float32)
        w = rng.standard_normal(shape[-1]).astype(np.float32)
        b = rng.standard_normal(shape[-1]).astype(np.float32)
        assert oov._layer_norm(x, w, b, 1e-5).tobytes() == oov._layer_norm_reference(x, w, b, 1e-5).tobytes()
        assert oov._gelu_fast(x).tobytes() == oov._gelu_exact(x).tobytes()


def test_cached_decoder_matches_reference(model, words):
    model._memo.clear()
    for w in words[:300]:
        assert model(w) == model.greedy_reference(w), w


def test_batched_prefetch_matches_reference(model, words):
    rng = np.random.default_rng(2)
    i = 0
    while i < len(words):
        group = words[i : i + int(rng.integers(2, 10))]
        i += len(group)
        model._memo.clear()
        model.prefetch(group + group[:1])          # a duplicate must not matter
        for w in group:
            assert model._memo[w] == model.greedy_reference(w), (w, group)


def test_prefetch_leaves_too_long_words_to_the_normal_path(model):
    long_word = "a" * (model.max_positions + 5)
    model._memo.clear()
    model.prefetch([long_word, "zzvo"])
    assert not model.memoized(long_word) and model.memoized("zzvo")
    with pytest.raises(oov.OOVModelError):
        model(long_word)


def test_memo_is_bounded(model, monkeypatch):
    monkeypatch.setattr(oov, "MEMO_SIZE", 3)
    model._memo.clear()
    for w in ("zzva", "zzvb", "zzvc", "zzvd"):
        model(w)
    assert len(model._memo) == 3 and not model.memoized("zzva")


TEXTS = (
    "Rede Globo's website and a Globo-owned pay-per-view channel offer round-the-clock coverage.",
    "Mstislav Rostropovich met Philippe at the Waitemata harbour in Tamaki.",
    "The zzvo and an ozzaq went to Kathmandu; a ublorf followed.",
    "Nothing unknown here at all.",
)


def test_two_pass_g2p_equals_one_pass_with_the_real_model():
    g2p = nano_g2p.NanoG2P()
    model = g2p.fallback.model()
    calls = []
    real_prefetch = model.prefetch
    model.prefetch = lambda ws: (calls.append(list(ws)), real_prefetch(ws))[1]
    try:
        for text in TEXTS:
            model._memo.clear()
            reference = g2p._convert(text, g2p.fallback)[0]     # one pass, words decoded one at a time
            model._memo.clear()
            assert g2p(text)[0] == reference, text
    finally:
        del model.prefetch
    assert calls and not any("Nothing" in " ".join(c) for c in calls)
