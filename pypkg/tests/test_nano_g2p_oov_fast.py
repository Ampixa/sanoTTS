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
    return oov.OOVPhonemizer(table_path=None)  # decode every word: these tests check the decoder


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
        assert oov._softmax(x).tobytes() == oov._softmax_reference(x).tobytes()


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


def test_g2p_is_one_pass_with_the_real_model():
    """NanoG2P.__call__ converts once, decoding unknown words as it reaches
    them (the record-then-prefetch second pass is gone): its result is the
    plain one-pass conversion, unknown words end up memoised, and a text with
    no unknown word never touches the model."""
    g2p = nano_g2p.NanoG2P()
    model = g2p.fallback.model()
    decoded = []
    real_call = type(model).__call__
    for text in TEXTS:
        model._memo.clear()
        reference = g2p._convert(text, g2p.fallback)[0]
        model._memo.clear()
        before = set(model._memo)
        assert g2p(text)[0] == reference, text
        decoded.append(set(model._memo) - before)
    assert decoded[0] and decoded[2], decoded   # text 1's names are answered by the word table
    assert not decoded[3], decoded[3]
    assert real_call is type(model).__call__
