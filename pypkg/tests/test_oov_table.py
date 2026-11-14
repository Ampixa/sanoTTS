"""The precomputed OOV word table (g2p_data/oov_table_en_us.npz) is the model's
own output, is refused for another model, and answers without decoding.

Run:  python3 -m pytest pypkg/tests/test_oov_table.py -v
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pypkg"))

from sanotts import nano_g2p_oov as oov  # noqa: E402


@pytest.fixture(scope="module")
def decoder():
    return oov.OOVPhonemizer(table_path=None)


@pytest.fixture(scope="module")
def table(decoder):
    return oov._WordTable(oov.TABLE_PATH, decoder.model_sha256)


def _rows(table, idx):
    return table.items(idx)


def test_table_is_the_models_output(decoder, table):
    """A random sample of rows, decoded again here. The table was built on
    x86-64 (OpenBLAS); another BLAS rounds differently and could flip a
    near-tie, which this would show as a mismatch."""
    rng = np.random.default_rng(0)
    idx = rng.choice(table.size, 2000, replace=False)
    bad = [(k, v, decoder(k)) for k, v in _rows(table, idx) if decoder(k) != v]
    assert not bad, bad[:10]


def test_every_key_is_found(table):
    rng = np.random.default_rng(1)
    for key, value in _rows(table, rng.choice(table.size, 5000, replace=False)):
        assert table.get(key) == value


def test_words_strictly_sorted(table):
    keys = table.keys
    assert all(keys[i] < keys[i + 1] for i in range(table.size - 1))


def test_misses(table):
    for word in ("zzvoqx", "Zzvoqx", "", "the-the-the-zzq"):
        assert table.get(word) is None


def test_hits_do_not_decode(table, monkeypatch):
    model = oov.OOVPhonemizer()
    key, value = _rows(table, [0])[0]

    def boom(*_a, **_k):
        raise AssertionError("decoded a tabled word")

    monkeypatch.setattr(model, "_greedy", boom)
    assert model.memoized(key)
    assert model(key) == value
    model.prefetch([key])            # nothing left to decode


def test_pack_roundtrip_and_model_check(tmp_path):
    entries = {"Globo": "ɡlˈOboʊ", "globo": "ɡlˈOboʊ", "zeta": "zˈeɪtə", "naïve": "nɑˈiv"}
    path = tmp_path / "t.npz.xz"
    oov.write_table(path, entries, "a" * 64, {"source": "test"})
    t = oov._WordTable(path, "a" * 64)
    assert t.size == len(entries)
    assert all(t.get(k) == v for k, v in entries.items())
    with pytest.raises(oov.OOVModelError, match="built for model"):
        oov._WordTable(path, "b" * 64)


def test_prefixes_and_neighbours_miss(tmp_path):
    entries = {"alpha": "1", "alphabet": "2", "Alpha": "3", "beta": "4"}
    path = tmp_path / "c.npz.xz"
    oov.write_table(path, entries, "a" * 64, {})
    t = oov._WordTable(path, "a" * 64)
    assert [t.get(k) for k in entries] == ["1", "2", "3", "4"]
    for miss in ("alph", "alphab", "ALPHA", "a", "zeta", "Aaa"):
        assert t.get(miss) is None, miss


def test_missing_table_raises(tmp_path):
    model = oov.OOVPhonemizer(table_path=tmp_path / "absent.npz.xz")
    with pytest.raises(oov.OOVModelError, match="word table not found"):
        model("zzvo")


def test_corrupt_table_raises(tmp_path):
    path = tmp_path / "bad.npz.xz"
    path.write_bytes(b"not xz")
    with pytest.raises(oov.OOVModelError, match="could not read"):
        oov._WordTable(path, "a" * 64)
