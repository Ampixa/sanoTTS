"""Gate the espeak-free nano front end against the misaki reference it replaces.

The reference file was produced by real misaki (kokoro 0.9.4 `KPipeline.g2p` +
`en_tokenize`) on k2 -- the same two calls that built every nano training pack --
by `tools/compare_nano_frontends.py reference`. Replaying it here means the gate
runs with numpy alone: no torch, no kokoro, no espeak-ng.

Two thresholds, both below what was measured when the front end landed
(82.1% exact, 0.30% token error rate), so ordinary noise does not trip them but
a regression does.

Run:  python3 -m pytest pypkg/tests/test_nano_g2p.py -v
  or: python3 pypkg/tests/test_nano_g2p.py
"""

from __future__ import annotations

import pytest

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pypkg"))

from sanotts import nano_g2p  # noqa: E402
from sanotts.nano_g2p_oov import shared_phonemizer  # noqa: E402

# Tracked here rather than under artifacts/, which is gitignored, so a fresh
# clone can run the gate. It is the same reference, with the phoneme strings
# dropped: only row_id, text and ids are scored. The full comparison, including
# the espeak baseline and the OOV ablation, lives in
# artifacts/nano-g2p-espeak-free-20260904/.
REFERENCE = ROOT / "pypkg/tests/data/nano-g2p-misaki-reference.jsonl"

# The shipped espeak path scores 1.8% / 6.09% on this set; these are the floors
# the dictionary path has to stay above, not targets.
MIN_EXACT_MATCH = 0.80
MAX_TOKEN_ERROR_RATE = 0.005

# Frozen from the transformers run that validated the numpy port: the same 430
# probe words came back identical from `BartForConditionalGeneration.generate`
# and from `nano_g2p_oov`. These six are the regression tripwire.
OOV_EXPECTED = {
    "tokenization": "tˌOkənəzˈAʃən",
    "Euroclydon": "jˌɜɹOklˈIdᵊn",
    "grokking": "ɡɹˈɑkɪŋ",
    "pemberley": "pˈɛmbəɹli",
    "netherfield": "nˈɛðəɹfˌild",
    "cryptocurrency": "kɹˌɪptəkˈɜɹənsi",
}


def _load_reference() -> list[dict]:
    if not REFERENCE.is_file():
        raise FileNotFoundError(
            f"reference not found at {REFERENCE}; regenerate it with "
            "tools/compare_nano_frontends.py reference in the kokoro venv on k2"
        )
    rows = [json.loads(line) for line in REFERENCE.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) < 300:
        raise ValueError(f"{REFERENCE}: {len(rows)} rows, the gate wants at least 300")
    return rows


def _edit_distance(a: list[int], b: list[int]) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def test_oov_model_is_stable() -> None:
    # OOV_EXPECTED is the 751k model's (the word table's source), decoded without the table.
    from sanotts import nano_g2p_oov  # noqa: PLC0415
    model = nano_g2p_oov.OOVPhonemizer(nano_g2p_oov.TABLE_MODEL_PATH, table_path=None)
    for word, expected in OOV_EXPECTED.items():
        got = model(word)
        assert got == expected, f"{word!r}: expected {expected!r}, got {got!r}"


# The shipped stack (table, then the 96k model): four words come from the table, two are decoded.
SHIPPED_EXPECTED = dict(OOV_EXPECTED, Euroclydon="jˌʊɹOklˈIdᵊn")


def test_shipped_oov_path_is_stable() -> None:
    model = shared_phonemizer()
    for word, expected in SHIPPED_EXPECTED.items():
        got = model(word)
        assert got == expected, f"{word!r}: expected {expected!r}, got {got!r}"


def test_dictionary_path_beats_the_espeak_baseline() -> None:
    rows = _load_reference()
    engine = nano_g2p.shared_g2p()
    exact = 0
    distance = 0
    total = 0
    for row in rows:
        ids, _dropped = nano_g2p.phonemize(row["text"], max_tokens=100000, g2p=engine)
        if ids == row["ids"]:
            exact += 1
        distance += _edit_distance(row["ids"], ids)
        total += len(row["ids"])
    exact_match = exact / len(rows)
    token_error_rate = distance / total
    print(f"exact {exact_match:.1%}  TER {token_error_rate:.2%}  rows {len(rows)}")
    assert exact_match >= MIN_EXACT_MATCH, (
        f"exact-sequence match fell to {exact_match:.1%}, floor is {MIN_EXACT_MATCH:.0%}"
    )
    assert token_error_rate <= MAX_TOKEN_ERROR_RATE, (
        f"token error rate rose to {token_error_rate:.2%}, "
        f"ceiling is {MAX_TOKEN_ERROR_RATE:.2%}"
    )


def test_no_gpl_dependency_is_imported() -> None:
    """The whole point: phonemizer must not be reachable from this path."""
    for module in ("phonemizer", "espeakng_loader", "spacy", "num2words", "torch"):
        assert module not in sys.modules, f"{module} was imported by the espeak-free path"


if __name__ == "__main__":
    test_oov_model_is_stable()
    test_dictionary_path_beats_the_espeak_baseline()
    test_no_gpl_dependency_is_imported()
    print("nano_g2p gate: OK")


# Digit groups the misaki-derived tokenizer used to drop silently ("at 3:30" was read "at :"), found by listening
# to out-of-domain text on 2026-11-26. Each case must read the same as its spelled-out form.
@pytest.mark.parametrize("raw,spelled", [
    ("at 3:30 PM", "at three thirty p.m."),
    ("at 3:30pm", "at three thirty p.m."),
    ("at 10:05 AM", "at ten oh five a.m."),
    ("at 3:00", "at three o'clock"),
    ("at 12:00 p.m.", "at twelve p.m."),
    ("at 10 AM", "at ten a.m."),
    ("the score was 2:1", "the score was two to one"),
    ("it took 2:30:15", "it took two thirty fifteen"),
    ("a 100km walk", "a one hundred km walk"),
    ("an mp3 file", "an mp three file"),
])
def test_digit_groups_are_read(raw, spelled):
    from sanotts import nano_g2p
    assert nano_g2p.phonemize_to_string(raw) == nano_g2p.phonemize_to_string(spelled)


@pytest.mark.parametrize("text", ["the 2nd floor", "in the 1990s", "a 10x speedup", "a 4K screen", "COVID-19",
                                  "an A4 page", "EJ255", "U2", "B-52", "I am 10 years old"])
def test_number_forms_that_already_worked_are_unchanged(text):
    from sanotts import nano_g2p
    assert nano_g2p.normalize_times(text) == text
