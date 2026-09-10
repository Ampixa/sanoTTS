"""Gate the espeak-free piperlite mapping: full table coverage, no drift.

Two things are worth freezing here, and neither needs espeak-ng, torch or a
voice download -- the phoneme configs are tracked in the repo and the mapping is
pure string work:

1. **Coverage.** Every symbol the lexicon front end can emit reaches a symbol
   every voice's ``phoneme_id_map`` has an id for, or is one of the two the
   mapping deliberately drops because espeak-ng drops them too. If a voice is
   ever republished with a smaller table, or ``nano_g2p`` grows a symbol, this
   fails instead of silently skipping phonemes at synthesis time.
2. **The mapping itself.** A handful of cases that each pin one decision:
   the length-restoration rules, the ``i``/``iː`` context rule, the ``ɚ`` and
   ``ɜː`` majority rules, and the punctuation that espeak keeps versus drops.

Run:  python3 -m pytest pypkg/tests/test_piper_g2p.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pypkg"))

from sanotts import frontend, piper_g2p  # noqa: E402

RELEASE_DIR = ROOT / "releases/multivoice-20260713"

# Every published piperlite package, not just the five aliased voices: the
# 1p1m/1p8m amy variants share a table with amy but are separate files.
VOICE_PACKAGES = (
    "amy-en-1p1m",
    "amy-en-1p46m",
    "amy-en-1p8m",
    "hfc-en-1p8m",
    "kristin-en-1p4m",
    "id-newstts-1p46m",
    "vi-vais1000-1p46m",
)

# espeak-ng emits nothing at all for these (verified on probe sentences: an em
# dash, an en dash and a horizontal ellipsis all vanish from its output), so
# emitting nothing is what matches the baseline.
EXPECTED_DROPPED = ["—", "…"]


def _table(package: str) -> frontend.PhonemeTable:
    return frontend.load_phoneme_table(RELEASE_DIR / package / "piper-phoneme-config.json")


def test_every_voice_table_covers_the_whole_mapping() -> None:
    for package in VOICE_PACKAGES:
        report = piper_g2p.coverage_report(_table(package))
        assert report["unmappable_espeak_symbols"] == [], (
            f"{package}: mapping can emit symbols this voice has no id for: "
            f"{report['unmappable_espeak_symbols']}"
        )
        assert report["unmappable_lexicon_symbols"] == {}, (
            f"{package}: lexicon symbols with no landing place: "
            f"{report['unmappable_lexicon_symbols']}"
        )
        assert report["dropped_by_design"] == EXPECTED_DROPPED, (
            f"{package}: dropped set changed: {report['dropped_by_design']}"
        )


def test_mapping_cases() -> None:
    cases = {
        # diphthongs and affricates: one misaki codepoint, two espeak ones
        "mˈA": "mˈeɪ",
        "mˈI": "mˈaɪ",
        "mˈO": "mˈoʊ",
        "nˈW": "nˈaʊ",
        "bˈY": "bˈɔɪ",
        "ʤˈʌʤ": "dʒˈʌdʒ",
        "ʧˈɜɹʧ": "tʃˈɜːtʃ",
        # length restored where espeak-ng never prints the vowel short
        "bˈut": "bˈuːt",
        "kˈɑt": "kˈɑːt",
        "θˈɔt": "θˈɔːt",
        # ɜɹ -> ɜː and əɹ -> ɚ, both majority rules
        "bˈɜɹd": "bˈɜːd",
        "mˈɪstəɹ": "mˈɪstɚ",
        # the i/iː context rule
        "sˈi": "sˈiː",                 # stressed monosyllable
        "hˈæpi": "hˈæpi",              # the -y vowel stays short
        "mˈɪɹli": "mˈɪɹli",
        "təbi": "təbi",                # unstressed function word, two vowels
        "bi": "biː",                   # unstressed, i is the only vowel
        "ˈikwəli": "ˈiːkwəli",
        # the pre-vocalic guard: ɹ is the next syllable's onset here, so the
        # schwa stays a plain schwa instead of collapsing into ɚ
        "pˈiəɹɪəd": "pˈiəɹɪəd",
        "ˈɔθəɹɹˌIzd": "ˈɔːθɚɹˌaɪzd",
        # the flap and the syllabic marker
        "bˈɛTəɹ": "bˈɛɾɚ",
        "bˈʌtᵊn": "bˈʌtən",
        # punctuation espeak keeps, rewritten to its spelling
        '“hI,”': '"haɪ,"',
        # punctuation espeak drops, without leaving a doubled space behind
        "hˈI — ðˈɛɹ": "hˈaɪ ðˈɛɹ",
    }
    for misaki, expected in cases.items():
        assert piper_g2p.misaki_to_espeak_ipa(misaki) == expected, (
            f"{misaki!r} -> {piper_g2p.misaki_to_espeak_ipa(misaki)!r}, expected {expected!r}"
        )


def test_text_to_ids_runs_without_espeak() -> None:
    table = _table("amy-en-1p46m")
    ids, unmapped = piper_g2p.text_to_phoneme_ids(
        "The quick brown fox jumps over the lazy dog.", table)
    assert unmapped == ""
    assert ids[0] == frontend.BOS_ID and ids[-1] == frontend.EOS_ID
    assert len(ids) > 3


if __name__ == "__main__":
    test_every_voice_table_covers_the_whole_mapping()
    test_mapping_cases()
    test_text_to_ids_runs_without_espeak()
    print("ok")
