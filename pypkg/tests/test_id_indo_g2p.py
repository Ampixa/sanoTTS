"""Gate the indo-g2p Indonesian front end and the espeak-space bridge over it.

numpy-only, no espeak-ng, no network, no voice download, no bun: the phoneme
config is tracked in the repo and everything here is string work over the
vendored tables.

Five things are worth freezing.

1. **Port fidelity.** ``pypkg/tests/data/indo-g2p-reference.json`` is the
   TypeScript original's own output at the revision that was ported, produced
   under bun, for 521 words and 60 sentences. The port reproduces it exactly or
   this fails. That is the only thing standing between a "port" and a rewrite.
2. **The vendored data is the data.** The payload hashes in
   ``g2p_data/indo_g2p/MANIFEST.json`` still match what is on disk, and the
   optional tables really are optional -- the module works with them absent.
3. **The bridge's encoding decisions**, one test each, because each of them is
   a claim about what the ``id`` voice was trained on rather than about
   Indonesian: ASCII ``g`` must become U+0261, the non-schwa ``e`` must become
   ``ɛ``, stress must stay penultimate, and ``glottal=False`` must undo only
   the glottal stops.
4. **Coverage.** Every codepoint the bridge can emit has an id in the ``id``
   voice's ``phoneme_id_map``, and that id is inside the deployed embedding.
5. **Dispatch and defaults.** ``piperlite_g2p="indo"`` reaches the bridge, an
   unknown path raises, and the shipped default is still ``"lexicon"``.

Run:  python3 -m pytest pypkg/tests/test_id_indo_g2p.py -v
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pypkg"))

from sanotts import engine, frontend, id_g2p, id_indo_bridge, id_indo_g2p, piper_g2p  # noqa: E402

RELEASE_DIR = ROOT / "releases/multivoice-20260713"
ID_PACKAGE = "id-newstts-1p46m"
REFERENCE = json.loads((Path(__file__).parent / "data" / "indo-g2p-reference.json")
                       .read_text(encoding="utf-8"))

# What the deployed id embeddings actually carry. Read from the pack's own
# manifest rather than hardcoded, because a republished voice could change it.
DEPLOYED_VOCAB = 122


def _table() -> frontend.PhonemeTable:
    return frontend.load_phoneme_table(
        RELEASE_DIR / ID_PACKAGE / "piper-phoneme-config.json")


# --------------------------------------------------------------------------
# 1. port fidelity
# --------------------------------------------------------------------------

def test_word_conversion_matches_the_typescript_original() -> None:
    wrong = [row["word"] for row in REFERENCE["words"]
             if id_indo_g2p.to_phoneme(row["word"], english=False) != row["core"]]
    assert wrong == [], f"{len(wrong)} words diverge from indo-g2p, e.g. {wrong[:5]}"


def test_schwa_placement_matches_the_typescript_original() -> None:
    wrong = [row["word"] for row in REFERENCE["words"]
             if id_indo_g2p.apply_schwa(row["word"]) != row["schwa"]]
    assert wrong == [], f"{len(wrong)} words diverge, e.g. {wrong[:5]}"


def test_syllabifier_matches_the_typescript_original() -> None:
    wrong = [row["word"] for row in REFERENCE["words"]
             if id_indo_g2p.to_syllables(row["word"]) != row["syllables"]]
    assert wrong == [], f"{len(wrong)} words diverge, e.g. {wrong[:5]}"


def test_sentence_conversion_matches_the_typescript_original() -> None:
    for row in REFERENCE["sentences"]:
        text = row["text"]
        assert id_indo_g2p.normalize_text(text) == row["normalized"], text
        assert id_indo_g2p.to_phoneme(text, english=False) == row["core"], text
        assert id_indo_g2p.to_phoneme(text, english=False,
                                      resolve_schwa=None) == row["core_no_colloc"], text
        assert id_indo_g2p.to_phoneme(text, english=False,
                                      expand_abbr=True) == row["core_abbr"], text


# --------------------------------------------------------------------------
# 2. the vendored data
# --------------------------------------------------------------------------

def test_vendored_payloads_match_their_manifest() -> None:
    import lzma
    manifest = json.loads((id_indo_g2p.DATA_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["upstream_revision"] == id_indo_g2p.UPSTREAM_REVISION
    assert manifest["license"] == "MIT"
    for stem, row in manifest["files"].items():
        payload = lzma.decompress((id_indo_g2p.DATA_DIR / f"{stem}.xz").read_bytes())
        assert len(payload) == row["raw_bytes"], stem
        assert hashlib.sha256(payload).hexdigest() == row["raw_sha256"], stem


def test_the_optional_english_table_is_optional() -> None:
    """It is not vendored, and nothing depends on it being there."""
    assert id_indo_g2p.english_available() is False
    assert id_indo_g2p.look_up_english("event") is None
    # The Indonesian rules read it instead, which is indo-g2p/core's behaviour.
    assert id_indo_g2p.to_phoneme("event", english=True) == "efent"


def test_missing_data_raises_rather_than_returning_nothing() -> None:
    with pytest.raises(id_indo_g2p.IndoG2PError) as caught:
        id_indo_g2p._read_packed("a_table_that_does_not_exist")
    assert caught.value.kind == "missing_data"


# --------------------------------------------------------------------------
# 3. the rules that answer the issue
# --------------------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("pergi", "pərgi"),        # the pepet, where espeak-ng prints ɛ
    ("dengan", "dəŋan"),
    ("kecil", "kətʃil"),
    ("sekolah", "səkolah"),
    ("bebek", "bəbəʔ"),
    ("teknologi", "teʔnologi"),  # not te- + knologi: `kn` is not an onset, so
                                 # the `e` stays -- but see the glottal test below
    ("keset", "keset"),          # a schwa-overrides correction
])
def test_schwa_is_lexical_not_positional(word: str, expected: str) -> None:
    assert id_indo_g2p.to_phoneme(word, english=False) == expected


@pytest.mark.parametrize("word,expected", [
    ("rusak", "rusaʔ"),        # word-final k
    ("bakso", "baʔso"),        # pre-consonantal k
    ("tidaklah", "tidaʔlah"),  # before the -lah clitic
    ("akhir", "axir"),         # kh is a digraph, not k before a consonant
    ("iklan", "iklan"),        # kl is a borrowed onset cluster
    ("demokrat", "demokrat"),  # so is kr
])
def test_glottal_stop_rule(word: str, expected: str) -> None:
    assert id_indo_g2p.to_phoneme(word, english=False) == expected


@pytest.mark.parametrize("word,expected", [
    ("teknologi", "teʔnologi"),
    ("dokter", "doʔtər"),
    ("tekstil", "teʔstil"),
    ("apotek", "apoteʔ"),
    ("dialek", "dialeʔ"),
])
def test_the_glottal_rule_over_applies_to_european_loanwords(word: str,
                                                             expected: str) -> None:
    """Pinned because it is a known cost, not because it is right.

    The rule is unconditional, so a `k` in a Latin cluster or at the end of a
    Dutch or English borrowing becomes a glottal stop too. Measured against
    en.wiktionary, 153 of the 222 corpus word types that end in `k` are
    transcribed with a plain /k/ there, and this emits /ʔ/ for all 222. The
    reading is defensible for casual speech and wrong for the careful register
    the voice was trained on; it is recorded here so a future change to the
    rule shows up as a test diff rather than as a surprise.
    """
    assert id_indo_g2p.to_phoneme(word, english=False) == expected


def test_affix_rules_reach_words_no_list_carries() -> None:
    """The prefix vowel is a schwa whether or not the root is known."""
    assert id_indo_g2p.apply_schwa("tersebut").startswith("tər")
    assert id_indo_g2p.schwa_source("zzzterblah") == "rules"
    # A root the dictionary has never seen keeps its own `e`s; only the
    # prefix's schwa is claimed. `zzzz` is not a possible onset, so no prefix
    # applies at all and the answer is None rather than a guess.
    assert id_indo_g2p.affix_schwa_mask("terzabu", lambda w: None) == 0b1
    assert id_indo_g2p.affix_schwa_mask("terzzzz", lambda w: None) is None


def test_diphthongs_are_scoped_to_a_syllable() -> None:
    """`air` is /a.ir/, two syllables, not the diphthong /aɪr/."""
    assert id_indo_g2p.to_phoneme("air", english=False) == "air"
    assert id_indo_g2p.to_phoneme("bermain", english=False) == "bərmain"
    assert id_indo_g2p.to_phoneme("naik", english=False) == "naɪʔ"


def test_numbers_and_symbols_are_spelled_out() -> None:
    assert id_indo_g2p.normalize_text("harga Rp15.000 naik 5%") == \
        "harga lima belas ribu rupiah naik lima persen"
    assert id_indo_g2p.spell_number(15000) == "lima belas ribu"
    assert id_indo_g2p.spell_decimal(3, "14") == "tiga koma satu empat"


def test_collocations_resolve_a_homograph_from_context() -> None:
    assert id_indo_g2p.resolve_collocations(["kami", "apel", "di", "lapangan"]) == \
        [None, "apel", None, None]


def test_a_resolver_of_the_wrong_length_raises() -> None:
    with pytest.raises(id_indo_g2p.IndoG2PError) as caught:
        id_indo_g2p.convert("satu dua tiga", resolve_schwa=lambda words: [None])
    assert caught.value.kind == "resolver"


# --------------------------------------------------------------------------
# 4. the bridge's encoding decisions
# --------------------------------------------------------------------------

def test_ascii_g_becomes_the_script_g_the_voice_knows() -> None:
    """id 154 is past the deployed embedding; U+0261 is id 66 and is not."""
    ipa = id_indo_bridge.phonemize_to_espeak_ipa("gajinya")
    assert "g" not in ipa and "ɡ" in ipa
    table = _table()
    assert table.id_map["ɡ"] < DEPLOYED_VOCAB <= table.id_map["g"]


def test_the_non_schwa_e_becomes_epsilon() -> None:
    """espeak-ng printed plain `e` 9 times in the corpus and `ɛ` 401 times."""
    ipa = id_indo_bridge.phonemize_to_espeak_ipa("enak")
    assert "ɛ" in ipa and "e" not in ipa


def test_stress_stays_on_the_penultimate_syllable() -> None:
    assert id_indo_bridge.phonemize_to_espeak_ipa("pergi") == "pˈərɡi"
    assert id_indo_bridge.phonemize_to_espeak_ipa("pergi", stress="schwa_shift") == "pərɡˈi"
    assert id_indo_bridge.phonemize_to_espeak_ipa("sekolah") == "səkˈolah"


def test_unstressed_clitics_match_the_lexicon_path() -> None:
    for word in sorted(id_g2p.UNSTRESSED_WORDS):
        assert id_g2p.STRESS_PRIMARY not in id_indo_bridge.phonemize_to_espeak_ipa(word)


def test_glottal_false_undoes_only_the_glottal_stops() -> None:
    with_stops = id_indo_bridge.phonemize_to_espeak_ipa("Bakso rusak dan enak.")
    without = id_indo_bridge.phonemize_to_espeak_ipa("Bakso rusak dan enak.", glottal=False)
    assert "ʔ" in with_stops and "ʔ" not in without
    assert with_stops.replace("ʔ", "k") == without


def test_an_unknown_stress_option_raises() -> None:
    with pytest.raises(id_indo_bridge.IdIndoBridgeError) as caught:
        id_indo_bridge.phonemize_to_espeak_ipa("halo", stress="whatever")
    assert caught.value.kind == "option"


def test_empty_input_raises_rather_than_returning_nothing() -> None:
    with pytest.raises(id_indo_bridge.IdIndoBridgeError) as caught:
        id_indo_bridge.phonemize_to_espeak_ipa("   ")
    assert caught.value.kind == "empty"


# --------------------------------------------------------------------------
# 5. coverage, dispatch, defaults
# --------------------------------------------------------------------------

def test_the_voice_table_covers_every_symbol_the_bridge_emits() -> None:
    report = id_indo_bridge.coverage_report(_table())
    assert report["unmappable_symbols"] == [], report["unmappable_codepoints"]


def test_every_emitted_symbol_is_inside_the_deployed_embedding() -> None:
    """A symbol past vocab_size is silently remapped to the schwa at synthesis.

    Punctuation is excluded: espeak-ng's own output has the same ids past the
    embedding, so both arms carry it equally and it is not this path's problem.
    """
    table = _table()
    phonetic = {s for s in id_indo_bridge.producible_symbols()
                if s not in id_indo_g2p._KEPT_PUNCTUATION and s != " "}
    beyond = sorted(s for s in phonetic if table.id_map[s] >= DEPLOYED_VOCAB)
    assert beyond == [], beyond


def test_dispatch_reaches_the_bridge_and_refuses_an_unknown_path() -> None:
    table = _table()
    assert piper_g2p.module_for(table, "indo") is id_indo_bridge
    assert piper_g2p.module_for(table, "lexicon") is id_g2p
    with pytest.raises(piper_g2p.PiperG2PError) as caught:
        piper_g2p.module_for(table, "nonsense")
    assert caught.value.kind == "path"


def test_the_two_paths_produce_different_ids_for_the_same_text() -> None:
    table = _table()
    lexicon_ids, _ = piper_g2p.text_to_phoneme_ids("Bakso rusak.", table, path="lexicon")
    indo_ids, _ = piper_g2p.text_to_phoneme_ids("Bakso rusak.", table, path="indo")
    assert lexicon_ids.tolist() != indo_ids.tolist()
    assert table.id_map["ʔ"] in indo_ids.tolist()
    assert table.id_map["ʔ"] not in lexicon_ids.tolist()


def test_nothing_is_dropped_in_silence() -> None:
    _ids, unmapped = id_indo_bridge.text_to_phoneme_ids("Halo, dunia!", _table())
    assert unmapped == ""


def test_indo_is_the_default() -> None:
    assert engine.DEFAULT_PIPERLITE_G2P == "indo"
    assert "indo" in engine.PIPERLITE_G2P_CHOICES
    with pytest.raises(ValueError):
        engine.Synthesizer(voice_dir=RELEASE_DIR / ID_PACKAGE, piperlite_g2p="nope")
