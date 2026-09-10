"""Gate the Indonesian and Vietnamese espeak-free front ends.

numpy-only, no espeak-ng, no network, no voice download: the two phoneme
configs are tracked in the repo and the front ends are pure string work.

Four things are worth freezing.

1. **Coverage.** Every symbol either module can emit has an id in its voice's
   ``phoneme_id_map``. If a voice is republished with a smaller table, or a
   rule grows a symbol, this fails instead of silently skipping phonemes at
   synthesis time.
2. **The rules themselves.** One case per decision that was hard to get right,
   each named after what it pins: Indonesian's positional ``e``, its penult
   stress and its prefix rule; Vietnamese's tone slot, its `ay`/`ai` length
   contrast, its palatal codas and its `ây`/`âu` irregulars.
3. **Dispatch.** ``piper_g2p.text_to_phoneme_ids`` routes each voice to the
   right language and refuses an unknown one rather than reading it as English.
4. **Nothing is dropped in silence.** A string the rules cannot read comes back
   in the second return value.

Run:  python3 -m pytest pypkg/tests/test_idvi_g2p.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pypkg"))

from sanotts import frontend, id_g2p, piper_g2p, vi_g2p  # noqa: E402

RELEASE_DIR = ROOT / "releases/multivoice-20260713"
ID_PACKAGE = "id-newstts-1p46m"
VI_PACKAGE = "vi-vais1000-1p46m"


def _table(package: str) -> frontend.PhonemeTable:
    return frontend.load_phoneme_table(RELEASE_DIR / package / "piper-phoneme-config.json")


# --------------------------------------------------------------------------
# 1. coverage
# --------------------------------------------------------------------------

def test_id_voice_table_covers_every_symbol_the_rules_emit() -> None:
    report = id_g2p.coverage_report(_table(ID_PACKAGE))
    assert report["espeak_voice"] == "id"
    assert report["unmappable_symbols"] == [], (
        f"id rules can emit symbols the voice has no id for: {report['unmappable_symbols']}"
    )


def test_vi_voice_table_covers_every_symbol_the_rules_emit() -> None:
    report = vi_g2p.coverage_report(_table(VI_PACKAGE))
    assert report["espeak_voice"] == "vi"
    assert report["unmappable_symbols"] == [], (
        f"vi rules can emit symbols the voice has no id for: {report['unmappable_symbols']}"
    )


def test_corpus_output_stays_inside_each_voice_table() -> None:
    """The static claim above, re-checked dynamically on real sentences."""
    cases = {
        ID_PACKAGE: (id_g2p, [
            "Selamat pagi, apa kabar hari ini?",
            "Aku menerjemahkan kalimat ini dua kali.",
            "Dia membeli 27 buku di toko itu.",
            '"Mau pergi ke bioskop tidak?" "Ya, ayo."',
        ]),
        VI_PACKAGE: (vi_g2p, [
            "Tôi rất vui được gặp bạn.",
            "Quay lại làm việc đi. Đồ lười!",
            "Chúng ta đã tấn công vào cơ sở dữ liệu.",
            "Tôi có 27 quyển sách.",
        ]),
    }
    for package, (module, texts) in cases.items():
        table = _table(package)
        for text in texts:
            ids, unmapped = module.text_to_phoneme_ids(text, table)
            assert unmapped == "", f"{package}: {text!r} produced unmapped {unmapped!r}"
            assert len(ids) > 3
            assert ids[0] == frontend.BOS_ID and ids[-1] == frontend.EOS_ID


# --------------------------------------------------------------------------
# 2. the rules
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    # penultimate primary stress, marked immediately before the nucleus
    ("aku", "ˈaku"),
    ("kalimat", "kalˈimat"),
    ("bahasa", "bahˈasa"),
    # `e` is ɛ under stress and ə everywhere else -- the same word shows both
    ("kereta", "kərˈɛta"),
    ("dompet", "dˈompət"),
    # the schwa prefixes do not host the secondary mark; a full-vowel first
    # syllable does
    ("menggunakan", "məŋɡunˈakan"),
    ("digunakan", "dˌiɡunˈakan"),
    ("menerjemahkan", "mənˌɛrdʒəmˈahkan"),
    # digraphs and the two falling diphthongs
    ("punya", "pˈuɲa"),
    ("khawatir", "xawˈatir"),
    ("tinggal", "tˈiŋɡal"),
    ("balai", "bˈalaɪ"),
    ("kalau", "kˈalaʊ"),
    # `o` lowers before a rhotic
    ("kantor", "kˈantɔr"),
    # obstruent-plus-liquid is an onset, s-plus-stop is not
    ("istri", "ˈistri"),
    ("diskusikan", "dˌiskusˈikan"),
    # the monosyllabic clitics carry no stress at all
    ("yang", "jaŋ"),
])
def test_id_word_cases(text: str, expected: str) -> None:
    assert id_g2p.phonemize_to_espeak_ipa(text)[0] == expected


@pytest.mark.parametrize("text,expected", [
    # the tone slot sits after the nucleus and before the coda, and the
    # unmarked ngang tone puts nothing in it
    ("tôi", "t̪ˈoj"),
    ("bạn", "bˈaː6n"),
    ("làm", "lˈaː2m"),
    ("có", "kˈɔɜ"),
    ("thể", "tˈe4"),
    ("sẽ", "sˈɛ5"),
    # `ai`/`ao` are long, `ay`/`au` are the short vowel
    ("hai", "hˈaːj"),
    ("hay", "hˈaj"),
    ("nào", "nˈaː2w"),
    ("nhau", "ɲˈaw"),
    # Hanoi mergers: d/gi/r all /z/, s/x both /s/, tr/ch both /tʃ/
    ("ra", "zˈaː"),
    ("gì", "zˈi2"),
    ("gia", "zˈaː"),
    ("xong", "sˈɔŋ"),
    ("trong", "tʃˈɔŋ"),
    # palatal codas front the `a`, and print espeak's hyphenated nucleus
    ("anh", "ˈe-ɲ"),
    ("cách", "kˈe-ɜc"),
    ("mình", "mˈi2ɲ"),
    # final c prints `k` only after a rounded back vowel
    ("học", "hˈɔ6k"),
    ("lúc", "lˈuɜc"),
    # the two `â` irregulars
    ("ấy", "ˈəɪɜ"),
    ("câu", "kˈə1w"),
    ("cậu", "kˈə6w"),
    # the labiovelar on-glide, and `qu` before `uô`
    ("hoàn", "hwˈaː2n"),
    ("luật", "lwˈə6t̪"),
    ("quốc", "kˈuəɜc"),
    ("quay", "kwˈaj"),
    # rising diphthongs: open `ia` stays `iə`, closed `iê` lowers to `iɛ`
    ("kia", "kˈiə"),
    ("biết", "bˈiɛɜt̪"),
    ("được", "ɗˈyə6c"),
    # `ưu` is the one place `ư` does not print `y`
    ("lưu", "lˈiw"),
])
def test_vi_syllable_cases(text: str, expected: str) -> None:
    assert vi_g2p.phonemize_to_espeak_ipa(text)[0] == expected


def test_vi_solid_loanwords_split_into_syllables() -> None:
    """Vietnamese writes one syllable per token; loans like `ôtô` do not."""
    assert vi_g2p.split_syllables("ôtô") == ["ô", "tô"]
    assert vi_g2p.split_syllables("capô") == ["ca", "pô"]
    assert vi_g2p.split_syllables("không") == ["không"]      # not `khô|ng`
    assert vi_g2p.split_syllables("zzz") is None


@pytest.mark.parametrize("value,words", [
    (0, ["nol"]), (11, ["sebelas"]), (15, ["lima", "belas"]),
    (21, ["dua", "puluh", "satu"]), (100, ["seratus"]),
    (1000, ["seribu"]), (2500, ["dua", "ribu", "lima", "ratus"]),
])
def test_id_numerals(value: int, words: list[str]) -> None:
    assert id_g2p.spell_number(value) == words


@pytest.mark.parametrize("value,words", [
    (0, ["không"]), (10, ["mười"]), (15, ["mười", "lăm"]),
    (21, ["hai", "mươi", "mốt"]), (25, ["hai", "mươi", "lăm"]),
    (105, ["một", "trăm", "lẻ", "năm"]),
    (2500, ["hai", "nghìn", "năm", "trăm"]),
])
def test_vi_numerals(value: int, words: list[str]) -> None:
    assert vi_g2p.spell_number(value) == words


def test_numerals_are_read_as_words_not_digits() -> None:
    for module, text in ((id_g2p, "ada 3 buku"), (vi_g2p, "có 3 quyển")):
        ipa, _unreadable = module.phonemize_to_espeak_ipa(text)
        assert not any(character.isdigit() and character not in "12456" for character in ipa), ipa


# --------------------------------------------------------------------------
# 3. dispatch
# --------------------------------------------------------------------------

def test_dispatch_picks_the_language_from_the_voice_config() -> None:
    assert piper_g2p.module_for(_table(ID_PACKAGE)) is id_g2p
    assert piper_g2p.module_for(_table(VI_PACKAGE)) is vi_g2p
    assert piper_g2p.module_for(_table("amy-en-1p46m")) is None


def test_dispatch_refuses_a_language_with_no_espeak_free_path() -> None:
    table = frontend.PhonemeTable(espeak_voice="hi", id_map=_table(ID_PACKAGE).id_map)
    with pytest.raises(piper_g2p.PiperG2PError) as caught:
        piper_g2p.text_to_phoneme_ids("नमस्ते", table)
    assert caught.value.kind == "language"


def test_dispatch_reaches_the_same_ids_as_calling_the_module_directly() -> None:
    for package, module, text in ((ID_PACKAGE, id_g2p, "Selamat pagi, apa kabar?"),
                                  (VI_PACKAGE, vi_g2p, "Tôi rất vui được gặp bạn.")):
        table = _table(package)
        through_dispatch, _ = piper_g2p.text_to_phoneme_ids(text, table)
        direct, _ = module.text_to_phoneme_ids(text, table)
        assert list(through_dispatch) == list(direct)


# --------------------------------------------------------------------------
# 4. nothing is dropped in silence
# --------------------------------------------------------------------------

def test_vi_reports_the_tokens_it_could_not_read() -> None:
    ipa, unreadable = vi_g2p.phonemize_to_espeak_ipa("Xin chào Giselle.")
    assert unreadable == "Giselle"
    assert ipa                                     # still spoken, as letter names


def test_id_reports_characters_outside_the_alphabet() -> None:
    _ipa, unmapped = id_g2p.phonemize_to_espeak_ipa("halo ☃ dunia")
    assert "☃" in unmapped


@pytest.mark.parametrize("module", [id_g2p, vi_g2p])
def test_empty_and_wrong_typed_input_raise_rather_than_return_junk(module) -> None:
    with pytest.raises(RuntimeError) as empty:
        module.phonemize_to_espeak_ipa("   ")
    assert empty.value.kind == "empty"
    with pytest.raises(RuntimeError) as wrong_type:
        module.phonemize_to_espeak_ipa(None)
    assert wrong_type.value.kind == "type"


@pytest.mark.parametrize("module", [id_g2p, vi_g2p])
def test_negative_numbers_raise_rather_than_spelling_nonsense(module) -> None:
    with pytest.raises(RuntimeError) as caught:
        module.spell_number(-1)
    assert caught.value.kind == "value"
