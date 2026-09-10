"""Gate the Chinese pinyin front end against piper's own g2pW output.

numpy-only, no network, no g2pW, no voice download: the voice's phoneme config
and 156 rows of piper's own output are tracked beside this file, so the
comparison the evidence file reports can be re-run in two seconds without the
152 MB BERT that produced the reference.

Five things are worth freezing.

1. **Parity.** The measured numbers -- how many held-out sentences get a
   byte-identical id sequence, and how many individual ids differ -- are
   asserted, not just computed. A change that quietly makes the front end worse
   fails here.
2. **The deterministic half.** Number expansion, sentence splitting and the id
   framing contributed zero divergences over the whole corpus. Each is pinned
   separately, because a regression in one of them would otherwise hide inside
   the parity number.
3. **Coverage.** Every symbol the tables can emit has an id in the voice's map,
   except the four syllabic-nasal readings piper cannot map either.
4. **Dispatch.** A pinyin table routes to ``zh_g2p``; an espeak table with a
   Chinese voice string is refused rather than fed pinyin it has no ids for.
5. **The other voices are untouched.** ``load_phoneme_table`` still refuses a
   multi-codepoint key for an espeak voice.

Run:  python3 -m pytest pypkg/tests/test_zh_g2p.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pypkg"))

from sanotts import frontend, piper_g2p, zh_g2p  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
CONFIG = DATA / "zh-xiaoya-piper-phoneme-config.json"
REFERENCE = DATA / "zh-pinyin-g2pw-reference.jsonl"


@pytest.fixture(scope="module")
def table() -> frontend.PhonemeTable:
    return frontend.load_phoneme_table(CONFIG)


@pytest.fixture(scope="module")
def reference() -> list[dict]:
    return [json.loads(line) for line in REFERENCE.open(encoding="utf-8") if line.strip()]


def _levenshtein(a: list[int], b: list[int]) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        current = [i]
        for j, bj in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ai != bj)))
        previous = current
    return previous[-1]


# --------------------------------------------------------------------------
# 1. parity against g2pW
# --------------------------------------------------------------------------

def test_config_is_a_pinyin_voice(table: frontend.PhonemeTable) -> None:
    assert table.phoneme_type == frontend.PHONEME_TYPE_PINYIN
    assert len(table.id_map) == 85
    assert max(table.id_map.values()) == 72
    # A multi-codepoint key is the whole reason this table needs its own type.
    assert table.id_map["zh"] == 18 and table.id_map["ang"] == 36


def test_phoneme_id_parity_against_g2pw(table, reference) -> None:
    """The gate. Frozen at the numbers in the evidence file, held-out rows only.

    140 FLORES rows the distillation driver holds out plus 16 out-of-domain
    Tatoeba rows. Measured on 2026-09-10: 107 of 156 sentences identical
    (68.6%) and 81 of 26,030 ids different (0.311%).
    """
    identical = 0
    edits = 0
    oracle_ids = 0
    for row in reference:
        ids, unmapped = zh_g2p.text_to_phoneme_ids(row["text"], table)
        assert not unmapped, f"{row['id']}: symbols with no id: {unmapped!r}"
        got = ids.tolist()
        oracle_ids += len(row["ids"])
        if got == row["ids"]:
            identical += 1
        else:
            edits += _levenshtein(got, row["ids"])

    assert identical >= 107, (
        f"sentence-level id parity regressed: {identical}/{len(reference)} identical, "
        f"was 107"
    )
    assert edits <= 81, (
        f"id error rate regressed: {edits} edits over {oracle_ids} ids, was 81"
    )


def test_every_row_is_close_even_when_it_differs(table, reference) -> None:
    """No row may drift far. The worst held-out row is 5 ids in 2026-09-10."""
    worst = 0
    for row in reference:
        ids, _ = zh_g2p.text_to_phoneme_ids(row["text"], table)
        worst = max(worst, _levenshtein(ids.tolist(), row["ids"]))
    assert worst <= 5, f"a single row now differs by {worst} ids, was at most 5"


# --------------------------------------------------------------------------
# 2. the deterministic half, pinned separately
# --------------------------------------------------------------------------

@pytest.mark.parametrize("digits,expected", [
    ("0", "〇"), ("10", "十"), ("12", "十二"), ("20", "二十"),
    ("100", "一百"), ("101", "一百〇一"), ("110", "一百一十"),
    ("1000", "一千"), ("1001", "一千〇一"), ("1010", "一千〇一十"),
    ("2015", "二千〇一十五"), ("1999", "一千九百九十九"),
    ("10000", "一万"), ("12345", "一万二千三百四十五"),
    ("100000", "十万"), ("100000000", "一亿"),
    ("1234567890", "十二亿三千四百五十六万七千八百九十"),
    # the awkward corners of unicode_rbnf, reproduced rather than corrected
    ("-7", "负七"), ("3.14", "三点一四"), ("0.5", "〇点五"), ("0.05", "〇点〇五"),
    ("-3.5", "负三"), ("-0.5", "负"), ("-0", "〇"),
    ("15.00", "十五"), ("15.50", "十五点五〇"), ("007", "七"),
])
def test_number_expansion_matches_unicode_rbnf(digits: str, expected: str) -> None:
    assert zh_g2p.format_number(digits) == expected


def test_piper_number_rewrites() -> None:
    assert zh_g2p.numbers_to_words("今天-7°C。") == "今天零下七度。"
    assert zh_g2p.numbers_to_words("升了7℃") == "升了七度"
    assert zh_g2p.numbers_to_words("98.76%") == "百分之九十八点七六"
    assert zh_g2p.numbers_to_words("增长77％") == "增长百分之七十七"
    # The percent rewrite passes Chinese numerals through untouched.
    assert zh_g2p.numbers_to_words("百分之七十七") == "百分之七十七"


def test_sentence_splitting() -> None:
    assert zh_g2p.split_sentences("第一句。第二句！第三句？") == [
        "第一句。", "第二句！", "第三句？"]
    # Trailing closers ride with the sentence they close.
    assert zh_g2p.split_sentences("他说：“好。”然后走了。") == [
        "他说：“好。”", "然后走了。"]
    # A comma is not a boundary; a full stop with no following capital is not
    # one either.
    assert zh_g2p.split_sentences("甲，乙，丙。") == ["甲，乙，丙。"]
    assert len(zh_g2p.split_sentences("Dr. Smith 来了。")) == 1


def test_id_framing_is_the_pinyin_one_not_the_espeak_one(table) -> None:
    """A pad after each tone, not between every symbol."""
    ids, _ = zh_g2p.text_to_phoneme_ids("中国", table)
    assert ids.tolist() == [
        frontend.BOS_ID,
        table.id_map["zh"], table.id_map["ong"], table.id_map["1"], frontend.PAD_ID,
        table.id_map["g"], table.id_map["uo"], table.id_map["2"], frontend.PAD_ID,
        frontend.EOS_ID,
    ]


def test_zero_initial_and_tone_split(table) -> None:
    assert zh_g2p.split_initial_final_tone("hang2") == ("h", "ang", "2")
    assert zh_g2p.split_initial_final_tone("ai3") == ("", "ai", "3")
    assert zh_g2p.split_initial_final_tone("zhuang1") == ("zh", "uang", "1")
    # The u-umlaut family: g2pW prints `u:` or `ü`, the map is keyed on `v`.
    assert zh_g2p.normalize_syllable("nu:3") == "nv3"
    assert zh_g2p.normalize_syllable("lüe4") == "lve4"
    assert zh_g2p.split_initial_final_tone("jvan3") == ("j", "van", "3")
    ids, _ = zh_g2p.text_to_phoneme_ids("女", table)
    assert ids.tolist()[1:4] == [table.id_map["n"], table.id_map["v"], table.id_map["3"]]


def test_punctuation_passes_through_only_when_the_voice_has_an_id(table) -> None:
    ids, _ = zh_g2p.text_to_phoneme_ids("好，好。", table)
    assert table.id_map["，"] in ids.tolist()
    assert table.id_map["。"] in ids.tolist()
    # A bracket has no id in this voice and must not become one.
    plain, _ = zh_g2p.text_to_phoneme_ids("好好", table)
    bracketed, _ = zh_g2p.text_to_phoneme_ids("（好好）", table)
    assert plain.tolist() == bracketed.tolist()


def test_the_two_corrections_actually_fire(table) -> None:
    """The Taiwan remap and the 一/不 sandhi, each on a case it decides."""
    lexicon = zh_g2p.shared_lexicon()
    # pypinyin alone gives the mainland reading; the teacher says the other.
    assert lexicon.chars["和"] == "he2"
    assert zh_g2p.sentence_syllables("你和我")[1] == "han4"
    # 一 is yí before a fourth tone, yì before the others, yī at the end.
    assert zh_g2p.sentence_syllables("一个")[0] == "yi2"
    assert zh_g2p.sentence_syllables("一天")[0] == "yi4"
    assert zh_g2p.sentence_syllables("第一")[1] == "yi1"
    # 不 is bú only before a fourth tone.
    assert zh_g2p.sentence_syllables("不是")[0] == "bu2"
    assert zh_g2p.sentence_syllables("不能")[0] == "bu4"


# --------------------------------------------------------------------------
# 3. coverage
# --------------------------------------------------------------------------

def test_coverage_report(table) -> None:
    report = zh_g2p.coverage_report(table)
    assert report["phoneme_type"] == "pinyin"
    assert report["lexicon_characters"] > 40000
    assert report["lexicon_phrases"] > 40000
    assert report["teacher_reading_corrections"] == len(zh_g2p.TEACHER_READINGS)
    # The only symbols with no id are the syllabic nasals -- 嗯, 呣 and two
    # rarer characters. piper drops them too: the voice's inventory is
    # initial+final+tone and these readings have no final.
    assert report["unmappable_symbols"] == ["m2", "n2", "n3", "n4"]
    assert report["characters_whose_default_reading_has_no_final_id"] == 4


def test_nothing_is_dropped_in_silence(table) -> None:
    ids, unmapped = zh_g2p.text_to_phoneme_ids("嗯，好。", table)
    assert "n2" in unmapped or "n3" in unmapped or "n4" in unmapped
    assert ids.size > 3


def test_empty_text_is_rejected_not_guessed(table) -> None:
    for bad in ("", "   ", "\n"):
        with pytest.raises(zh_g2p.ZhG2PError) as excinfo:
            zh_g2p.text_to_phoneme_ids(bad, table)
        assert excinfo.value.kind == "empty"


# --------------------------------------------------------------------------
# 4. dispatch
# --------------------------------------------------------------------------

def test_a_pinyin_table_routes_to_zh_g2p(table) -> None:
    assert piper_g2p.module_for(table) is zh_g2p
    ids, unmapped = piper_g2p.text_to_phoneme_ids("你好，世界。", table)
    assert not unmapped
    assert ids.tolist() == zh_g2p.text_to_phoneme_ids("你好，世界。", table)[0].tolist()


def test_an_espeak_chinese_table_is_refused_not_fed_pinyin(table) -> None:
    """The old espeak `cmn` voices must not be handed initials and finals."""
    espeak_cmn = frontend.PhonemeTable(espeak_voice="cmn", id_map=table.id_map)
    with pytest.raises(piper_g2p.PiperG2PError) as excinfo:
        piper_g2p.module_for(espeak_cmn)
    assert excinfo.value.kind == "language"


def test_espeak_cannot_be_used_as_a_fallback_for_a_pinyin_voice(table) -> None:
    with pytest.raises(frontend.FrontendError, match="phoneme_type"):
        frontend.text_to_phoneme_ids("你好", table)


def test_espeak_tables_still_refuse_multi_codepoint_keys(tmp_path: Path) -> None:
    """The relaxation is gated on phoneme_type and nothing else."""
    broken = tmp_path / "config.json"
    broken.write_text(json.dumps({
        "espeak": {"voice": "en-us"},
        "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "zh": [3]},
    }), encoding="utf-8")
    with pytest.raises(frontend.FrontendError, match="multi-codepoint"):
        frontend.load_phoneme_table(broken)
