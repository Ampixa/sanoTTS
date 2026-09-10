"""Chinese text -> the pinyin phoneme ids the `zh_CN` piperlite voice is keyed on.

`zh_CN-xiao_ya-medium`, the teacher this voice is distilled from, is not an
espeak voice: its `phoneme_type` is `pinyin` and its `phoneme_id_map` is 85
symbols over 73 ids -- 24 initials, 35 finals, 5 tones, punctuation and the
three framing symbols. Piper reaches that inventory with **g2pW**, a
BERT-based polyphone disambiguator: `g2pw.onnx` is 158,785,270 parameters and
152 MB, a hundred times the 1.55M voice it would serve. This module reaches
the same inventory from two dictionaries and no model at all.

WHAT PIPER DOES, and what is reproduced here, in order
(`piper.phonemize_chinese.ChinesePhonemizer.phonemize`):

1. delete `"`, `“` and `”`;
2. split into sentences with `sentence-stream`;
3. expand digits per sentence with `unicode_rbnf`'s CLDR `zh` rules, plus
   piper's own temperature and percent rewrites;
4. ask g2pW for one pinyin syllable per character (`None` for punctuation);
5. split each syllable into initial / final / tone and emit three symbols,
   with a zero initial written `Ø`;
6. punctuation passes through when the voice's map has an id for it;
7. ids are framed `bos, ..., eos` with a pad after each tone and each pause --
   NOT the pad-between-every-symbol framing the espeak voices use.

Steps 1, 2, 3, 5, 6 and 7 are deterministic and are reproduced exactly: over
the 1,937-sentence Chinese corpus in
`experiments/evidence/zh-pinyin-g2p-20260910.json` they contribute **zero**
divergences. Step 4 is the only place a lexicon can differ from a BERT, and
every measured error is there.

WHERE THE LEXICON COMES FROM. `g2p_data/zh_pinyin/` is pypinyin's character
and phrase dictionaries (MIT; upstream `pinyin-data` and `phrase-pinyin-data`,
both MIT), vendored by `tools/vendor_zh_pinyin.py` and read here by a
reimplementation of pypinyin's own twenty lines of lookup: forward maximum
match over the phrase keys, the phrase's readings when one matches, the
character's first reading otherwise. 358 KB compressed, against 152 MB.
g2pW's own `MONOPHONIC_CHARS.txt` / `POLYPHONIC_CHARS.txt` were measured as an
alternative and rejected -- they bought 0.7 points of sentence parity and the
model archive they ship in states no licence at all.

TWO CORRECTIONS SIT ON TOP OF THE LEXICON, both measured:

* `TEACHER_READINGS` -- g2pW is trained on Taiwanese Mandarin and the teacher
  inherited its readings. `和` as a conjunction comes out `han4`, not `he2`;
  `期` is `qi2`, `质` is `zhi2`, `危` and `微` are `wei2`. Matching the voice means
  matching that, so 54 (character, reading) pairs are remapped. Alone this
  moves held-out sentence parity from 27.1% to 51.4%.
* `_SANDHI` -- the tone of `一` and `不` depends on the tone that follows, which
  a per-word dictionary cannot see. Alone this moves 27.1% to 33.6%; with the
  remap, to 67.9%.

MEASURED, on held-out text the voice never trained on: 67.9% of sentences get
a **byte-identical** id sequence and 0.305% of individual ids differ
(Levenshtein over 25,256 ids). 95% of the residue is one polyphone chosen
differently -- `为` wei4/wei2, `行` hang2/xing2, `长` zhang3/chang2 -- which is
what the BERT is for. Number expansion, sentence segmentation, punctuation and
character coverage each contribute exactly zero. The audio cost of that
residue is in the evidence file.
"""

from __future__ import annotations

import logging
import lzma
import re
import unicodedata
from pathlib import Path

import numpy as np

from . import frontend

logger = logging.getLogger("sanotts.zh_g2p")

DATA_DIR = Path(__file__).resolve().parent / "g2p_data" / "zh_pinyin"


class ZhG2PError(RuntimeError):
    """`kind` separates an expected rejection (empty input) from a bug."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------
# 1. text normalisation, sentence splitting
# ---------------------------------------------------------------------------

# piper deletes these three before anything else. A straight double quote is
# in the voice's phoneme_id_map, so this is a deliberate deletion and not an
# encoding accident.
_QUOTES = re.compile('["“”]')

_ZH_CLOSERS = "”’」』）》】〕〉）"
_ZH_ENDERS = "。！？"
_ZH_BOUNDARY = re.compile(rf"(?:[{_ZH_ENDERS}]|……|…)+[{_ZH_CLOSERS}]*")

_SENTENCE_END = r"[.!?…]|[؟]|[।॥]"
_ASCII_CLOSERS = r"['\"\)\]\}’”»]*"
# sentence-stream writes the lookahead into the pattern with `\p{Lu}`; `re` has
# no Unicode property classes, so it is checked in python below instead.
_ASCII_BOUNDARY = re.compile(rf"(?:{_SENTENCE_END}+){_ASCII_CLOSERS}", re.DOTALL)
_ASCII_LOOKAHEAD_NUMBER = re.compile(r"\s+\d+[.)]{1,2}\s+")
_BLANK_LINES = re.compile(r"(?:\r?\n){2,}")

_WORD_ASTERISKS = re.compile(r"\*+([^\*]+)\*+")
# `re` has no variable-width lookbehind; `(?:\A|(?<=\n))` is the same
# zero-width position as sentence-stream's `(?<=^|\n)` without re.MULTILINE.
_LINE_ASTERISKS = re.compile(r"(?:\A|(?<=\n))\s*\*+")

_TITLECASE_CATEGORIES = frozenset({"Lu", "Lt", "Lo"})
_LETTER_CATEGORIES = frozenset({"Lu", "Ll", "Lt", "Lm", "Lo"})


def _remove_asterisks(text: str) -> str:
    return _LINE_ASTERISKS.sub("", _WORD_ASTERISKS.sub(r"\1", text))


def _is_abbreviation(tail: str) -> bool:
    """sentence-stream's `ABBREVIATION_RE`: `\\b\\p{Lu}(?:\\p{L}{1,2})?\\.$`."""
    if not tail.endswith("."):
        return False
    stem = tail[:-1]
    for length in (1, 2, 3):
        if len(stem) < length:
            break
        candidate = stem[-length:]
        if unicodedata.category(candidate[0]) != "Lu":
            continue
        if any(unicodedata.category(c) not in _LETTER_CATEGORIES for c in candidate[1:]):
            continue
        before = stem[:-length]
        if before and (before[-1].isalnum() or before[-1] == "_"):
            continue  # the \b is not satisfied
        return True
    return False


def _ascii_boundary(text: str) -> re.Match | None:
    """The first ASCII sentence boundary whose lookahead actually holds."""
    position = 0
    while True:
        match = _ASCII_BOUNDARY.search(text, position)
        if match is None:
            return None
        rest = text[match.end():]
        stripped = rest.lstrip()
        if len(rest) != len(stripped) and stripped and \
                unicodedata.category(stripped[0]) in _TITLECASE_CATEGORIES:
            return match
        if _ASCII_LOOKAHEAD_NUMBER.match(rest):
            return match
        position = match.start() + 1


def split_sentences(text: str) -> list[str]:
    """`sentence_stream.stream_to_sentences([text])`, for one complete string.

    Only the whole-string case is reproduced, because that is the only one
    piper uses: it hands the detector a single chunk and then calls `finish()`,
    so the streaming state machine collapses -- the buffer never runs out
    mid-sentence, and the hold-off for a Chinese boundary flush against the end
    of the buffer fires exactly once, on the final sentence, which `finish()`
    emits anyway.
    """
    sentences: list[str] = []
    remaining = text
    current = ""

    while remaining:
        blank = _BLANK_LINES.search(remaining)
        chinese = _ZH_BOUNDARY.search(remaining)
        ascii_match = _ascii_boundary(remaining)

        if chinese and ascii_match:
            punctuation = chinese if chinese.start() < ascii_match.start() else ascii_match
        else:
            punctuation = chinese or ascii_match

        if blank and punctuation:
            first = blank if blank.start() < punctuation.start() else punctuation
        else:
            first = blank or punctuation
        if first is None:
            break
        if first is chinese and first.end() == len(remaining):
            break

        chunk = remaining[:first.end()]
        if not current:
            if _is_abbreviation(chunk[-5:]):
                current = chunk
            else:
                output = _remove_asterisks(chunk.strip())
                if output:
                    sentences.append(output)
        elif _is_abbreviation(current[-5:]):
            current += chunk
        else:
            output = _remove_asterisks(current.strip())
            if output:
                sentences.append(output)
            current = chunk

        if current and not _is_abbreviation(current[-5:]):
            output = _remove_asterisks(current.strip())
            if output:
                sentences.append(output)
            current = ""

        remaining = remaining[first.end():]

    tail = _remove_asterisks((current + remaining).strip())
    if tail:
        sentences.append(tail)
    return sentences


# ---------------------------------------------------------------------------
# 2. numbers -> words: CLDR's zh `spellout-numbering`
# ---------------------------------------------------------------------------
#
# piper expands digits with `unicode_rbnf.RbnfEngine.for_language("zh")`, which
# interprets CLDR's `zh.xml`. Only one ruleset is reachable from
# `format_number` and it is 20 rules long; the six private rulesets it calls
# differ from each other in two integers each. Transcribing them is smaller
# than an XML rule engine, and the transcription is checked against the engine
# rather than by eye: identical on 218,974 distinct inputs from 0 to 10^18,
# including every integer below 100,001, negatives and decimals.

_DIGITS = "〇一二三四五六七八九"

# (scale, unit word, does its private ruleset write a 〇 filler from 20 up,
#  and below what remainder does it write one at all). `百`'s ruleset -- CLDR's
# `number2` -- is the odd one out and writes no filler from 20 up.
_SCALES: tuple[tuple[int, str, bool, int], ...] = (
    (10 ** 16, "京", True, 10 ** 12),
    (10 ** 12, "兆", True, 10 ** 7),
    (10 ** 8, "亿", True, 10 ** 4),
    (10 ** 4, "万", True, 10 ** 3),
    (10 ** 3, "千", True, 10 ** 2),
    (10 ** 2, "百", False, 20),
)


def _remainder(value: int, zero_from_20: bool, plain_from: int) -> str:
    """One of CLDR's private `number*` rulesets, applied to a nonzero value."""
    if value < 10:
        return _DIGITS[0] + _numbering(value)
    if value < 20:
        return (_DIGITS[0] if zero_from_20 else "") + _DIGITS[1] + _numbering(value)
    if value < plain_from:
        return (_DIGITS[0] if zero_from_20 else "") + _numbering(value)
    return _numbering(value)


def _numbering(value: int) -> str:
    """CLDR zh `spellout-numbering`, for a non-negative integer."""
    if value >= 10 ** 18:
        return f"{value:,}"          # CLDR's own fallback to the decimal format
    for scale, unit, zero_from_20, plain_from in _SCALES:
        if value >= scale:
            high, low = divmod(value, scale)
            out = _numbering(high) + unit
            if low:
                out += _remainder(low, zero_from_20, plain_from)
            return out
    if value >= 20:
        high, low = divmod(value, 10)
        return _DIGITS[high] + "十" + (_DIGITS[low] if low else "")
    if value >= 10:
        low = value - 10
        return "十" + (_DIGITS[low] if low else "")
    return _DIGITS[value]


def format_number(text: str) -> str:
    """`RbnfEngine.for_language("zh").format_number(text).text`, reproduced."""
    negative = text.startswith("-")
    body = text[1:] if negative else text
    whole, _, fraction = body.partition(".")
    value = int(whole or "0")
    # An all-zero fraction is not spoken: "15.00" -> 十五. One that merely ends
    # in zeros is: "15.50" -> 十五点五〇.
    if fraction and set(fraction) == {"0"}:
        fraction = ""
    if negative and value == 0 and not fraction:
        negative = False
    if fraction:
        if negative:
            # unicode_rbnf's own behaviour: the fraction is dropped for a
            # negative decimal ("-3.5" -> 负三, "-0.5" -> 负 and nothing else).
            # Reproduced, not endorsed -- the target is the id stream the voice
            # was distilled on.
            return "负" + (_numbering(value) if value else "")
        return _numbering(value) + "点" + "".join(_DIGITS[int(d)] for d in fraction)
    out = _numbering(value)
    return "负" + out if negative else out


# piper's own rewrites, ahead of the generic digit pass.
_TEMPERATURE = re.compile(r"(?P<sign>[-−])?(?P<num>\d+)\s*(?:°\s*C|℃)")
_PERCENT = re.compile(
    r"(?P<num>-?\d+(?:\.\d+)?|"
    r"[零〇一二三四五六七八九十百千万亿两点]+)\s*(?:%|％)")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def numbers_to_words(text: str) -> str:
    """`ChinesePhonemizer._numbers_to_words`."""

    def replace_temperature(match: re.Match) -> str:
        words = format_number(match.group("num"))
        # "零下" rather than "负" for a temperature: piper's choice, kept.
        return f"零下{words}度" if match.group("sign") else f"{words}度"

    text = _TEMPERATURE.sub(replace_temperature, text)

    def replace_percent(match: re.Match) -> str:
        number = match.group("num")
        words = format_number(number) if _NUMBER.fullmatch(number) else number
        return f"百分之{words}"

    text = _PERCENT.sub(replace_percent, text)
    return _NUMBER.sub(lambda m: format_number(m.group(0)), text)


# ---------------------------------------------------------------------------
# 3. the lexicon: pypinyin's lookup over the vendored tables
# ---------------------------------------------------------------------------

# `pypinyin.constants.RE_HANS`, which decides what counts as a Han run.
RE_HANS = re.compile(
    "^(?:[〇-礼㐀-䶿一-鿿豈-﫿"
    "\U00020000-\U0002A6DF\U0002A703-\U0002B73F\U0002B740-\U0002B81D"
    "\U0002B825-\U0002BF6E\U0002C029-\U0002CE93\U0002D016"
    "\U0002D11B-\U0002EBD9\U0002F80A-\U0002FA1F"
    "\U00030000-\U0003134A\U000300F7-\U00031288\U00030EDD])+$"
)


def _read_packed(name: str) -> str:
    """One vendored table, xz-decompressed."""
    path = DATA_DIR / f"{name}.xz"
    if not path.is_file():
        raise ZhG2PError(
            "missing_data",
            f"vendored pinyin table not found: {path}. Reinstall the package, "
            f"or regenerate it with tools/vendor_zh_pinyin.py",
        )
    try:
        return lzma.decompress(path.read_bytes()).decode("utf-8")
    except (lzma.LZMAError, UnicodeDecodeError) as exc:
        raise ZhG2PError("corrupt_data", f"could not read {path}: {exc}") from exc


class Lexicon:
    """pypinyin's resolution, over flat tables instead of the package.

    Three pieces of data and twenty lines of control flow, and no model:
    the phrase KEYS drive a forward maximum-match segmenter, the phrase VALUES
    give a reading per character, and the character table is the default for
    whatever the segmenter left standing alone.
    """

    def __init__(self) -> None:
        self.chars: dict[str, str] = {}
        for line in _read_packed("chars").split("\n"):
            if not line:
                continue
            char, _, reading = line.partition("\t")
            self.chars[char] = reading

        self.phrases: dict[str, list[str]] = {}
        for line in _read_packed("phrases").split("\n"):
            if not line:
                continue
            phrase, _, readings = line.partition("\t")
            self.phrases[phrase] = readings.split(" ")
        # Keys whose readings are exactly the per-character defaults. Their
        # readings are not stored, but the keys are: the segmenter matches on
        # the key set, so deleting one lets a different phrase win the match.
        for phrase in _read_packed("phrase_keys").split("\n"):
            if phrase:
                self.phrases.setdefault(phrase, [])

        self.prefixes: set[str] = set()
        for phrase in self.phrases:
            for index in range(len(phrase)):
                self.prefixes.add(phrase[:index + 1])

    def cut(self, text: str):
        """`pypinyin.seg.mmseg.Seg.cut` in its strict (`no_non_phrases`) mode."""
        remaining = text
        while remaining:
            last_valid_word = ""
            last_valid_index = 0
            for index in range(len(remaining)):
                word = remaining[:index + 1]
                if word in self.prefixes:
                    if word in self.phrases:
                        last_valid_word = word
                        last_valid_index = index + 1
                else:
                    if last_valid_word:
                        yield last_valid_word
                        remaining = remaining[last_valid_index:]
                    else:
                        yield remaining[0]
                        remaining = remaining[1:]
                    break
            else:
                if last_valid_word:
                    yield last_valid_word
                    remaining = remaining[last_valid_index:]
                else:
                    if remaining not in self.phrases:
                        yield from remaining
                    else:
                        yield remaining
                    break

    def syllables(self, text: str) -> list[str | None]:
        """One reading per character, `None` where there is no reading."""
        out: list[str | None] = []
        for run in _split_han_runs(text):
            if not RE_HANS.match(run):
                out.extend([None] * len(run))
                continue
            for word in self.cut(run):
                readings = self.phrases.get(word)
                if readings:
                    out.extend(readings)
                else:
                    out.extend(self.chars.get(char) for char in word)
        return out


def _split_han_runs(text: str) -> list[str]:
    """`pypinyin.seg.simpleseg._seg`: alternating Han and non-Han runs."""
    runs: list[str] = []
    current = ""
    current_is_han: bool | None = None
    for char in text:
        is_han = bool(RE_HANS.match(char))
        if current_is_han is None or is_han == current_is_han:
            current += char
        else:
            runs.append(current)
            current = char
        current_is_han = is_han
    runs.append(current)
    return runs


_LEXICON: Lexicon | None = None


def shared_lexicon() -> Lexicon:
    """The process-wide lexicon; the tables are read once."""
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = Lexicon()
    return _LEXICON


# ---------------------------------------------------------------------------
# 4. the two corrections
# ---------------------------------------------------------------------------

# g2pW is trained on Taiwanese Mandarin -- its inventory is Bopomofo and its
# character lists come from a Taiwanese ministry dictionary -- and the teacher
# voice inherited its readings wholesale even though the voice is zh_CN. These
# are the (character, pypinyin reading) -> (what the teacher actually says)
# pairs, learned on the 1,781 training rows of the Chinese corpus at a support
# of 3 and a purity of 0.8, and measured on the 140 held-out rows the voice
# never saw. 54 entries; almost all of them are the Taiwan reading of a
# character whose mainland reading pypinyin gives correctly.
#
# This table is what the voice SAYS, not what Chinese IS. 和 as a conjunction is
# hé in Putonghua and this makes it hàn, because that is the token the weights
# were trained on. The counts in the comments are occurrences in the training
# corpus.
TEACHER_READINGS: dict[tuple[str, str], str] = {
    # the four that dominate: 和 "and", the measure word 个, and two Taiwan tones
    ("和", "he2"): "han4",       # 553
    ("个", "ge4"): "ge5",        # 275
    ("期", "qi1"): "qi2",        # 80
    ("质", "zhi4"): "zhi2",      # 60
    # Taiwan readings of common characters
    ("括", "kuo4"): "gua1",      # 54
    ("尽", "jin3"): "jin4",      # 51
    ("击", "ji1"): "ji2",        # 43
    ("究", "jiu1"): "jiu4",      # 39
    ("识", "shi2"): "shi4",      # 31
    ("息", "xi1"): "xi2",        # 22
    ("危", "wei1"): "wei2",      # 21
    ("塞", "sai1"): "se4",       # 21
    ("微", "wei1"): "wei2",      # 20
    ("播", "bo1"): "bo4",        # 19
    ("筑", "zhu4"): "zhu2",      # 18
    ("拥", "yong1"): "yong3",    # 15
    ("勒", "lei1"): "le4",       # 14
    ("穴", "xue2"): "xue4",      # 14
    ("熟", "shu2"): "shou2",     # 14
    ("菌", "jun1"): "jun4",      # 14
    ("绩", "ji4"): "ji1",        # 13
    ("场", "chang2"): "chang3",  # 13
    ("迹", "ji4"): "ji1",        # 12
    ("突", "tu1"): "tu2",        # 10
    ("档", "dang4"): "dang3",    # 9
    ("艘", "sou1"): "sao1",      # 8
    ("血", "xue4"): "xie3",      # 8
    ("干", "gan4"): "gan1",      # 8
    ("携", "xie2"): "xi1",       # 8
    ("脏", "zang4"): "zang1",    # 6
    ("壳", "qiao4"): "ke2",      # 6
    ("泊", "po1"): "bo2",        # 6
    ("片", "pian1"): "pian4",    # 6
    ("锡", "xi1"): "xi2",        # 6
    ("宜", "yi5"): "yi2",        # 6
    ("匹", "pi3"): "pi1",        # 5
    ("薄", "bao2"): "bo2",       # 5
    ("削", "xue1"): "xue4",      # 5
    ("斗", "dou4"): "dou3",      # 5
    ("驯", "xun4"): "xun2",      # 5
    ("暂", "zan4"): "zhan4",     # 4
    ("掷", "zhi4"): "zhi2",      # 4
    ("佛", "fu2"): "fo2",        # 4
    ("堤", "di1"): "ti2",        # 4
    ("妮", "ni1"): "ni2",        # 4
    ("差", "cha4"): "cha1",      # 4
    ("柏", "bai3"): "bo2",       # 4
    ("氰", "qing2"): "qing1",    # 4
    ("夕", "xi1"): "xi4",        # 3
    ("发", "fa4"): "fa1",        # 3
    ("的", "di2"): "de5",        # 3
    ("的", "di1"): "de5",        # 3
    ("储", "chu3"): "chu2",      # 3
    ("咖", "ga1"): "ka1",        # 3
}

# The tone of 一 and 不 is a function of the tone that follows, which no
# per-word dictionary can see. Derived from the teacher's own output on the
# training rows and identical to the textbook rule: 一 is yí before a fourth or
# neutral tone and yì before the others, 不 is bú only before a fourth tone.
# Word-final 一 keeps its dictionary yī, which is what the missing right
# context leaves in place.
_SANDHI: dict[str, dict[str, str]] = {
    "一": {"1": "yi4", "2": "yi4", "3": "yi4", "4": "yi2", "5": "yi2"},
    "不": {"1": "bu4", "2": "bu4", "3": "bu4", "4": "bu2", "5": "bu4"},
}


def sentence_syllables(sentence: str) -> list[str | None]:
    """One pinyin syllable per character of `sentence`, or `None`.

    Aligned character-for-character with the input, exactly like the list
    g2pW hands back, so the caller can zip the two.
    """
    syllables = shared_lexicon().syllables(sentence)
    syllables = [TEACHER_READINGS.get((char, syllable), syllable) if syllable else None
                 for char, syllable in zip(sentence, syllables)]
    for index, char in enumerate(sentence):
        table = _SANDHI.get(char)
        if table is None or syllables[index] is None:
            continue
        following = syllables[index + 1] if index + 1 < len(syllables) else None
        if following:
            syllables[index] = table[following[-1]]
    return syllables


# ---------------------------------------------------------------------------
# 5. syllable -> phonemes -> ids, piper's own framing
# ---------------------------------------------------------------------------

# Longest first: `zh`/`ch`/`sh` have to beat `z`/`c`/`s`.
PINYIN_INITIALS: tuple[str, ...] = (
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "z", "c", "s", "y", "w",
)
ZERO_INITIAL = "Ø"

_SYLLABLE = re.compile(r"^([a-züv:]+?)([1-5])$")

# Ids get a pad after these, not between every symbol -- `phonemes_to_ids` in
# piper.phonemize_chinese, which is a different function from the espeak one.
GROUP_END_PHONEMES: frozenset[str] = frozenset(
    "12345") | frozenset({
        "。", "？", "！", ".", "?", "!",
        "—", "…", "、", "，", "：", "；",
        ",", ":", ";", " ",
    })


def normalize_syllable(syllable: str) -> str:
    """`_normalize_g2pw_syllable`: the u-umlaut family becomes ASCII `v`."""
    match = _SYLLABLE.match(syllable)
    if not match:
        return syllable
    base, tone = match.group(1), match.group(2)
    return base.replace("u:", "v").replace("ü", "v") + tone


def split_initial_final_tone(syllable: str) -> tuple[str, str, str | None]:
    """`hang2` -> `('h', 'ang', '2')`; `ai3` -> `('', 'ai', '3')`."""
    match = re.match(r"^([a-zvü]+?)([1-5])$", syllable)
    if not match:
        return "", "", None
    base, tone = match.group(1), match.group(2)
    initial = ""
    for candidate in PINYIN_INITIALS:
        if base.startswith(candidate):
            initial = candidate
            break
    return initial, base[len(initial):] if initial else base, tone


def sentence_phonemes(sentence: str, syllables: list[str | None],
                      table: frontend.PhonemeTable) -> list[str]:
    """The inner loop of `ChinesePhonemizer.phonemize`, for one sentence."""
    out: list[str] = []
    for syllable, char in zip(syllables, sentence):
        if syllable is None:
            # Punctuation, or a character with no reading. piper emits it only
            # if the voice has an id for it, and drops it silently otherwise.
            if char in table.id_map:
                out.append(char)
            continue
        syllable = normalize_syllable(syllable)
        initial, final, tone = split_initial_final_tone(syllable)
        if not final:
            # Not a pinyin syllable at all. piper appends it whole, which the
            # id lookup then discards; kept so the two behave the same.
            out.append(syllable)
            continue
        out.extend((initial or ZERO_INITIAL, final, tone))
    return out


def phonemes_to_ids(phonemes: list[str], table: frontend.PhonemeTable) -> list[int]:
    """`piper.phonemize_chinese.phonemes_to_ids`.

    Not `frontend.phonemes_to_ids`: the espeak voices get a pad between every
    pair of symbols, the pinyin voices get one after each complete syllable
    (i.e. after its tone) and after each pause. Using the wrong one produces a
    sequence twice the right length that the duration model has never seen.
    """
    ids: list[int] = [frontend.BOS_ID]
    missing = 0
    for phoneme in phonemes:
        phoneme_id = table.id_map.get(phoneme)
        if phoneme_id is None:
            missing += 1
            continue
        ids.append(phoneme_id)
        if phoneme in GROUP_END_PHONEMES:
            ids.append(frontend.PAD_ID)
    if missing:
        logger.warning(
            "sanotts: %d phoneme(s) missing from the voice's phoneme_id_map, skipped",
            missing)
    ids.append(frontend.EOS_ID)
    return ids


def phonemize(text: str, table: frontend.PhonemeTable) -> list[str]:
    """Text -> the flat phoneme list, sentence boundaries already applied.

    piper phonemizes per sentence and synthesizes each separately; the sanotts
    runtime renders one utterance, so the sentences are concatenated here --
    which is exactly what `tools/audition_voice_package.py` does with piper's
    own output when it scores this voice.
    """
    if not isinstance(text, str) or not text.strip():
        raise ZhG2PError("empty", "text must be a non-empty string")
    phonemes: list[str] = []
    for sentence in split_sentences(_QUOTES.sub("", text)):
        sentence = numbers_to_words(sentence)
        phonemes.extend(sentence_phonemes(sentence, sentence_syllables(sentence), table))
    if not phonemes:
        raise ZhG2PError("empty", f"phonemization produced no symbols for text: {text!r}")
    return phonemes


def text_to_phoneme_ids(text: str, table: frontend.PhonemeTable) -> tuple[np.ndarray, str]:
    """Drop-in for `frontend.text_to_phoneme_ids`, with the unmapped symbols."""
    phonemes = phonemize(text, table)
    unmapped = " ".join(sorted({p for p in phonemes if p not in table.id_map}))
    ids = phonemes_to_ids(phonemes, table)
    if len(ids) <= 3:
        raise ZhG2PError("empty", f"phonemization produced no usable phonemes for: {text!r}")
    return np.asarray(ids, dtype=np.int64), unmapped


# ---------------------------------------------------------------------------
# 6. what this path can emit, and what a voice refuses
# ---------------------------------------------------------------------------

def producible_symbols() -> set[str]:
    """Every symbol these tables can emit, enumerated from the tables.

    Not reasoned about: every reading in both vendored tables, plus every
    correction, is normalised and split, and the pieces collected. Punctuation
    is whatever a voice's own map holds and is reported separately.
    """
    lexicon = shared_lexicon()
    readings = set(lexicon.chars.values())
    for entry in lexicon.phrases.values():
        readings.update(entry)
    readings.update(TEACHER_READINGS.values())
    for table in _SANDHI.values():
        readings.update(table.values())

    produced: set[str] = {ZERO_INITIAL}
    for reading in readings:
        if not reading:
            continue
        initial, final, tone = split_initial_final_tone(normalize_syllable(reading))
        if not final:
            produced.add(normalize_syllable(reading))
            continue
        produced.add(initial or ZERO_INITIAL)
        produced.add(final)
        produced.add(tone)
    return produced


def coverage_report(table: frontend.PhonemeTable) -> dict[str, object]:
    """What these tables can emit, and what this voice's map has no id for."""
    lexicon = shared_lexicon()
    produced = producible_symbols()
    unmappable = sorted(symbol for symbol in produced if symbol not in table.id_map)

    unreadable_chars = sorted(
        char for char, reading in lexicon.chars.items()
        if split_initial_final_tone(normalize_syllable(reading))[1] not in table.id_map)
    return {
        "phoneme_type": table.phoneme_type,
        "espeak_voice": table.espeak_voice,
        "table_size": len(table.id_map),
        "lexicon_characters": len(lexicon.chars),
        "lexicon_phrases": len(lexicon.phrases),
        "teacher_reading_corrections": len(TEACHER_READINGS),
        "producible_symbols": len(produced),
        "unmappable_symbols": unmappable,
        "characters_whose_default_reading_has_no_final_id": len(unreadable_chars),
        "characters_whose_default_reading_has_no_final_id_sample": unreadable_chars[:40],
    }
