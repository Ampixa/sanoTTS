"""Espeak-free English text -> phoneme ids for the nano voices (heart, heart-nano).

Why this exists
---------------
`nano_frontend.py` runs espeak-ng on every word and then applies misaki's `E2M`
rewrite. espeak-ng is GPL-3.0, and it is the only reason the published `sanotts`
wheel has to be GPL. It is also not what the nano voices were trained on:
`tools/make_pack_from_text.py` built every training and eval pack with real
misaki (`KPipeline.g2p` + `KPipeline.en_tokenize`, kokoro 0.9.4), which is
dictionary-first and reaches espeak only for words its dictionaries do not have.
The shipped front end therefore runs misaki's *fallback* path for everything.

This module rebuilds the dictionary-first path from permissively licensed parts:

    text -> tokenise + tag -> misaki us_gold/us_silver lookup
         -> neural fallback for out-of-vocabulary words
         -> the same 62-symbol tokenisation nano_frontend.py already uses

Nothing here calls espeak-ng, and nothing here imports torch. Both paths stay
selectable: `nano_frontend.phonemize` is untouched and remains the baseline this
one is measured against (see docs/nano-espeak-free-frontend.md).

Provenance of the vendored data is in `g2p_data/NOTICE.md`. In short: the
dictionaries are misaki's own `us_gold.json` / `us_silver.json` (Apache-2.0,
byte-identical to the misaki 0.9.4 wheel that built the packs), and the OOV
model is PeterReid/graphemes_to_phonemes_en_us (Apache-2.0), trained on those
same dictionaries rather than distilled from espeak.

What is ported and what is approximated
---------------------------------------
The lexicon, the stress rules, the sub-tokeniser, the token context and the
chunker are direct ports of `misaki/en.py` and `kokoro/pipeline.py`; where a
function corresponds to an upstream one its docstring names it, so the two can
be diffed. Two pieces are deliberately NOT ports, because their upstream
implementations are exactly the dependencies being removed:

  * misaki gets its tokens and part-of-speech tags from spaCy
    (`en_core_web_sm`), which would pull spacy + spacy-curated-transformers into
    a package whose whole selling point is "numpy only". `tokenize` here is a
    rule tokeniser and `_tag` a closed-class lookup. Consequences are measured
    in the design note, not guessed at: the tag only reaches the output through
    punctuation, a dozen special-case function words, and the 790 gold entries
    that are keyed by tag -- and most of those are keyed on 'None', which
    depends on the following vowel rather than on any tag.
  * misaki spells numbers with `num2words`, which is LGPL. `cardinal_words`,
    `ordinal_words`, `year_words` and `decimal_words` reimplement the English
    subset misaki asks for and match num2words' output on those forms.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .nano_frontend import (
    DEFAULT_MAX_TOKENS,
    FrontendError,
    phonemes_to_token_ids,
)
from .nano_g2p_oov import OOVModelError, OOVPhonemizer, shared_phonemizer

DATA_DIR = Path(__file__).resolve().parent / "g2p_data"
GOLD_PATH = DATA_DIR / "us_gold.json"
SILVER_PATH = DATA_DIR / "us_silver.json"

# --------------------------------------------------------------------------
# Constants ported verbatim from misaki/en.py (Apache-2.0). Names are kept so
# the two files can be diffed line by line when misaki moves.
# --------------------------------------------------------------------------

DIPHTHONGS = frozenset("AIOQWYʤʧ")
SUBTOKEN_JUNKS = frozenset("',-._‘’/")
PUNCTS = frozenset(';:,.!?—…"“”')
NON_QUOTE_PUNCTS = frozenset(p for p in PUNCTS if p not in '"“”')

PUNCT_TAGS = frozenset([".", ",", "-LRB-", "-RRB-", "``", '""', "''", ":", "$", "#", "NFP"])
PUNCT_TAG_PHONEMES = {"-LRB-": "(", "-RRB-": ")", "``": chr(8220), '""': chr(8221),
                      "''": chr(8221)}

LEXICON_ORDS = frozenset([39, 45, *range(65, 91), *range(97, 123)])
CONSONANTS = frozenset("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ")
US_TAUS = frozenset("AIOWYiuæɑəɛɪɹʊʌ")

CURRENCIES = {"$": ("dollar", "cent"), "£": ("pound", "pence"), "€": ("euro", "cent")}
ORDINALS = frozenset(["st", "nd", "rd", "th"])
ADD_SYMBOLS = {".": "dot", "/": "slash"}
SYMBOLS = {"%": "percent", "&": "and", "+": "plus", "@": "at"}

US_VOCAB = frozenset("AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʤʧˈˌθᵊᵻʔ")

STRESSES = "ˌˈ"
SECONDARY_STRESS = STRESSES[0]
PRIMARY_STRESS = STRESSES[1]
VOWELS = frozenset("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ")

# kokoro/pipeline.py: the chunk cap KModel enforces, and the punctuation
# classes en_tokenize falls back through when it has to split a long input.
MAX_PS_CHARS = 510
WATERFALL = ["!.?…", ":;", ",—"]
WATERFALL_BUMPS = [")", "”"]


# --------------------------------------------------------------------------
# Token
# --------------------------------------------------------------------------


@dataclass
class Token:
    """The fields of misaki's `MToken` that the English pipeline actually reads.

    misaki stores the second group on an `addict.Dict` under `_`; flattening
    them here drops a dependency and changes nothing about the algorithm.
    """

    text: str
    tag: str
    whitespace: str
    phonemes: str | None = None
    is_head: bool = True
    alias: str | None = None
    stress: float | None = None
    currency: str | None = None
    num_flags: str = ""
    prespace: bool = False
    rating: int | None = None


def stress_weight(ps: str | None) -> int:
    """misaki/en.py: stress_weight."""
    return sum(2 if c in DIPHTHONGS else 1 for c in ps) if ps else 0


def merge_tokens(tokens: list[Token], unk: str | None = None) -> Token:
    """misaki/en.py: merge_tokens."""
    if not tokens:
        raise ValueError("merge_tokens needs at least one token")
    stresses = {tk.stress for tk in tokens if tk.stress is not None}
    currencies = {tk.currency for tk in tokens if tk.currency is not None}
    ratings = {tk.rating for tk in tokens}
    if unk is None:
        phonemes = None
    else:
        phonemes = ""
        for tk in tokens:
            if tk.prespace and phonemes and not phonemes[-1].isspace() and tk.phonemes:
                phonemes += " "
            phonemes += unk if tk.phonemes is None else tk.phonemes
    return Token(
        text="".join(tk.text + tk.whitespace for tk in tokens[:-1]) + tokens[-1].text,
        tag=max(tokens, key=lambda tk: sum(1 if c == c.lower() else 2 for c in tk.text)).tag,
        whitespace=tokens[-1].whitespace,
        phonemes=phonemes,
        is_head=tokens[0].is_head,
        alias=None,
        stress=next(iter(stresses)) if len(stresses) == 1 else None,
        currency=max(currencies) if currencies else None,
        num_flags="".join(sorted({c for tk in tokens for c in tk.num_flags})),
        prespace=tokens[0].prespace,
        rating=None if None in ratings else min(r for r in ratings if r is not None),
    )


def apply_stress(ps: str | None, stress: float | None) -> str | None:
    """misaki/en.py: apply_stress."""

    def restress(text: str) -> str:
        indexed: list[tuple[float, str]] = list(enumerate(text))
        moved = {
            i: next(j for j, v in indexed[int(i):] if v in VOWELS)
            for i, p in indexed if p in STRESSES
        }
        for i, j in moved.items():
            indexed[int(i)] = (j - 0.5, indexed[int(i)][1])
        return "".join(p for _, p in sorted(indexed, key=lambda pair: pair[0]))

    if ps is None or stress is None:
        return ps
    if stress < -1:
        return ps.replace(PRIMARY_STRESS, "").replace(SECONDARY_STRESS, "")
    if stress == -1 or (stress in (0, -0.5) and PRIMARY_STRESS in ps):
        return ps.replace(SECONDARY_STRESS, "").replace(PRIMARY_STRESS, SECONDARY_STRESS)
    if stress in (0, 0.5, 1) and all(s not in ps for s in STRESSES):
        if all(v not in ps for v in VOWELS):
            return ps
        return restress(SECONDARY_STRESS + ps)
    if stress >= 1 and PRIMARY_STRESS not in ps and SECONDARY_STRESS in ps:
        return ps.replace(SECONDARY_STRESS, PRIMARY_STRESS)
    if stress > 1 and all(s not in ps for s in STRESSES):
        if all(v not in ps for v in VOWELS):
            return ps
        return restress(PRIMARY_STRESS + ps)
    return ps


def is_digit(text: str) -> bool:
    """misaki/en.py: is_digit."""
    return bool(re.match(r"^[0-9]+$", text))


# --------------------------------------------------------------------------
# Sub-tokeniser
# --------------------------------------------------------------------------

# misaki/en.py builds this with the `regex` module so it can use \p{L}, \p{Lu}
# and \p{Ll}. `re` has no Unicode property classes, so: \p{L} becomes
# [^\W\d_], which is exactly "word character that is neither digit nor
# underscore", i.e. any Unicode letter; the two case-sensitive alternatives
# become ASCII ranges. That narrowing only affects CamelCase splitting of
# non-ASCII words, and a non-ASCII word cannot be in the lexicon anyway
# (LEXICON_ORDS is ASCII), so it goes to the neural fallback either way.
_LETTER = r"[^\W\d_]"
_QUOTES = "'‘’"
_SUBTOKEN_RE = re.compile(
    rf"^[{_QUOTES}]+"
    rf"|[A-Z](?=[A-Z][a-z])"
    rf"|(?:^-)?(?:\d?[,.]?\d)+"
    rf"|[-_]+"
    rf"|[{_QUOTES}]{{2,}}"
    rf"|{_LETTER}*?(?:[{_QUOTES}]{_LETTER})*?[a-z](?=[A-Z])"
    rf"|{_LETTER}+(?:[{_QUOTES}]{_LETTER})*"
    rf"|[^\w{_QUOTES}-]"
    rf"|[{_QUOTES}]+$"
)


def subtokenize(word: str) -> list[str]:
    """misaki/en.py: subtokenize (the SUBTOKEN_REGEX findall)."""
    return _SUBTOKEN_RE.findall(word)


# --------------------------------------------------------------------------
# Numbers: the English subset of num2words that misaki asks for
# --------------------------------------------------------------------------

_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_SCALES = ("", "thousand", "million", "billion", "trillion", "quadrillion")

_ORDINAL_WORDS = {"one": "first", "two": "second", "three": "third", "five": "fifth",
                  "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


class NumberError(ValueError):
    """A number this reimplementation cannot spell; never silently ignored."""


def _under_hundred(value: int) -> str:
    if value < 20:
        return _ONES[value]
    tens, ones = divmod(value, 10)
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"


def _under_thousand(value: int) -> str:
    hundreds, rest = divmod(value, 100)
    if hundreds == 0:
        return _under_hundred(rest)
    head = f"{_ONES[hundreds]} hundred"
    return head if rest == 0 else f"{head} and {_under_hundred(rest)}"


def cardinal_words(value: int) -> str:
    """num2words(value) for English integers, including its comma/'and' rules.

    Groups of three are joined with ", ", except that a final group below one
    hundred is joined with " and " -- which is why num2words gives "one
    thousand, one hundred" but "one thousand and one".
    """
    if value < 0:
        return f"minus {cardinal_words(-value)}"
    if value < 1000:
        return _under_thousand(value)
    groups: list[int] = []
    rest = value
    while rest:
        rest, group = divmod(rest, 1000)
        groups.append(group)
    if len(groups) > len(_SCALES):
        raise NumberError(f"{value} is larger than this spell-out table covers")
    parts = [
        f"{_under_thousand(group)} {_SCALES[power]}".strip()
        for power, group in reversed(list(enumerate(groups))) if group
    ]
    if len(parts) == 1:
        return parts[0]
    tail = groups[0]
    if 0 < tail < 100:
        return ", ".join(parts[:-1]) + " and " + parts[-1]
    return ", ".join(parts)


def ordinal_words(value: int) -> str:
    """num2words(value, to='ordinal'): the cardinal with its last word ordinalised."""
    words = cardinal_words(value)
    match = re.search(r"[a-z]+$", words)
    if match is None:
        raise NumberError(f"cannot ordinalise the spelling of {value}: {words!r}")
    return words[: match.start()] + _ordinalise(match.group())


def _ordinalise(word: str) -> str:
    if word in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[word]
    if word.endswith("y"):
        return word[:-1] + "ieth"
    return word + "th"


def year_words(value: int) -> str:
    """num2words(value, to='year'): two-digit pairs unless the pattern is 00XX/X00X."""
    if value < 0:
        raise NumberError("negative years are not supported")
    high, low = divmod(value, 100)
    if high == 0 or (high % 10 == 0 and low < 10) or high >= 100:
        return cardinal_words(value)
    if low == 0:
        return f"{cardinal_words(high)} hundred"
    if low < 10:
        return f"{cardinal_words(high)} oh-{cardinal_words(low)}"
    return f"{cardinal_words(high)} {cardinal_words(low)}"


def decimal_words(text: str) -> str:
    """num2words(float(text)): integer part, then 'point', then digit by digit."""
    whole, _, frac = text.partition(".")
    head = cardinal_words(int(whole)) if whole else "zero"
    if not frac:
        return head
    return head + " point " + " ".join(_ONES[int(d)] for d in frac)


# --------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenContext:
    """misaki/en.py: TokenContext."""

    future_vowel: bool | None = None
    future_to: bool = False


class Lexicon:
    """misaki/en.py: Lexicon, for American English only (british=False).

    The British tables are not vendored: the nano voices are American English
    and shipping gb_gold/gb_silver would add 6.5 MB nothing reads.
    """

    cap_stresses = (0.5, 2)

    def __init__(self, gold_path: Path = GOLD_PATH, silver_path: Path = SILVER_PATH) -> None:
        self.golds = Lexicon.grow_dictionary(_load_dictionary(gold_path))
        self.silvers = Lexicon.grow_dictionary(_load_dictionary(silver_path))
        for word, value in self.golds.items():
            if isinstance(value, dict):
                if "DEFAULT" not in value:
                    raise FrontendError(
                        "lexicon", f"gold entry {word!r} is tag-keyed but has no DEFAULT"
                    )
                for variant in value.values():
                    _check_vocab(word, variant)
            else:
                _check_vocab(word, value)

    @staticmethod
    def grow_dictionary(entries: dict[str, Any]) -> dict[str, Any]:
        """misaki/en.py: Lexicon.grow_dictionary."""
        grown: dict[str, Any] = {}
        for key, value in entries.items():
            if len(key) < 2:
                continue
            if key == key.lower():
                if key != key.capitalize():
                    grown[key.capitalize()] = value
            elif key == key.lower().capitalize():
                grown[key.lower()] = value
        grown.update(entries)
        return grown

    def get_NNP(self, word: str) -> tuple[str | None, int | None]:
        """misaki/en.py: Lexicon.get_NNP -- spell the word out letter by letter."""
        letters = [self.golds.get(c.upper()) for c in word if c.isalpha()]
        if not letters or None in letters:
            return None, None
        spelled = apply_stress("".join(str(p) for p in letters), 0)
        if spelled is None:
            return None, None
        head, _, tail = spelled.rpartition(SECONDARY_STRESS)
        if not head:
            return spelled, 3
        return head + PRIMARY_STRESS + tail, 3

    def get_special_case(self, word: str, tag: str, stress: float | None,
                         ctx: TokenContext) -> tuple[str | None, int | None]:
        """misaki/en.py: Lexicon.get_special_case."""
        if tag == "ADD" and word in ADD_SYMBOLS:
            return self.lookup(ADD_SYMBOLS[word], None, -0.5, ctx)
        if word in SYMBOLS:
            return self.lookup(SYMBOLS[word], None, None, ctx)
        if ("." in word.strip(".") and word.replace(".", "").isalpha()
                and len(max(word.split("."), key=len)) < 3):
            return self.get_NNP(word)
        if word in ("a", "A"):
            return ("ɐ" if tag == "DT" else "ˈA"), 4
        if word in ("am", "Am", "AM"):
            if tag.startswith("NN"):
                return self.get_NNP(word)
            if ctx.future_vowel is None or word != "am" or (stress is not None and stress > 0):
                return self.golds["am"], 4
            return "ɐm", 4
        if word in ("an", "An", "AN"):
            if word == "AN" and tag.startswith("NN"):
                return self.get_NNP(word)
            return "ɐn", 4
        if word == "I" and tag == "PRP":
            return SECONDARY_STRESS + "I", 4
        if word in ("by", "By", "BY") and Lexicon.get_parent_tag(tag) == "ADV":
            return "bˈI", 4
        if word in ("to", "To") or (word == "TO" and tag in ("TO", "IN")):
            return {None: self.golds["to"], False: "tə", True: "tʊ"}[ctx.future_vowel], 4
        if word in ("in", "In") or (word == "IN" and tag != "NNP"):
            prefix = PRIMARY_STRESS if ctx.future_vowel is None or tag != "IN" else ""
            return prefix + "ɪn", 4
        if word in ("the", "The") or (word == "THE" and tag == "DT"):
            return ("ði" if ctx.future_vowel is True else "ðə"), 4
        if tag == "IN" and re.match(r"(?i)vs\.?$", word):
            return self.lookup("versus", None, None, ctx)
        if word in ("used", "Used", "USED"):
            if tag in ("VBD", "JJ") and ctx.future_to:
                return self.golds["used"]["VBD"], 4
            return self.golds["used"]["DEFAULT"], 4
        return None, None

    @staticmethod
    def get_parent_tag(tag: str | None) -> str | None:
        """misaki/en.py: Lexicon.get_parent_tag."""
        if tag is None:
            return tag
        if tag.startswith("VB"):
            return "VERB"
        if tag.startswith("NN"):
            return "NOUN"
        if tag.startswith("ADV") or tag.startswith("RB"):
            return "ADV"
        if tag.startswith("ADJ") or tag.startswith("JJ"):
            return "ADJ"
        return tag

    def is_known(self, word: str, tag: str | None) -> bool:
        """misaki/en.py: Lexicon.is_known."""
        if word in self.golds or word in SYMBOLS or word in self.silvers:
            return True
        if not word.isalpha() or not all(ord(c) in LEXICON_ORDS for c in word):
            return False
        if len(word) == 1:
            return True
        if word == word.upper() and word.lower() in self.golds:
            return True
        return word[1:] == word[1:].upper()

    def lookup(self, word: str, tag: str | None, stress: float | None,
               ctx: TokenContext | None) -> tuple[str | None, int | None]:
        """misaki/en.py: Lexicon.lookup."""
        is_NNP = None
        if word == word.upper() and word not in self.golds:
            word = word.lower()
            is_NNP = tag == "NNP"
        entry: Any = self.golds.get(word)
        rating: int | None = 4
        if entry is None and not is_NNP:
            entry, rating = self.silvers.get(word), 3
        if isinstance(entry, dict):
            if ctx is not None and ctx.future_vowel is None and "None" in entry:
                tag = "None"
            elif tag not in entry:
                tag = Lexicon.get_parent_tag(tag)
            entry = entry.get(tag, entry["DEFAULT"])
        if entry is None or (is_NNP and PRIMARY_STRESS not in entry):
            spelled, spelled_rating = self.get_NNP(word)
            if spelled is not None:
                return spelled, spelled_rating
        return apply_stress(entry, stress), rating

    def _s(self, stem: str | None) -> str | None:
        """misaki/en.py: Lexicon._s (the -s suffix)."""
        if not stem:
            return None
        if stem[-1] in "ptkfθ":
            return stem + "s"
        if stem[-1] in "szʃʒʧʤ":
            return stem + "ᵻz"
        return stem + "z"

    def stem_s(self, word: str, tag: str | None, stress: float | None,
               ctx: TokenContext | None) -> tuple[str | None, int | None]:
        """misaki/en.py: Lexicon.stem_s."""
        if len(word) < 3 or not word.endswith("s"):
            return None, None
        if not word.endswith("ss") and self.is_known(word[:-1], tag):
            stem = word[:-1]
        elif ((word.endswith("'s") or (len(word) > 4 and word.endswith("es")
                                       and not word.endswith("ies")))
              and self.is_known(word[:-2], tag)):
            stem = word[:-2]
        elif len(word) > 4 and word.endswith("ies") and self.is_known(word[:-3] + "y", tag):
            stem = word[:-3] + "y"
        else:
            return None, None
        stem_ps, rating = self.lookup(stem, tag, stress, ctx)
        return self._s(stem_ps), rating

    def _ed(self, stem: str | None) -> str | None:
        """misaki/en.py: Lexicon._ed (the -ed suffix)."""
        if not stem:
            return None
        if stem[-1] in "pkfθʃsʧ":
            return stem + "t"
        if stem[-1] == "d":
            return stem + "ᵻd"
        if stem[-1] != "t":
            return stem + "d"
        if len(stem) < 2:
            return stem + "ɪd"
        if stem[-2] in US_TAUS:
            return stem[:-1] + "ɾᵻd"
        return stem + "ᵻd"

    def stem_ed(self, word: str, tag: str | None, stress: float | None,
                ctx: TokenContext | None) -> tuple[str | None, int | None]:
        """misaki/en.py: Lexicon.stem_ed."""
        if len(word) < 4 or not word.endswith("d"):
            return None, None
        if not word.endswith("dd") and self.is_known(word[:-1], tag):
            stem = word[:-1]
        elif (len(word) > 4 and word.endswith("ed") and not word.endswith("eed")
              and self.is_known(word[:-2], tag)):
            stem = word[:-2]
        else:
            return None, None
        stem_ps, rating = self.lookup(stem, tag, stress, ctx)
        return self._ed(stem_ps), rating

    def _ing(self, stem: str | None) -> str | None:
        """misaki/en.py: Lexicon._ing (the -ing suffix)."""
        if not stem:
            return None
        if len(stem) > 1 and stem[-1] == "t" and stem[-2] in US_TAUS:
            return stem[:-1] + "ɾɪŋ"
        return stem + "ɪŋ"

    def stem_ing(self, word: str, tag: str | None, stress: float | None,
                 ctx: TokenContext | None) -> tuple[str | None, int | None]:
        """misaki/en.py: Lexicon.stem_ing."""
        if len(word) < 5 or not word.endswith("ing"):
            return None, None
        if len(word) > 5 and self.is_known(word[:-3], tag):
            stem = word[:-3]
        elif self.is_known(word[:-3] + "e", tag):
            stem = word[:-3] + "e"
        elif (len(word) > 5 and re.search(r"([bcdgklmnprstvxz])\1ing$|cking$", word)
              and self.is_known(word[:-4], tag)):
            stem = word[:-4]
        else:
            return None, None
        stem_ps, rating = self.lookup(stem, tag, stress, ctx)
        return self._ing(stem_ps), rating

    def get_word(self, word: str, tag: str | None, stress: float | None,
                 ctx: TokenContext) -> tuple[str | None, int | None]:
        """misaki/en.py: Lexicon.get_word."""
        ps, rating = self.get_special_case(word, tag or "", stress, ctx)
        if ps is not None:
            return ps, rating
        lowered = word.lower()
        if (len(word) > 1 and word.replace("'", "").isalpha() and word != lowered
                and (tag != "NNP" or len(word) > 7)
                and word not in self.golds and word not in self.silvers
                and (word == word.upper() or word[1:] == word[1:].lower())
                and (lowered in self.golds or lowered in self.silvers
                     or any(fn(lowered, tag, stress, ctx)[0]
                            for fn in (self.stem_s, self.stem_ed, self.stem_ing)))):
            word = lowered
        if self.is_known(word, tag):
            return self.lookup(word, tag, stress, ctx)
        if word.endswith("s'") and self.is_known(word[:-2] + "'s", tag):
            return self.lookup(word[:-2] + "'s", tag, stress, ctx)
        if word.endswith("'") and self.is_known(word[:-1], tag):
            return self.lookup(word[:-1], tag, stress, ctx)
        for stemmer, stem_stress in ((self.stem_s, stress), (self.stem_ed, stress),
                                     (self.stem_ing, 0.5 if stress is None else stress)):
            stemmed, rating = stemmer(word, tag, stem_stress, ctx)
            if stemmed is not None:
                return stemmed, rating
        return None, None

    @staticmethod
    def is_currency(word: str) -> bool:
        """misaki/en.py: Lexicon.is_currency."""
        if "." not in word:
            return True
        if word.count(".") > 1:
            return False
        cents = word.split(".")[1]
        return len(cents) < 3 or set(cents) == {"0"}

    @staticmethod
    def is_number(word: str, is_head: bool) -> bool:
        """misaki/en.py: Lexicon.is_number."""
        if all(not is_digit(c) for c in word):
            return False
        for suffix in ("ing", "'d", "ed", "'s", *ORDINALS, "s"):
            if word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        return all(is_digit(c) or c in ",." or (is_head and i == 0 and c == "-")
                   for i, c in enumerate(word))

    def get_number(self, word: str, currency: str | None, is_head: bool,
                   num_flags: str) -> tuple[str | None, int | None]:
        """misaki/en.py: Lexicon.get_number, with cardinal_words for num2words."""
        match = re.search(r"[a-z']+$", word)
        suffix = match.group() if match else None
        if suffix:
            word = word[: -len(suffix)]
        result: list[tuple[str | None, int | None]] = []
        if word.startswith("-"):
            result.append(self.lookup("minus", None, None, None))
            word = word[1:]

        def extend_num(num: str, first: bool = True, escape: bool = False) -> None:
            splits = re.split(r"[^a-z]+", num if escape else cardinal_words(int(num)))
            for i, part in enumerate(splits):
                if not part:
                    continue
                if part != "and" or "&" in num_flags:
                    if first and i == 0 and len(splits) > 1 and part == "one" and "a" in num_flags:
                        result.append(("ə", 4))
                    else:
                        result.append(
                            self.lookup(part, None, -2 if part == "point" else None, None)
                        )
                elif part == "and" and "n" in num_flags and result:
                    result[-1] = ((result[-1][0] or "") + "ən", result[-1][1])

        if is_digit(word) and suffix in ORDINALS:
            extend_num(ordinal_words(int(word)), escape=True)
        elif not result and len(word) == 4 and currency not in CURRENCIES and is_digit(word):
            extend_num(year_words(int(word)), escape=True)
        elif not is_head and "." not in word:
            num = word.replace(",", "")
            if num[0] == "0" or len(num) > 3:
                for digit in num:
                    extend_num(digit, first=False)
            elif len(num) == 3 and not num.endswith("00"):
                extend_num(num[0])
                if num[1] == "0":
                    result.append(self.lookup("O", None, -2, None))
                    extend_num(num[2], first=False)
                else:
                    extend_num(num[1:], first=False)
            else:
                extend_num(num)
        elif word.count(".") > 1 or not is_head:
            first = True
            for num in word.replace(",", "").split("."):
                if not num:
                    pass
                elif num[0] == "0" or (len(num) != 2 and any(n != "0" for n in num[1:])):
                    for digit in num:
                        extend_num(digit, first=False)
                else:
                    extend_num(num, first=first)
                first = False
        elif currency in CURRENCIES and Lexicon.is_currency(word):
            pairs = [(int(num) if num else 0, unit) for num, unit
                     in zip(word.replace(",", "").split("."), CURRENCIES[currency])]
            if len(pairs) > 1:
                if pairs[1][0] == 0:
                    pairs = pairs[:1]
                elif pairs[0][0] == 0:
                    pairs = pairs[1:]
            for i, (num, unit) in enumerate(pairs):
                if i > 0:
                    result.append(self.lookup("and", None, None, None))
                extend_num(str(num), first=i == 0)
                result.append(
                    self.stem_s(unit + "s", None, None, None)
                    if abs(num) != 1 and unit != "pence"
                    else self.lookup(unit, None, None, None)
                )
        else:
            cleaned = word.replace(",", "")
            if is_digit(cleaned):
                spelled = (ordinal_words(int(cleaned)) if suffix in ORDINALS
                           else cardinal_words(int(cleaned)))
            elif "." not in cleaned:
                raise NumberError(f"cannot spell {word!r} as a number")
            elif cleaned[0] == ".":
                spelled = "point " + " ".join(_ONES[int(n)] for n in cleaned[1:])
            else:
                spelled = decimal_words(cleaned)
            extend_num(spelled, escape=True)

        if not result:
            return None, None
        joined = " ".join(p for p, _ in result if p)
        ratings = [r for _, r in result if r is not None]
        rating = min(ratings) if ratings else None
        if suffix in ("s", "'s"):
            return self._s(joined), rating
        if suffix in ("ed", "'d"):
            return self._ed(joined), rating
        if suffix == "ing":
            return self._ing(joined), rating
        return joined, rating

    def append_currency(self, ps: str | None, currency: str | None) -> str | None:
        """misaki/en.py: Lexicon.append_currency."""
        if not currency:
            return ps
        pair = CURRENCIES.get(currency)
        if pair is None:
            return ps
        unit = self.stem_s(pair[0] + "s", None, None, None)[0]
        return f"{ps} {unit}" if unit else ps

    @staticmethod
    def numeric_if_needed(char: str) -> str:
        """misaki/en.py: Lexicon.numeric_if_needed."""
        if not char.isdigit():
            return char
        value = unicodedata.numeric(char)
        return str(int(value)) if value == int(value) else char

    def __call__(self, token: Token, ctx: TokenContext) -> tuple[str | None, int | None]:
        """misaki/en.py: Lexicon.__call__."""
        word = (token.text if token.alias is None else token.alias)
        word = word.replace(chr(8216), "'").replace(chr(8217), "'")
        word = unicodedata.normalize("NFKC", word)
        word = "".join(Lexicon.numeric_if_needed(c) for c in word)
        stress = None if word == word.lower() else Lexicon.cap_stresses[int(word == word.upper())]
        ps, rating = self.get_word(word, token.tag, stress, ctx)
        if ps is not None:
            return apply_stress(self.append_currency(ps, token.currency), token.stress), rating
        if Lexicon.is_number(word, token.is_head):
            ps, rating = self.get_number(word, token.currency, token.is_head, token.num_flags)
            return apply_stress(ps, token.stress), rating
        # misaki distinguishes "outside the lexicon alphabet" from "inside it but
        # unknown" here and returns the same thing for both; the caller's next
        # step is the fallback either way.
        return None, None


def _load_dictionary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FrontendError(
            "lexicon",
            f"dictionary not found at {path}; run tools/build_nano_g2p_assets.py to fetch it",
        )
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrontendError("lexicon", f"could not read {path}: {exc}") from exc
    if not isinstance(entries, dict) or not entries:
        raise FrontendError("lexicon", f"{path} is not a non-empty JSON object")
    return entries


def _check_vocab(word: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise FrontendError("lexicon", f"entry {word!r} maps to {type(value).__name__}, not str")
    unknown = [c for c in value if c not in US_VOCAB]
    if unknown:
        raise FrontendError(
            "lexicon", f"entry {word!r} uses symbols outside the American vocab: {unknown}"
        )


# --------------------------------------------------------------------------
# Tokeniser and tagger (the spaCy replacements)
# --------------------------------------------------------------------------

# Words whose part of speech is fixed enough to hard-code. Only tags that can
# change the phonemes are worth listing: the punctuation tags, 'DT' for "a" and
# "the", 'IN' for "in", 'PRP' for "I", 'TO', and the coarse NOUN/VERB/ADJ that
# picks a variant for the 790 tag-keyed gold entries.
_CLOSED_CLASS: dict[str, str] = {
    "a": "DT", "an": "DT", "the": "DT", "this": "DT", "that": "IN", "these": "DT",
    "those": "DT", "every": "DT", "each": "DT", "another": "DT", "any": "DT",
    "some": "DT", "no": "DT", "both": "DT", "either": "DT", "neither": "DT",
    "all": "PDT", "such": "JJ",
    "to": "TO",
    "of": "IN", "in": "IN", "on": "IN", "at": "IN", "by": "IN", "for": "IN",
    "with": "IN", "from": "IN", "into": "IN", "about": "IN", "as": "IN",
    "than": "IN", "if": "IN", "while": "IN", "though": "IN", "although": "IN",
    "because": "IN", "since": "IN", "until": "IN", "unless": "IN", "upon": "IN",
    "over": "IN", "under": "IN", "between": "IN", "among": "IN", "through": "IN",
    "during": "IN", "without": "IN", "within": "IN", "against": "IN",
    "toward": "IN", "towards": "IN", "before": "IN", "after": "IN", "behind": "IN",
    "beyond": "IN", "near": "IN", "per": "IN", "via": "IN", "despite": "IN",
    "above": "IN", "below": "IN", "across": "IN", "along": "IN", "around": "IN",
    "beside": "IN", "besides": "IN", "beneath": "IN", "throughout": "IN",
    "and": "CC", "or": "CC", "but": "CC", "nor": "CC", "yet": "CC", "so": "RB",
    "i": "PRP", "you": "PRP", "he": "PRP", "she": "PRP", "it": "PRP", "we": "PRP",
    "they": "PRP", "me": "PRP", "him": "PRP", "us": "PRP", "them": "PRP",
    "myself": "PRP", "yourself": "PRP", "himself": "PRP", "herself": "PRP",
    "itself": "PRP", "ourselves": "PRP", "themselves": "PRP",
    "my": "PRP$", "your": "PRP$", "his": "PRP$", "its": "PRP$", "our": "PRP$",
    "their": "PRP$", "her": "PRP$",
    "can": "MD", "could": "MD", "will": "MD", "would": "MD", "shall": "MD",
    "should": "MD", "may": "MD", "might": "MD", "must": "MD",
    "am": "VBP", "are": "VBP", "is": "VBZ", "was": "VBD", "were": "VBD",
    "be": "VB", "been": "VBN", "being": "VBG",
    "have": "VBP", "has": "VBZ", "had": "VBD", "having": "VBG",
    "do": "VBP", "does": "VBZ", "did": "VBD", "done": "VBN", "doing": "VBG",
    "not": "RB", "never": "RB", "very": "RB", "too": "RB", "also": "RB",
    "just": "RB", "only": "RB", "then": "RB", "now": "RB", "again": "RB",
    "still": "RB", "quite": "RB", "rather": "RB", "almost": "RB", "always": "RB",
    "often": "RB", "here": "RB", "well": "RB", "even": "RB", "much": "RB",
    "more": "RBR", "most": "RBS", "less": "RBR", "least": "RBS",
    "there": "EX", "when": "WRB", "where": "WRB", "why": "WRB", "how": "WRB",
    "who": "WP", "whom": "WP", "what": "WP", "whose": "WP$", "which": "WDT",
}

# Punctuation character -> Penn tag. misaki reads the tag, not the character:
# ';' has to arrive as ':' and '!' as '.' or the phoneme lookup misses.
_PUNCT_TAG_BY_CHAR: dict[str, str] = {
    ",": ",", ".": ".", "!": ".", "?": ".", ";": ":", ":": ":",
    # A bare ASCII hyphen is HYPH, not ':': spaCy calls it that even when it is
    # standing in for a dash, and HYPH is outside PUNCT_TAGS, so misaki voices
    # nothing for it. Real en/em dashes do get ':' and become the "—" symbol.
    "-": "HYPH", "–": ":", "—": ":", "…": ":",
    "(": "-LRB-", "[": "-LRB-", "{": "-LRB-",
    ")": "-RRB-", "]": "-RRB-", "}": "-RRB-",
    "$": "$", "£": "$", "€": "$", "#": "#",
    "“": "``", "‘": "``", "”": "''", "’": "''",
}

_PREFIX_CHARS = "\"`([{“‘«¿¡$£€#*…"
_SUFFIX_CHARS = ".,;:!?\")]}”»…%"

# Abbreviations whose trailing period belongs to the word, hand-written here
# rather than taken from spaCy's exception table. Only the ones the gold
# dictionary can actually pronounce, plus the common titles and Latin
# abbreviations, are listed; anything else keeps the period as its own token,
# which is also what misaki does when spaCy does not recognise the form.
_ABBREVIATIONS = frozenset([
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "Jr.", "Sr.", "Esq.", "Rev.",
    "Gen.", "Gov.", "Sen.", "Rep.", "Capt.", "Lt.", "Col.", "Sgt.", "Adm.",
    "Mt.", "Ft.", "Co.", "Corp.", "Inc.", "Ltd.", "Bros.",
    "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.",
    "Oct.", "Nov.", "Dec.",
    "etc.", "vs.", "e.g.", "i.e.", "a.m.", "p.m.",
])

_QUOTE_CHARS = "\"“”"

# spaCy splits a double hyphen that sits between two word characters into its
# own token ("saw--and" -> "saw", "--", "and"); misaki then reads the ':' tag
# and voices nothing for it. Without this the whole run reaches the fallback as
# one unpronounceable string.
_DOUBLE_HYPHEN_RE = re.compile(r"(?<=\w)(-{2,})(?=\w)")


def _punct_tag(text: str, position: str, quote_open: bool) -> str | None:
    """Penn tag for a punctuation-only token, or None if it is not punctuation."""
    if any(c.isalnum() for c in text):
        return None
    if text in SYMBOLS:
        # "%", "&", "+", "@" are words, not punctuation: misaki's
        # get_special_case spells them out. Tagging them here would put them in
        # PUNCT_TAGS, where they phonemize to nothing at all.
        return "NN"
    if text and set(text) <= set(_QUOTE_CHARS):
        if text in ("“", "”"):
            return "``" if text == "“" else "''"
        if position == "prefix":
            return "``"
        if position == "suffix":
            return "''"
        return "``" if quote_open else "''"
    if set(text) == {"'"}:
        return "''"
    tags = {_PUNCT_TAG_BY_CHAR.get(c) for c in text}
    if len(tags) == 1:
        only = tags.pop()
        if only is not None:
            return only
    return "NFP"


def _tag(text: str, position: str, sentence_start: bool, quote_open: bool) -> str:
    """A Penn tag for one token; the stand-in for spaCy's statistical tagger.

    See the module docstring for what this costs. It is deliberately blunt: a
    wrong tag on an ordinary content word changes nothing, because the lexicon
    only consults the tag for punctuation, a dozen function words, and the gold
    entries that carry per-tag variants.
    """
    punct = _punct_tag(text, position, quote_open)
    if punct is not None:
        return punct
    if Lexicon.is_number(text, True):
        return "CD"
    if text == "I":
        return "PRP"
    lowered = text.lower()
    closed = _CLOSED_CLASS.get(lowered)
    if closed is not None:
        return closed
    if text[:1].isupper() and not sentence_start:
        return "NNP"
    return "NN"


def _split_chunk(chunk: str) -> list[tuple[str, str]]:
    """One whitespace-delimited chunk -> [(token text, position)] as spaCy splits it.

    Position is "prefix", "core" or "suffix" and exists only so a straight
    double quote can be called opening or closing by where it sits: a quote
    peeled off the front of a chunk opens, one peeled off the back closes. That
    beats alternating from the start of the string, which gets the direction
    wrong for every sentence that begins inside a quotation.

    Prefix and suffix punctuation is peeled one character at a time. Apostrophes
    are never split: misaki's own sub-tokeniser keeps "don't" and "boys'" whole
    and `get_word` has the rules for both, whereas a split apostrophe would be
    tagged as a closing quote and voiced.
    """
    if not chunk:
        return []
    prefixes: list[str] = []
    suffixes: list[str] = []
    core = chunk
    changed = True
    while changed and len(core) > 1:
        changed = False
        if core[0] in _PREFIX_CHARS:
            prefixes.append(core[0])
            core = core[1:]
            changed = True
            continue
        if core[-1] in _SUFFIX_CHARS and not _keeps_final_period(core):
            suffixes.insert(0, core[-1])
            core = core[:-1]
            changed = True
    cores = [part for part in _DOUBLE_HYPHEN_RE.split(core) if part] if core else []
    return [*((p, "prefix") for p in prefixes),
            *((c, "core") for c in cores),
            *((x, "suffix") for x in suffixes)]


def _keeps_final_period(word: str) -> bool:
    """True when a trailing '.' is part of the token, not sentence punctuation."""
    if not word.endswith("."):
        return False
    if word in _ABBREVIATIONS or word.lower() in _ABBREVIATIONS:
        return True
    # Dotted initialisms ("U.S.", "a.m.") -- the same shape misaki's
    # get_special_case spells out letter by letter.
    stripped = word.strip(".")
    return ("." in stripped and word.replace(".", "").isalpha()
            and len(max(word.split("."), key=len)) < 3)


def tokenize(text: str) -> list[Token]:
    """Text -> tagged tokens, in place of `spacy.load('en_core_web_sm')`."""
    tokens: list[Token] = []
    quote_open = True
    sentence_start = True
    for chunk in text.split():
        pieces = _split_chunk(chunk)
        for index, (piece, position) in enumerate(pieces):
            tag = _tag(piece, position, sentence_start, quote_open)
            if tag in ("``", "''"):
                quote_open = tag == "''"
            sentence_start = tag == "." or (sentence_start and tag in ("``", "-LRB-"))
            tokens.append(Token(text=piece, tag=tag,
                                whitespace=" " if index == len(pieces) - 1 else ""))
    if tokens:
        tokens[-1].whitespace = ""
    _retag_verbs(tokens)
    _retag_that(tokens)
    return tokens


def _retag_that(tokens: list[Token]) -> None:
    """"that" is the one homograph a single lookahead resolves.

    us_gold keys it as {'DEFAULT': 'ðæt', 'DT': 'ðˈæt'}, so only the
    determiner reading is stressed. Demonstrative "that" is followed by a noun
    or an adjective ("that cup", "that lively air"); the complementiser and
    relative readings are followed by a pronoun, a determiner or a verb. On the
    measured set this rule is right 11 times and wrong twice.
    """
    for i, token in enumerate(tokens):
        if token.text.lower() != "that" or token.tag != "IN":
            continue
        following = tokens[i + 1].tag if i + 1 < len(tokens) else None
        if following is not None and (following.startswith("NN") or following.startswith("JJ")):
            token.tag = "DT"


# Auxiliaries after which an open-class word is a past participle rather than a
# bare verb; "read" is keyed on VBN and VBD separately, so VB alone is no use.
_PERFECT_AUXILIARIES = frozenset(["have", "has", "had", "having",
                                  "is", "are", "was", "were", "be", "been", "being"])


def _retag_verbs(tokens: list[Token]) -> None:
    """Promote a following open-class word to a verb after "to" or an auxiliary.

    The 790 tag-keyed gold entries include a few dozen noun/verb pairs whose
    vowel changes ("live", "tear", "use", "close"), and without spaCy the only
    signal available is the word in front. "to X" and "will X" are verbs;
    "had X" and "was X" are past participles. Both are safe enough to apply
    unconditionally -- an open-class word in those slots is a verb in ordinary
    English -- and neither fires on a word the closed-class table already tagged.
    """
    previous_tag = None
    previous_text = ""
    for token in tokens:
        if token.tag in ("NN", "JJ"):
            if previous_tag in ("TO", "MD"):
                token.tag = "VB"
            elif previous_text in _PERFECT_AUXILIARIES:
                token.tag = "VBN"
        if token.tag not in ("RB", "RBR", "RBS"):
            previous_tag = token.tag
            previous_text = token.text.lower()


# --------------------------------------------------------------------------
# G2P
# --------------------------------------------------------------------------


class NanoG2P:
    """misaki/en.py: G2P, with the spaCy front and the espeak fallback replaced."""

    def __init__(self, fallback: Callable[[Token], tuple[str | None, int | None]] | None = None,
                 unk: str = "") -> None:
        self.lexicon = Lexicon()
        self.unk = unk
        self.fallback = fallback if fallback is not None else _NeuralFallback()

    @staticmethod
    def fold_left(tokens: list[Token]) -> list[Token]:
        """misaki/en.py: G2P.fold_left."""
        result: list[Token] = []
        for token in tokens:
            if result and not token.is_head:
                token = merge_tokens([result.pop(), token], unk="")
            result.append(token)
        return result

    @staticmethod
    def retokenize(tokens: list[Token]) -> list[Token | list[Token]]:
        """misaki/en.py: G2P.retokenize."""
        words: list[Token | list[Token]] = []
        currency: str | None = None
        for i, token in enumerate(tokens):
            if token.alias is None and token.phonemes is None:
                pieces = subtokenize(token.text)
                if not pieces:
                    pieces = [token.text]
                tks = [replace(token, text=piece, whitespace="", is_head=True,
                               num_flags=token.num_flags, stress=token.stress,
                               prespace=False, phonemes=None, alias=None, currency=None,
                               rating=None)
                       for piece in pieces]
            else:
                tks = [token]
            tks[-1].whitespace = token.whitespace
            for j, tk in enumerate(tks):
                if tk.alias is not None or tk.phonemes is not None:
                    pass
                elif tk.tag == "$" and tk.text in CURRENCIES:
                    currency = tk.text
                    tk.phonemes = ""
                    tk.rating = 4
                elif tk.tag == ":" and tk.text in ("-", "–"):
                    tk.phonemes = "—"
                    tk.rating = 3
                elif (tk.tag in PUNCT_TAGS
                      and not all(97 <= ord(c.lower()) <= 122 for c in tk.text)):
                    tk.phonemes = PUNCT_TAG_PHONEMES.get(
                        tk.tag, "".join(c for c in tk.text if c in PUNCTS)
                    )
                    tk.rating = 4
                elif currency is not None:
                    if tk.tag != "CD":
                        currency = None
                    elif (j + 1 == len(tks)
                          and (i + 1 == len(tokens) or tokens[i + 1].tag != "CD")):
                        tk.currency = currency
                elif (0 < j < len(tks) - 1 and tk.text == "2"
                      and (tks[j - 1].text[-1] + tks[j + 1].text[0]).isalpha()):
                    tk.alias = "to"
                if tk.alias is not None or tk.phonemes is not None:
                    words.append(tk)
                elif words and isinstance(words[-1], list) and not words[-1][-1].whitespace:
                    tk.is_head = False
                    words[-1].append(tk)
                else:
                    words.append(tk if tk.whitespace else [tk])
        return [w[0] if isinstance(w, list) and len(w) == 1 else w for w in words]

    @staticmethod
    def token_context(ctx: TokenContext, ps: str | None, token: Token) -> TokenContext:
        """misaki/en.py: G2P.token_context."""
        vowel = ctx.future_vowel
        if ps:
            vowel = next(
                (None if c in NON_QUOTE_PUNCTS else (c in VOWELS)
                 for c in ps
                 if any(c in group for group in (VOWELS, CONSONANTS, NON_QUOTE_PUNCTS))),
                vowel,
            )
        future_to = token.text in ("to", "To") or (token.text == "TO"
                                                   and token.tag in ("TO", "IN"))
        return TokenContext(future_vowel=vowel, future_to=future_to)

    @staticmethod
    def resolve_tokens(tokens: list[Token]) -> None:
        """misaki/en.py: G2P.resolve_tokens."""
        text = "".join(tk.text + tk.whitespace for tk in tokens[:-1]) + tokens[-1].text
        prespace = (" " in text or "/" in text
                    or len({0 if c.isalpha() else (1 if is_digit(c) else 2)
                            for c in text if c not in SUBTOKEN_JUNKS}) > 1)
        for i, tk in enumerate(tokens):
            if tk.phonemes is None:
                if i == len(tokens) - 1 and tk.text in NON_QUOTE_PUNCTS:
                    tk.phonemes = tk.text
                    tk.rating = 3
                elif all(c in SUBTOKEN_JUNKS for c in tk.text):
                    tk.phonemes = ""
                    tk.rating = 3
            elif i > 0:
                tk.prespace = prespace
        if prespace:
            return
        indices = [(PRIMARY_STRESS in tk.phonemes, stress_weight(tk.phonemes), i)
                   for i, tk in enumerate(tokens) if tk.phonemes]
        if len(indices) == 2 and len(tokens[indices[0][2]].text) == 1:
            i = indices[1][2]
            tokens[i].phonemes = apply_stress(tokens[i].phonemes, -0.5)
            return
        if len(indices) < 2 or sum(b for b, _, _ in indices) <= (len(indices) + 1) // 2:
            return
        for _, _, i in sorted(indices)[: len(indices) // 2]:
            tokens[i].phonemes = apply_stress(tokens[i].phonemes, -0.5)

    def __call__(self, text: str) -> tuple[str, list[Token]]:
        """misaki/en.py: G2P.__call__ (preprocess reduced to a left strip)."""
        tokens: list[Token | list[Token]] = NanoG2P.retokenize(
            NanoG2P.fold_left(tokenize(text.lstrip()))
        )
        ctx = TokenContext()
        for _i, word in reversed(list(enumerate(tokens))):
            if not isinstance(word, list):
                if word.phonemes is None:
                    word.phonemes, word.rating = self.lexicon(word, ctx)
                if word.phonemes is None and self.fallback is not None:
                    word.phonemes, word.rating = self.fallback(word)
                ctx = NanoG2P.token_context(ctx, word.phonemes, word)
                continue
            left, right = 0, len(word)
            should_fallback = False
            while left < right:
                if any(tk.alias is not None or tk.phonemes is not None
                       for tk in word[left:right]):
                    merged = None
                else:
                    merged = merge_tokens(word[left:right])
                ps, rating = (None, None) if merged is None else self.lexicon(merged, ctx)
                # `ps` is None whenever `merged` is, but only the reader knows
                # that; naming both keeps the invariant checkable.
                if merged is not None and ps is not None:
                    word[left].phonemes = ps
                    word[left].rating = rating
                    for other in word[left + 1:right]:
                        other.phonemes = ""
                        other.rating = rating
                    ctx = NanoG2P.token_context(ctx, ps, merged)
                    right = left
                    left = 0
                elif left + 1 < right:
                    left += 1
                else:
                    right -= 1
                    tk = word[right]
                    if tk.phonemes is None:
                        if all(c in SUBTOKEN_JUNKS for c in tk.text):
                            tk.phonemes = ""
                            tk.rating = 3
                        elif self.fallback is not None:
                            should_fallback = True
                            break
                    left = 0
            if should_fallback:
                merged = merge_tokens(word)
                word[0].phonemes, word[0].rating = self.fallback(merged)
                for other in word[1:]:
                    other.phonemes = ""
                    other.rating = word[0].rating
            else:
                NanoG2P.resolve_tokens(word)
        flat = [merge_tokens(tk, unk=self.unk) if isinstance(tk, list) else tk for tk in tokens]
        # misaki applies this whenever version != '2.0', which is kokoro's setting.
        for tk in flat:
            if tk.phonemes:
                tk.phonemes = tk.phonemes.replace("ɾ", "T").replace("ʔ", "t")
        result = "".join((self.unk if tk.phonemes is None else tk.phonemes) + tk.whitespace
                         for tk in flat)
        return result, flat


# A word-like run for the fallback: letters, plus the hyphens, apostrophes and
# periods that sit between letters. The model's grapheme vocabulary contains
# "-", "'" and ".", because misaki's dictionary keys do, so hyphenated names
# stay in-domain instead of being chopped into pieces it never saw.
_FALLBACK_PIECE_RE = re.compile(
    r"[^\W\d_]+(?:['’\-.][^\W\d_]+)*|\s+|.", re.DOTALL
)


def _fold_to_model_alphabet(word: str) -> str:
    """Drop the diacritics the model has no grapheme token for.

    It was trained on misaki's dictionary keys, whose alphabet is ASCII letters
    plus "'", "-" and "."; an accented letter would become <unk> and take the
    surrounding phonemes down with it. NFD-decomposing and dropping the
    combining marks turns "café" into "cafe", which the model can read.
    """
    decomposed = unicodedata.normalize("NFD", word.replace(chr(8217), "'"))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", stripped).strip("-.'")


def _split_for_model(word: str, max_positions: int) -> list[str]:
    """Cut a word that will not fit the model's 64-position table.

    Ordinary English words never reach this; runs of letters this long are URLs
    with the separators already stripped, or someone leaning on a key. Cutting
    is better than raising, because raising loses the whole utterance over one
    pathological token.
    """
    limit = max_positions - 2  # the model's own bos and eos
    if len(word) <= limit:
        return [word]
    return [word[i:i + limit] for i in range(0, len(word), limit)]


class _NeuralFallback:
    """The OOV path: PeterReid's BART where misaki calls espeak-ng.

    misaki hands the fallback whatever the back-off loop could not resolve,
    which is not always a single clean word: a token like "forgave;--that" gets
    here whole because spaCy never split it and no prefix of it is in the
    dictionary. espeak phonemizes the word runs and keeps the punctuation. This
    does the same -- word runs through the model, characters that are in PUNCTS
    passed through, everything else dropped -- rather than handing the model a
    string full of characters it has no token for.
    """

    def __init__(self, model: OOVPhonemizer | None = None) -> None:
        self._model = model

    def __call__(self, token: Token) -> tuple[str | None, int | None]:
        text = token.text.strip()
        if not text:
            return "", 2
        if self._model is None:
            self._model = shared_phonemizer()
        out: list[str] = []
        for piece in _FALLBACK_PIECE_RE.findall(text):
            if piece.isspace():
                if out and not out[-1].endswith(" "):
                    out.append(" ")
                continue
            if piece[0].isalpha():
                cleaned = _fold_to_model_alphabet(piece)
                if not cleaned:
                    continue
                for part in _split_for_model(cleaned, self._model.max_positions):
                    try:
                        phonemes = self._model(part)
                    except OOVModelError as exc:
                        raise FrontendError(
                            "oov", f"the fallback model could not phonemize {part!r}: {exc}"
                        ) from exc
                    if phonemes:
                        out.append(phonemes)
            elif piece in PUNCTS:
                out.append(piece)
        joined = "".join(out).strip()
        return joined, 2


# --------------------------------------------------------------------------
# Chunking and the public entry point
# --------------------------------------------------------------------------


def _tokens_to_ps(tokens: list[Token]) -> str:
    """kokoro/pipeline.py: KPipeline.tokens_to_ps."""
    return "".join((tk.phonemes or "") + (" " if tk.whitespace else "")
                   for tk in tokens).strip()


def _waterfall_last(tokens: list[Token], next_count: int) -> int:
    """kokoro/pipeline.py: KPipeline.waterfall_last."""
    for group in WATERFALL:
        z = next((i for i, t in reversed(list(enumerate(tokens))) if t.phonemes in set(group)),
                 None)
        if z is None:
            continue
        z += 1
        if z < len(tokens) and tokens[z].phonemes in WATERFALL_BUMPS:
            z += 1
        if next_count - len(_tokens_to_ps(tokens[:z])) <= MAX_PS_CHARS:
            return z
    return len(tokens)


def en_chunks(tokens: list[Token]) -> list[str]:
    """kokoro/pipeline.py: KPipeline.en_tokenize, reduced to the phoneme strings.

    The chunk boundaries matter beyond length: each chunk is wrapped in its own
    <bos>/<eos> when the ids are built, which is how the training packs were
    laid out.
    """
    chunks: list[str] = []
    held: list[Token] = []
    pcount = 0
    for token in tokens:
        if token.phonemes is None:
            token.phonemes = ""
        next_ps = token.phonemes + (" " if token.whitespace else "")
        next_pcount = pcount + len(next_ps.rstrip())
        if next_pcount > MAX_PS_CHARS:
            z = _waterfall_last(held, next_pcount)
            ps = _tokens_to_ps(held[:z])
            if ps:
                chunks.append(ps)
            held = held[z:]
            pcount = len(_tokens_to_ps(held))
            if not held:
                next_ps = next_ps.lstrip()
        held.append(token)
        pcount += len(next_ps)
    if held:
        ps = _tokens_to_ps(held)
        if ps:
            chunks.append(ps)
    return chunks


_SHARED_G2P: NanoG2P | None = None


def shared_g2p() -> NanoG2P:
    """One process-wide G2P.

    Building it reads 6.1 MB of JSON into 365,368 dict entries (misaki's
    `grow_dictionary` adds the capitalised and lower-cased variants), which
    takes about 0.2 s and 68 MB of resident memory. Worth doing once.
    """
    global _SHARED_G2P
    if _SHARED_G2P is None:
        _SHARED_G2P = NanoG2P()
    return _SHARED_G2P


def phonemize(text: str, *, vocabulary: dict[str, int] | None = None,
              max_tokens: int = DEFAULT_MAX_TOKENS,
              g2p: NanoG2P | None = None) -> tuple[list[int], str]:
    """Text -> (phoneme ids, dropped symbols), with no espeak-ng anywhere.

    Drop-in for `nano_frontend.phonemize` apart from the `voice` argument, which
    only ever selected an espeak language and has no meaning here: these
    dictionaries are American English.
    """
    if not isinstance(text, str) or not text.strip():
        raise FrontendError("empty", "text must be a non-empty string")
    engine = g2p if g2p is not None else shared_g2p()
    _result, tokens = engine(text)
    chunks = en_chunks(tokens)
    if not chunks:
        raise FrontendError("empty", f"phonemization produced no symbols for text: {text!r}")
    ids: list[int] = []
    dropped: list[str] = []
    for chunk in chunks:
        if len(chunk) > MAX_PS_CHARS:
            chunk = chunk[:MAX_PS_CHARS]
        # The per-chunk cap is disabled (a chunk can never exceed its own length
        # plus BOS/EOS) so that the real limit is checked once, on the whole
        # sequence, and reports the number the caller cares about.
        chunk_ids, chunk_dropped = phonemes_to_token_ids(
            chunk, vocabulary, max_tokens=len(chunk) + 2
        )
        ids.extend(chunk_ids)
        dropped.append(chunk_dropped)
    if len(ids) > max_tokens:
        raise FrontendError(
            "too_long",
            f"phoneme sequence has {len(ids)} tokens including BOS/EOS; maximum is {max_tokens}",
        )
    return ids, "".join(dropped)


def phonemize_to_string(text: str, *, g2p: NanoG2P | None = None) -> str:
    """The phoneme string alone, for comparing against misaki without the ids."""
    if not isinstance(text, str) or not text.strip():
        raise FrontendError("empty", "text must be a non-empty string")
    engine = g2p if g2p is not None else shared_g2p()
    return "".join(en_chunks(engine(text)[1]))
