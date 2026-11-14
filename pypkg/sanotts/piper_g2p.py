"""Text -> Piper phoneme ids for the piperlite voices, without espeak-ng.

This module is the entry point and the English half of the work.
``text_to_phoneme_ids`` dispatches on the voice's own espeak voice string:
English voices go through the misaki-lexicon mapping documented below,
``id`` through ``sanotts.id_g2p`` and ``vi`` through ``sanotts.vi_g2p``, which
are rule front ends written from Indonesian and Vietnamese orthography rather
than from any dictionary. An unknown language raises instead of falling back to
English.

The piperlite voices (amy, kristin, hfc, id, vi) are distilled from Piper/VITS
teachers whose own front end is espeak-ng, so the default path stays
``frontend.py``: espeak-ng through phonemizer-fork, then piper's exact
``phonemes_to_ids`` framing. This module is a *second, selectable* path that
reaches the same phoneme-id space from the espeak-free lexicon front end built
for the nano voices (``nano_g2p.py`` -- misaki's Apache-2.0 dictionaries plus a
numpy out-of-vocabulary model).

Two encodings have to be reconciled.

``nano_g2p`` emits **misaki's compressed alphabet**: one codepoint per
diphthong or affricate (``A``=eɪ, ``I``=aɪ, ``O``=oʊ, ``W``=aʊ, ``Y``=ɔɪ,
``ʤ``=dʒ, ``ʧ``=tʃ, ``T``=ɾ), no length marks at all, ``ᵊ`` for a syllabic
schwa and ``ᵻ`` for the ROSES vowel. A piperlite ``phoneme_id_map`` instead
wants **the codepoints espeak-ng actually printed**, NFD-decomposed: ``eɪ`` as
two symbols, ``iː``/``uː``/``ɑː``/``ɔː``/``ɜː`` with the length mark, ``ɚ`` as
one codepoint.

``nano_frontend.E2M`` is the forward direction of that map (espeak -> misaki).
It is a sequential list of ``str.replace`` calls, and it is lossy in three
places, which is why the inverse here is a fresh longest-match table and not a
reversed ``E2M``:

* **Length is destroyed.** ``E2M`` ends with ``.replace("ː", "")``. Four of the
  five long vowels can be restored exactly, because espeak-ng en-us never
  prints them short -- over the 330-sentence development set ``u``, ``ɑ``,
  ``ɜ`` and bare ``e``/``o`` occur 0 times without a following ``ː`` or
  off-glide. ``i`` is the exception and gets the context rule in
  ``_long_i_positions``.
* **``ɐ`` and ``ə`` are merged** by ``("ɐ", "ə")``. Nothing can undo that from
  the misaki side -- but nothing has to: misaki's own dictionaries distinguish
  the two, so the lexicon path emits ``ɐ`` where it means ``ɐ`` and the symbol
  passes straight through.
* **``ɚ`` becomes ``əɹ``** and **``ɜː``/``ɜːɹ`` both become ``ɜɹ``**. A
  pre-vocalic guard recovers the first exactly (379/379 on the development set)
  and the second 165 times in 176; the residue is the 11 word-final ``ɜːɹ``,
  which is not separable from ``ɜː`` on the misaki side.

Symbols with no home in a given voice's table are collected and returned to the
caller rather than dropped in silence; ``coverage_report`` states the same thing
statically for a whole voice.

Measured end to end in
``experiments/evidence/piperlite-espeak-free-ab-20260904.json``.
"""

from __future__ import annotations

import logging
import re
import unicodedata

import numpy as np

from . import frontend, nano_g2p

logger = logging.getLogger("sanotts.piper_g2p")


class PiperG2PError(RuntimeError):
    """`kind` separates an expected rejection (empty) from a bug."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# Punctuation espeak-ng keeps (frontend.PUNCTUATION_MARKS) written the way
# misaki writes it. misaki's tokenizer turns a straight double quote into a
# curly one by position, and both are the same espeak token.
_PUNCT_REWRITE: dict[str, str] = {
    "“": '"',   # LEFT DOUBLE QUOTATION MARK
    "”": '"',   # RIGHT DOUBLE QUOTATION MARK
    "‘": "'",   # LEFT SINGLE QUOTATION MARK
    "’": "'",   # RIGHT SINGLE QUOTATION MARK
}

# Punctuation misaki voices that espeak-ng drops on the floor. Verified by
# running the shipped frontend on probe sentences: an em dash, a horizontal
# ellipsis and curly quotes all vanish from espeak's output, so emitting
# nothing here is what *matches* the baseline rather than what loses to it.
_PUNCT_DROPPED: frozenset[str] = frozenset("—–…")

# Longest-match first. A single left-to-right scan rather than a chain of
# str.replace calls, so no rule can consume the output of another.
#
# A rule with ``before_vowel=False`` is skipped when the next symbol is a vowel.
# Both such rules restore an espeak symbol that only exists in a coda: over the
# 330-sentence development set espeak-ng printed ``ɚ`` before a vowel 0 times in
# 376, and plain ``ɜː`` before a vowel 0 times in 159, while all 3 of its
# genuine ``əɹ`` sequences (``pˈiəɹɪəd``) are exactly the pre-vocalic case. The
# guard makes ``əɹ`` exact on that set and stops ``ɜɹ`` from emitting a bigram
# espeak never emits.
M2E_RULES: tuple[tuple[str, str, bool], ...] = (
    # -- two-symbol misaki sequences that were one espeak codepoint, or that
    #    lost a length mark between them.
    ("ɜɹ", "ɜː", False),        # ɜɹ -> ɜː  (159 vs 17 for ɜːɹ)
    ("əɹ", "ɚ", False),         # əɹ -> ɚ   (376 vs 3 for a real əɹ)
    # -- misaki's one-codepoint diphthongs and affricates.
    ("A", "eɪ", True),
    ("I", "aɪ", True),
    ("O", "oʊ", True),
    ("W", "aʊ", True),
    ("Y", "ɔɪ", True),
    ("ʤ", "dʒ", True),
    ("ʧ", "tʃ", True),
    ("T", "ɾ", True),           # the flap
    # -- misaki's syllabic marker. espeak prints "ən" 241 times and "n̩" 11
    #    times on the development set, so the plain schwa is the majority form.
    ("ᵊ", "ə", True),
    # -- length restored where espeak-ng en-us never prints the vowel short:
    #    bare u, ɑ and ɜ occur 0 times without the length mark, and bare ɔ only
    #    inside ɔɪ, which "Y" above has already consumed.
    ("u", "uː", True),
    ("ɑ", "ɑː", True),
    ("ɔ", "ɔː", True),
    ("ɜ", "ɜː", True),
)

_MAX_RULE_LEN = max(len(key) for key, _, _ in M2E_RULES)
_RULE_MAP: dict[str, tuple[str, bool]] = {
    key: (replacement, before_vowel) for key, replacement, before_vowel in M2E_RULES
}

_STRESS_MARKS = "ˈˌ"          # ˈ ˌ
# A misaki token's phonemes end where its trailing punctuation begins.
_TRAILING_PUNCT = ".,;:!?\")'”’—…"
# misaki's vowel set plus ɐ, which the article/"am"/"an" special cases emit
# directly and which is outside nano_g2p.US_VOCAB.
_VOWELS = frozenset(nano_g2p.VOWELS) | {"ɐ"}
_MULTI_SPACE = re.compile(r" {2,}")


def _long_i_positions(token: str) -> set[int]:
    """Which ``i`` in one whitespace-delimited misaki token were ``iː``.

    ``E2M`` strips every length mark, so misaki's ``i`` covers both espeak's
    FLEECE ``iː`` (399 occurrences on the 330-sentence development set) and its
    unstressed happy vowel ``i`` (262). The two are separable by position:
    espeak prints ``iː`` in the stressed syllable and in stressed monosyllables,
    and bare ``i`` everywhere else.

    * ``i`` directly before ``ə`` is short. espeak writes ``pˈiəɹɪəd``,
      ``ɪtˈæliən``; the trigram ``iːə`` does not occur at all on the
      development set, so lengthening here would emit a sequence the teacher
      never saw.
    * with a stress mark earlier in the token, the ``i`` is long only when it is
      the nucleus of that stressed syllable -- no other vowel stands between the
      mark and it. ``sˈiː`` and ``ɪmˈiːdɪətli`` qualify; the ``-y``/``-ly``/
      ``-ily`` vowels of ``mˈɪɹli``, ``ɹˈɛdili``, ``bjˈuːɾifəl`` do not.
    * with the token's only stress mark *after* the ``i``, the ``i`` is a
      pre-tonic full vowel and long (``kɹiːˈeɪt``).
    * with no stress mark at all the token is an unstressed function word:
      long if the ``i`` is its only vowel (``biː``, ``hiː``), short otherwise
      (``təbi`` for "to be").

    Measured on the development set with the G2P held fixed -- the round-trip
    control, where the same espeak call produced both sides -- this is wrong
    about 7 of 638 words containing ``i``. The simpler "token-final and
    unstressed" rule it replaced was wrong about 34.
    """
    core = token.rstrip(_TRAILING_PUNCT)
    long_positions: set[int] = set()
    for index, char in enumerate(token):
        if char != "i":
            continue
        if token[index + 1:index + 2] == "ə":
            continue
        prior = token[:index]
        last_stress = max((i for i, c in enumerate(prior) if c in _STRESS_MARKS), default=-1)
        if last_stress >= 0:
            between = token[last_stress + 1:index]
            if not any(c in _VOWELS for c in between):
                long_positions.add(index)
        elif any(c in _STRESS_MARKS for c in token[index + 1:]):
            long_positions.add(index)
        elif not any(c in _VOWELS for j, c in enumerate(core) if j != index):
            long_positions.add(index)
    return long_positions


def misaki_to_espeak_ipa(phoneme_string: str) -> str:
    """misaki's compressed alphabet -> the IPA codepoints espeak-ng prints.

    Whitespace and punctuation pass through; the return value is *not* yet
    NFD-normalized or filtered against any voice table.
    """
    if not isinstance(phoneme_string, str):
        raise PiperG2PError("type", f"phoneme_string must be str, got {type(phoneme_string)!r}")

    out: list[str] = []
    for token in phoneme_string.split(" "):
        if not token:
            out.append("")
            continue
        long_i = _long_i_positions(token)
        index = 0
        pieces: list[str] = []
        while index < len(token):
            matched = False
            for width in range(min(_MAX_RULE_LEN, len(token) - index), 0, -1):
                rule = _RULE_MAP.get(token[index:index + width])
                if rule is None:
                    continue
                replacement, before_vowel = rule
                if not before_vowel and token[index + width:index + width + 1] in _VOWELS:
                    continue
                pieces.append(replacement)
                index += width
                matched = True
                break
            if matched:
                continue
            char = token[index]
            if char == "i":
                pieces.append("iː" if index in long_i else "i")
            elif char in _PUNCT_REWRITE:
                pieces.append(_PUNCT_REWRITE[char])
            elif char in _PUNCT_DROPPED:
                pass
            else:
                pieces.append(char)
            index += 1
        out.append("".join(pieces))
    # A token that mapped to nothing -- an em dash, an ellipsis -- would leave a
    # doubled space behind, and espeak-ng never emits one; the space is an id in
    # every voice's table, so the collapse is not cosmetic.
    return _MULTI_SPACE.sub(" ", " ".join(out)).strip()


def phonemize_to_espeak_ipa(text: str, *, g2p: nano_g2p.NanoG2P | None = None) -> str:
    """Text -> the espeak-shaped IPA string, with no espeak-ng anywhere."""
    if not isinstance(text, str) or not text.strip():
        raise PiperG2PError("empty", "text must be a non-empty string")
    engine = g2p if g2p is not None else nano_g2p.shared_g2p()
    chunks = nano_g2p.en_chunks(engine(text)[1])
    if not chunks:
        raise PiperG2PError("empty", f"phonemization produced no symbols for text: {text!r}")
    return misaki_to_espeak_ipa(" ".join(chunks))


def _symbols(ipa: str) -> list[str]:
    """The codepoint list piper's phonemes_to_ids consumes."""
    return list(unicodedata.normalize("NFD", ipa))


def english_text_to_phoneme_ids(
    text: str,
    table: frontend.PhonemeTable,
    *,
    g2p: nano_g2p.NanoG2P | None = None,
) -> tuple[np.ndarray, str]:
    """The English path: misaki's dictionaries, mapped back into espeak IPA.

    Returns ``(ids, unmapped)`` where ``unmapped`` is every symbol this voice's
    ``phoneme_id_map`` has no entry for, in order of occurrence. The baseline
    espeak path logs and discards those; here they are handed back so a caller
    can refuse or report rather than guess.
    """
    ipa = phonemize_to_espeak_ipa(text, g2p=g2p)
    symbols = _symbols(ipa)
    unmapped = "".join(symbol for symbol in symbols if symbol not in table.id_map)
    ids = frontend.phonemes_to_ids(symbols, table)
    if len(ids) <= 3:
        raise PiperG2PError("empty", f"phonemization produced no usable phonemes for: {text!r}")
    return np.asarray(ids, dtype=np.int64), unmapped


# Which espeak voice each espeak-free path stands in for. The key is the
# ``espeak.voice`` string in the voice's own piper-phoneme-config.json, which
# is what actually selects the language -- not the alias in tables/voices.json.
# ``kristin`` asks for a bare ``en``, which newer espeak-ng builds no longer
# expose, so the English entry covers both spellings.
ENGLISH_VOICES: frozenset[str] = frozenset({"en", "en-us", "en-gb"})


# Which Indonesian front end ``path`` selects. Only ``id`` has more than one.
#   "lexicon"  id_g2p        -- espeak-ng's own behaviour, reproduced by rule.
#   "indo"     id_indo_bridge -- the indo-g2p port, which fixes the schwa and
#                               the glottal stops and is therefore a token
#                               distribution the voice was not distilled on.
ID_PATHS: frozenset[str] = frozenset({"lexicon", "indo"})


def module_for(table: frontend.PhonemeTable, path: str = "lexicon"):
    """The espeak-free module that serves this voice, or raise.

    Refusing an unknown language is the point: silently falling back to the
    English lexicon would read Indonesian as English, which is exactly the
    failure mode a MOS score cannot see.

    ``path`` picks between the Indonesian front ends and is ignored for every
    other language, which has only one. An unknown value raises rather than
    quietly selecting the default.
    """
    if path not in ID_PATHS:
        raise PiperG2PError("path", f"path must be one of {sorted(ID_PATHS)}, got {path!r}")
    voice = table.espeak_voice
    if voice in ENGLISH_VOICES:
        return None
    if voice == "id":
        if path == "indo":
            from . import id_indo_bridge  # noqa: PLC0415
            return id_indo_bridge
        from . import id_g2p  # noqa: PLC0415
        return id_g2p
    if voice == "vi":
        from . import vi_g2p  # noqa: PLC0415
        return vi_g2p
    raise PiperG2PError(
        "language",
        f"no espeak-free front end for espeak voice {voice!r}; "
        f"known: {sorted(ENGLISH_VOICES)} + ['id', 'vi']",
    )


def text_to_phoneme_ids(
    text: str,
    table: frontend.PhonemeTable,
    *,
    g2p: nano_g2p.NanoG2P | None = None,
    path: str = "lexicon",
) -> tuple[np.ndarray, str]:
    """Drop-in for ``frontend.text_to_phoneme_ids`` with an extra return value.

    Dispatches on the voice's own espeak voice string: American English goes
    through the misaki lexicon above, ``id`` through ``id_g2p`` (or, with
    ``path="indo"``, through the indo-g2p port) and ``vi`` through
    ``vi_g2p``. Every path returns ``(ids, unmapped)`` and drops nothing
    quietly.
    """
    module = module_for(table, path)
    if module is None:
        return english_text_to_phoneme_ids(text, table, g2p=g2p)
    if g2p is not None:
        raise PiperG2PError(
            "language",
            f"the g2p argument is for the English lexicon path only; "
            f"espeak voice {table.espeak_voice!r} does not use it",
        )
    return module.text_to_phoneme_ids(text, table)


def lexicon_alphabet() -> set[str]:
    """Every symbol ``nano_g2p`` can put in a phoneme string.

    Closed, and checked against the assets rather than assumed: the value
    alphabet of ``us_gold.json`` + ``us_silver.json`` and the OOV model's
    ``phoneme_chars`` are both exactly ``nano_g2p.US_VOCAB``. Two things sit
    outside it -- ``ɐ``, which the article/``am``/``an`` special cases emit
    directly, and the punctuation ``Lexicon`` copies through.
    """
    return (
        set(nano_g2p.US_VOCAB)
        | {"ɐ"}
        | set(nano_g2p.PUNCT_TAG_PHONEMES.values())
        | set(nano_g2p.PUNCTS)
    )


def producible_symbols() -> set[str]:
    """Every codepoint this path can emit, NFD-decomposed.

    Enumerated exhaustively rather than reasoned about: no rule in
    ``M2E_RULES`` is longer than two symbols and no context test looks past an
    adjacent symbol, a stress mark earlier in the token, or the token end, so
    running every ordered pair over the closed alphabet in three token shapes
    reaches every reachable output.
    """
    alphabet = sorted(lexicon_alphabet())
    produced: set[str] = set()
    for first in alphabet:
        for second in alphabet:
            for shape in (f"{first}{second}", f"ˈ{first}{second}",
                          f"ˈx{first}{second} y"):
                produced.update(_symbols(misaki_to_espeak_ipa(shape)))
    produced.discard("x")
    produced.discard("y")
    return produced


def coverage_report(table: frontend.PhonemeTable) -> dict[str, object]:
    """What this path can emit for one voice, and what its table refuses.

    Four categories, none of them silent:

    * ``mapped`` -- lexicon symbols whose espeak form is entirely in the table.
    * ``rewritten`` -- lexicon symbols spelled differently on the espeak side
      (misaki's curly quotes are espeak's straight ones).
    * ``dropped_by_design`` -- lexicon symbols that produce nothing, because
      espeak-ng itself produces nothing for them (verified on probe text).
    * ``unmappable`` -- produced espeak symbols with no id in this voice.
    """
    produced = producible_symbols()
    unmappable = sorted(symbol for symbol in produced if symbol not in table.id_map)

    mapped: dict[str, str] = {}
    rewritten: dict[str, str] = {}
    dropped: list[str] = []
    missing: dict[str, str] = {}
    for symbol in sorted(lexicon_alphabet()):
        espeak_form = misaki_to_espeak_ipa(symbol)
        if not espeak_form:
            dropped.append(symbol)
            continue
        pieces = _symbols(espeak_form)
        absent = [piece for piece in pieces if piece not in table.id_map]
        if absent:
            missing[symbol] = espeak_form
        elif espeak_form == symbol:
            mapped[symbol] = espeak_form
        else:
            rewritten[symbol] = espeak_form
    return {
        "espeak_voice": table.espeak_voice,
        "table_size": len(table.id_map),
        "lexicon_alphabet_size": len(lexicon_alphabet()),
        "producible_espeak_symbols": len(produced),
        "identity": sorted(mapped),
        "rewritten": rewritten,
        "dropped_by_design": dropped,
        "unmappable_lexicon_symbols": missing,
        "unmappable_espeak_symbols": unmappable,
        "unmappable_espeak_codepoints": [f"U+{ord(s):04X}" for s in unmappable],
    }
