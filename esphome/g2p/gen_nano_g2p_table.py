#!/usr/bin/env python3
"""Generate `nano_g2p_table.h` -- the e12nano 62-symbol id table and the
espeak-IPA -> misaki-character rewrite table, as C data.

WHY THIS EXISTS
---------------
`mcu/ports/esp32s3/firmware/main/cp_id_table.h` is a 157-entry Piper/Kristin
table. It is the WRONG lineage for the e12nano: that voice is a Kokoro
`af_heart` distillation whose token vocabulary is the 62-entry corpus vocab
built by `tools/kokoro_extract_front.py`, documented in the module docstring of
`tools/make_pack_from_text.py` and frozen in
`artifacts/kokoro-corpus-af_heart-20260713/kokoro_vocab.json`.

WHERE THE REWRITE RULES COME FROM
---------------------------------
They are NOT invented here. They are transcribed from the two implementations
the repo already ships and already measures:

  * `pypkg/sanotts/nano_frontend.py`  -- `E2M` + `apply_e2m()`, the front end
    the `sanotts` package ships and the one scored in
    `experiments/evidence/nano-frontend-ab-20260904.json`;
  * `web/trellis_frontend.js`         -- the browser twin, whose header records
    that its `E2M` was dumped from misaki itself with
    `python -c "from misaki.espeak import EspeakFallback; print(EspeakFallback.E2M)"`.

Both files are hashed into the generated header, so a drift in either is
visible as a hash mismatch rather than as silent phonetic damage.

The full contract those two implement, and that `nano_g2p.c` reproduces, is:

    text
      -> phonemizer Punctuation.preserve      (punctuation split out)
      -> espeak-ng 1.52.0 IPA *with U+0361 ties*
      -> phonemizer EspeakBackend._postprocess_line   (tie U+0361 -> '^')
      -> phonemizer Punctuation.restore
      -> misaki EspeakFallback E2M + tail rewrites    (apply_e2m)
      -> 62-symbol vocabulary filter, <bos> ... <eos>

The tie is load-bearing. E2M matches on "a^ɪ", "e^ɪ", "d^ʒ", "t^ʃ", "ɔ^ɪ", so
if espeak is asked for plain IPA (no tie) NOT ONE diphthong rule fires and the
model is fed `ɪ`/`ʊ`/`ʃ`/`ʒ` where it was trained on `I`/`W`/`A`/`Y`/`ʧ`/`ʤ`.
That is the single largest failure mode of a naive port, and it is exactly what
`experiments/evidence/espeak-vs-misaki-frontend-20260822.json` measured: raw,
untied espeak IPA confuses misaki `I`->espeak `ɪ` 251 times and `A`->`ɪ` 230
times over 249 rows.

USAGE
-----
    python3 esphome/g2p/gen_nano_g2p_table.py            # writes the header
    python3 esphome/g2p/gen_nano_g2p_table.py --check    # verify it is current

No heavy dependencies: standard library only. Safe to run on the 18 GB host.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
# Two identical copies, written by the same render so they cannot drift:
#   * the canonical one, next to this script;
#   * one inside the IDF component, because ESPHome's add_idf_component() copies
#     the component directory alone -- a `#include "../../nano_g2p_table.h"`
#     would resolve here but not in `.esphome/build/<name>/`.
OUT_PATHS = [
    _HERE / "nano_g2p_table.h",
    _HERE / "idf" / "espeak_ng" / "nano_g2p_table.h",
]

VOCAB_PATH = ROOT / "artifacts" / "kokoro-corpus-af_heart-20260713" / "kokoro_vocab.json"

# Files the rewrite table is transcribed from. Hashed into the header so that a
# change to either is a visible mismatch, not a silent phonetic regression.
PROVENANCE_PATHS = [
    VOCAB_PATH,
    ROOT / "pypkg" / "sanotts" / "nano_frontend.py",
    ROOT / "web" / "trellis_frontend.js",
    ROOT / "src" / "saanotts" / "kokoro_frontend.py",
]

SPECIAL_TOKENS = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
EXPECTED_VOCAB_SIZE = 62

SYLLABIC = "̩"  # COMBINING VERTICAL LINE BELOW
NASAL = "̃"  # COMBINING TILDE

# --------------------------------------------------------------------------
# misaki EspeakFallback.E2M, in the order python produces it (sorted by
# -len(key), stable over insertion order). Transcribed verbatim from
# pypkg/sanotts/nano_frontend.py; identical list in web/trellis_frontend.js.
# ORDER IS PART OF THE CONTRACT: "e^ɪ" must be tried before the bare "e".
# --------------------------------------------------------------------------
E2M: list[tuple[str, str]] = [
    ("ʔˌ" + "n" + SYLLABIC, "ʔn"),
    ("ʔn" + SYLLABIC, "ʔn"),
    ("a^ɪ", "I"),
    ("a^ʊ", "W"),
    ("d^ʒ", "ʤ"),
    ("e^ɪ", "A"),
    ("t^ʃ", "ʧ"),
    ("ɔ^ɪ", "Y"),
    ("ə^l", "ᵊl"),
    ("ʲo", "jo"),
    ("ʲə", "jə"),
    ("e", "A"),
    ("ʲ", ""),
    ("ɚ", "əɹ"),
    ("r", "ɹ"),
    ("x", "k"),
    ("ç", "k"),
    ("ɐ", "ə"),
    ("ɬ", "l"),
    (NASAL, ""),
]

# apply_e2m()'s tail, AFTER the syllabic regex. Also order-sensitive:
# "ɜːɹ" before "ɜː", and the bare "ː" strip after both.
E2M_TAIL: list[tuple[str, str]] = [
    ("o^ʊ", "O"),
    ("ɜːɹ", "ɜɹ"),
    ("ɜː", "ɜɹ"),
    ("ɪə", "iə"),
    ("ː", ""),
    ("o", "ɔ"),  # espeak < 1.52 leaves a bare 'o'
    ("ɾ", "T"),  # misaki: version != '2.0'
    ("ʔ", "t"),
    ("^", ""),  # drop the tie last, after every tie-matching rule
]

# The punctuation marks the SHIPPED chain uses. This is
# `pypkg/sanotts/frontend.py:60`'s PUNCTUATION_MARKS, which nano_frontend.py
# hands to EspeakBackend(punctuation_marks=...) -- NOT phonemizer's own
# Punctuation._DEFAULT_MARKS (";:,.!?¡¿—…\"«»“”(){}[]"), and NOT the
# DEFAULT_MARKS constant in web/trellis_frontend.js, which copied phonemizer's
# defaults instead of the package's override. The two sets disagree, so the
# browser port and the Python package do not split punctuation identically.
# The Python package is the reference here: it is what produced every recorded
# `shipped-espeak.jsonl` row, and it is what the ESPHome component must match.
PUNCTUATION_MARKS = "!'(),-.:;?\""


class GeneratorError(RuntimeError):
    """Raised for any condition that would produce a wrong header."""


def sha256_of(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise GeneratorError(f"cannot hash {path}: {exc}") from exc


def load_vocabulary(path: Path) -> dict[str, int]:
    """Read and validate the frozen 62-entry corpus vocabulary."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GeneratorError(f"cannot read vocabulary {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GeneratorError(f"vocabulary {path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise GeneratorError("vocabulary must be a non-empty JSON object")
    for symbol, index in raw.items():
        if not isinstance(symbol, str) or not isinstance(index, int) or isinstance(index, bool):
            raise GeneratorError(f"vocabulary entry {symbol!r} -> {index!r} is not str -> int")
    if len(raw) != EXPECTED_VOCAB_SIZE:
        raise GeneratorError(
            f"vocabulary has {len(raw)} entries, expected {EXPECTED_VOCAB_SIZE}; "
            "the e12nano checkpoints declare vocab_size 62"
        )
    for token, expected in SPECIAL_TOKENS.items():
        if raw.get(token) != expected:
            raise GeneratorError(f"vocabulary must map {token} -> {expected}")
    ids = sorted(raw.values())
    if ids != list(range(len(raw))):
        raise GeneratorError("vocabulary ids must be unique and contiguous from zero")

    for symbol in raw:
        if symbol in SPECIAL_TOKENS:
            continue
        if len(symbol) != 1:
            raise GeneratorError(
                f"symbol {symbol!r} is not a single Unicode codepoint; the on-device "
                "tokenizer is codepoint-indexed and cannot represent it"
            )
    return raw


def c_escape_utf8(text: str) -> str:
    """A C string literal for `text`, escaped byte-by-byte as UTF-8.

    Hex-escaping every non-ASCII byte keeps the header ASCII-clean and immune
    to a compiler with a non-UTF-8 default source charset. `"" ` between a hex
    escape and a following literal digit stops C's greedy hex-escape parse.
    """
    out: list[str] = []
    prev_was_hex = False
    for byte in text.encode("utf-8"):
        char = chr(byte)
        if byte < 0x20 or byte >= 0x7F:
            out.append(f"\\x{byte:02x}")
            prev_was_hex = True
            continue
        if prev_was_hex and char in "0123456789abcdefABCDEF":
            out.append('" "')
        if char in ('"', "\\"):
            out.append("\\" + char)
        else:
            out.append(char)
        prev_was_hex = False
    return "".join(out)


def rule_comment(needle: str, replacement: str) -> str:
    """A readable `U+XXXX -> U+YYYY` comment for a rewrite rule."""

    def describe(text: str) -> str:
        if text == "":
            return "(delete)"
        return " ".join(f"U+{ord(ch):04X}" for ch in text)

    return f"{describe(needle)} -> {describe(replacement)}"


def render_header(vocabulary: dict[str, int]) -> str:
    phoneme_entries = sorted(
        ((ord(symbol), index) for symbol, index in vocabulary.items() if symbol not in SPECIAL_TOKENS),
        key=lambda pair: pair[0],
    )
    if len(phoneme_entries) != EXPECTED_VOCAB_SIZE - len(SPECIAL_TOKENS):
        raise GeneratorError(
            f"expected {EXPECTED_VOCAB_SIZE - len(SPECIAL_TOKENS)} codepoint symbols, "
            f"got {len(phoneme_entries)}"
        )
    codepoints = [cp for cp, _ in phoneme_entries]
    if len(set(codepoints)) != len(codepoints):
        raise GeneratorError("duplicate codepoint in vocabulary")
    if codepoints != sorted(codepoints):
        raise GeneratorError("internal error: table is not sorted")

    generated = datetime.date.today().isoformat()
    lines: list[str] = []
    add = lines.append

    add("/* nano_g2p_table.h -- GENERATED, DO NOT EDIT BY HAND.")
    add(" *")
    add(" * Regenerate with:   python3 esphome/g2p/gen_nano_g2p_table.py")
    add(" * Verify with:       python3 esphome/g2p/gen_nano_g2p_table.py --check")
    add(f" * Generated:         {generated}")
    add(" *")
    add(" * The 62-symbol token vocabulary of the `en_us_e12nano` voice (294,642")
    add(" * params, Kokoro af_heart lineage) plus the espeak-IPA -> misaki-character")
    add(" * rewrite table needed to reach that alphabet from raw espeak-ng output.")
    add(" *")
    add(" * This is NOT the 157-entry Piper table in")
    add(" * mcu/ports/esp32s3/firmware/main/cp_id_table.h -- that one belongs to the")
    add(" * Kristin lineage and is wrong for this voice.")
    add(" *")
    add(" * SOURCES (sha256 at generation time):")
    for path in PROVENANCE_PATHS:
        add(f" *   {path.relative_to(ROOT).as_posix()}")
        add(f" *     {sha256_of(path)}")
    add(" *")
    add(" * The rewrite rules are transcribed from nano_frontend.py's E2M table and")
    add(" * apply_e2m() tail, which web/trellis_frontend.js records as a verbatim dump")
    add(" * of misaki.espeak.EspeakFallback.E2M. They are not reconstructed from")
    add(" * memory and must not be 'improved': they are the model's input contract.")
    add(" */")
    add("")
    add("#ifndef NANO_G2P_TABLE_H")
    add("#define NANO_G2P_TABLE_H")
    add("")
    add("#include <stdint.h>")
    add("")
    add("/* Special tokens. Not codepoints, so they are not in NANO_CP_ID. */")
    add("#define NANO_ID_PAD 0")
    add("#define NANO_ID_BOS 1")
    add("#define NANO_ID_EOS 2")
    add("")
    add(f"#define NANO_VOCAB_SIZE {EXPECTED_VOCAB_SIZE}")
    add("")
    add("/* configs.front.max_tokens, including <bos> and <eos>. Mirrors")
    add(" * nano_frontend.DEFAULT_MAX_TOKENS / trellis_frontend.js DEFAULT_MAX_TOKENS. */")
    add("#define NANO_MAX_TOKENS 207")
    add("")
    add("typedef struct {")
    add("    uint32_t cp;   /* Unicode codepoint */")
    add("    int16_t  id;   /* token id in the 62-entry corpus vocabulary */")
    add("} nano_cp_id_t;")
    add("")
    add(f"/* {len(phoneme_entries)} single-codepoint symbols, sorted by codepoint for binary search.")
    add(" * ASCII 'A I O T W Y' are misaki's diphthong/flap letters, not letters:")
    add(" *   A = /eɪ/   I = /aɪ/   O = /oʊ/   W = /aʊ/   Y = /ɔɪ/   T = alveolar flap")
    add(" * U+1D4A MODIFIER LETTER SMALL SCHWA and U+1D7B SMALL CAPITAL I WITH STROKE")
    add(" * are misaki-only symbols with no standard IPA spelling. */")
    add(f"static const nano_cp_id_t NANO_CP_ID[{len(phoneme_entries)}] = {{")
    for cp, index in phoneme_entries:
        char = chr(cp)
        shown = repr(char)[1:-1] if char.isprintable() else "?"
        add(f"    {{ 0x{cp:04X}, {index:2d} }},  /* U+{cp:04X}  '{shown}' */")
    add("};")
    add(f"#define NANO_CP_ID_COUNT {len(phoneme_entries)}")
    add("")
    add("typedef struct {")
    add("    const char *from;   /* UTF-8 needle */")
    add("    const char *to;     /* UTF-8 replacement, may be empty */")
    add("} nano_rewrite_t;")
    add("")
    add("/* espeak's tie, requested from espeak_TextToPhonemes() via")
    add(" * (espeakPHONEMES_IPA | espeakPHONEMES_TIE | (0x0361 << 8)); phonemizer")
    add(" * rewrites it to '^' before misaki sees it, and every diphthong rule below")
    add(" * matches on '^'. Ask espeak for untied IPA and none of them fire. */")
    add('#define NANO_TIE_ESPEAK "\\xcd\\xa1"   /* U+0361 COMBINING DOUBLE INVERTED BREVE */')
    add('#define NANO_TIE_MISAKI "^"')
    add("#define NANO_TIE_CODEPOINT 0x0361u")
    add("")
    add('#define NANO_SYLLABIC "\\xcc\\xa9"     /* U+0329 COMBINING VERTICAL LINE BELOW */')
    add("#define NANO_SYLLABIC_CODEPOINT 0x0329u")
    add('#define NANO_SCHWA_MODIFIER "\\xe1\\xb5\\x8a"  /* U+1D4A, the regex replacement */')
    add("")
    add("/* misaki EspeakFallback.E2M, applied in order, each as a full")
    add(" * left-to-right non-overlapping pass. Order is load-bearing. */")
    add(f"static const nano_rewrite_t NANO_E2M[{len(E2M)}] = {{")
    for needle, replacement in E2M:
        add(
            f'    {{ "{c_escape_utf8(needle)}", "{c_escape_utf8(replacement)}" }},'
            f"  /* {rule_comment(needle, replacement)} */"
        )
    add("};")
    add(f"#define NANO_E2M_COUNT {len(E2M)}")
    add("")
    add("/* apply_e2m()'s tail, applied AFTER the syllabic regex rewrite. */")
    add(f"static const nano_rewrite_t NANO_E2M_TAIL[{len(E2M_TAIL)}] = {{")
    for needle, replacement in E2M_TAIL:
        add(
            f'    {{ "{c_escape_utf8(needle)}", "{c_escape_utf8(replacement)}" }},'
            f"  /* {rule_comment(needle, replacement)} */"
        )
    add("};")
    add(f"#define NANO_E2M_TAIL_COUNT {len(E2M_TAIL)}")
    add("")
    add("/* pypkg/sanotts/frontend.py PUNCTUATION_MARKS, as codepoints.")
    add(" * phonemizer splits these out of the text BEFORE espeak sees it and splices")
    add(" * them back into the phoneme string afterwards. Skip that and every comma")
    add(" * and full stop -- which this vocabulary has ids for and the model was")
    add(" * trained to pause on -- is lost.")
    add(" *")
    add(" * This is the package's OVERRIDE, not phonemizer's default mark set and not")
    add(" * the DEFAULT_MARKS in web/trellis_frontend.js. Note it includes the")
    add(" * apostrophe and the hyphen, so \"it's\" and \"twenty-two\" are split into")
    add(" * separate espeak calls. That is measured behaviour of the shipped front")
    add(" * end, not an oversight to correct here. */")
    marks = sorted({ord(ch) for ch in PUNCTUATION_MARKS})
    add(f"static const uint32_t NANO_PUNCT_MARKS[{len(marks)}] = {{")
    for start in range(0, len(marks), 8):
        chunk = marks[start : start + 8]
        add("    " + " ".join(f"0x{cp:04X}," for cp in chunk))
    add("};")
    add(f"#define NANO_PUNCT_MARK_COUNT {len(marks)}")
    add("")
    add("#endif /* NANO_G2P_TABLE_H */")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed header differs from a fresh render",
    )
    parser.add_argument(
        "--out",
        type=Path,
        action="append",
        default=None,
        help="header path to write (repeatable); defaults to both tracked copies",
    )
    args = parser.parse_args()
    targets = args.out if args.out else list(OUT_PATHS)

    try:
        vocabulary = load_vocabulary(VOCAB_PATH)
        rendered = render_header(vocabulary)
    except GeneratorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # The generation date legitimately differs between renders; compare the rest
    # so --check catches real drift and not the calendar.
    def strip_date(text: str) -> str:
        return "\n".join(
            line for line in text.splitlines() if not line.startswith(" * Generated:")
        )

    if args.check:
        failed = False
        for target in targets:
            try:
                current = target.read_text(encoding="utf-8")
            except OSError as exc:
                print(f"error: cannot read {target} for --check: {exc}", file=sys.stderr)
                return 2
            if strip_date(current) != strip_date(rendered):
                print(f"error: {target} is out of date; re-run without --check", file=sys.stderr)
                failed = True
            else:
                print(f"ok: {target} matches the sources")
        return 1 if failed else 0

    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot write {target}: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {target} ({len(rendered)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
