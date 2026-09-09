#!/usr/bin/env python3
"""Pack misaki's us_gold/us_silver dictionaries into a flash-resident C table.

Regenerate with:   python3 esphome/g2p/lex/gen_lex_tables.py
Verify with:       python3 esphome/g2p/lex/gen_lex_tables.py --check

Why this shape
--------------
`pypkg/sanotts/nano_g2p.py` reads 6.1 MB of JSON into 365,368 Python dict
entries and calls that 68 MB of resident memory. An ESP32-S3 has ~512 KB of
internal SRAM, so the dictionary has to live in flash and be read in place.
That rules out a hash map (needs a RAM-resident bucket array to be worth
anything, and its probe order defeats the flash cache) and it rules out 90,000
C string literals (the linker relocation table alone would be larger than the
data).

What is emitted instead is one sorted, self-delimiting record blob per
dictionary plus a block index, searched by:

    binary search the block index on each block's head key
      -> linear scan of at most NANO_LEX_BLOCK records inside that block

Keys inside a block are front-coded against their predecessor (shared-prefix
length + suffix), which is why the scan is linear rather than a second binary
search: reconstructing key i needs key i-1. Measured on the real data this
costs 8 record decodes per lookup and saves 1.4 MB of flash against a plain
length-prefixed blob with a dense uint32 offset per entry.

Byte encoding
-------------
Every control byte in the blob is biased by '0' (48) and every phoneme is a
small code, also biased by 48, so the whole blob is printable ASCII in
[0x27, 0x76] and can be emitted as ordinary C string literals. That makes the
generated .c about 3.4 MB instead of the 13 MB a `{0x41, 0x42, ...}`
initialiser would need, and it keeps the table readable in a diff. The C side
undoes the bias with one subtraction; see NANO_LEX_UNBIAS.

Record layout (offsets are byte counts into the blob):

    first record of a block:   [48+klen]  key[klen]
    any other record:          [48+shared] [48+suffixlen] suffix[suffixlen]
    then, for every record:    [48+vcode]  payload

    vcode == 0            the entry's value is JSON null
    1 <= vcode <= 60      plain value, length vcode-1, payload is that many
                          phoneme code bytes
    61 <= vcode <= 70     tag-keyed value with vcode-60 variants; payload is
                          that many [48+tagcode] [48+vlen1] phonemes[vlen1]
                          groups, where vlen1 == 0 means the variant is null

`shared` is the number of leading bytes this key has in common with the
previous key in the same block; key bytes themselves are raw ASCII (misaki's
keys use only `'`, `-`, `.`, A-Z and a-z, checked below, all printable and none
of them `"` or `\\`).

Sources, their sha256, the entry counts and the packed byte counts are written
into the generated header so a firmware binary can be traced back to the exact
dictionary revision it was built from.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pypkg"))

from sanotts import nano_g2p as NG  # noqa: E402
from sanotts import nano_frontend as NF  # noqa: E402

GOLD_PATH = ROOT / "pypkg" / "sanotts" / "g2p_data" / "us_gold.json"
SILVER_PATH = ROOT / "pypkg" / "sanotts" / "g2p_data" / "us_silver.json"

BLOCK = 8
BIAS = ord("0")

# Codes 60..62 are phonemes the dictionaries contain but the 62-symbol model
# vocabulary does not: `ɾ` and `ʔ` are rewritten to `T`/`t` by G2P.__call__'s
# tail, and `…` arrives from PUNCTS and is dropped by the id filter. They need
# codes because they exist in intermediate strings, not because they survive.
EXTRA_PHONEMES = ["ɾ", "ʔ", "…"]

# Every Penn tag any part of this port can produce or read, plus the four
# coarse parents get_parent_tag folds them into and the two pseudo-tags the
# gold dictionary uses as variant keys ("None", "DEFAULT").
TAGS = [
    "NONE_TAG",  # the C sentinel for Python's `tag is None`
    "#", "$", "''", ",", "-LRB-", "-RRB-", ".", ":", "``", '""', "NFP", "HYPH",
    "ADD", "CC", "CD", "DT", "EX", "IN", "JJ", "MD", "NN", "NNP", "NNS", "PDT",
    "PRP", "PRP$", "RB", "RBR", "RBS", "TO", "VB", "VBD", "VBG", "VBN", "VBP",
    "VBZ", "WDT", "WP", "WP$", "WRB",
    "ADJ", "ADV", "NOUN", "VERB",
    "None", "DEFAULT",
]

TAG_IDENT = {
    "NONE_TAG": "NLG_TAG_NONE", "#": "NLG_TAG_HASH", "$": "NLG_TAG_DOLLAR",
    "''": "NLG_TAG_CLOSEQ", ",": "NLG_TAG_COMMA", "-LRB-": "NLG_TAG_LRB",
    "-RRB-": "NLG_TAG_RRB", ".": "NLG_TAG_PERIOD", ":": "NLG_TAG_COLON",
    "``": "NLG_TAG_OPENQ", '""': "NLG_TAG_DQUOTE", "NFP": "NLG_TAG_NFP",
    "HYPH": "NLG_TAG_HYPH", "ADD": "NLG_TAG_ADD", "CC": "NLG_TAG_CC",
    "CD": "NLG_TAG_CD", "DT": "NLG_TAG_DT", "EX": "NLG_TAG_EX",
    "IN": "NLG_TAG_IN", "JJ": "NLG_TAG_JJ", "MD": "NLG_TAG_MD",
    "NN": "NLG_TAG_NN", "NNP": "NLG_TAG_NNP", "NNS": "NLG_TAG_NNS",
    "PDT": "NLG_TAG_PDT", "PRP": "NLG_TAG_PRP", "PRP$": "NLG_TAG_PRPS",
    "RB": "NLG_TAG_RB", "RBR": "NLG_TAG_RBR", "RBS": "NLG_TAG_RBS",
    "TO": "NLG_TAG_TO", "VB": "NLG_TAG_VB", "VBD": "NLG_TAG_VBD",
    "VBG": "NLG_TAG_VBG", "VBN": "NLG_TAG_VBN", "VBP": "NLG_TAG_VBP",
    "VBZ": "NLG_TAG_VBZ", "WDT": "NLG_TAG_WDT", "WP": "NLG_TAG_WP",
    "WP$": "NLG_TAG_WPS", "WRB": "NLG_TAG_WRB", "ADJ": "NLG_TAG_ADJ",
    "ADV": "NLG_TAG_ADV", "NOUN": "NLG_TAG_NOUN", "VERB": "NLG_TAG_VERB",
    "None": "NLG_TAG_NONEKEY", "DEFAULT": "NLG_TAG_DEFAULT",
}

MAX_KEY_LEN = 63          # the encoding's ceiling; the real maximum is asserted
MAX_PLAIN_VALUE = 59      # vcode 1..60 encodes length 0..59
MAX_VARIANTS = 10         # vcode 61..70


class GeneratorError(RuntimeError):
    """Anything that would make the generated table wrong. Never swallowed."""


# --------------------------------------------------------------------------
# Phoneme alphabet
# --------------------------------------------------------------------------


def build_phoneme_alphabet() -> tuple[list[str], dict[str, int]]:
    """code 1..59 are the model's own symbols in id order, then the extras.

    Taking the order from `nano_frontend.DEFAULT_VOCABULARY` rather than
    writing it out means the code -> token-id table below is a subtraction, and
    it cannot drift from the vocabulary the voice was exported with.
    """
    ordered = [sym for sym, tid in sorted(NF.DEFAULT_VOCABULARY.items(), key=lambda kv: kv[1])
               if sym not in NF.SPECIAL_IDS]
    if len(ordered) != 59:
        raise GeneratorError(f"expected 59 non-special vocabulary symbols, got {len(ordered)}")
    for offset, sym in enumerate(ordered):
        if NF.DEFAULT_VOCABULARY[sym] != offset + 3:
            raise GeneratorError(
                f"vocabulary is not contiguous from id 3: {sym!r} has id "
                f"{NF.DEFAULT_VOCABULARY[sym]}, expected {offset + 3}"
            )
    symbols = ordered + EXTRA_PHONEMES
    codes = {sym: i + 1 for i, sym in enumerate(symbols)}
    if len(codes) != len(symbols):
        raise GeneratorError("duplicate symbol in the phoneme alphabet")
    if len(symbols) + 1 + BIAS > 0x7E:
        raise GeneratorError("phoneme alphabet no longer fits the printable bias window")
    return symbols, codes


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------


def load_dictionary(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GeneratorError(f"cannot read {path}: {exc}") from exc
    try:
        entries = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GeneratorError(f"{path} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(entries, dict) or not entries:
        raise GeneratorError(f"{path} is not a non-empty JSON object")
    return entries


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise GeneratorError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def encode_value(value: object, name: str, codes: dict[str, int],
                 tag_index: dict[str, int]) -> bytes:
    """One entry's value part: the vcode byte and its payload."""
    if value is None:
        return bytes([BIAS])
    if isinstance(value, str):
        if len(value) > MAX_PLAIN_VALUE:
            raise GeneratorError(f"value for {name!r} is {len(value)} phonemes, over "
                                 f"the {MAX_PLAIN_VALUE} the encoding allows")
        return bytes([BIAS + 1 + len(value)]) + encode_phonemes(value, name, codes)
    if isinstance(value, dict):
        if "DEFAULT" not in value:
            raise GeneratorError(f"tag-keyed entry {name!r} has no DEFAULT variant")
        if not 1 <= len(value) <= MAX_VARIANTS:
            raise GeneratorError(f"entry {name!r} has {len(value)} variants, over "
                                 f"the {MAX_VARIANTS} the encoding allows")
        out = bytearray([BIAS + 60 + len(value)])
        # DEFAULT first so the C fallback can stop at variant 0 without a scan.
        ordered = ["DEFAULT"] + sorted(k for k in value if k != "DEFAULT")
        for tag in ordered:
            if tag not in tag_index:
                raise GeneratorError(f"entry {name!r} uses unknown variant tag {tag!r}")
            variant = value[tag]
            if variant is None:
                out += bytes([BIAS + tag_index[tag], BIAS])
                continue
            if not isinstance(variant, str):
                raise GeneratorError(f"entry {name!r} variant {tag!r} is "
                                     f"{type(variant).__name__}, not str")
            if len(variant) > MAX_PLAIN_VALUE:
                raise GeneratorError(f"entry {name!r} variant {tag!r} is too long")
            out += bytes([BIAS + tag_index[tag], BIAS + 1 + len(variant)])
            out += encode_phonemes(variant, name, codes)
        return bytes(out)
    raise GeneratorError(f"entry {name!r} maps to {type(value).__name__}, not str/dict/null")


def encode_phonemes(value: str, name: str, codes: dict[str, int]) -> bytes:
    out = bytearray()
    for char in value:
        code = codes.get(char)
        if code is None:
            raise GeneratorError(f"entry {name!r} uses phoneme {char!r} (U+{ord(char):04X}) "
                                 f"which is outside the packed alphabet")
        out.append(BIAS + code)
    return bytes(out)


def pack_dictionary(entries: dict[str, object], codes: dict[str, int],
                    tag_index: dict[str, int]) -> tuple[bytes, list[int], dict[str, int]]:
    """-> (blob, block offsets, stats). Keys are sorted by raw byte value."""
    keys = sorted(entries)
    for key in keys:
        if not key:
            raise GeneratorError("empty dictionary key")
        if len(key) > MAX_KEY_LEN:
            raise GeneratorError(f"key {key!r} is {len(key)} bytes, over the encoding's "
                                 f"{MAX_KEY_LEN}")
        for char in key:
            if not 0x20 <= ord(char) < 0x7F or char in '"\\':
                raise GeneratorError(f"key {key!r} contains {char!r}, which the printable "
                                     f"string-literal encoding cannot carry")
    if [k.encode("ascii") for k in keys] != sorted(k.encode("ascii") for k in keys):
        raise GeneratorError("python string order and byte order disagree on these keys")

    blob = bytearray()
    offsets: list[int] = []
    key_bytes = 0
    value_bytes = 0
    for index, key in enumerate(keys):
        if index % BLOCK == 0:
            offsets.append(len(blob))
            blob.append(BIAS + len(key))
            blob += key.encode("ascii")
            key_bytes += 1 + len(key)
        else:
            previous = keys[index - 1]
            shared = 0
            limit = min(len(key), len(previous))
            while shared < limit and key[shared] == previous[shared]:
                shared += 1
            suffix = key[shared:]
            if not suffix:
                raise GeneratorError(f"duplicate key {key!r} after sorting")
            blob.append(BIAS + shared)
            blob.append(BIAS + len(suffix))
            blob += suffix.encode("ascii")
            key_bytes += 2 + len(suffix)
        encoded = encode_value(entries[key], key, codes, tag_index)
        blob += encoded
        value_bytes += len(encoded)
    stats = {
        "entries": len(keys),
        "blocks": len(offsets),
        "blob_bytes": len(blob),
        "key_bytes": key_bytes,
        "value_bytes": value_bytes,
        "index_bytes": len(offsets) * 4,
        "max_key_len": max(len(k) for k in keys),
    }
    return bytes(blob), offsets, stats


# --------------------------------------------------------------------------
# Emitters
# --------------------------------------------------------------------------


def c_string_literals(blob: bytes, per_line: int = 96) -> str:
    """The blob as concatenated C string literals, one per source line."""
    out: list[str] = []
    for start in range(0, len(blob), per_line):
        piece = blob[start:start + per_line]
        text = []
        for byte in piece:
            if byte == 0x5C:
                text.append("\\\\")
            elif byte == 0x22:
                text.append('\\"')
            elif byte == 0x3F:
                # Escaped so no run of bytes can form a C99 trigraph.
                text.append("\\?")
            elif 0x20 <= byte < 0x7F:
                text.append(chr(byte))
            else:
                raise GeneratorError(f"byte 0x{byte:02X} escaped the printable window")
        out.append('    "' + "".join(text) + '"')
    return "\n".join(out) if out else '    ""'


def c_u32_array(values: list[int], per_line: int = 12) -> str:
    lines = []
    for start in range(0, len(values), per_line):
        chunk = values[start:start + per_line]
        lines.append("    " + " ".join(f"{v}u," for v in chunk))
    return "\n".join(lines)


def c_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def build_header(meta: dict) -> str:
    phonemes = meta["phoneme_symbols"]
    lines: list[str] = []
    a = lines.append
    a("/* nano_lex_tables.h -- GENERATED, DO NOT EDIT BY HAND.")
    a(" *")
    a(" * Regenerate with:   python3 esphome/g2p/lex/gen_lex_tables.py")
    a(" * Verify with:       python3 esphome/g2p/lex/gen_lex_tables.py --check")
    a(f" * Generated:         {meta['generated']}")
    a(" *")
    a(" * misaki's American English pronunciation dictionaries, packed for")
    a(" * in-place lookup out of flash. Every array in nano_lex_tables.c is")
    a(" * `const` and therefore lands in .rodata; nothing here is copied to RAM.")
    a(" *")
    a(" * LICENCE: the dictionary data is misaki's us_gold.json / us_silver.json,")
    a(" * Apache-2.0, (c) hexgrad and the misaki contributors. Full provenance,")
    a(" * including the upstream commit, is in")
    a(" * pypkg/sanotts/g2p_data/NOTICE.md and pypkg/sanotts/g2p_data/LICENSE.misaki.txt.")
    a(" *")
    a(" * SOURCES (sha256 at generation time):")
    for source in meta["sources"]:
        a(f" *   {source['path']}")
        a(f" *     {source['sha256']}")
        a(f" *     {source['bytes']} B, {source['entries']} entries")
    a(" *")
    a(" * PACKED SIZE:")
    for name in ("gold", "silver"):
        st = meta[name]
        a(f" *   {name:<7} {st['entries']:>6} entries -> blob {st['blob_bytes']:>8} B"
          f" + index {st['index_bytes']:>7} B = {st['blob_bytes'] + st['index_bytes']:>8} B")
    total = sum(meta[n]["blob_bytes"] + meta[n]["index_bytes"] for n in ("gold", "silver"))
    a(f" *   total   {total} B")
    a(" */")
    a("")
    a("#ifndef NANO_LEX_TABLES_H")
    a("#define NANO_LEX_TABLES_H")
    a("")
    a("#include <stdint.h>")
    a("")
    a("#ifdef __cplusplus")
    a('extern "C" {')
    a("#endif")
    a("")
    a("/* Compile with -DNANO_LEX_WITH_SILVER=0 to drop us_silver.json entirely.")
    a(f" * That saves {meta['silver']['blob_bytes'] + meta['silver']['index_bytes']} B of flash"
      " and costs accuracy on the words only silver has. */")
    a("#ifndef NANO_LEX_WITH_SILVER")
    a("#define NANO_LEX_WITH_SILVER 1")
    a("#endif")
    a("")
    a("/* Every control byte and phoneme code in the blob is stored biased by '0'")
    a(" * so the whole table is printable ASCII and can be a string literal. */")
    a("#define NANO_LEX_BIAS '0'")
    a("#define NANO_LEX_UNBIAS(b) ((uint8_t)((uint8_t)(b) - (uint8_t)NANO_LEX_BIAS))")
    a("")
    a(f"#define NANO_LEX_BLOCK {BLOCK}")
    a(f"#define NANO_LEX_MAX_KEY {meta['max_key_len']}")
    a(f"#define NANO_LEX_MAX_VALUE {MAX_PLAIN_VALUE}")
    a("")
    a("/* vcode conventions, mirrored from gen_lex_tables.py's docstring. */")
    a("#define NANO_LEX_VCODE_NULL 0")
    a("#define NANO_LEX_VCODE_PLAIN_BASE 1")
    a("#define NANO_LEX_VCODE_TAGGED_BASE 61")
    a("")
    a("/* ---- phoneme alphabet ------------------------------------------------ */")
    a("")
    a(f"#define NANO_LEX_PH_COUNT {len(phonemes)}")
    a("")
    a("/* Codes 1..59 are the model vocabulary in token-id order, so")
    a(" * token id == code + 2 for those. Codes 60.. are intermediate-only")
    a(" * symbols with no id; NANO_LEX_PH_ID gives them -1. */")
    for symbol, code in zip(phonemes, range(1, len(phonemes) + 1)):
        ident = meta["phoneme_idents"][symbol]
        a(f"#define {ident} {code}  /* U+{ord(symbol):04X} */")
    a("")
    a("/* code -> 62-symbol token id, or -1 when the symbol has no id. */")
    a(f"extern const int16_t NANO_LEX_PH_ID[{len(phonemes) + 1}];")
    a("/* code -> UTF-8 spelling, for diagnostics only. */")
    a(f"extern const char *const NANO_LEX_PH_UTF8[{len(phonemes) + 1}];")
    a("/* code -> Unicode codepoint; index 0 is unused. */")
    a(f"extern const uint32_t NANO_LEX_PH_CP[{len(phonemes) + 1}];")
    a("/* code -> bitmask of NANO_LEX_F_* */")
    a(f"extern const uint8_t NANO_LEX_PH_FLAGS[{len(phonemes) + 1}];")
    a("")
    a("#define NANO_LEX_F_VOWEL      0x01u")
    a("#define NANO_LEX_F_CONSONANT  0x02u")
    a("#define NANO_LEX_F_DIPHTHONG  0x04u")
    a("#define NANO_LEX_F_PUNCT      0x08u  /* misaki PUNCTS */")
    a("#define NANO_LEX_F_NONQ_PUNCT 0x10u  /* misaki NON_QUOTE_PUNCTS */")
    a("#define NANO_LEX_F_US_TAU     0x20u")
    a("")
    a("/* ---- Penn tags ------------------------------------------------------- */")
    a("")
    a("typedef enum {")
    for tag in TAGS:
        a(f"    {TAG_IDENT[tag]} = {meta['tag_index'][tag]},")
    a(f"    NLG_TAG_COUNT = {len(TAGS)}")
    a("} nlg_tag_t;")
    a("")
    a("extern const char *const NANO_LEX_TAG_NAME[NLG_TAG_COUNT];")
    a("/* get_parent_tag: VB.. -> VERB, NN.. -> NOUN, RB../ADV -> ADV, JJ../ADJ -> ADJ. */")
    a("extern const uint8_t NANO_LEX_TAG_PARENT[NLG_TAG_COUNT];")
    a("")
    a("/* ---- packed dictionaries --------------------------------------------- */")
    a("")
    a("typedef struct {")
    a("    const char     *blob;        /* biased, printable, NUL-terminated */")
    a("    const uint32_t *block_off;   /* byte offset of each block's first record */")
    a("    uint32_t        blob_bytes;")
    a("    uint32_t        block_count;")
    a("    uint32_t        entry_count;")
    a("} nano_lex_dict_t;")
    a("")
    a("extern const nano_lex_dict_t NANO_LEX_GOLD;")
    a("#if NANO_LEX_WITH_SILVER")
    a("extern const nano_lex_dict_t NANO_LEX_SILVER;")
    a("#endif")
    a("")
    a("/* ---- word lists the tokeniser needs ---------------------------------- */")
    a("")
    a("typedef struct {")
    a("    const char *word;")
    a("    uint8_t     tag;")
    a("} nano_lex_word_tag_t;")
    a("")
    a(f"#define NANO_LEX_CLOSED_COUNT {meta['closed_count']}")
    a("/* nano_g2p._CLOSED_CLASS, sorted by word for binary search. */")
    a("extern const nano_lex_word_tag_t NANO_LEX_CLOSED[NANO_LEX_CLOSED_COUNT];")
    a("")
    a(f"#define NANO_LEX_ABBREV_COUNT {meta['abbrev_count']}")
    a("/* nano_g2p._ABBREVIATIONS, lower-cased and sorted. */")
    a("extern const char *const NANO_LEX_ABBREV[NANO_LEX_ABBREV_COUNT];")
    a("")
    a(f"#define NANO_LEX_PERFECT_AUX_COUNT {meta['aux_count']}")
    a("/* nano_g2p._PERFECT_AUXILIARIES, sorted. */")
    a("extern const char *const NANO_LEX_PERFECT_AUX[NANO_LEX_PERFECT_AUX_COUNT];")
    a("")
    a("/* nano_g2p._PUNCT_TAG_BY_CHAR: codepoint -> tag, sorted by codepoint. */")
    a("typedef struct {")
    a("    uint32_t cp;")
    a("    uint8_t  tag;")
    a("} nano_lex_punct_tag_t;")
    a(f"#define NANO_LEX_PUNCT_TAG_COUNT {meta['punct_tag_count']}")
    a("extern const nano_lex_punct_tag_t NANO_LEX_PUNCT_TAG[NANO_LEX_PUNCT_TAG_COUNT];")
    a("")
    a("/* nano_g2p._PREFIX_CHARS / _SUFFIX_CHARS as sorted codepoint arrays. */")
    a(f"#define NANO_LEX_PREFIX_COUNT {meta['prefix_count']}")
    a(f"#define NANO_LEX_SUFFIX_COUNT {meta['suffix_count']}")
    a("extern const uint32_t NANO_LEX_PREFIX_CP[NANO_LEX_PREFIX_COUNT];")
    a("extern const uint32_t NANO_LEX_SUFFIX_CP[NANO_LEX_SUFFIX_COUNT];")
    a("")
    a("/* The 62-symbol vocabulary's specials, mirrored from nano_frontend.py. */")
    a("#define NANO_LEX_ID_PAD 0")
    a("#define NANO_LEX_ID_BOS 1")
    a("#define NANO_LEX_ID_EOS 2")
    a(f"#define NANO_LEX_MAX_TOKENS {NF.DEFAULT_MAX_TOKENS}")
    a(f"#define NANO_LEX_MAX_PS_CHARS {NG.MAX_PS_CHARS}")
    a("")
    a("#ifdef __cplusplus")
    a("}")
    a("#endif")
    a("")
    a("#endif /* NANO_LEX_TABLES_H */")
    return "\n".join(lines) + "\n"


def build_source(meta: dict, gold: tuple[bytes, list[int]],
                 silver: tuple[bytes, list[int]]) -> str:
    phonemes = meta["phoneme_symbols"]
    lines: list[str] = []
    a = lines.append
    a("/* nano_lex_tables.c -- GENERATED, DO NOT EDIT BY HAND.")
    a(" *")
    a(" * Regenerate with: python3 esphome/g2p/lex/gen_lex_tables.py")
    a(f" * Generated:       {meta['generated']}")
    a(" *")
    a(" * Everything below is `const` so it links into .rodata (flash on an")
    a(" * ESP32-S3) and costs no RAM. `size(1)` on the host build is the check.")
    a(" *")
    a(" * Dictionary data: misaki us_gold.json / us_silver.json, Apache-2.0.")
    a(" * See pypkg/sanotts/g2p_data/NOTICE.md.")
    a(" */")
    a("")
    a('#include "nano_lex_tables.h"')
    a("")
    a("#include <stddef.h>")
    a("")

    a(f"const int16_t NANO_LEX_PH_ID[{len(phonemes) + 1}] = {{")
    ids = [-1] + [NF.DEFAULT_VOCABULARY.get(sym, -1) for sym in phonemes]
    a("    " + " ".join(f"{v}," for v in ids))
    a("};")
    a("")
    a(f"const char *const NANO_LEX_PH_UTF8[{len(phonemes) + 1}] = {{")
    a('    "",')
    for symbol in phonemes:
        escaped = "".join(f"\\x{b:02x}" for b in symbol.encode("utf-8"))
        a(f'    "{escaped}",   /* U+{ord(symbol):04X} */')
    a("};")
    a("")
    a(f"const uint32_t NANO_LEX_PH_CP[{len(phonemes) + 1}] = {{")
    a("    0u, " + " ".join(f"0x{ord(s):04X}u," for s in phonemes))
    a("};")
    a("")
    a(f"const uint8_t NANO_LEX_PH_FLAGS[{len(phonemes) + 1}] = {{")
    flags = [0]
    for symbol in phonemes:
        value = 0
        if symbol in NG.VOWELS:
            value |= 0x01
        if symbol in NG.CONSONANTS:
            value |= 0x02
        if symbol in NG.DIPHTHONGS:
            value |= 0x04
        if symbol in NG.PUNCTS:
            value |= 0x08
        if symbol in NG.NON_QUOTE_PUNCTS:
            value |= 0x10
        if symbol in NG.US_TAUS:
            value |= 0x20
        flags.append(value)
    a("    " + " ".join(f"0x{v:02x}," for v in flags))
    a("};")
    a("")

    a("const char *const NANO_LEX_TAG_NAME[NLG_TAG_COUNT] = {")
    for tag in TAGS:
        a(f'    "{c_escape(tag)}",')
    a("};")
    a("")
    a("const uint8_t NANO_LEX_TAG_PARENT[NLG_TAG_COUNT] = {")
    for tag in TAGS:
        parent = NG.Lexicon.get_parent_tag(None if tag == "NONE_TAG" else tag)
        parent_tag = "NONE_TAG" if parent is None else parent
        if parent_tag not in meta["tag_index"]:
            raise GeneratorError(f"parent tag {parent_tag!r} of {tag!r} is not in TAGS")
        a(f"    {meta['tag_index'][parent_tag]},  /* {c_escape(tag)} -> {c_escape(parent_tag)} */")
    a("};")
    a("")

    for name, (blob, offsets) in (("GOLD", gold), ("SILVER", silver)):
        if name == "SILVER":
            a("#if NANO_LEX_WITH_SILVER")
        lower = name.lower()
        a(f"static const char NANO_LEX_{name}_BLOB[] =")
        a(c_string_literals(blob))
        a(";")
        a("")
        a(f"static const uint32_t NANO_LEX_{name}_OFF[{len(offsets)}] = {{")
        a(c_u32_array(offsets))
        a("};")
        a("")
        a(f"const nano_lex_dict_t NANO_LEX_{name} = {{")
        a(f"    NANO_LEX_{name}_BLOB,")
        a(f"    NANO_LEX_{name}_OFF,")
        a(f"    {len(blob)}u,")
        a(f"    {len(offsets)}u,")
        a(f"    {meta[lower]['entries']}u,")
        a("};")
        a("")
        if name == "SILVER":
            a("#endif /* NANO_LEX_WITH_SILVER */")
            a("")

    a("const nano_lex_word_tag_t NANO_LEX_CLOSED[NANO_LEX_CLOSED_COUNT] = {")
    for word in sorted(NG._CLOSED_CLASS):
        tag = NG._CLOSED_CLASS[word]
        if tag not in meta["tag_index"]:
            raise GeneratorError(f"closed-class word {word!r} has unknown tag {tag!r}")
        a(f'    {{ "{c_escape(word)}", {meta["tag_index"][tag]} }},  /* {c_escape(tag)} */')
    a("};")
    a("")
    a("const char *const NANO_LEX_ABBREV[NANO_LEX_ABBREV_COUNT] = {")
    for word in sorted({w.lower() for w in NG._ABBREVIATIONS}):
        a(f'    "{c_escape(word)}",')
    a("};")
    a("")
    a("const char *const NANO_LEX_PERFECT_AUX[NANO_LEX_PERFECT_AUX_COUNT] = {")
    for word in sorted(NG._PERFECT_AUXILIARIES):
        a(f'    "{c_escape(word)}",')
    a("};")
    a("")
    a("const nano_lex_punct_tag_t NANO_LEX_PUNCT_TAG[NANO_LEX_PUNCT_TAG_COUNT] = {")
    for char in sorted(NG._PUNCT_TAG_BY_CHAR, key=ord):
        tag = NG._PUNCT_TAG_BY_CHAR[char]
        if tag not in meta["tag_index"]:
            raise GeneratorError(f"punctuation {char!r} has unknown tag {tag!r}")
        a(f"    {{ 0x{ord(char):04X}u, {meta['tag_index'][tag]} }},"
          f"  /* {c_escape(tag)} */")
    a("};")
    a("")
    a("const uint32_t NANO_LEX_PREFIX_CP[NANO_LEX_PREFIX_COUNT] = {")
    a("    " + " ".join(f"0x{ord(c):04X}u," for c in sorted(set(NG._PREFIX_CHARS), key=ord)))
    a("};")
    a("")
    a("const uint32_t NANO_LEX_SUFFIX_CP[NANO_LEX_SUFFIX_COUNT] = {")
    a("    " + " ".join(f"0x{ord(c):04X}u," for c in sorted(set(NG._SUFFIX_CHARS), key=ord)))
    a("};")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def generate() -> tuple[str, str, dict]:
    symbols, codes = build_phoneme_alphabet()
    tag_index = {tag: i for i, tag in enumerate(TAGS)}

    gold_entries = load_dictionary(GOLD_PATH)
    silver_entries = load_dictionary(SILVER_PATH)

    gold_blob, gold_off, gold_stats = pack_dictionary(gold_entries, codes, tag_index)
    silver_blob, silver_off, silver_stats = pack_dictionary(silver_entries, codes, tag_index)

    # A packed table nothing can read back is worse than no table: decode the
    # whole thing here and diff it against the JSON before writing a byte.
    verify_roundtrip(gold_blob, gold_off, gold_entries, symbols, tag_index, "us_gold")
    verify_roundtrip(silver_blob, silver_off, silver_entries, symbols, tag_index, "us_silver")

    ident_for = {}
    for symbol in symbols:
        ident_for[symbol] = f"NLG_PH_{ord(symbol):04X}"

    meta = {
        "generated": datetime.date.today().isoformat(),
        "sources": [
            {"path": str(GOLD_PATH.relative_to(ROOT)), "sha256": sha256_of(GOLD_PATH),
             "bytes": GOLD_PATH.stat().st_size, "entries": len(gold_entries)},
            {"path": str(SILVER_PATH.relative_to(ROOT)), "sha256": sha256_of(SILVER_PATH),
             "bytes": SILVER_PATH.stat().st_size, "entries": len(silver_entries)},
        ],
        "gold": gold_stats,
        "silver": silver_stats,
        "max_key_len": max(gold_stats["max_key_len"], silver_stats["max_key_len"]),
        "phoneme_symbols": symbols,
        "phoneme_idents": ident_for,
        "tag_index": tag_index,
        "closed_count": len(NG._CLOSED_CLASS),
        "abbrev_count": len({w.lower() for w in NG._ABBREVIATIONS}),
        "aux_count": len(NG._PERFECT_AUXILIARIES),
        "punct_tag_count": len(NG._PUNCT_TAG_BY_CHAR),
        "prefix_count": len(set(NG._PREFIX_CHARS)),
        "suffix_count": len(set(NG._SUFFIX_CHARS)),
    }
    header = build_header(meta)
    source = build_source(meta, (gold_blob, gold_off), (silver_blob, silver_off))
    return header, source, meta


def verify_roundtrip(blob: bytes, offsets: list[int], entries: dict[str, object],
                     symbols: list[str], tag_index: dict[str, int], name: str) -> None:
    """Decode the blob exactly as the C does and compare against the JSON."""
    tag_name = {i: t for t, i in tag_index.items()}
    keys = sorted(entries)
    pos = 0
    previous = ""
    decoded: dict[str, object] = {}
    for index, expected_key in enumerate(keys):
        if index % BLOCK == 0:
            if offsets[index // BLOCK] != pos:
                raise GeneratorError(f"{name}: block offset {index // BLOCK} is wrong")
            klen = blob[pos] - BIAS
            pos += 1
            key = blob[pos:pos + klen].decode("ascii")
            pos += klen
        else:
            shared = blob[pos] - BIAS
            suffix_len = blob[pos + 1] - BIAS
            pos += 2
            key = previous[:shared] + blob[pos:pos + suffix_len].decode("ascii")
            pos += suffix_len
        previous = key
        vcode = blob[pos] - BIAS
        pos += 1
        if vcode == 0:
            value: object = None
        elif vcode <= 60:
            length = vcode - 1
            value = "".join(symbols[b - BIAS - 1] for b in blob[pos:pos + length])
            pos += length
        else:
            variants: dict[str, object] = {}
            for _ in range(vcode - 60):
                tag = tag_name[blob[pos] - BIAS]
                vlen1 = blob[pos + 1] - BIAS
                pos += 2
                if vlen1 == 0:
                    variants[tag] = None
                else:
                    length = vlen1 - 1
                    variants[tag] = "".join(symbols[b - BIAS - 1]
                                            for b in blob[pos:pos + length])
                    pos += length
            value = variants
        decoded[key] = value
        if key != expected_key:
            raise GeneratorError(f"{name}: decoded key {key!r} != {expected_key!r}")
    if pos != len(blob):
        raise GeneratorError(f"{name}: decoder consumed {pos} of {len(blob)} bytes")
    if decoded != entries:
        differing = [k for k in entries if decoded.get(k) != entries[k]]
        raise GeneratorError(f"{name}: {len(differing)} entries do not round-trip, "
                             f"first is {differing[0]!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory and fail if the tracked files differ")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args(argv)

    try:
        header, source, meta = generate()
    except GeneratorError as exc:
        print(f"gen_lex_tables: {exc}", file=sys.stderr)
        return 2

    header_path = args.out_dir / "nano_lex_tables.h"
    source_path = args.out_dir / "nano_lex_tables.c"

    if args.check:
        problems = []
        for path, wanted in ((header_path, header), (source_path, source)):
            try:
                have = path.read_text(encoding="utf-8")
            except OSError as exc:
                problems.append(f"{path}: {exc}")
                continue
            # The date line changes every day; ignore only that one line.
            if _strip_date(have) != _strip_date(wanted):
                problems.append(f"{path} differs from a fresh generation")
        if problems:
            for problem in problems:
                print(f"gen_lex_tables --check: {problem}", file=sys.stderr)
            return 1
        print("gen_lex_tables --check: tracked tables match a fresh generation")
        return 0

    try:
        header_path.write_text(header, encoding="utf-8")
        source_path.write_text(source, encoding="utf-8")
    except OSError as exc:
        print(f"gen_lex_tables: cannot write output: {exc}", file=sys.stderr)
        return 2

    total = sum(meta[n]["blob_bytes"] + meta[n]["index_bytes"] for n in ("gold", "silver"))
    json_bytes = sum(s["bytes"] for s in meta["sources"])
    print(f"wrote {header_path} ({header_path.stat().st_size} B)")
    print(f"wrote {source_path} ({source_path.stat().st_size} B)")
    for name in ("gold", "silver"):
        st = meta[name]
        print(f"  {name:<7} {st['entries']:>6} entries  blob {st['blob_bytes']:>8} B"
              f"  index {st['index_bytes']:>7} B"
              f"  keys {st['key_bytes']:>8} B  values {st['value_bytes']:>8} B")
    print(f"  packed total {total} B, {100.0 * total / json_bytes:.1f}% of the "
          f"{json_bytes} B of JSON")
    return 0


def _strip_date(text: str) -> str:
    return "\n".join(line for line in text.splitlines()
                     if not line.strip().startswith("* Generated:"))


if __name__ == "__main__":
    raise SystemExit(main())
