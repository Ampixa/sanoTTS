# Third-party assets in `sanotts/g2p_data/`

Two licences, both permissive. The English assets at the top of this file are
**Apache-2.0**; the Indonesian ones in `indo_g2p/` are **MIT**, with one
Apache-2.0 upstream behind a table that is deliberately *not* vendored.
Nothing here is derived from espeak-ng, and none of it carries a copyleft
obligation. `tools/build_nano_g2p_assets.py` fetches each English file from the
pinned revision below and refuses to write it if its sha256 has changed;
`tools/vendor_indo_g2p.py` does the same job for `indo_g2p/` and records every
payload's sha256 in `indo_g2p/MANIFEST.json`.

---

## `us_gold.json`, `us_silver.json`

- **Upstream:** <https://github.com/hexgrad/misaki>, `misaki/data/`
- **Revision:** `e820629b96334db28227df37f280e4836d46fadb` (2025-04-05), the last
  commit that touched either file as of 2026-09-04. Fetched over
  `raw.githubusercontent.com` at that exact commit, not at `main`.
- **License:** Apache-2.0. Declared by the repository's own `LICENSE` (vendored
  here as `LICENSE.misaki.txt`), by the GitHub licence API, and by the
  `License :: OSI Approved :: Apache Software License` classifier on the
  `misaki` PyPI release.
- **Copyright:** hexgrad and the misaki contributors.
- **sha256:**
  - `us_gold.json` — `dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f` (3,000,469 B, 90,201 entries)
  - `us_silver.json` — `de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475` (3,099,517 B, 93,361 entries)
- **Modifications:** none. Both files are byte-identical to the upstream blobs
  and to the copies inside the installed `misaki` 0.9.4 wheel that produced the
  nano training and eval packs, which is why the vendored lookup and the
  training reference are the same table rather than two tables that agree today.
- **Not vendored:** `gb_gold.json` and `gb_silver.json`. The nano voices are
  American English and the British tables would add 6.5 MB nothing reads.

Note that the *dictionaries* are Apache-2.0 but `pip install misaki[en]` is not
a substitute for them: that extra pulls in `phonemizer-fork` (GPL-3.0),
`espeakng-loader`, `spacy` and `spacy-curated-transformers`, which is both the
licence this work removes and the dependency weight this package refuses.

## `oov_bart_en_us.npz`

- **Upstream:** <https://huggingface.co/PeterReid/graphemes_to_phonemes_en_us>
- **Revision:** `a5631b285d18d59483c32c0c3379cb9fac924f4b`
- **License:** `apache-2.0`, declared in the model card's front matter and
  reported by the Hugging Face model API.
- **Copyright:** Peter Reid.
- **Source file:** `model.safetensors`, sha256
  `dc4a02e62d4fcb4bb4097ecf00db89b8e1a12a549a52ab6adfbba220b80a55c5`
  (3,011,692 B, 751,551 float32 parameters), together with `config.json`, sha256
  `8deb3537fb29c63cd9f20d75515ae06e4c92f1b6db0703a2d45bca95b33a53a4`.
- **Modifications:** format only. `tools/build_nano_g2p_assets.py` reads the
  safetensors with a 20-line header parser, checks every tensor's name and
  shape against the topology `nano_g2p_oov.py` implements, and writes the same
  float32 values into a compressed `.npz` along with the architecture fields
  from `config.json`. No weight is retrained, quantised or reordered.
- **Provenance of the training data:** the author's own
  `english_to_phonemes.py` (published in the same repository) trains this model
  on misaki's `us_gold.json` / `us_silver.json`, augmented with regular plurals
  derived from those entries by rule. It is not distilled from espeak-ng, which
  is what makes it usable here.

## `LICENSE.misaki.txt`

The Apache-2.0 licence text as published in the misaki repository at the commit
above, sha256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
It also covers the parts of `sanotts/nano_g2p.py` that are line-by-line ports of
`misaki/en.py`; those functions name their upstream counterpart in their
docstrings.

## Not vendored from anywhere

The tokeniser, the part-of-speech tagger and the number speller in
`sanotts/nano_g2p.py` are written for this package. They stand in for spaCy's
`en_core_web_sm` and for `num2words` (LGPL), and no word list, model or table
was copied from either.

---

## Nothing vendored for `id_g2p.py` or `vi_g2p.py`

`sanotts/id_g2p.py` and `sanotts/vi_g2p.py` -- the rule front ends that
reproduce espeak-ng's behaviour for the `id` and `vi` piperlite voices -- carry
**no third-party asset at all**. No dictionary, no model, no rule file. They are
written from Indonesian and Vietnamese orthography, which is why there is
nothing in this directory for them. Recorded here because the *absence* of a
vendored asset is the load-bearing fact: it is what keeps the two languages free
of the copyleft this work removes.

The second Indonesian path, `sanotts/id_indo_g2p.py`, *does* vendor data, and it
is all MIT; see `indo_g2p/` below.

Two sources were evaluated and **rejected**, both on licence grounds:

### espeak-ng — GPL-3.0, no output exception

espeak-ng is the front end these voices were distilled through, and it is the
reason this work exists. It was used here only as a **black-box oracle**: run on
a held-out corpus and diffed against the rules, to settle which codepoints it
prints for each phoneme and to measure the residual disagreement. Its
`id_rules`, `id_list`, `vi_rules`, `vi_list` and source tree were not read, and
none of its output was used as training data. The agreement figures are in
`experiments/evidence/idvi-espeak-free-ab-20260904.json`.

### misaki's `vi.py` — Apache-2.0 declared, GPL-3.0 in fact

misaki (<https://github.com/hexgrad/misaki>) ships `misaki/vi.py`, and the
repository's `LICENSE`, its GitHub licence API entry and its PyPI classifier all
say Apache-2.0. That file is nonetheless a port of **vPhon**
(<https://github.com/kirbyj/vPhon>, "A Vietnamese phonetizer" by James Kirby),
which is **GPL-3.0** -- stated by its own `LICENSE.md` (the GPLv3 text) and by
the GitHub licence API.

Checked rather than assumed, on 2026-09-04. Comparing misaki `vi.py` at
`9e02a0b6269ceeabd5b355fdef27dc6f8b5eb673` (sha256 of the file:
`be333eac8211063eafd3304b13eafa2f2250af0a950b47f231f7758fb08d951e`) against
vPhon `vPhon.py` at `89d8ffede60047797b6deb8e790c18a97b00f40b`, the two
`trans()` functions are **90.5% line-identical** once comments and whitespace
are stripped (`difflib.SequenceMatcher` over the code lines: 124 in misaki, 128
in vPhon, 99 distinct lines shared verbatim), and **33 comments survive word for
word**, including dated and cited ones that could
not arise independently:

    # Modified 20 Sep 2008 to fix aberrant 33 error
    # There is also this reverse fronting, see Thompson 1965:94 ff.
    # Monophthongization (Southern dialects: Thompson 1965: 86; Hoàng 1985: 181)
    # labialized allophony (added 17.09.08)

misaki's variables are vPhon's with a `Cus_` prefix (`Cus_onsets`, `Cus_nuclei`,
`Cus_codas`, `Cus_offglides`, `Cus_onglides`, `Cus_onoffglides`, `Cus_tones_p`).
Vendoring or porting it would trade espeak-ng's GPL-3.0 for vPhon's, so nothing
from it is used -- not the code, not the tables, not the alphabet.

It would not have fitted anyway. `misaki[vi]` declares `underthesea`, `spacy`
and `spacy-curated-transformers`, which is the dependency weight this package
refuses, and its output alphabet (`ʐ` for `r`, `ɓ`/`ɗ`, `ŋ͡m`, tone digits in a
different slot) is not the alphabet `vi-vais1000-1p46m`'s `phoneme_id_map` is
keyed on.

### Evaluation text

The Indonesian and Vietnamese sentences the two front ends were measured on are
Tatoeba's, CC BY 2.0 FR, and live outside the installed package in
`data/textsets/idvi-espeak-free-20260904/` with their own `NOTICE.md`. No
sentence from them informed a rule.


---

## `indo_g2p/` — snowfluke/indo-g2p, MIT

`sanotts/id_indo_g2p.py` is a port of **indo-g2p**, and the files in
`indo_g2p/` are its data. This is the front end that places the Indonesian
schwa lexically and emits the glottal stops espeak-ng does not; the measurement
that justified adopting it is
`experiments/evidence/id-indo-g2p-ab-20260904.json` and the design note is
`docs/id-indo-g2p-frontend.md`.

- **Upstream:** <https://github.com/snowfluke/indo-g2p>
- **Revision:** `dd5f102cba7345ba46adef8c2ad9fa261587ea4e`, npm version `0.1.2`,
  pushed 2026-09-02, cloned and read on 2026-09-04.
- **License:** **MIT**. Read in the repository's own `LICENSE` (`MIT License /
  Copyright (c) 2026 snowfluke`), and stated again as `"license": "MIT"` in
  `package.json`. Not taken from a badge or an API summary.
- **Copyright:** snowfluke, plus the upstreams listed below, each of which the
  project's own `NOTICE.md` names.
- **Modifications:** format only. Each file below is the payload of one
  `export const NAME = '...'` string literal in `src/data/*.ts`, unescaped and
  xz-compressed, byte for byte. `tools/vendor_indo_g2p.py` regenerates all of
  them from a checkout and prints each payload's sha256; the same hashes are in
  `indo_g2p/MANIFEST.json` and are asserted by
  `pypkg/tests/test_id_indo_g2p.py`. No entry was added, removed or edited.

### What is vendored, and the upstream behind each

| file | from | upstream of that data | licence |
| --- | --- | --- | --- |
| `schwa_dict.xz` | `src/data/schwa-dict.ts` | [Wikidepia/g2p-id](https://github.com/Wikidepia/g2p-id) | MIT |
| `schwa_overrides.xz` | `src/data/schwa-overrides.ts` | indo-g2p's own corrections | MIT |
| `lexicon.xz` | `src/data/lexicon.ts` | [bookbot-kids/g2p_id](https://github.com/bookbot-kids/g2p_id) lexicon, reduced to schwa bitmasks | Apache-2.0 |
| `syllabifier_state.xz` | `src/data/syllabifier-model.ts` | Wikidepia/g2p-id's CRF syllabifier weights | MIT |
| `collocations.xz` | `src/data/collocations.ts` | indo-g2p's own rules | MIT |

`Wikidepia/g2p-id` is MIT, Copyright (c) 2026 Akmal; indo-g2p carries its
licence text at `licenses/g2p-id-MIT.txt`. `bookbot-kids/g2p_id` is
Apache-2.0, Copyright 2023 PT BOOKBOT INDONESIA (<https://bookbot.id/>) —
**this product includes software developed at PT BOOKBOT INDONESIA** — and
indo-g2p carries that licence text at `licenses/bookbot-Apache-2.0.txt`. Both
are permissive and both are compatible with redistributing this package under
its own terms; the attributions above are the whole of what either asks for.

The port of the *algorithm* — `src/g2p.ts`, `src/schwa.ts`, `src/affix.ts`,
`src/syllabifier.ts`, `src/crf-model.ts`, `src/normalize.ts`, `src/number.ts`,
`src/collocations.ts`, `src/constants.ts` — is MIT under indo-g2p's own licence,
and indo-g2p's `NOTICE.md` records that its phoneme rules are themselves a port
of Wikidepia/g2p-id's `g2p.py` (MIT).

### What is deliberately **not** vendored

- **`english.ts` (2.0 MB raw, 547 KB compressed).** English pronunciations from
  [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) (MIT,
  Copyright (c) 2016 Yuchen Zhang and contributors), for reading borrowed words
  and foreign names. Measured on the 28,194-sentence Tatoeba Indonesian export
  it answers for 710 of 11,098 word types and **1.39% of tokens**, and 656 of
  those 2,152 tokens are the single name `mary`, which is an artefact of that
  corpus rather than a fact about Indonesian. 547 KB for that is not a trade
  worth making by default. `tools/vendor_indo_g2p.py --with-english` writes it
  for anyone who disagrees, and `id_indo_g2p.english_available()` reports
  whether it is there. Without it the Indonesian rules read the word, which is
  exactly what the upstream `indo-g2p/core` entry point does.
- **`pos-model.ts` (3.3 MB raw, 640 KB compressed) and `homographs.ts`.** An
  averaged-perceptron POS tagger from bookbot-kids/g2p_id (Apache-2.0) that
  resolves homographs such as `apel` from their part of speech. Over the
  438-sentence development corpus it changed the output of **2 sentences**, and
  on one of those two it disagreed with — and was wrong against — the 672-byte
  collocation rules that are vendored. `--with-pos` writes it; nothing in the
  package reads it.
- **`data/*.tsv`** (`homographs-verified`, `homographs-collocations`,
  `indonesian-proper-nouns`, `schwa-overrides`). Build inputs for the packed
  tables above, not runtime data.

### Wiktionary was an oracle, not a source

The correctness claims in the evidence file were checked against English
Wiktionary's Indonesian entries, read through
[kaikki.org](https://kaikki.org/dictionary/Indonesian/)'s machine-readable
extract, which is **CC BY-SA 4.0**. Nothing from it is redistributed: the
repository keeps counts and accuracy rates, not entries, and no rule, table or
test fixture in this package is derived from it. It was chosen precisely
because it is independent of both espeak-ng and of the dictionary indo-g2p
ships, so the two can be checked against each other rather than against
themselves.
