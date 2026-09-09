# `nano_lex_g2p` — espeak-free English text → e12nano phoneme ids, in C

An embeddable C99 port of the dictionary-first path in
`pypkg/sanotts/nano_g2p.py`. It turns `"The front door is unlocked."` into the
dense id sequence `1 37 42 3 21 47 54 … 9 2` that the `en_us_e12nano` voice
expects, on an ESP32-S3, with no espeak-ng, no `malloc`, and no libc.

```c
#include "nano_lex_g2p.h"

int32_t ids[NANO_LEX_MAX_TOKENS];               /* 207 */
int n = nano_lex_g2p_text_to_ids("The front door is unlocked.", ids, 207);
if (n < 0) {
    ESP_LOGE(TAG, "g2p: %s", nano_lex_g2p_strerror(n));
}
```

## Why this and not the espeak-ng port

|                              | on-chip espeak-ng | `nano_lex_g2p` |
| ---------------------------- | ----------------: | -------------: |
| internal SRAM, unconditional | 62,090 B          | **18,648 B**   |
| flash (code + data)          | 222,744 B + 281 KB SPIFFS | **3,090,784 B, no partition** |
| task stack                   | crashed WiFi at 28 KB, moved to PSRAM | **1,257 B** |
| licence                      | GPL-3.0           | **Apache-2.0** |
| token error rate vs misaki   | 6.09 %            | **1.47 %**     |

espeak is smaller in flash and much larger in RAM; this is the other way round.
Flash is the cheap resource on an S3 module and internal SRAM is the expensive
one, so the trade is deliberate. The licence and the accuracy both move the
same way, which is the actual argument.

## Files

| path | what it is |
| --- | --- |
| `gen_lex_tables.py` | packs misaki's `us_gold.json` / `us_silver.json` into the C table; `--check` verifies the tracked output still matches |
| `nano_lex_tables.[ch]` | **generated**, 3.3 MB of C; the packed dictionaries and every constant table taken from `nano_g2p.py` |
| `nano_lex_g2p.[ch]` | the implementation and the public contract |
| `test_nano_lex_g2p.c` | host driver, self-test, stack probe |
| `run_parity.py` | measures the C against `nano_g2p.py`, imported directly |
| `ha_corpus.txt` | 52 Home Assistant announcements written for this port |
| `edge_corpus.txt` | 16 adversarial inputs covering the branches an announcement never reaches |
| `Makefile` | `make`, `selftest`, `parity`, `sizes`, `sizes-elf`, `stack`, `stack-usage`, `tables`, `check` |

## The contract

```c
int         nano_lex_g2p_text_to_ids(const char *text, int32_t *out, int cap);
const char *nano_lex_g2p_strerror(int rc);
size_t      nano_lex_g2p_workspace_bytes(void);
```

* Returns the number of ids written, or a negative code. Twelve distinct codes,
  each with its own message; no failure path is shared with another and none is
  swallowed.
* Emits `<bos>` (1) first and `<eos>` (2) last, dense, no pad and no interior
  specials — the framing of `mcu/test/fixtures/en_us_e12nano/r00_ids.bin`.
* Never writes past `cap`. When the sequence does not fit, it writes exactly
  `cap` ids — a real prefix with `<eos>` forced into the last slot, so the
  buffer stays well framed — and returns `NANO_LEX_E_CAP`. Truncation is always
  visible; it is never silent and it never overruns. The self-test checks the
  byte after `cap` is untouched.
* No `malloc`, no `free`, no `printf`, no libc at all: `memcpy`/`memcmp`/`strlen`
  are five-line statics so the module builds `-ffreestanding`.
* **Not reentrant.** One static workspace, so one task at a time.
* Two extras beyond the three required entry points, both diagnostic and both
  droppable: `nano_lex_g2p_get_stats()` (see *How often OOV actually fires*) and
  `nano_lex_g2p_text_to_phonemes()` (the UTF-8 phoneme string, used by the
  parity harness).

## Measured numbers

All produced on this machine on 2026-09-09; none is an estimate. `HOST-REF:`
marks a host-computed constant, per
`mcu/ports/esp32s3/measurements/README.md`. Nothing was run on hardware.

### Packed table — `python3 gen_lex_tables.py`

| | entries | key bytes | value bytes | blob | block index | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `us_gold.json`   | 90,201 | 518,226 | 932,969 | 1,451,195 | 45,104 | **1,496,299** |
| `us_silver.json` | 93,361 | 546,781 | 954,386 | 1,501,167 | 46,684 | **1,547,851** |
| total | 183,562 | | | | | **3,044,150** |

`HOST-REF:` 3,044,150 B is **49.9 %** of the 6,099,986 B of source JSON, and
**31.3 % smaller** than the 4,429,520 B a plain length-prefixed blob with a
dense `uint32` offset per entry would need. The saving comes from front-coding
keys inside blocks of 8 and indexing only the block heads.

**Dropping silver saves 1,547,851 B** — compile everything with
`-DNANO_LEX_WITH_SILVER=0`. Measured, not assumed:

| gold-only vs gold+silver | exact sequence | token error rate | OOV word groups |
| --- | ---: | ---: | ---: |
| Home Assistant (52) | **52/52 = 100 %** | **0/2,086 = 0.000 %** | 2/350, unchanged |
| Gutenberg (330) | 301/330 = 91.2 % | 114/30,674 = 0.372 % | 63 → 77 of 5,606 |

Silver contributes **nothing at all** to the announcement corpus: the two builds
produce identical ids on all 52 sentences. It only starts to matter on
literary text. If flash is tight, cut silver first — it halves the table for no
measurable loss on the traffic this device actually sees. The gold-only table
object is 1,498,498 B against 3,046,350 B (host `size`).

### Sections — `make sizes-elf`

Bare-metal ELF, `clang --target=arm-none-eabi -std=c99 -Os -ffreestanding`.
This target rather than the host because Mach-O puts relocatable `const` pointer
arrays in `__DATA,__const`, which is a dyld artefact; a statically linked
firmware image does not do that, and the ELF build proves it.

| object | `.text` | `.rodata` (+`.str1.1`) | `.data` | `.bss` |
| --- | ---: | ---: | ---: | ---: |
| `nano_lex_g2p.o`    | 41,096 | 336 + 1,030 | **0** | **18,648** |
| `nano_lex_tables.o` | 0 | 3,046,928 + 1,394 | **0** | **0** |
| total | 41,096 | 3,049,688 | **0** | **18,648** |

`HOST-REF:` flash 3,090,784 B; RAM 18,648 B. `nm` finds exactly one writable
symbol across both objects — `g_ws`, the workspace — and none in the table
object. The generated header states the same claim and `make sizes` re-checks
it.

These are host cross-compilation figures for an ARM target. The xtensa
toolchain is not installed on this machine, so the ESP32-S3 `.text` figure will
differ (xtensa is usually a little larger); `.rodata` and `.bss` are data and
will not.

### Workspace and stack

* `nano_lex_g2p_workspace_bytes()` → **18,648 bytes**, all `.bss`, identical to
  the `.bss` figure above. Breakdown: 6,400 B token array, 3,840 B source-token
  and group arrays, 3,072 B phoneme arena, 1,430 B word slots, 1,024 B decoded
  codepoints, 512 B chunk buffer, and 2,370 B of number, piece and stress
  scratch.
* The phoneme arena's high-water mark over the 398 sentences below is **235
  bytes** of the 3,072 reserved, and overflow returns `NANO_LEX_E_ARENA` rather
  than corrupting anything.
* Peak stack for one call, `make stack`: **1,257 bytes** (`-Os`, arm64 host,
  measured by painting the stack below the call and scanning the survivors;
  the figure includes the probe's own frame, so it is an over-estimate). Worst
  of six probes including a numeral-heavy and an out-of-vocabulary sentence.
  `make stack-usage` corroborates from `-fstack-usage`: the deepest chain is
  `text_to_ids(160) → run(320) → lexicon_call(208) → … → lookup(128) →
  grown_find(128) → dict_find(176)`.
* Tune with `NANO_LEX_MAX_CHARS` (512 input codepoints) and
  `NANO_LEX_MAX_TOKEN` (320 tokens) in `nano_lex_g2p.h`; both are hard limits
  that return `NANO_LEX_E_TEXT_LONG` / `NANO_LEX_E_TOKENS` rather than
  truncating.

### Parity against the Python oracle — `make parity`

The oracle is `pypkg/sanotts/nano_g2p.py` itself, imported and called by
`run_parity.py`. Nothing is reimplemented.

**Port correctness** — C vs `nano_g2p.py` with its neural fallback disabled,
which is the algorithm this file implements:

| corpus | sentences | exact sequence | token error rate |
| --- | ---: | ---: | ---: |
| `pypkg/tests/data/nano-g2p-misaki-reference.jsonl` (330 Gutenberg sentences, the repo's own eval set for this front end) | 330 | **330/330 = 100.00 %** | **0/30,674 = 0.000 %** |
| `ha_corpus.txt` (Home Assistant, written for this port) | 52 | **52/52 = 100.00 %** | **0/2,086 = 0.000 %** |
| `edge_corpus.txt` (adversarial) | 16 | **15/15 = 100.00 %** | **0/1,063 = 0.000 %** |

The one edge row not compared is `café naïve résumé Zzyzx Bingley Fitzwilliam
Euroclydon`: every word is out of dictionary, so with no fallback both the
Python and the C produce nothing and both refuse the row — agreement, counted
separately rather than as a pass.

**Phonetic fidelity** — against the recorded real-misaki ids the nano training
and eval packs were built from, on the same 330 sentences:

| system | exact sequence | token error rate |
| --- | ---: | ---: |
| `nano_frontend.py` — espeak on every word (shipped baseline, from `docs/nano-espeak-free-frontend.md`) | 1.8 % | 6.09 % |
| `nano_g2p.py` + neural OOV model (reproduced here) | 82.12 % | 0.299 % |
| **`nano_lex_g2p.c` (no OOV model)** | **78.18 %** | **1.470 %** |

So dropping the 2.8 MB neural fallback costs 3.94 points of whole-sentence match
and takes the token error rate from 0.299 % to 1.470 % — still **4.1× better
than the espeak path it replaces**, on the metric that matters, which is
agreement with what the voice was distilled from.

## What is ported, approximated, and dropped

### Ported, faithfully enough to score 100 % against the oracle

The tokeniser and closed-class tagger, `_retag_verbs` / `_retag_that`,
`subtokenize` (the nine-alternative regex, hand-scanned), `retokenize` with its
currency and `2`-as-`to` special cases, the gold+silver lexicon with
`grow_dictionary`, tag-keyed variant selection, `get_special_case`,
`get_NNP`, `is_known`, the `-s` / `-ed` / `-ing` stemmers, `apply_stress` and
`restress`, `TokenContext`, `resolve_tokens`, the back-off loop, `merge_tokens`,
the `ɾ→T` / `ʔ→t` tail, `en_chunks` with its waterfall, and the 62-symbol
tokenisation.

**Numbers are in scope and are ported in full**: `cardinal_words`,
`ordinal_words`, `year_words`, `decimal_words` and the whole of
`Lexicon.get_number`, including currency pairs, digit-by-digit spelling,
three-digit "four oh five", the `s`/`'s`/`ed`/`'d`/`ing` suffixes and the minus
sign. A house assistant says "twenty two degrees" and "ten minutes" constantly;
a G2P that mangles digits would not be worth shipping. Every numeral in
`edge_corpus.txt` matches the oracle exactly.

### Approximated — every one of these is marked `DEVIATION:` in the source

* **Unicode.** `isalpha`/`upper`/`lower` are exact for ASCII. Above U+007F,
  `nlg_is_letter` treats a codepoint as a letter unless it is in the
  punctuation, symbol or combining-mark ranges listed in the source, and case
  mapping is ASCII-only. A word containing a non-ASCII letter can never be a
  dictionary key (misaki's `LEXICON_ORDS` is ASCII), so this can only change
  how such a word reaches the out-of-vocabulary path, which it does either way.
* **NFKC** is not implemented. It is the identity on every ASCII string, which
  is every string that can reach the dictionaries. The two curly apostrophes
  *are* normalised, because `"don’t"` is a real input.
  `Lexicon.numeric_if_needed` likewise only ever rewrites non-ASCII digits.
* **Astral-plane codepoints** (emoji) fold to U+FFFD. They are non-letters
  either way.
* **Bounded workspace.** A token longer than 64 codepoints is treated as
  out-of-vocabulary; no dictionary key is longer than 45. Text over 512
  codepoints or 320 tokens is rejected with a distinct code rather than
  truncated.
* **Numerals over 18 digits** return `NANO_LEX_E_NUMBER`. Python raises
  `NumberError` past 10^18 for the same reason (`_SCALES` runs out), so the two
  agree about *whether* it fails, not about the exact boundary.
* **`num_flags`** has three branches in misaki's `extend_num`. Nothing in
  `nano_g2p.py` ever sets a non-empty `num_flags`, so those branches are dead;
  the C implements the `num_flags == 0` case and returns
  `NANO_LEX_E_INTERNAL` if it is ever called with flags, rather than silently
  pretending to handle them.
* **The `tag == "ADD"` branch** of `get_special_case` is omitted:
  `nano_g2p._tag` never returns spaCy's `ADD` tag, so it is unreachable.

### Dropped

**The neural out-of-vocabulary phonemizer** (`oov_bart_en_us.npz`, 2.8 MB,
751,551 float32 parameters, a one-layer BART with greedy decoding). Not ported:
the weights alone are almost as large as the dictionaries, and the forward pass
needs float matmuls this device would rather spend on the vocoder.

The behaviour when a word is unresolved is **defined and counted, never a silent
skip**: the C reproduces `NanoG2P` with `fallback = None` exactly — the back-off
loop shrinks the group, the unresolved sub-token contributes no phonemes, and
`nano_lex_g2p_get_stats()->oov_words` counts the word group it was in. Read it
and log it:

```c
nano_lex_g2p_stats_t st;
nano_lex_g2p_get_stats(&st);
if (st.oov_words) {
    ESP_LOGW(TAG, "%u of %u words had no pronunciation", st.oov_words, st.lexicon_words);
}
```

### How often OOV actually fires

| corpus | word groups | unresolved | rate |
| --- | ---: | ---: | ---: |
| Home Assistant (52 sentences) | 350 | **2** | **0.57 %** |
| Gutenberg (330 sentences) | 5,606 | 63 | 1.12 % |
| edge cases (16 sentences, adversarial by construction) | 119 | 6 | 5.04 % |

The two Home Assistant misses are **not unknown words**. They are the `:` inside
`6:30` and `2:15` — the colon is not in `SUBTOKEN_JUNKS`, so it is left
unvoiced while the digits around it are spelled correctly ("six thirty", "two
fifteen"). Across all 52 announcements there is not one word the dictionaries
cannot pronounce. The Gutenberg misses are 49 distinct tokens, almost all
19th-century proper nouns — Darcy (5), Bennet (3), Bingley, Fitzwilliam,
Gardiner, Euroclydon — where the reference is espeak's guess anyway.

Worth recording: on clock times the C is **better** than the Python with its
neural fallback on. `nano_g2p.py` hands the whole `6:30` group to the model,
which passes punctuation through and drops digits, yielding a bare `":"`; the
no-fallback path keeps backing off and spells the numbers. That accounts for
both of the two Home Assistant rows where the C and the neural Python differ.

## Licence

The dictionary data in `nano_lex_tables.c` is misaki's `us_gold.json` and
`us_silver.json` — **Apache-2.0**, © hexgrad and the misaki contributors,
pinned at commit `e820629b96334db28227df37f280e4836d46fadb`, byte-identical to
the copies in the misaki 0.9.4 wheel that built the nano packs. Full provenance,
with sha256 for every file, is in
[`pypkg/sanotts/g2p_data/NOTICE.md`](../../../pypkg/sanotts/g2p_data/NOTICE.md);
the licence text is
[`pypkg/sanotts/g2p_data/LICENSE.misaki.txt`](../../../pypkg/sanotts/g2p_data/LICENSE.misaki.txt).
The generated header records both source paths and their sha256 so a firmware
image can be traced back to the exact revision it was built from.

The algorithm in `nano_lex_g2p.c` is a port of `pypkg/sanotts/nano_g2p.py`,
whose lexicon parts are themselves line-by-line ports of `misaki/en.py`
(Apache-2.0) and whose tokeniser, tagger and number speller were written for
this repository and copied from nothing.

**No espeak-ng anywhere.** Nothing here is GPL, and nothing here is derived from
espeak-ng output.

## Reproducing everything above

```sh
cd esphome/g2p/lex
make tables      # regenerate nano_lex_tables.[ch] from the misaki JSON
make check       # ... and verify the tracked files match a fresh generation
make selftest    # 20 contract checks: framing, truncation, every error path
make parity      # C vs nano_g2p.py on all three corpora
make sizes       # host objects, writable symbols, workspace bytes
make sizes-elf   # bare-metal ELF section sizes
make stack       # measured peak stack
make stack-usage # per-function frames from -fstack-usage
```

`make parity` needs `numpy` (`nano_g2p.py` imports the OOV module at import
time) and exits non-zero if port correctness is anything but 100 %.

## Verdict

Good enough to be the on-device text path for a Home Assistant announcement
device, and better than the alternative on every axis that was measured.

* It reproduces the Python front end **exactly** — 397 of 397 comparable
  sentences, 33,823 reference tokens, zero edit distance. There is no accuracy
  argument left between the C and the Python.
* Against what the voice was actually trained on it is 1.47 % token error,
  against 6.09 % for the espeak path it replaces.
* On real announcement text it has **no unknown words at all**; the only two
  unresolved sub-tokens in 52 sentences are colons in clock times, and the
  digits around them are still spoken correctly.
* 18,648 B of SRAM and 1,257 B of stack, against espeak's 62,090 B and a stack
  that had to be moved to PSRAM.

The honest costs, in order:

1. **3.0 MB of flash.** That is the whole argument against it. On a 4 MB module
   it does not fit next to a model; on 8 MB or 16 MB it is comfortable. Gold
   alone is 1.5 MB and gives **byte-identical ids on all 52 announcements**, so
   the escape hatch is real and measured — `-DNANO_LEX_WITH_SILVER=0`.
2. **A colon in a clock time is silent.** `6:30` reads as "six thirty" with no
   pause. Fixable in the caller by rewriting `H:MM` before the call; not fixed
   here, because fixing it in the C would break parity with the Python oracle.
3. **Unusual proper nouns get no pronunciation at all** rather than a guess.
   1.12 % of words on 19th-century literature, 0 % on the announcement corpus.
   If a deployment names its rooms after Brontë characters, that is the number
   to watch — read `oov_words` and log it.
4. Bounded input: 512 codepoints, 320 tokens, 64-codepoint words. Every limit
   returns a distinct error rather than truncating silently, but a caller
   feeding it a paragraph will get `NANO_LEX_E_TEXT_LONG` and must chunk.
