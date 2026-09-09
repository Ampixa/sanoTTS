# Text-to-phoneme for the e12nano voice: two paths, one shipped

The component in `../components/sanotts/` needs to turn `"The front door is
unlocked."` into e12nano phoneme ids **on the chip**. Two routes were built and
costed. One ships; the other is kept because the cost that ruled it out is the
useful part.

Both had to solve the same problem first: **the e12nano's vocabulary is 62
symbols, not the 157-entry Piper/Kristin table** in
`mcu/ports/esp32s3/firmware/main/cp_id_table.h`. That existing on-chip espeak
G2P produces ids in the wrong id space for this voice. The 62-symbol map is
`artifacts/kokoro-corpus-af_heart-20260713/kokoro_vocab.json` — a pure
single-codepoint -> id table including misaki's ASCII diphthong letters
(`A I O W Y T`) and its `ᵊ`/`ᵻ`, with no length mark, no `ɚ`, no `ʔ`.

## `lex/` — SHIPPED

A C99 port of the dictionary-first path in `pypkg/sanotts/nano_g2p.py`: misaki's
`us_gold` pronunciation dictionary, number spelling, stress rules, the
sub-tokeniser and the chunker. **Apache-2.0 data, no espeak, no GPL.**

- 1,500,446 B of flash (gold only; `us_silver` doubles it and is off by default,
  measured byte-identical on 52 Home Assistant announcements)
- 18,648 B of static RAM, no `malloc`
- HOST-REF: 1,257 B peak stack; 100.00% exact-sequence agreement with the Python
  oracle over 397 sentences; 1.470% token error rate against recorded real
  misaki; 0.57% out-of-vocabulary on a Home Assistant announcement corpus

Details, including what was ported and what was not, in `lex/README.md`.

## `idf/espeak_ng/` and `nano_g2p_table.h` — NOT SHIPPED

A vendored espeak-ng 1.52.0 translator-only ESP-IDF component plus the
espeak-IPA -> misaki rewrite table, following the existing port at
`mcu/ports/esp32s3/firmware/components/espeak-ng/`. It works and is left here
because the numbers behind the decision are worth keeping:

- **62,090 B of internal SRAM in `.bss`+`.data`, unconditional from boot** —
  over half of it `phoneme_list` and friends belonging to espeak's formant
  synthesiser, which a G2P-only port never calls but still links
- 222,744 B of flash plus a 281,248 B SPIFFS data partition
- a task stack that crashed WiFi TX at 28 KB internal in the earlier firmware
  and had to be moved to PSRAM
- **GPL-3.0**, which is fine for device firmware but must be stated
- and it is *phonetically worse*: 6.09% token error rate against misaki, against
  the lexicon path's 1.47%

62 KB of immovable internal SRAM is not affordable in a budget where the whole
question is whether a ~100 KB contiguous arena fits. That is why it did not
ship. Nothing in the ESPHome build references this directory.

`nano_g2p_table.h` and `gen_nano_g2p_table.py` belong to that path; `host/`
holds its parity harness.
