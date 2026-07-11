# Minimal en-US espeak-ng data (275 KB)

The voice data espeak-ng loads at runtime, flashed to the `espeak` SPIFFS
partition and mounted at `/espeak`. This is the **minimal en-US subset**, not
the full 17.5 MB multi-language install.

Provenance: derived from piper-tts's bundled `espeak-ng-data`. Only the files
en-US phonemization touches are kept:

| file | purpose |
| --- | --- |
| `en_dict` | English pronunciation dictionary + rules (164 KB) |
| `phontab`, `phonindex` | phoneme tables |
| `intonations` | intonation data |
| `phondata` | **4 KB stub** — espeak checks its version header but never reads the body for phoneme-only output (the real 542 KB file is the formant synthesizer, unused here) |
| `lang/gmw/en*` | en voice definitions |

Everything else (other languages, the full `phondata`) is dropped. Verified: this
275 KB set produces byte-identical phoneme IDs to the full install.
