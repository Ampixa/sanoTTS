# en_us_e13b Golden Fixture

E13 Arm B final: standard-ops w62 decoder (tiny-vocos-student.pt, step 250000) over the h31 QAT acoustic + the E12 nano duration student; 294,279 params; exported on k2 2026-09-02

Layout, contract and verification are identical to
`en_us_e12nano` (see its README): blobs + frozen durations +
float PyTorch reference waveforms + the split noise contract.
Assembled by `tools/make_nano_fixture.sh` from a
`tools/export_e12_nano_q8.py` export; the export report is
`export-report.json` beside the original export.

Operators (from `mcu/models/en_us_e13b/nano_q8_meta.h`):
norm_type=0 (0=LayerNorm 1=DyT), act_type=0
(0=GELU 1=ReLU), decoder width 62.

```bash
cd mcu/test/fixtures/en_us_e13b && shasum -a 256 -c SHA256SUMS && cd -
make -C mcu test-nano NANO_GOLDEN=test/fixtures/en_us_e13b NANO_MODEL=models/en_us_e13b
```
