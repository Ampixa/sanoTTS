# Embedded golden model

The firmware `EMBED_FILES` the five `*.bin` here (int8 model + a reference
utterance for the boot self-test). They are **not stored in this directory** —
they are the canonical test fixtures at `mcu/test/fixtures/en_us_r7/` (checksummed
there). Stage them before building:

```bash
./stage.sh      # copies model_q8/front_q8/e2e_{ids,durs,audio}.bin from the fixtures
```

`build.sh` runs this automatically.
