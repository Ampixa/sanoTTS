# Adding a Voice

Date: 2026-09-24

How a new language or voice gets from "a Piper teacher exists" to "it works in
the browser, in pip, and in Home Assistant".

`docs/roota-language-porting-recipe.md` already covers the first half -- the
distillation itself -- and `docs/distillation-recipe.md` covers why the
pipeline has the shape it does. **This document is the second half**, which was
undocumented until now. That gap is not theoretical: PR #17 (Polish) shipped a
correct, well-made voice that was missing seven of the steps below, including
one that would have silently deleted it.

Two rules before anything else:

- **Gate the teacher first.** A student cannot be more intelligible than what
  it copied, so a bad teacher is an answer, not a setback -- and it costs
  minutes to find out instead of a training run.
- **Several files below are GENERATED.** Hand-editing them works until someone
  regenerates, at which point your voice vanishes with no error. They are
  marked ⚠ GENERATED. Change the source, re-run the generator, commit both.

---

## Part A -- make the voice

### A1. Audition the teacher

```bash
python3 tools/audition_piper_teacher.py \
    --voices pl_PL-gosia-medium \
    --texts flores24.json --out audition.json
```

The gate is **CER ≤ 0.10**. Above that the teacher is not worth distilling.
Read the gap, not the absolute: Whisper's own error floor differs by language.

Chinese needs `zhconv` installed or every CER is inflated roughly 2x by
traditional/simplified orthography the voice never pronounced. We rejected a
perfectly good Chinese teacher twice on that mistake.

### A2. Distil

```bash
python3 tools/train_voice_from_piper.py --voice pl_PL-gosia-medium --auto-text
```

One command: teacher download, preflight, the four teacher packs, the decoder
cut, three students, the two decoder adaptation stages, the joint finetune, the
quality gate, and the export. Runs on k2/omac/cdjk, never on the laptop.

Output is `artifacts/voices/<VOICE>/` containing the `.pt` checkpoints and a
`package/` directory.

**Keep those checkpoints.** Everything in Part B derives from them, and if they
are lost the package can only be reconstructed by transcoding the web blobs --
possible (the round-trip is byte-identical), but you should not need to.

### A3. Score the student

```bash
python3 tools/audition_voice_package.py \
    --package artifacts/voices/pl_PL-gosia-medium/package \
    --teacher-audition audition.json \
    --texts flores24.json --out student-audition.json
```

Report the **retention gap** against the teacher, not the raw number.

---

## Part B -- ship it

Twelve surfaces. Missing any one of them produces a voice that is broken in
exactly one place, usually silently.

### B1. Teacher config into the repo

```
mcu/ports/wasm/voice-configs/<VOICE>.onnx.json
```

Copy the teacher's `.onnx.json`. The phoneme ids are the training-time
contract, so this must be reproducible on any checkout rather than sitting
behind a machine-specific path.

### B2. ⚠ GENERATED -- the phoneme id table

```
mcu/ports/wasm/cp_id_tables_multi.h     <- generated
tools/gen_cp_id_tables.py               <- the source: edit the VOICES list
```

Append your voice to `VOICES` in `tools/gen_cp_id_tables.py`, then:

```bash
.venv/bin/python tools/gen_cp_id_tables.py
```

**APPEND ONLY.** Slot numbers are the JS-visible contract baked into
`web/index.html`; renumbering silently repoints every existing voice at the
wrong id table.

This is the trap PR #17 fell into. The header was hand-written -- correctly,
all 152 entries matched the real config -- but `VOICES` was never touched, so
the next regeneration would have dropped Polish with no error anywhere.

### B3. eSpeak data for the browser

```
mcu/ports/wasm/espeak-data-multi/<lang>_dict          # at the root
mcu/ports/wasm/espeak-data-multi/lang/<family>/<code> # e.g. lang/zlw/pl
```

Taken from piper-tts 1.4.2's bundled `espeak-ng-data` so the WASM ids match
python `PiperVoice` exactly. `build_g2p.sh` preloads the whole directory, so no
build-script change is needed -- just put the files in the right place.

A very large dictionary is a judgement call: Russian's `ru_dict` is 8.2 MB raw
and is deliberately NOT bundled. The browser assembles it per utterance from
`web/g2p-lazy/ru/` and compiles it in-module instead. If your dictionary is
multi-megabyte, follow that path rather than doubling the download for
everyone.

### B4. Rebuild the G2P WASM module

```bash
mcu/ports/wasm/build_g2p.sh        # needs emcc on PATH
```

Regenerates `web/snt_g2p.{js,wasm,data}`. Verify the data actually grew by the
size of what you added -- that one arithmetic check catches a stale build:

```
new .data size - old .data size == <lang>_dict + lang/<family>/<code>
```

Emscripten bakes the **builder's absolute path** into `snt_g2p.js` three times.
Neutralise it to `web/snt_g2p.data` before committing; it is only the
run-dependency bookkeeping key, and a home directory should not travel in a
public repo.

### B5. ⚠ GENERATED -- the web voice bundle

```bash
.venv/bin/python tools/export_voice_bundle.py <key>
```

Produces `web/voices/<key>/{front_f32.bin,dec_f32.bin,meta.json}` from the
checkpoints.

### B6. Narrow to f16

```bash
python tools/shrink_voice_bundle_f16.py <key>
```

**Not optional for a new voice.** Every 1.56M voice ships ~3.0 MiB; f32 is
~6.0 MiB, double the download, and it falsifies the site's own "under 4 MB per
voice" claim.

This also writes the fields the integrity check needs -- `front`, `dec`,
`weights`, and both `*_widened_sha256` digests. Without them
`voiceBytesOk()` reads `expect.sha256` as `undefined` and accepts whatever
bytes arrive, so the voice ships with no checksum at all.

Narrowing is lossy, so **fp16 becomes the reference** and any CER/WER you
publish must be re-measured on it. In practice the cost is nil -- for Polish,
CER 0.0524 (f32) against 0.0519 (f16), which is ASR decode jitter -- but
measure it rather than asserting it. Do not run this over the voices released
as f32; the tool's `F32_ONLY` set names them.

### B7. The distributable package

```bash
python tools/export_roota_self_contained_package.py \
    --package-name pl-gosia-1p57m --language pl_PL --voice pl_PL-gosia-medium \
    --acoustic-checkpoint ... --duration-checkpoint ... --decoder-checkpoint ... \
    --piper-config mcu/ports/wasm/voice-configs/<VOICE>.onnx.json
```

**Naming is round-half-up on the parameter count**, and it is load-bearing:
`pypkg` resolves that exact string against Hugging Face, so a wrong name is a
name that can never exist. 1,565,324 params is `1p57m`, not `1p56m` -- German
has the identical count and ships as `de-thorsten-1p57m`.

### B8. Upload to Hugging Face

Two separate uploads to `ampixa/sanoTTS`, and both are needed:

```
<package-name>/           # for pip and Home Assistant
web/voices/<key>/         # for the website
```

There is no tool for this; it is `huggingface_hub` or the web UI. The website
tries **hf → gh → local** in that order, so skipping the second upload
silently demotes the voice to the slower GitHub raw mirror.

### B9. The website

`web/index.html`, four places:

- the `VOICES` array -- `key`, `group`, `label`, `params`, `exact`, `flag`, `emj`
- `DEFAULT_TEXT` -- a native sentence for the box
- a row in the language table
- the voice/language counts in the bullets at the top

`params` is the rounded label and `exact` is the true count; keep them
consistent with B7.

**The WER column is the Tatoeba out-of-domain run**
(`experiments/evidence/ood-tatoeba-20260908.json`): 16 sentences, numpy
runtime, per-language random draw. It is *not* the FLORES number from A3 --
different corpus, row count and runtime. Leave the cell as `&mdash;` with
class `lt-m none` unless you have a number from that protocol. An unmeasured
cell is styled differently from a measured one on purpose.

### B10. pip and Home Assistant

```
pypkg/sanotts/tables/voices.json        # package name, language, sample rate
custom_components/sanotts/const.py      # VoiceInfo(alias, label, lang)
```

The `package` string must equal the HF directory from B7 exactly. Both files
advertise the voice to users, so adding them before B8 ships a hard failure to
anyone who selects it.

Sizes in `const.py` are **measured**, not read off the pack name.

### B11. The gates

```bash
node mcu/ports/wasm/verify_g2p_node.mjs      # wasm ids == PiperVoice ids
node mcu/ports/wasm/verify_voice_node.mjs <key> ...   # end-to-end render
```

Add a row to `verify_g2p_node.mjs` (`lang`, `text`, `slot`, `espeakVoice`,
`onnx`) and a sentence to `SENTENCES` in `verify_voice_node.mjs`. Neither is
optional: a voice with no row is a voice nobody will notice breaking.

`verify_g2p_node.mjs` computes ground truth by loading the teacher ONNX through
python `PiperVoice`, so it needs `models/teachers/<VOICE>/<VOICE>.onnx` present
and it must run on k2/omac/cdjk, not the laptop.

**Known gap, documented in that file:** the WASM G2P drops a comma *and the
word boundary after it* in every language. No existing gate row contains a
comma, which is why it has never fired. Keep new rows comma-free until that is
fixed, or you will be debugging a fleet-wide bug you did not cause.

### B12. Docs and the public mirror

```
docs/public/README.public.md      # the voices table and the language counts
docs/public/manifest.txt          # only if you added a new TOOL
```

`manifest.txt` is an allowlist and `publish_public.py` mirrors only
**git-tracked** files. Directory entries (`web/`, `pypkg/`, `mcu/`,
`custom_components/`) sweep in everything beneath them, so voice files travel
automatically -- but a new tool does not, and a published tool that imports an
unpublished one produces a public repo containing a trainer that cannot start.

```bash
python3 tools/check_public_manifest_closure.py
```

---

## If you are an outside contributor

Right now you can do Part A and almost none of Part B, because the tools that
ship a voice are not in the public mirror:

| tool | public? |
| --- | --- |
| `audition_piper_teacher.py` | yes |
| `train_voice_from_piper.py` | yes |
| `audition_voice_package.py` | **no** |
| `gen_cp_id_tables.py` | **no** |
| `export_voice_bundle.py` | **no** |
| `shrink_voice_bundle_f16.py` | **no** |
| `export_roota_self_contained_package.py` | **no** |

This is the direct cause of what PR #17 ran into. That contributor hand-wrote
the id table and hand-built an f32 bundle -- both reasonable responses to the
generators being invisible, and both of the things that then needed fixing. The
id table they produced was in fact perfect, which says the gap is in the
tooling's visibility, not in their work.

Until that is resolved, open the PR with Part A done and the voice rendering,
and say plainly which of Part B you could not reach. That is a good PR.

## Where the work lives

`Ampixa/saanotts` is private and authoritative. `Ampixa/sanoTTS` is a **mirror**
produced by `publish_public.py`.

A PR merged into the public repo alone will be **overwritten** by the next
publish, because every path in Part B sits under a directory entry in
`manifest.txt`. Contributor work has to be ported back to the private repo, or
it disappears on the next mirror run.

Merge contributor PRs and improve on top of them. Never close and reimplement.

---

## Checklist

```
[ ] A1  teacher auditioned, CER <= 0.10
[ ] A2  distilled on a remote box
[ ] A3  student scored, retention reported
[ ] B1  voice-configs/<VOICE>.onnx.json
[ ] B2  gen_cp_id_tables.py VOICES appended, header REGENERATED
[ ] B3  espeak <lang>_dict + lang/<family>/<code>
[ ] B4  build_g2p.sh re-run, .data growth checked, baked path neutralised
[ ] B5  export_voice_bundle.py
[ ] B6  shrink_voice_bundle_f16.py, CER re-measured on f16
[ ] B7  package exported, name = round-half-up param count
[ ] B8  BOTH HF uploads: <package-name>/ and web/voices/<key>/
[ ] B9  index.html: VOICES, DEFAULT_TEXT, table row, counts
[ ] B10 voices.json + const.py, package string matches B7 exactly
[ ] B11 rows added to BOTH gates, both run green
[ ] B12 README.public.md, manifest closure check
[ ] --  ported to the PRIVATE repo
```
