# saanoTTS Language Porting Recipe

Date: 2026-06-29

This is the reusable saanoTTS recipe for making a tiny Piper-distilled neural
TTS voice for a new language. It is based on the Root A Nepali, English, Hindi,
and Chinese runs.

The recipe is not "train TTS from scratch." It is:

```text
one deterministic Piper/VITS teacher
-> Piper-compatible phoneme IDs
-> tiny duration student
-> tiny acoustic latent student
-> tiny decoder student
-> raw fp16 neural package
```

The current package boundary is phoneme IDs in, waveform out. A complete
arbitrary-text product still needs the language frontend packaged or reimplemented.

## When This Recipe Is Appropriate

Use this recipe when:

- your language already has a usable Piper/VITS voice;
- the teacher voice is clean enough that distilling it is worth doing;
- you can accept a Piper/eSpeak frontend dependency at first;
- your target is a tiny neural runtime around `1.4M-1.5M` parameters, not SOTA
  naturalness;
- you are willing to judge with teacher/oracle/full-student lanes, not just
  parameter count.

Do not use it when:

- the only teacher voice is noisy, fast, unstable, or stylistically wrong;
- the license does not allow distillation/release;
- you need a fully standalone text frontend on day one;
- you only have disconnected audio/text data but no working teacher model.

## Required Inputs

For one language/voice:

```text
TEACHER_ONNX=/path/to/voice.onnx
TEACHER_JSON=/path/to/voice.onnx.json
TEACHER_DECODER_ONNX=/path/to/decoder-from-generator-input.onnx
SOURCE_JSONL=/path/to/text-or-transcript-lines.jsonl
LANGUAGE_KEY=<language-slug>
VOICE_KEY=<voice-slug>
SAMPLE_RATE=<read-from-TEACHER_JSON>
ROOT=artifacts/sub10m-search/root-a-piper-vits
```

Each JSONL row should have at least text. Rows with source duration metadata are
better, but text-only rows are allowed because the teacher generates the target
audio/tensors.

Before doing anything else, open the actual config and record:

- `phoneme_type`
- `phoneme_id_map`
- `audio.sample_rate`
- `espeak.voice` if present
- speaker count
- license/provenance

Do not assume these fields from the language name.

## Stage 0: Teacher Check

First decide whether the teacher is good enough. If the teacher is unpleasant,
do not distill it. A tiny student will mostly preserve the wrong target.

Probe the graph and render a few deterministic examples:

```bash
.venv/bin/python tools/probe_piper_vits_onnx.py \
  --model "$TEACHER_ONNX" \
  --config "$TEACHER_JSON" \
  --out-dir "$ROOT/$VOICE_KEY-probe" \
  --text "This is a short deterministic teacher check." \
  --noise-scale 0 \
  --length-scale 1 \
  --noise-w 0
```

The important tensors are:

- `phoneme_ids`
- `w_ceil`
- `generator_input`
- waveform output

For Piper/VITS voices, `generator_input` is the acoustic target. Do not convert
the problem to mel-spectrograms unless you deliberately change the whole recipe.

## Stage 1: Build Native Teacher Packs

Build three packs from the same teacher:

- smoke/listening pack;
- held-out eval pack;
- train pack.

Example:

```bash
.venv/bin/python tools/build_piper_vits_roota_probe_pack.py \
  --model "$TEACHER_ONNX" \
  --config "$TEACHER_JSON" \
  --source-jsonl "$SOURCE_JSONL" \
  --out-dir "$ROOT/$VOICE_KEY-smoke12-piper-native" \
  --max-rows 12 \
  --allow-text-only-source \
  --tensor-mode decoder \
  --noise-scale 0 \
  --length-scale 1 \
  --noise-w 0

.venv/bin/python tools/build_piper_vits_roota_probe_pack.py \
  --model "$TEACHER_ONNX" \
  --config "$TEACHER_JSON" \
  --source-jsonl "$SOURCE_JSONL" \
  --out-dir "$ROOT/$VOICE_KEY-eval128-piper-native" \
  --max-rows 128 \
  --skip-rows 12 \
  --allow-text-only-source \
  --tensor-mode decoder \
  --noise-scale 0 \
  --length-scale 1 \
  --noise-w 0

.venv/bin/python tools/build_piper_vits_roota_probe_pack.py \
  --model "$TEACHER_ONNX" \
  --config "$TEACHER_JSON" \
  --source-jsonl "$SOURCE_JSONL" \
  --out-dir "$ROOT/$VOICE_KEY-train2048-piper-native" \
  --max-rows 2048 \
  --skip-rows 140 \
  --allow-text-only-source \
  --tensor-mode acoustic \
  --noise-scale 0 \
  --length-scale 1 \
  --noise-w 0
```

For decoder training, create a first-chunk train subset that retains
feature-rich decoder tensors:

```bash
.venv/bin/python tools/build_piper_vits_roota_probe_pack.py \
  --model "$TEACHER_ONNX" \
  --config "$TEACHER_JSON" \
  --source-jsonl "$SOURCE_JSONL" \
  --out-dir "$ROOT/$VOICE_KEY-train512-decoder-piper-native" \
  --max-rows 512 \
  --skip-rows 140 \
  --allow-text-only-source \
  --tensor-mode decoder \
  --noise-scale 0 \
  --length-scale 1 \
  --noise-w 0
```

Validation rule: pack rows must be deterministic and internally coherent. The
sum of `w_ceil` should match `generator_input` frames, and waveform length
should match the decoder hop.

After the smoke pack exists, create and validate a decoder-cut ONNX that accepts
the stored `generator_input` directly:

```bash
.venv/bin/python tools/extract_piper_vits_decoder_cut.py \
  --model "$TEACHER_ONNX" \
  --pack-dir "$ROOT/$VOICE_KEY-smoke12-piper-native" \
  --out-dir "$ROOT/$VOICE_KEY-decoder-cut-smoke"
```

Set `TEACHER_DECODER_ONNX` to the resulting
`*-decoder-from-generator-input.onnx`. Decoder training and signature extraction
should feed the latent from the pack, not replay the whole teacher graph and
hope the stochastic upstream path lands on the same frames.

## Stage 2: Train Duration Student

Start small. Duration is rarely the main blocker once the model gets close to
the teacher frame count.

Reference shape:

```bash
.venv/bin/python tools/train_roota_piper_duration_student.py \
  --pack-dir "$ROOT/$VOICE_KEY-train2048-piper-native" \
  --eval-pack-dir "$ROOT/$VOICE_KEY-eval128-piper-native" \
  --out-dir "$ROOT/$VOICE_KEY-a5-duration-h64d3-4000" \
  --hidden 64 \
  --depth 3 \
  --kernel-size 5 \
  --steps 4000 \
  --device auto
```

Promote duration when:

- predicted/teacher total frame ratio is close to `1.0`;
- token duration MAE is stable on held-out;
- oracle-duration and learned-duration full-stack lanes sound similar.

If oracle duration fixes the model, tune duration. If it barely changes the
model, stop spending compute there.

## Stage 3: Train Acoustic Latent Student

The acoustic model predicts `generator_input`, not waveform.

Reference shape:

```bash
.venv/bin/python tools/train_roota_piper_latent_student.py \
  --pack-dir "$ROOT/$VOICE_KEY-train2048-piper-native" \
  --eval-pack-dir "$ROOT/$VOICE_KEY-eval128-piper-native" \
  --out-dir "$ROOT/$VOICE_KEY-a16-acoustic-tokenctx-h96" \
  --architecture token_context \
  --hidden 96 \
  --token-depth 3 \
  --depth 4 \
  --kernel-size 5 \
  --norm-l1-weight 0.25 \
  --delta-l1-weight 0.10 \
  --channel-stat-weight 0.05 \
  --steps 2000 \
  --device auto
```

Do not promote from latent cosine alone. Always render through a decoder lane.
The recurring failure is that latent metrics look good while compact-decoder
audio collapses because the predicted latent is off the decoder's accepted
manifold.

## Stage 4: Verify Teacher Decoder Parity

Before shrinking the decoder, verify that the local PyTorch teacher-decoder
reconstruction matches Piper's ONNX decoder path:

```bash
.venv/bin/python tools/verify_piper_decoder_torch_parity.py \
  --decoder "$TEACHER_DECODER_ONNX" \
  --pack-dir "$ROOT/$VOICE_KEY-eval128-piper-native" \
  --out-dir "$ROOT/$VOICE_KEY-a72-piper-decoder-torch-parity" \
  --rows 32 \
  --export-checkpoint
```

Do not train a compact decoder against a teacher implementation that has not
passed parity. Otherwise you cannot tell whether failures come from compression
or from a mismatched teacher path.

## Stage 5: Build Decoder Activation Signatures

Activation signatures give the compact decoder more teacher-structure signal
than waveform loss alone.

Do not rely on the script's default nodes for a new language. Those defaults are
historical Chitwan-era node names. First inspect the teacher ONNX and record the
actual decoder value names. Official Piper exports usually expose names like:

```text
/dec/conv_pre/Conv_output_0
/dec/ups.0/ConvTranspose_output_0
/dec/Div_output_0
/dec/ups.1/ConvTranspose_output_0
/dec/Div_1_output_0
/dec/ups.2/ConvTranspose_output_0
/dec/Div_2_output_0
/dec/conv_post/Conv_output_0
/dec/Tanh_output_0
```

Then pass those names explicitly:

```bash
.venv/bin/python tools/build_piper_vits_decoder_signature_pack.py \
  --model "$TEACHER_ONNX" \
  --pack-dir "$ROOT/$VOICE_KEY-train512-decoder-piper-native" \
  --out-dir "$ROOT/$VOICE_KEY-a26-decoder-signatures-train512" \
  --feed-latent-from-pack \
  --dtype float16 \
  --node /dec/conv_pre/Conv_output_0 \
  --node /dec/ups.0/ConvTranspose_output_0 \
  --node /dec/Div_output_0 \
  --node /dec/ups.1/ConvTranspose_output_0 \
  --node /dec/Div_1_output_0 \
  --node /dec/ups.2/ConvTranspose_output_0 \
  --node /dec/Div_2_output_0 \
  --node /dec/conv_post/Conv_output_0 \
  --node /dec/Tanh_output_0
```

For advanced runs, add exact features only after proving the base signature
recipe works. Exact features can be large and can over-constrain the wrong
nodes.

## Stage 6: Train Compact Decoder

Start with a teacher-initialized `piperlite` decoder. Random tiny decoders were
not the winning route.

Reference compact decoder around 640k params:

```bash
.venv/bin/python tools/train_roota_piper_decoder_student.py \
  --pack-dir "$ROOT/$VOICE_KEY-train512-decoder-piper-native" \
  --eval-pack-dir "$ROOT/$VOICE_KEY-eval128-piper-native" \
  --teacher-decoder "$TEACHER_DECODER_ONNX" \
  --teacher-init-checkpoint "$ROOT/$VOICE_KEY-a72-piper-decoder-torch-parity/piper-decoder-teacher.pt" \
  --teacher-init-method importance \
  --acoustic-checkpoint "$ROOT/$VOICE_KEY-a16-acoustic-tokenctx-h96/latent-student.pt" \
  --signature-pack-dir "$ROOT/$VOICE_KEY-a26-decoder-signatures-train512" \
  --out-dir "$ROOT/$VOICE_KEY-a91-decoder-piperlite-640k" \
  --variant piperlite \
  --channels 160,80,40,20 \
  --activation leaky_relu \
  --signature-hint-weight 0.05 \
  --signature-temporal-weight 0.4 \
  --feature-hint-weight 0.05 \
  --quiet-ceiling-weight 0.010 \
  --quiet-ceiling-margin-db 0.5 \
  --adv-weight 0.075 \
  --adv-feature-weight 0.75 \
  --adv-start-step 1500 \
  --steps 2800 \
  --lr 2e-4 \
  --device auto
```

The exact width/rank should be adjusted to your budget. The proven pattern is:

- teacher initialization first;
- decoder-cut ONNX as the training teacher;
- direct waveform and feature losses;
- activation signatures;
- quiet/click artifact gates;
- adversarial losses as training-only helpers, not runtime dependencies.

Strict compression branch for English used factorized `piperlite`:

```text
channels=152,76,38,19
factorized_pre_rank=96
piper_res_factor_rank_ratio=0.65
```

That landed at `1,390,936` total params for Kristin A105, but quality was still
below promotion. Treat factorization as a budget lever, not as a guaranteed
quality win.

## Stage 7: Evaluate With Lanes

A candidate is not real until the lane decomposition is run.

Minimum lanes:

```text
teacher Piper output
teacher latent -> full Piper decoder
teacher latent -> compact decoder
student duration + student acoustic -> compact decoder
oracle duration + student acoustic -> compact decoder
```

Run the single-language loop:

```bash
.venv/bin/python tools/run_roota_feedback_loop.py \
  --eval-pack "$ROOT/$VOICE_KEY-eval128-piper-native" \
  --acoustic-checkpoint "$ROOT/$VOICE_KEY-a16-acoustic-tokenctx-h96/latent-student.pt" \
  --duration-checkpoint "$ROOT/$VOICE_KEY-a5-duration-h64d3-4000/duration-student.pt" \
  --piper-model "$TEACHER_ONNX" \
  --piper-config "$TEACHER_JSON" \
  --decoder-report "$ROOT/$VOICE_KEY-a91-decoder-piperlite-640k/train-report.json" \
  --candidate-label "$VOICE_KEY-a91" \
  --out-dir "$ROOT/feedback-loops/$VOICE_KEY-a91" \
  --rows 128 \
  --param-budget 1500000 \
  --force-render \
  --force-score \
  --device auto
```

Then summarize gates:

```bash
.venv/bin/python tools/summarize_roota_feedback_loop.py \
  --scoreq-summary "$ROOT/feedback-loops/$VOICE_KEY-a91/scoreq-summary.json" \
  --out-dir "$ROOT/feedback-loops/$VOICE_KEY-a91/summary" \
  --param-budget 1500000
```

If the teacher itself scores badly under ASR or MOS predictors, do not blindly
use those as absolute gates. Use teacher-calibrated deltas and manual listening.

## Stage 8: Serve Arbitrary Text Dashboard

Closed-set dashboards are not enough. Serve arbitrary text before claiming a
voice works:

```bash
.venv/bin/python tools/serve_roota_arbitrary_tts_dashboard.py \
  --host 0.0.0.0 \
  --port 8900 \
  --acoustic-checkpoint "$ROOT/$VOICE_KEY-a16-acoustic-tokenctx-h96/latent-student.pt" \
  --duration-checkpoint "$ROOT/$VOICE_KEY-a5-duration-h64d3-4000/duration-student.pt" \
  --decoder-backend student \
  --decoder-student-checkpoint "$ROOT/$VOICE_KEY-a91-decoder-piperlite-640k/decoder-student.pt" \
  --piper-model "$TEACHER_ONNX" \
  --piper-config "$TEACHER_JSON" \
  --out-dir "$ROOT/live/$VOICE_KEY-a91" \
  --duration-length-scale 1.0 \
  --noise-scale 0 \
  --length-scale 1 \
  --noise-w 0 \
  --dashboard-title "$VOICE_KEY Root A tiny TTS"
```

Listen for:

- robotic buzz;
- clicks;
- quiet high-frequency hiss;
- duration drift;
- fast/slow speaking rate;
- pronunciation regressions on numbers, loan words, and uncommon names;
- mismatch between teacher-latent decoder and full-student lanes.

## Stage 9: Export Package

Once the component tuple is chosen, export raw fp16 runtime assets:

```bash
.venv/bin/python tools/export_roota_self_contained_package.py \
  --package-name "$VOICE_KEY-roota-tiny" \
  --language "$LANGUAGE_KEY" \
  --voice "$VOICE_KEY" \
  --acoustic-checkpoint "$ROOT/$VOICE_KEY-a16-acoustic-tokenctx-h96/latent-student.pt" \
  --duration-checkpoint "$ROOT/$VOICE_KEY-a5-duration-h64d3-4000/duration-student.pt" \
  --decoder-checkpoint "$ROOT/$VOICE_KEY-a91-decoder-piperlite-640k/decoder-student.pt" \
  --piper-config "$TEACHER_JSON" \
  --sample-rate "$SAMPLE_RATE" \
  --duration-length-scale 1.0 \
  --out-dir "$ROOT/packages/$VOICE_KEY-roota-tiny-fp16"
```

The package should contain:

```text
weights.fp16.bin
manifest.json
piper-phoneme-config.json
runtime-kernels.md
README.md
```

It should not contain:

```text
*.pt
*.onnx
optimizer state
training logs required for inference
```

The manifest is the source of truth for parameter counts and byte sizes.

## Failure Diagnosis

Use this table before changing knobs:

| Symptom | Likely blocker | Next action |
| --- | --- | --- |
| Teacher itself sounds bad | Teacher choice | Pick another Piper voice. |
| Teacher-latent -> compact decoder is bad | Decoder | Change decoder topology/init/loss. Do not retrain duration/acoustic first. |
| Teacher-latent decoder is good, full student is bad | Acoustic/interface | Add interface-aware acoustic calibration; inspect latent interpolation. |
| Oracle duration fixes full student | Duration | Tune duration scale/model. |
| Only arbitrary text fails | Frontend/text normalization | Fix phonemization/normalization, not model size. |
| Metrics improve but listening worsens | Bad metric/overfit | Demote that metric; add listening and artifact gates. |
| Training sentences sound good but held-out fails | Memorization | Rebuild train/eval split and reduce closed-set claims. |

## Minimum Release Checklist

- [ ] Teacher license and provenance recorded.
- [ ] Teacher audition passed by listening.
- [ ] Piper config and phoneme contract frozen.
- [ ] Smoke, train, and held-out packs generated from the same teacher path.
- [ ] Duration, acoustic, and decoder checkpoints trained per language.
- [ ] Teacher decoder parity verified before decoder compression.
- [ ] Fixed held-out lane evaluation completed.
- [ ] Arbitrary-text dashboard listened to.
- [ ] Package exported as raw fp16 assets.
- [ ] Manifest confirms parameter and byte budgets.
- [ ] README states that frontend is external unless packaged.

## What To Publish

A responsible release should say:

```text
This is a tiny Piper-distilled neural runtime package. It starts from
Piper-compatible phoneme IDs and produces waveform audio. It is not yet a fully
standalone arbitrary-text TTS runtime unless the frontend is included.
```

A responsible paper/report claim is:

```text
We show a repeatable recipe for distilling Piper/VITS voices into roughly
1.4M-1.5M parameter neural stacks across multiple languages, and we provide a
lane-based diagnostic loop that identifies whether duration, acoustic latent,
or decoder compression is the current quality bottleneck.
```
