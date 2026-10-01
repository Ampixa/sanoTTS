# Distilling Kokoro

Date: 2026-09-25

How a Kokoro voice becomes a shipped sanoTTS voice. This is the chain behind
`heart` (2,272,145 params) and `heartnano` (294,279) — the **nano lineage**.

`docs/adding-a-voice.md` is the Piper path and does not apply here. Almost none
of it transfers, because **Kokoro is StyleTTS2-family, not VITS**: there is no
teacher latent to cut, slice or graft. The Piper path distils a teacher's
internals. This one renders the teacher to audio and distils from the audio.

Read `docs/MEL100-ROUTE.md` first for why the chain has this shape, and
`docs/FEEDBACK-LOOP.md` before you quote any number at anybody.

> **A note on paths.** This document is mirrored into the public repo from the
> private research repo. Most `tools/` scripts it names are NOT in the public
> mirror, and references marked *(internal)* point at documents that stay
> private. The commands are recorded so the recipe is legible and auditable,
> not because every one of them is runnable from a public checkout.

---

## The one sentence that matters

**The recipe is intact and re-runnable from text. The artifact is not, and no
number measured before 2026-08-21 can be compared to anything you produce
today.**

Two undocumented wipes took the corpora, every training-split pack, and both
shipped checkpoints. `~/saanotts/artifacts/` does not exist on k2 at all. The
shipped voices survive **only as exported blobs**. And because the original
corpus recorded no Kokoro version, voice or speed, a regenerated corpus is a
*different teacher* — so build your own paired control and do not compare
against the historical table below.

---

## The chain

```
text
  --Kokoro 82M render-------------> 24 kHz audio + manifest        (stage 1)
  --Kokoro front, no decode-------> phonemes, token_ids, pred_dur  (stage 2)
  --pack build--------------------> rows.json + PCM16 + NPZ        (stage 3)
  --Vocos featurizer--------------> mel-100 targets                (stage 4)
  --frozen Vocos------------------> teacher waveforms              (stage 5)

espeak/misaki -> ids
  --duration student--------------> frames per phoneme             (stage 6)
  --acoustic student--------------> mel-100 per frame              (stage 7)
  --precompute predicted mel------> input-matched distil source    (stage 8)
  --ConvNeXt-1D + iSTFT decoder---> 24 kHz waveform                (stage 9)
  --render two lanes + gate------------------------------------->  (stage 10)
  --export--------------------------------------------------------> (stage 11)
```

Note what the decoder's teacher is: **frozen `charactr/vocos-mel-24khz`**, a
third-party 13M vocoder — *not* Kokoro. Kokoro supplies the corpus audio and the
durations. Vocos supplies the phase competence. Only two genuinely
Kokoro-internal signals are used anywhere: `pred_dur`, and optionally F0/N
(abandoned, see below).

---

## Before you start

### Three virtualenvs, and getting it wrong fails late

| env | holds | used by |
| --- | --- | --- |
| `~/kokoro-venv` | kokoro 0.9.4 + misaki 0.9.4 | stages 1, 2 |
| `venv` (repo) | torch, scoreq, torchcodec | stages 3, 6–11 |
| `~/venv-vocos` | vocos 0.1.0, CPU, **no onnxruntime** | stages 4, 5, Vocos renderers |

SCOREQ **silently returns NaN without torchcodec**. Everything runs `nice`'d on
k2/omac/cdjk, never on the laptop.

### ⚠ Six of the eleven tools are not where the docs say

Campaign tools live in **two layouts on purpose**: `tools/<name>.py` on k2,
`tools/campaign-e8/<name>.py` in the repo. Every docstring and every
reproduction block in every doc writes the k2 path. `tools/nano_paths.py` exists
solely to make both work — but it fixes the paths *inside* the tools, not the
commands you copy out of a doc.

In the repo, these are under `tools/campaign-e8/`: `build_mel100_targets.py`,
`build_e8a_vocos_teacher_waves.py`, `train_tiny_vocos_student.py`,
`render_tiny_vocos_student.py`, `render_fullstack_tiny.py`,
`render_fullstack_mel100.py`.

### There is no driver

Eleven loose tools, no chain script, no chain document. `kokoro_83_chain.sh` and
`e13_armb_orchestrate.sh` exist only on k2 and drive the **abandoned mel-80
route**; so does the only in-repo chain script,
`tools/run_kokoro_wavehax_chain.sh`. You are hand-sequencing. Writing the driver
is the highest-value thing anyone could add here.

### Disk

The 49,750-row corpus decompresses to ~37 GB and the packs are larger. Check
free space on the training host *before* stage 1, not at stage 3.

`A=artifacts/kokoro-corpus-af_heart-<date>` throughout.

---

## Stage 1 — render the teacher corpus

```bash
~/kokoro-venv/bin/python tools/build_kokoro_corpus.py \
    --texts corpus.txt --out-dir $A --device mps \
    --voice af_heart --lang-code a --speed 1.0 --repo-id hexgrad/Kokoro-82M
```

Produces `audio/000001.flac` (24 kHz mono), `manifest.jsonl` (`{i, text, wav}`),
`corpus-provenance.json`.

**One sentence per line, and keep them short.** Kokoro chunks past 510 phoneme
characters and stage 3 hard-rejects any row that chunked, so a long paragraph
costs you the row.

**Bring your own text.** The 49,999-row corpus behind `heart-nano` came from a
script that is not in this repo. `data/textsets/multilang-distill-v1/en_US.train50k.jsonl`
holds 50,000 usable rows. This reproduces the *recipe*, not that corpus.

The provenance file exists because the original run recorded none, which is why
the shipped teacher is not verifiably reproducible. Do not skip it.

## Stage 2 — extract Kokoro's front

```bash
~/kokoro-venv/bin/python tools/kokoro_extract_front.py \
    --manifest $A/manifest.jsonl --out $A/kokoro-front.jsonl \
    --voice af_heart --lang-code a --speed 1.0 --device auto \
    --validate-against-audio --max-frame-drift 2 --vocab-out $A/kokoro_vocab.json
```

Reproduces `KModel.forward_with_tokens` steps 1–7 and stops before the decoder.
The duration unit is **600 samples** (`prod(upsample_rates) * gen_istft_hop_size
= 300`, times one more stride-2 ConvTranspose in `Decoder.decode[3]`).

⚠ **Hard-coded macOS paths.** Lines 44–47 pin
`/opt/homebrew/lib/libespeak-ng.dylib` and
`/opt/homebrew/share/espeak-ng-data`. Non-brew hosts fail at import.

⚠ **The 62-symbol vocab is built corpus-locally**, not taken from Kokoro:
`<pad>0 <bos>1 <eos>2` then observed characters *sorted* from 3 up. **Two
corpora with different coverage produce silently incompatible vocabs**, and
therefore incompatible checkpoints. Keep `kokoro_vocab.json` with the
checkpoints forever.

## Stage 3 — build the Root-A-schema pack

```bash
venv/bin/python tools/build_kokoro_pack.py \
    --front $A/kokoro-front.jsonl --audio-root $A --out-dir $A/packs/train \
    --row-modulo 200 --row-remainder 0 --invert-selection \
    --sample-drift-tolerance-frames 2
# eval split: same, minus --invert-selection, --out-dir $A/packs/eval
```

Writes **PCM16 mono WAV** (not the source FLAC — the decoder trainer's loader
uses Python's `wave` and rejects FLAC) plus NPZ with `phoneme_ids`, `w_ceil`,
`generator_input`.

The 40 fps → 93.75 fps grid conversion is the load-bearing arithmetic:

```
cum_samples = cumsum(durations_kokoro) * 600
cum_mel     = round(cum_samples / 256)
cum_mel[-1] = target_frames            # "adjust last token"
w_ceil      = diff(monotonic_clamp(cum_mel))
```

`generator_input` is a **misnomer kept to satisfy the Piper loaders** — it holds
an 80-bin log-mel of the row's own audio, not a teacher internal. Stage 7 loads
and discards it, but you must still build it.

**Do not pass `--f0-conditioning` or `--n-pred-channel`.** Those build the
82/83-channel packs of the abandoned mel-80 route.

## Stage 4 — Vocos-native mel-100 targets

```bash
~/venv-vocos/bin/python tools/campaign-e8/build_mel100_targets.py \
    --pack-dir $A/packs/train --out-dir $A/packs-mel100/train-targets
```

This is the stage the whole route turns on, and it self-validates three ways: it
asserts ten fields of the real `vocos.feature_extractor` (sr 24000, n_fft 1024,
hop 256, win 1024, n_mels 100, **power 1.0**, center True, **mel_scale htk**);
it trims `center=True`'s extra frame to `[:T]` and aborts if a +1-frame shift
correlates better on more than half the rows; and it re-checks
`audio.shape[0] == target_frames * 256`.

## Stage 5 — frozen-Vocos teacher waveforms

```bash
~/venv-vocos/bin/python tools/campaign-e8/build_e8a_vocos_teacher_waves.py \
    --target-dir $A/packs-mel100/train-targets --out-dir $A/teacher-waves/train
```

Float32 WAV, `T` frames → exactly `(T-1)*256` samples, matching the student's
`torch.istft(center=True)` by construction. **These files are the decoder's
teacher**, and their STFT is its phase supervision.

> Stages 1–5 were verified end-to-end on k2 from a bare text file on 2026-09-08
> (commit `6bdf0b3`): corpus renders, front validates 3/3, pack builds at drift
> 0.0, mel-100 at corr 0.93–0.95, teacher waves out. **Stages 6–11 have no such
> recent end-to-end record.**

## Stage 6 — duration student (the Piper tool, unchanged)

```bash
venv/bin/python tools/train_roota_piper_duration_student.py \
    --pack-dir $A/packs/train --eval-pack-dir $A/packs/eval \
    --hidden 64 --depth 3 --kernel-size 5 --steps 8000 --out-dir $A/runs/duration
```

`hidden 64` → 131,652 params (heart); `hidden 26` → 22,858 (heartnano).

`--eval-pack-dir` does **no** checkpoint selection — the final file is
bit-identical to the last periodic save, and there is no `best_`/`argmin`
anywhere in any of the three trainers. It neither leaks nor selects.

⚠ **`packs/eval` is the DEV set, not the test set.** It is the right thing to
pass here and the right thing to watch, but numbers measured on it may not be
quoted outward: it is also what arms get killed on, so it accumulates selection
bias across a campaign. Headline numbers come from the sealed test set. Before
you start, prove the pack is disjoint from both:

```bash
python3 tools/check_eval_seal.py --train-pack $A/packs/train --dev-pack $A/packs/eval
```

See `docs/FEEDBACK-LOOP.md` §7. This is not hypothetical — one system trained on
233 of the 249 dev rows and was caught only because somebody read its
run-config.

## Stage 7 — acoustic student, retargeted to mel-100

```bash
venv/bin/python tools/train_roota_piper_latent_student.py \
    --pack-dir $A/packs/train \
    --target-dir $A/packs-mel100/train-targets --target-key mel100 \
    --architecture token_context --hidden 96 --depth 4 --token-depth 3 \
    --kernel-size 5 --vocab-size 62 --steps 200000 \
    --out-dir $A/runs/e6-mel100-fullcorpus
```

`out_channels` is inferred from the target NPZ shape, not passed. `h96 d4` →
681,227 (heart); `h31 d3 --qat c-int8` → 65,299 (heartnano).

## Stage 8 — precompute the acoustic's predicted mel-100

```bash
venv/bin/python tools/precompute_acoustic_mel100.py \
    --checkpoint $A/runs/e6-mel100-fullcorpus/latent-student.pt \
    --pack-dir $A/packs/train --out-dir $A/packs-mel100/pred-train \
    --device cpu --threads 4 --expected-channels 100 --target-key mel100
```

Feeds stage 9's input-matched distillation. Asserts `out_channels == 100`.

## Stage 9 — the decoder

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 nice -n 19 venv/bin/python \
  tools/campaign-e8/train_tiny_vocos_student.py \
    --target-dir $A/packs-mel100/train-targets \
    --teacher-dir $A/teacher-waves/train \
    --mix-pred-dir $A/packs-mel100/pred-train --mix-prob 0.5 \
    --out-dir $A/runs/e8h-tinyvocos-noisefed \
    --width 192 --num-layers 5 --noise-channels 4 \
    --phase-loss-mode warmup-off --phase-loss-steps 50000 \
    --crop-frames 32 --batch-size 8 --steps 250000 \
    --lr 1e-4 --rms-weight 5.0 \
    --adv-weight 0.1 --adv-feature-weight 0.5 --adv-start-step 1 --disc-lr 5e-5 \
    --adam-beta1 0.8 --adam-beta2 0.99 --adam-eps 1e-9 \
    --checkpoint-every 25000 --periodic-disc-checkpoints drop \
    --seed 8801 --device mps
```

heart ships step **200000**; heartnano is `--width 62 --num-layers 4
--norm-type layernorm --act-type gelu` at step **250000**.

Architecture: mel-100 → Conv1d embed k7 + zero-init noise adapter → N
ConvNeXt-1D blocks (dw k7, 3× pointwise, LayerNorm, GELU, LayerScale) → Linear
to 1026 = 513 log-mags + 513 cos/sin phases, *exactly Vocos' ISTFTHead
parametrisation* → `torch.istft` → DC-blocking high-pass. **No upsampling
anywhere**: the trunk runs at frame rate, one iSTFT frame per mel frame.

Four fixes are baked into those flags, each from a failed run:

- **DC offset (E8f)** was phase collapse at bin 0 — the head settled at φ=0 so
  per-frame DC was always `+|mag|`. Fixed structurally: bins 0/Nyquist zeroed,
  plus a first-order high-pass `R=0.9973` applied identically to student and
  teacher so sub-10 Hz can never be the discriminator's tell.
- **Over-drive (E8g)**: the GAN drove 1.9–2.2× over-drive and clipped at int16.
  No limiter — `--rms-weight` only.
- **Input-matched distillation (E8g)**: `--mix-prob 0.5` feeds the *predicted*
  mel while the wave target stays the GT-mel Vocos wave, so the decoder learns
  to compensate acoustic blur.
- **Crop edges**: a crop's iSTFT lacks overlap-add context, so losses drop 2
  edge frames or 512 edge samples per side.

**Auto-abort**: the GAN exits code 3 if mean discriminator loss over
`adv-start+899..+1099` falls below 0.05 — the trivial-tell collapse signature.

⚠ **Do not explain this recipe by the noise channels.** The registered
hypothesis was that supplied noise carries stochastic content;
`X2c-noise-dose-response` **refuted it one day later** — the decoder does not
depend on its noise input. The 4 channels are in the shipped graph and counted,
but the win belongs to phase-loss release, crop-32 and 200k steps. The stack
works; its original rationale does not.

⚠ **MPS NaN**: torch 2.12.1 intermittently returns NaN gradients for batches
that replay finite and bit-identical on CPU. Persistent non-finite steps abort.

## Stage 10 — render both lanes and gate

```bash
# oracle lane: decoder fed ground-truth mel-100
venv/bin/python tools/campaign-e8/render_tiny_vocos_student.py \
    --checkpoint $A/runs/e8h-tinyvocos-noisefed/tiny-vocos-student.pt \
    --target-dir $A/packs-mel100/eval-targets \
    --teacher-dir $A/teacher-waves/eval --out-dir $A/runs/.../renders-oracle

# full stack
venv/bin/python tools/campaign-e8/render_fullstack_tiny.py \
    --duration-checkpoint $A/runs/duration/duration-student.pt \
    --acoustic-checkpoint $A/runs/e6-mel100-fullcorpus/latent-student.pt \
    --decoder-checkpoint  $A/runs/e8h-tinyvocos-noisefed/tiny-vocos-student.pt \
    --pack-dir $A/packs/eval --out-dir $A/runs/.../renders-fullstack
```

Noise is seeded per row from `sha256(row_id)`, so renders are bit-reproducible.
`render_fullstack_mel100.py` is the same chain through **frozen Vocos** — the
oracle ceiling — and must run in `~/venv-vocos`.

**Always read the two lanes together.** Oracle isolates the decoder; the gap to
full stack is the acoustic lane. For heart, 53% of the deficit is the decoder;
for the nano, 79%.

⚠ These renders are on **dev** (`packs/eval`). Use them to choose. For any
number that leaves the building, render the sealed test set
(`data/textsets/multilang-distill-v1/en_US.sealed-test.jsonl`, 224 rows, packed
with `tools/make_pack_from_text.py`) and report that instead — **once**, after the
shipping decision is already made on dev.

## Stage 11 — export

```bash
venv/bin/python tools/export_e12_nano_q8.py --base ~/saanotts \
    --duration-checkpoint ... --acoustic-checkpoint ... --decoder-checkpoint ... \
    --pack $A/packs/eval --out <export-dir> \
    --weights f32 --golden-rows 8 --calib-rows 8 --smoothquant-alpha 0.0 \
    --length-scale 1.0
```

Writes `front_f32.bin`/`model_f32.bin` (or `_q8`), `nano_q8_meta.h`, `golden/`,
`export-report.json`. Then `tools/make_nano_fixture.sh` for the MCU fixture.

**int8 is a gate, not an aspiration**: waveform correlation ≥ 0.98 per row.
heartnano passes at 0.981. **heart ships f32 because its int8 export failed** —
min corr 0.951106 on row `000002_i400`.

Then Part B of `docs/adding-a-voice.md` ships it, with one difference: the nano
lineage runs under the `snt_nano` runtime and exports `model_f32.bin`, not
`dec_f32.bin`.

---

## Gates

Tier discipline from `docs/FEEDBACK-LOOP.md`: **Tier 0 is the owner's ear and it
is never skipped.** A Tier-1 metric has reproduced the Tier-0 ordering on the
current calibration set, and must re-earn that status when the set grows. When a
Tier-1 metric and the ear disagree, **the ear wins and the metric is demoted on
the spot**.

| gate | tool | bar |
| --- | --- | --- |
| MOS panel | `tools/eval_mos_all.py` | — |
| batch scoring | `tools/paper-figures/score_size_curve_eval249.py` | — |
| whistle screen | `tools/campaign-e8/whistle_metric.py` | clean ≤ 1814, dirty ≥ 3818 |
| intelligibility | `tools/eval_scorecard.py` | — |
| WER | `e13_watch/wer_ljtest150.sh` — **not in the repo, lives on cdjk** | student − teacher ≤ +0.012 |
| automated battery | `tools/e14_watch/gate_final.sh` + `final_verdict.py` | see below |

`final_verdict.py` is the only file that hardcodes thresholds: `PARAM_CAP
300000`, `PRIMARY_SCOREQ 2.90`, `GOAL_SCOREQ 3.50`, `WHISTLE_SCREEN 1814`, and
only two are hard — `params_under_300k` and `fullstack_scoreq_primary`.
Repo-wide: teacher SCOREQ ≥ 3.5, oracle/teacher ≥ 0.85, fullstack/teacher ≥
0.75, teacher ASR CER ≤ 0.10.

⚠ **The whistle metric is a screen, not a ranker.** Recorded verdict: *"an
artifact detector, not a quality ranker."* It ranks the 82M Kokoro teacher as
the dirtiest system at n=249, frozen Vocos fails its own clean band on 15% of
clips, and it is non-monotone on an injected chirp. Never gate a single clip.

### Historical numbers — for shape only, not for comparison

Every row below is a **dev** number (`packs/eval`, n=249) except where noted,
measured before the 2026-09-30 seal. They are the best within-curve comparisons
available and they are not clean outward claims.

| system | SCOREQ | UTMOS | DNS-SIG |
| --- | --- | --- | --- |
| Kokoro af_heart teacher (82M) | 4.809 | 4.511 | 3.661 |
| frozen Vocos on GT mel-100 (ceiling) | 4.787 | 4.457 | 3.669 |
| heart 2.27M — oracle lane | 4.060 | 3.754 | 3.545 |
| **heart 2.27M — full stack** | **3.412** | 3.307 | 3.451 |
| e12nano 294,642 — full stack | 2.041 | 2.113 | 2.952 |
| shipped heartnano 294,279 | 2.294 (diverse24) | | |

A human MOS pilot put the 294,279 voice at **1.52** [1.32, 1.74] against a
hidden reference of 4.12. Predictors are generous at this size; the ear is not.

---

## Do not retry

- **The mel-80 interface.** Griffin-Lim from the teacher's *own* mel-80 already
  has the buzz, so the cap was the interface, not the architecture or the data.
  Random-init on mel-80 scored 1.41; an amy-body graft reached 3.53 fullstack
  after weeks. Mel-100's win was **the Tier-0 ear pass, not the scalar** — both
  routes reach ≈3.5 SCOREQ. A guide that leans on the number misattributes it.
- **Decoder architecture search at this budget.** baseline-leaky 10,396 /
  hiftlite-NSF 9,912 / wavehax-faithful 10,674 whistle — statistically
  identical. Budget fixes phase; architecture does not.
- **Temporal mel smoothing as a whistle fix.** It raises the ridge score.
- **From-scratch width increases.** Wide-256 from scratch at 200k = 2.40.
  Lineage beats width.
- **F0 conditioning** (`--with-prosody`, `--f0-conditioning`,
  `--n-pred-channel`). Mel-80-route only; absent from the entire shipped
  lineage. A probable unit bug lurks there too: the E16 trainer assumes
  `f0_log = ln(f0/200)` while the pack builder writes plain `ln(f0_hz)`.
- **Long decoder-only runs after joint training.** +100k decoder steps regressed
  the full stack 3.53 → 3.23 by undoing joint co-adaptation. Always re-couple.
- **New adversarial losses in a short polish.** Discriminators are not
  checkpointed; a 20k restart with fresh ones cost 0.3 SCOREQ. Bake them in from
  step 1 of a long run.

---

## Known problems

Ranked by how soon a fresh run hits them.

1. **No driver script and no chain document.** Eleven loose tools.
2. **Two-layout path trap.** Six stages are under `tools/campaign-e8/` in the
   repo while every docstring says `tools/`.
3. **Hard-coded `/opt/homebrew` espeak paths** in `kokoro_extract_front.py`.
4. **Three venvs**, and the wrong one fails late.
5. **Your corpus will not be the shipped teacher** — no version/voice/speed was
   ever recorded. Build a paired control.
6. **The corpus-local 62-symbol vocab** silently diverges between corpora.
7. **The shipped front end is not the training front end.** Training used
   misaki; `pypkg/sanotts/nano_frontend.py` and `web/trellis_frontend.js` use
   espeak-ng → misaki's E2M rewrite table → 62 ids, i.e. misaki's *fallback*
   branch on every word. Token agreement 93.3–93.9% at n=249; only 1–3 rows of
   249 reproduce exactly. Audio A/B CIs include zero, so it is inaudible — but
   it is not determinism. A better path exists unused in
   `pypkg/sanotts/nano_g2p.py` (82.1% exact-sequence, TER 0.30%).
8. **Both shipped checkpoints are gone from every machine.** Only exported
   blobs survive. `tools/repack_package_to_checkpoints.py` round-trips
   byte-identically, but that was never meant to be the only path.
9. **No shipped-binary parameter audit** for `heart` or `heartnano`.
   `tools/audit_e12_nano_parameters.py` hardcodes `EXPECTED_EXECUTED_TOTAL
   294_642` and the `en_us_e12nano` paths. Parameterising it is an afternoon.
10. **Ten of the eleven tools are private.** Only
    `tools/export_e12_nano_q8.py` is in `docs/public/manifest.txt`, so an
    outside reader can export a nano voice and cannot train one. The entire
    Piper path, by contrast, is published.

### Naming traps

- **`heartnano` is 294,279** (`en_us_e13b`). **294,642** is a *different* stack
  (`en_us_e12nano`) that ships as the MCU example and the paper's size-curve
  point. Different binaries, different sha256, decoder width 62 vs 48.
  `BOARDS.md` calls 294,642 "the 294k nano" throughout.
- **E13 does not test what its name says.** DyT/ReLU + QAT was refuted by Arm
  A's kill and amended out before Arm B's decoder started. Shipped heartnano is
  a **pure width reallocation at standard ops**. Read the `deviations` block,
  not the `hypothesis`.
- **`MEL100-ROUTE.md`'s "Runway to 4.5" was climbed and abandoned.** E7-wide
  4.043, E9-XL 4.183, E10-flow-refiner 4.358 — all oracle-lane, none shipped.
  heart's acoustic is still E6 at 681,227. `train_e15_joint_nano_finetune.py`
  never shipped and has no rejection record.
- **`docs/HANDOFF-sub300k-and-harmonic-source-20260827.md` *(internal)* §7.4 and
  §8 are stale.** The 49,999-row front file it says "survives" does not exist, and
  neither do the e16-v2 / e17 checkpoints it places on k2. Do not cite it.
- **Three incompatible gate-numbering conventions** coexist in the experiment
  registrations. Say which one you mean. `"gate_0_98"` is the int8 correlation
  gate and has nothing to do with "Gate 0".
