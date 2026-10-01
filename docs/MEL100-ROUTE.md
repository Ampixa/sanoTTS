# The mel-100 route — first ear-clean sanoTTS full stack (2026-07-30)

> **A note on paths.** This document is mirrored into the public repo from the
> private research repo. Most `tools/` scripts it names are NOT in the public
> mirror, and references marked *(internal)* point at documents that stay
> private. The commands are recorded so the recipe is legible and auditable,
> not because every one of them is runnable from a public checkout.

**Status: SPRINT DELIVERABLE, Tier-0 ear-passed.** First full student chain in
campaign history with no radio-tuning whistle by the user's ear.

## The chain

```
text --espeak--> phoneme_ids (vocab 62)
     --duration student (131k params, duration_conv h64 d3 k5)--> frames/phoneme
     --length-regulate--> frame-rate sequence (hop 256 @ 24 kHz)
     --acoustic student (681k params, token_context h96 d4 k5)--> mel-100/frame
     --frozen Vocos (charactr/vocos-mel-24khz, ~13M, PUBLIC PRETRAINED)--> waveform
```

The interface is **Vocos' native mel-100** (n_fft 1024, hop 256, win 1024,
power-1 magnitude, HTK scale, `log(clip(x, 1e-7))`) — computed by
`tools/build_mel100_targets.py` byte-identically to `vocos.feature_extractor`,
frame-aligned to the pack grid by trimming the center-pad extra frame
(verified: trim beats +1-shift on 40/40 rows by frame-energy correlation).

## Numbers (heldout-8, 2026-07-30)

| lane | SCOREQ | UTMOS | DNS-SIG | whistle (8-clip mean) |
|---|---|---|---|---|
| oracle (teacher mel-100 → Vocos) | 4.80 | 4.48 | 3.66 | 1,514 |
| acoustic pred (teacher durations) | 3.50 | 3.33 | 3.41 | 1,879 |
| **fullstack (student durations)** | **3.53** | 3.40 | 3.44 | 2,046 |

Clean band ≤ 1,814; every pre-route system ≥ 3,818 (probes ~10k). Teacher
SCOREQ 4.81. Duration student: mean |frame delta| 4.0%, max 6.2%.

## Why it works (evidence chain, all Tier-0-anchored)

1. The radio-tuning whistle = decoder phase invention gone wrong at small
   training budgets. GL-from-teacher-mel80 dirty (2026-07-28) localized it.
2. Architecture does NOT fix it at our budget: baseline-leaky 10,396 /
   hiftlite-NSF 9,912 / wavehax-faithful 10,674 — statistically identical
   (E3/E4, 2026-07-29). E1 discipline: −12% only.
3. Budget DOES fix it: frozen Vocos (~1M pretrained steps) from teacher
   mel-100 = user "sounds perfect" (D3 V100, 2026-07-29).
4. Our duration + acoustic lanes were always good (Whisper-verbatim, exact
   timing, 0.99 cosine) → retarget acoustic to Vocos' features, freeze the head.
5. Smoothing-sensitivity (D4) + E5/E6 confirmed student-predicted mels stay in
   the clean band through the frozen head.

## Artifacts (k2: ~/saanotts/artifacts/kokoro-corpus-af_heart-20260713/)

- Acoustic: `runs/e6-mel100-fullcorpus/latent-student.pt` (+ train-report)
- Duration: `runs/duration/duration-student.pt`
- Targets: `packs-mel100/{train,probe32,eval8}-targets` (`build_mel100_targets.py`)
- Renders: `runs/e6-mel100-fullcorpus/renders-{vocos,fullstack}/`
- Render tools: `tools/render_mel100_vocos.py`, `tools/render_fullstack_mel100.py`
- Vocoder env: `~/venv-vocos` (vocos 0.1.0, CPU)
- Registry: `runs/experiments-registry.jsonl` rows E5/E6; ear verdicts in
  `docs/calibration-registry.md`

## Falsified along the way (do not retry)

- Temporal mel smoothing as whistle fix: RAISES ridge score (2,237–2,315 vs 1,879).
- e2e VITS at our budget (E1: 8,642 at 25k steps, closed).
- Decoder-architecture search at our budget (E3/E4).

## Runway to 4.5 (standing target)

Gap pred 3.53 → oracle 4.80 is entirely acoustic-lane mel sharpness.
Levers, in registered order:
1. **E7-acoustic-wide** (live): hidden 192, depth 6, 200k steps.
2. Generative acoustic objective if regression blur floors out (flow/diffusion
   mel head) — see risk questions.
3. Vocos finetune on af_heart (only if oracle itself needs lifting — it doesn't yet).

## Known risks / open questions

- **Edge deployment**: Vocos ~13M params, CPU-RTF fine on desktop, but the
  ESP32/MCU story cannot ship it. Open research question (see deliberation).
- **OOD mels**: clean verified on 8 heldout rows; long-text / prosody-extreme
  stress sweep not yet run.
- **Metric scope**: whistle metric calibrated on the old artifact family; new
  route could have artifact classes it is blind to — ear cards stay mandatory.
