# The Feedback Loop — binding structure for all voice experiments

> **A note on paths.** This document is mirrored into the public repo from the
> private research repo. Most `tools/` scripts it names are NOT in the public
> mirror, and references marked *(internal)* point at documents that stay
> private. The commands are recorded so the recipe is legible and auditable,
> not because every one of them is runnable from a public checkout.

Written 2026-07-29 after a week of wrong signals: quiet-frame comb metric
(passed models the ear failed), difference-detector (measured "any difference",
not the artifact), buzz-meter (ordering violated), probe harness (died silently
twice and was read as "in progress"). Every one of those failures came from
gating on an unvalidated signal. This document makes that structurally
impossible.

## 1. Signal hierarchy

- **Tier 0 — the ear (ground truth).** Ashish's listening verdicts. Expensive
  (minutes/day); never skipped at a gate. Every Tier-0 verdict is appended to
  the calibration registry (§2).
- **Tier 1 — validated metrics.** A metric is Tier 1 only after it reproduces
  the Tier-0 ordering on the current calibration set with clean separation,
  and its validation table is committed next to its code. Tier-1 metrics may
  gate experiments. When the calibration set grows, Tier-1 status must be
  re-earned.
- **Tier 2 — diagnostic signals.** Anything unvalidated (new metrics, losses,
  detector logits, SCOREQ deltas <0.1). Allowed for exploration and debugging.
  **Forbidden as gates or as claims in status reports.** Reporting a Tier-2
  number requires labeling it "(unvalidated)".

Current tier assignments:
| Signal | Tier | Note |
|---|---|---|
| user ear on listening page | 0 | ground truth |
| SCOREQ/UTMOS/DNS (large deltas ≥0.2) | 1 | validated by long usage; small deltas are Tier 2 |
| whistle metric | 2 → pending validation vs calibration set | |
| voiced-comb dB | 2 (demoted: passed dirty systems) | diagnostic only |
| detector logit / buzz-meter | 2 | localizer only |

## 2. Calibration registry

`docs/calibration-registry.md` — versioned table: clip path/system → Tier-0
verdict → date. Seeded with: CLEAN = teacher (gt-first8), Inflect Nano v2
samples, TinyTTS samples. DIRTY (radio-tuning whistle) = joint82, buzzkill,
joint-mrd, e2e-23k, expand fullstack renders. Every future listening card's
verdicts are appended. Metrics validate against the registry, never against
anecdotes.

## 3. Experiment contract (pre-registration)

No run launches without a row appended to `runs/experiments-registry.jsonl`
BEFORE launch:
```json
{"name": "...", "hypothesis": "...", "gates": ["whistle<=X", "scoreq>=Y"],
 "kill_criteria": "...", "max_wall_hours": 3, "slot": "k2-a|k2-b|omac-a|omac-b",
 "launched": "<ts>"}
```
Post-hoc goal shifting is a protocol violation: if a run misses its
pre-registered gate, it failed — findings go in the registry row's `result`,
and any new idea it inspired becomes a NEW pre-registered experiment.

## 4. The cycle

propose → pre-register (§3) → run (≤3 h wall, monitored with
absence-of-progress alarms) → auto-gate (Tier-1 only) → listening card
(≤60 s of audio, clearly labeled) → Tier-0 verdicts appended to registry →
metrics revalidated if registry grew → next round.

## 5. Disagreement protocol

When a Tier-1 metric and the ear disagree, the ear wins, the metric is demoted
to Tier 2 on the spot, the disagreement becomes a calibration entry, and the
metric is refit/rebuilt before it can gate again. Logged in
`docs/calibration-registry.md` under "demotions".

## 6. Monitor rules

Every run watcher must alarm when *nothing is progressing* (no process AND no
new output), not only on error strings. Silent death read as "in progress"
cost this campaign half a day twice.

## 7. Dev and sealed test

Amendment, 2026-09-30. Until now there was one holdout and it did two jobs.

`packs/eval` — the 249-row modulo-200 holdout — was the number watched during
training *and* a reported headline (heart 3.412, e12nano 2.041, the whole
size/quality curve). The split itself is clean: train and eval are disjoint by
construction, and no trainer selects a checkpoint on it, because there is no
`best_`, `save_best`, `argmin` or `argmax` anywhere in the three trainers — the
duration student's final file is bit-identical to its last periodic save. So
there was no classic leak.

There was still selection. Arms were killed, recipes chosen and a shipping
candidate picked by humans reading those 249 numbers, across some forty
registered rows. Selection by a person is selection. It biases the headline
optimistically by an amount nobody has measured, and the bias grows with every
arm compared on the same rows.

From now on the two jobs are separated.

**Dev** is `packs/eval`. Watch it, plot it, kill arms on it, choose recipes on
it. It remains the right instrument for controlled within-curve comparisons,
where every system sees the identical sentences and only one thing varies.

**Sealed test** is `data/textsets/multilang-distill-v1/en_US.sealed-test.jsonl`
(224 rows, sha256 `6e88738e60597a75f0e43123741200af41ba418870c5327d5868107575f7ac90`),
real prose, verified **zero** overlap with the real dev pack. Report
`data/textsets/multilang-distill-v1/en_US.diverse-heldout24.jsonl` beside it as
the out-of-pool read. Every externally quoted number — a README, the site, a
paper, a release note — comes from these.

**The seal is a FILE, not a formula.** Future packs must exclude these texts
explicitly, by text. Do not try to carve the sealed slice arithmetically out of
the textset: front extraction yields 49,999 rows from the 50,000-row textset, so
pack order diverges from textset order after the dropped row. The first draft of
this set was built as "pool order % 200 == 1" on the reasoning that dev is
remainder 0 — and collided with **176 of the 249 actual dev rows**, a 78.6%
contaminated "sealed" set. `tools/check_eval_seal.py` caught it. The dev pack's
recorded text is the only authority on what dev is. Full rule in
`en_US.sealed-test.provenance.json`.

The rule that makes it worth having: **no campaign decision may read a sealed
number.** Not a kill, not an arm selection, not a hyperparameter, not "let us
just check". Measure it when a candidate is finished and the decision to ship is
already made on dev. A sealed set consulted twice during a search is a dev set
with a longer name.

Pre-registration (§3) must now name which set each `gate` is measured on. A gate
whose set is unstated is measured on dev and may not be quoted outward.

### Enforcement

    python3 tools/check_eval_seal.py --train-pack <pack> --dev-pack <corpus>/packs/eval

Run it before a run starts. It matches on text, case-folded and
whitespace-collapsed, never on row ids — ids renumber between corpus builds,
which is exactly how a contaminated pack passes a glance.

This is not hypothetical. `tiny-vocos-r32-499k-20260821` trained on 233 of the
249 dev rows, 93.6% of the dev set, and so did the parent it was distilled
from. Both were excluded from the curve and written up in
`docs/size-quality-curve-eval249.md` *(internal)* — but they were caught because somebody
opened a run-config and read the row ids. That is luck, and it only covers the
runs someone audits. The checker reproduces that 93.6% as a regression test.

The mechanical half is now closed. The social half is not: nothing can stop a
person measuring the sealed set early and letting it steer a choice. That part
is a promise, which is why it is written here rather than only in a tool.

### The honest limitation of the current set

The 152 rows are assembled from two files that predate the seal, not freshly
drawn. `en_US.eval.jsonl` was the eval set for the 2026-07 8k-row
mel-80/Piper-era line — `en_US.train8k.noeval.jsonl` exists precisely to exclude
it — so some of these sentences have been rendered and heard before, by a
different lineage at a different interface, months ago. The defensible claim is
**sealed from 2026-09-30**, not never-before-seen. A set drawn fresh from text
outside this repo would be strictly better and is the obvious next improvement.

And no number measured before this amendment becomes retrospectively sealed.
Every figure in `docs/size-quality-curve-eval249.md` *(internal)* and in the evidence files
is a dev number. They are still the best within-curve comparisons available;
they are not clean outward claims, and they should be relabelled as dev when
next revised rather than quietly reused.
