# Calibration registry — Tier-0 (ear) verdicts

| system / clip source | verdict | character | date |
|---|---|---|---|
| teacher gt-first8 (Kokoro af_heart corpus audio) | CLEAN — "the best" | — | 2026-07-28 |
| Inflect Nano v2 official samples | CLEAN (passes) | — | 2026-07-28 |
| TinyTTS diverse24 archive renders | CLEAN (passes) | — | 2026-07-28 |
| gate-joint82-fullstack | DIRTY | buzz/whistle | 2026-07-15 |
| gate-buzzkill-fullstack | DIRTY | same buzz as joint82 | 2026-07-26 |
| gate-jointmrd-fullstack | DIRTY ("all ass") | radio-tuning tail | 2026-07-29 |
| vits-e2e-renders (7800) + renders2 (23400) | DIRTY | same tail, + earlier: underwater+hum at itail | 2026-07-28/29 |
| gate-expand-fullstack | DIRTY | buzz present | 2026-07-26 |
| GL-from-teacher-mel80 (cause-isolation) | DIRTY | has the artifact | 2026-07-28 |
| itail-phase clips | DIRTY | underwater + machine hum | 2026-07-28 |
| e3-hiftlite 15k probe (eval0-3) | DIRTY — WORSE | new artifact class, worse than baseline; radio-tuning tail indeterminable under it | 2026-07-29 |
| D3 V100: teacher → Vocos-native mel-100 → frozen pretrained Vocos | CLEAN — "sounds perfect" | phase invention is budget-solvable; interface-in-principle conviction OVERTURNED | 2026-07-29 |
| D3 V80: teacher → our mel-80 → pinv lift → Vocos | DIRTY ("isn't good") | confounded by lossy lift adapter — bounds mel-80 from above, not a clean conviction | 2026-07-29 |
| E6 FULLSTACK: student durations → mel-100 acoustic → frozen Vocos | **CLEAN — EAR PASS ("did it")** | first ear-passing full student chain in campaign history; whistle 2046, SCOREQ 3.53 | 2026-07-30 |
| BLIND A/B 2026-08-04: our 2.27M (E8h noise-fed) vs TinyTTS 1.22M | **OURS WINS** ("system Y is simply better"; Y = ours, sealed pre-verdict) | campaign goal achieved by blind ear: smallest-class TTS, beats the existence proof | 2026-08-04 |
| BLIND A/B 2026-08-19: amy 1.46M (shipped web voice) vs Trellis-RIFT 995k | **AMY WINS, BOTH SENTENCES** ("system B is good by 100 times"; counterbalanced, so the user picked B then A — opposite letters, same system) | decides that the public demo's live English voice stays amy; Trellis-RIFT ships as an additional research preview, not a replacement | 2026-08-19 |
| TinyTTS 1.2M vs our 24M flagship | **EAR-EQUAL** ("their TTS is just as good as ours in 1.2m"; "scoreq means nothing") | perceptual target at ~1M is EXISTENCE-PROVEN; SCOREQ deltas between ear-passing systems carry no perceptual weight — Tier-2 for cross-system comparisons among clean systems | 2026-08-01 |

## Demotions
| metric | demoted | reason |
|---|---|---|
| quiet-frame comb | 2026-07-25 | passed models the ear failed |
| voiced-comb dB | 2026-07-29 | +0.4 "teacher parity" on a system the ear failed |
| difference-detector logit | 2026-07-26 | measures any difference, not the artifact |
| injection buzz-meter | 2026-07-26 | ordering violations on real systems |

## Metric validations
| metric | status | criterion | caveat |
|---|---|---|---|
| whistle_metric ridge_track_score | **Tier 1 — SYSTEM-LEVEL GATES ONLY** (2026-07-29) | 8-clip system means: all dirty ≥ 3818, all clean ≤ 1814 (>2x margin, ordering matches ear) | per-clip separation fails on 1/46 pair (gt eval6 sibilant vs jointmrd eval5); NEVER gate single clips; amendment adopted openly, not threshold-shopped |
| whistle_metric cepstral_residual | Tier 2 | overlapping bands | diagnostic only |

## Tier-1 measurements pending Tier-0 (public web voices vs Trellis-RIFT), 2026-08-19

Run while deciding which English voice the public demo should synthesize live.
These are **Tier-1 metric outputs, not ear verdicts** — no Tier-0 listening has
been done on these three systems, so nothing here promotes or demotes anything.

| system | 8-clip mean whistle score | band | clip source |
|---|---:|---|---|
| Trellis-RIFT 995,058 (held-out predicted) | **74.7** | far inside clean (≤1814) | `artifacts/trellis-rift-selected-20260816/listen/audio/` |
| amy 1.46M web voice (student renders) | **1,968.8** | **between** clean ceiling 1814 and dirty floor 3818 | `artifacts/listen-all-20260713/inj/amy_146/` + `release-clips-20260713` |
| amy Piper teacher (same sentences) | 1,377.7 | inside clean | `artifacts/listen-all-20260713/inj/amy_146/` |

Reference points from earlier campaign runs: E6 fullstack (first ear-PASS)
scored 2,046; the 2.27M blind-A/B winner scored 499.2.

Honest caveats, all of which must survive into any claim built on this:
1. amy at 1,968.8 lands in the **gap between** the calibration bands. That is
   "not inside the verified-clean band", which is NOT the same as "proven
   dirty". The registry has no clip in that gap to calibrate against.
2. The lanes use different sentences and different voices (Trellis-RIFT on its
   af_heart held-out rows, amy on the 2026-07-13 injection set). The score is
   per-second normalized, but content is a genuine confound.
3. Whistle score measures the sweeping-inharmonic-tone artifact only. It says
   nothing about the other axes on which amy leads.

Counter-evidence on the other side, same three systems, same two sentences
(SCOREQ NR / DNSMOS, harness on `cdjk`): amy 4.09 SCOREQ / 3.34 DNSMOS-OVRL vs
Trellis-RIFT 2.52 / 3.00; Whisper-small WER tied at 0.036.

### RESOLVED by Tier-0, 2026-08-19 — and the whistle metric pointed the wrong way

The blind counterbalanced A/B was run and **amy won both sentences by a large
stated margin**. So on this pair:

- **SCOREQ and DNSMOS tracked the ear correctly** (both favoured amy, SCOREQ by
  1.57). The 2026-08-01 "scoreq means nothing" row does NOT license ignoring a
  gap this size: that row scoped SCOREQ to Tier-2 for comparing systems that are
  *both already ear-clean*, which is not the situation here.
- **The whistle metric did not predict preference.** It is validated for one
  artifact — the sweeping inharmonic radio-tuning tone — and it was probably
  right about that artifact: Trellis-RIFT genuinely has far less of it (74.7 vs
  1,968.8). But less of one artifact is not better overall, and amy sitting
  above the clean ceiling did not make it the worse voice.

**Binding lesson: whistle score is an artifact detector, not a quality ranker.**
Do not use it to choose between systems, only to ask whether a specific system
carries the whistle artifact. Choosing between systems remains Tier-0.

Note: the tier table in `docs/FEEDBACK-LOOP.md` §1 still lists the whistle
metric as "Tier 2 → pending validation"; it was superseded by the Tier-1
amendment recorded above and in commit `cbfd261`, and should be reconciled.
