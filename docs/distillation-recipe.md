# Distilling a sub-1.5M real-time TTS from a Piper teacher

A reproducible recipe for distilling a ~1.4M-parameter English text-to-speech
student that, on held-out text, beats every open model up to 15M parameters on
naturalness (SCOREQ/UTMOS) and runs real-time on a microcontroller. It is an
*honest* recipe: it names the levers that worked, the ones that didn't, and the
evaluation that keeps you from fooling yourself.

## What you get

On 24 LJSpeech-validation sentences disjoint from training (the "diverse24" gate),
no-reference metrics:

| model | params | SCOREQ | UTMOS |
| --- | --- | --- | --- |
| **this recipe** | **1.40M** | **4.09** | **3.98** |
| TinyTTS | 1.6M | 3.94 | 3.65 |
| Inflect Nano | 4.6M | 3.81 | 3.65 |
| Kitten TTS | 15M | 3.02 | 3.58 |
| Kokoro (reference) | 82M | 4.90 | 4.52 |

Best quality-per-parameter in the sub-2M class; the frontier only advances at ~60x
the size (Kokoro). A 745k int8 variant runs on an ESP32-S3 at real-time.

## The idea in one sentence

Do **not** try to relearn text→acoustics at 1M parameters — it memorizes and cannot
speak held-out text. Instead treat a good open teacher (Piper/VITS) as a fixed
oracle, and distill three small students **against its latent interface**: a
duration predictor, an acoustic model that predicts the teacher's latent, and an
iSTFT decoder that inverts that latent to waveform.

## Architecture

Three students, one 192-dim latent interface (the teacher's `generator_input`):

```
text --[Piper frontend]--> phoneme ids
        --[duration student ~36k]--> per-phoneme frame counts
        --[acoustic student ~359k, token-context]--> 192-dim latent (per frame)
        --[iSTFT decoder student ~1.0M, "piperlite"]--> 22.05 kHz waveform
```

Total inference ~1.40M. The decoder is teacher-initialized (channel-sliced from the
Piper generator) and never trained from scratch.

## Prerequisites

- A Piper/VITS teacher voice (`.onnx` + `.onnx.json`), e.g. `en_US-kristin-medium`.
- Python env with `torch`, `onnxruntime`, `piper-tts`, `soundfile`, `numpy`, `scipy`.
- ~8k lines of any English text for the acoustic; a few hundred for the decoder.
- Tools in `tools/` (all referenced below are committed).

## Pipeline

### 1. Native pack — render the teacher's latents
The teacher is the data source. Render it on text to dump, per utterance, its
phoneme ids, durations (`w_ceil`), the 192-dim latent (`generator_input`), and audio.
```
python tools/build_piper_vits_roota_probe_pack.py \
  --model  teacher.onnx --config teacher.onnx.json \
  --source-jsonl texts.jsonl --max-rows 8000 \
  --tensor-mode decoder --allow-text-only-source \
  --out-dir packs/train8k
```
Build an ~8k-row pack for the acoustic and a small (~512-row) one plus a held-out
eval pack for the decoder. **Acoustic quality is data-limited** — under-provisioning
here (e.g. 3k rows) caps the model; 8k is the sweet spot at this size.

### 2. CUT the teacher's decoder into an ONNX oracle
Extract a decoder-only graph that maps `generator_input -> waveform`; it is the
render primitive the decoder student learns to imitate and is validated to
round-trip the pack to <5e-4 error.
```
python tools/extract_piper_vits_decoder_cut.py \
  --model teacher.onnx --pack-dir packs/train8k \
  --latent-channels 192 --out-dir cut/
```

### 3. Acoustic student — predict the latent (with de-smoothing)
Token-context architecture, hidden 64. A plain L1/MSE regressor over-smooths; a
**latent-adversarial** discriminator (training-only, never shipped) keeps the
predicted latent sharp.
```
python tools/train_roota_piper_latent_student.py \
  --pack-dir packs/train8k --architecture token_context --hidden 64 --depth 5 --token-depth 3 \
  --steps 50000 --latent-adv-weight 0.1 --latent-adv-start-step 1000 \
  --out-dir students/acoustic
```

### 4. Duration student — from scratch (it's tiny)
```
python tools/train_roota_piper_duration_student.py \
  --pack-dir packs/train8k --eval-pack-dir packs/eval128 \
  --hidden 32 --depth 3 --steps 4000 --out-dir students/duration
```

### 5. Decoder student — teacher-init, then recovery -> z-mix -> joint
Initialize from the teacher-parity `piperlite` decoder and adapt in three stages of
40k steps each. **Recovery** trains on the teacher's own latents; **z-mix** mixes in
the acoustic student's latents (`--acoustic-latent-mix-prob 0.5`) so the decoder is
robust to student latent error; **joint** co-fine-tunes acoustic + decoder.
```
# recovery + z-mix: tools/train_roota_piper_decoder_student.py
#   --init-decoder-checkpoint <piperlite teacher-init> --teacher-decoder cut/*.onnx
#   --variant piperlite --channels 192,96,48,24 --rank-ratio 0.5 --steps 40000
#   (+ --acoustic-latent-mix-prob 0.5 --acoustic-checkpoint students/acoustic/*.pt for z-mix)
# joint: tools/train_roota_joint_z_finetune.py --acoustic-checkpoint ... --decoder-checkpoint ...
```
Skimping on steps here is the classic failure: at 10k/stage the decoder handles
teacher latents but is brittle to the acoustic's — the honest gate exposes it.

### 6. De-metal — an adversarial polish
A short fine-tune with a MultiPeriodDiscriminator + iSTFT-phase loss removes a
metallic tail. Keep `--acoustic-latent-mix-prob 0.5` on so it doesn't re-break the
acoustic->decoder handoff.

### 7. (Optional) Sibilant injection — fix the whistly /s ʃ z ʒ/
A deterministic acoustic predicts the *mean* latent, and the mean of a
broadband-noise fricative is a dull tone — so sibilants come out whistly. Retraining
won't fix it (architectural). Instead, at inference, add teacher-calibrated Gaussian
noise into the predicted latent **only at sibilant frames**:
```
python tools/calibrate_sibilant_noise.py --pack packs/train512 --out calib.npz
# then serve with: --sibilant-inject-beta 6.0 --sibilant-calib calib.npz
```
This trades a small aggregate-SCOREQ dip for an audibly correct sibilant — an
ear-validated choice a global metric can't make.

## Evaluate honestly (this is half the recipe)

1. **Use an unseen gate.** Templated/synthesis-adjacent text inflates every score by
   ~1.3 SCOREQ. Report the student-teacher gap on the same held-out gate.
2. **Report multiple predictors + listen.** SCOREQ/UTMOS (naturalness) and DNSMOS
   (signal) disagree by design; a metallic artifact scores *high* on the first and
   low on the second. The ear breaks ties.
   `python tools/eval_mos_all.py name:render_dir ...`
3. **Resolve per phoneme class.** A whole-utterance score misses a defect confined to
   one class. `tools/eval_phoneme_class_fidelity.py` measures per-articulatory-class
   spectral flatness across three lanes (teacher / teacher-latent-through-your-decoder
   / full student) and **attributes** each gap to the acoustic vs the decoder. This is
   what found the sibilant collapse when every global metric was blind to it.

## Lessons that transfer

- **Distill against the interface, not through it.** A decodable low-dim re-contract
  of the teacher's latent inherits generalization; a merged text→waveform net at this
  budget memorizes.
- **Teacher-init the decoder.** It survives channel-pruning; from-scratch sub-400k
  decoders are a dead class.
- **Data provisioning is per-student.** The acoustic needs thousands of rows
  (text variety); the decoder needs few hundred (latent→audio variety).
- **Deterministic acoustics can't emit noise.** Fricatives/aspiration are where a
  regression model visibly fails; fix it at inference, not with more training.
- **Your metric has blind spots. Build the probe that sees them.** Every real defect
  in this project was caught by a human ear or a resolution-increasing probe before a
  scalar metric agreed.
