import { ARCH, DESIGN, PARAMS, TRAINING } from "../data/paperFacts";
import { useViz } from "../data/store";

const PCT = (x: number) => `${((x / PARAMS.total) * 100).toFixed(1)}%`;

export function DecoderCh() {
  const trace = useViz((s) => s.trace);
  const D = TRAINING.decoder;
  return (
    <div>
      <div className="chapter-kicker">05 · waveform decoder</div>
      <h2>{PARAMS.decoder.toLocaleString()} parameters that turn a picture into pressure.</h2>

      <p>
        The decoder holds {PCT(PARAMS.decoder)} of the entire parameter budget,
        and it earns it: it must invert the mel's lossy compression back into a
        full spectrum, phase included. It works as a {ARCH.dec.dim}-channel
        temporal trunk — a stem that merges the mel with seeded Gaussian noise,
        then {ARCH.dec.blocks} sequential ConvNeXt-style blocks that sculpt it.
      </p>

      <h3>One block, exactly</h3>
      <p>
        Each block: <b>depthwise conv</b> (kernel {ARCH.dec.dwKernel}, 496 params —
        cheap per-channel time mixing) → LayerNorm → <b>1×1 conv {ARCH.dec.dim}→{ARCH.dec.pw}</b>{" "}
        (11,718) → GELU → <b>1×1 conv {ARCH.dec.pw}→{ARCH.dec.dim}</b> (11,594) →
        LayerScale + residual. <b>23,994 parameters per block</b>, 95,976 across
        the trunk. Around it: stem 45,384 (mel embed k={ARCH.dec.embedKernel} +
        noise adapter + LN), final LN 124, head 64,638 — and those four numbers
        sum to exactly {PARAMS.decoder.toLocaleString()}, the shipped total.
      </p>

      <h3>Why noise goes in</h3>
      <p>
        A mel spectrogram carries no phase and none of the fine variation within
        speech sounds — many different waveforms map to the same mel. The{" "}
        {ARCH.dec.noiseCh} Gaussian noise channels per frame give the decoder
        independent random coordinates with which to pick one plausible waveform
        from that cloud of possibilities. The noise comes from a seeded PRNG, so
        synthesis is fully deterministic: same ids, same seed, same waveform.
      </p>

      <h3>Trained in two phases, against a giant</h3>
      <p>
        The decoder trained for {D.updates.toLocaleString()} updates on random
        32-frame crops (0.33 s). For the first {D.warmupSteps.toLocaleString()},
        the loss included complex-spectrum and waveform L1 terms that pin down
        phase and magnitude alignment; afterwards only log-magnitude L1,
        multi-resolution STFT, and RMS losses remained. Ablations showed the
        early terms teach waveform formation, but keeping them too long punishes
        perfectly reasonable waveforms — the audible symptom was a metallic{" "}
        <i>/s/</i>. Throughout, a training-only multi-period discriminator
        (periods {D.mpdPeriods.join("·")}) pushed the small generator toward
        realistic speech; it never ships.
      </p>
      <p>
        Getting to this size took targeted compression: an early{" "}
        {DESIGN.factorizedDecoder.params.toLocaleString()}-parameter width-160
        decoder was shrunk by replacing its large channel matrices with rank-
        {DESIGN.factorizedDecoder.rank} factorizations, scoring{" "}
        {DESIGN.factorizedDecoder.scoreq} against {DESIGN.factorizedDecoder.controlScoreq}{" "}
        for a simply narrower width-{DESIGN.factorizedDecoder.controlWidth} control —
        factorization preserved capacity that a width cut destroyed.
      </p>

      <h3>This sentence</h3>
      {trace && (
        <p className="small">
          A [{ARCH.dec.dim}×{trace.T}] plane written in place, four times. On the
          right, watch the stem and each block's output stacked: voicing structure
          appears early; frication and fine detail sharpen toward block{" "}
          {ARCH.dec.blocks}. The top strip is the raw noise input.
        </p>
      )}

      <div className="lookright">
        Left: the block's anatomy with per-op parameter math. Right: the real
        activations — noise, stem, and all four blocks.
      </div>
    </div>
  );
}
