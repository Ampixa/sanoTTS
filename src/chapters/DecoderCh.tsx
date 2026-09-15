import { ARCH, PARAMS } from "../data/paperFacts";
import { useViz } from "../data/store";

const PCT = (x: number) => `${((x / PARAMS.total) * 100).toFixed(1)}%`;

export function DecoderCh() {
  const trace = useViz((s) => s.trace);
  return (
    <div>
      <div className="chapter-kicker">05 · waveform decoder</div>
      <h2>{PARAMS.decoder.toLocaleString()} parameters that turn a picture into pressure.</h2>

      <p>
        The decoder is {PCT(PARAMS.decoder)} of the whole parameter budget, and it
        earns it: it must invert the mel's lossy compression back into a full
        spectrum. It works as a 62-channel temporal trunk — a stem that merges the
        mel with seeded Gaussian noise, then four ConvNeXt1D blocks that sculpt it.
      </p>

      <h3>one block, exactly</h3>
      <p>
        Each block: <b>depthwise conv</b> (kernel {ARCH.dec.dwKernel}, 496 params —
        cheap per-channel time mixing) → LayerNorm → <b>1×1 conv 62→{ARCH.dec.pw}</b>{" "}
        (11,718) → GELU → <b>1×1 conv {ARCH.dec.pw}→62</b> (11,594) → γ-scale +
        residual. <b>23,994 parameters per block</b>, 95,976 across the trunk.
        Around it: stem 45,384 (mel embed k={ARCH.dec.embedKernel} + noise adapter
        + LN), final LN 124, head 64,638 — and those four numbers sum to exactly{" "}
        {PARAMS.decoder.toLocaleString()}, the shipped total.
      </p>
      <p>
        The noise channels ({ARCH.dec.noiseCh} per frame, from a seeded PRNG — fully
        deterministic) are what let the decoder hallucinate plausible phase and
        frication detail the mel discarded.
      </p>

      <h3>this sentence</h3>
      {trace && (
        <p className="small">
          a [62×{trace.T}] plane written in place, four times. On the right, watch
          the stem and each block's output stacked: voicing structure appears
          early; frication and fine detail sharpen toward block 4. The top strip
          is the raw noise input.
        </p>
      )}

      <div className="lookright">
        left: the block's anatomy with per-op parameter math. Right: the real
        activations — noise, stem, and all four blocks.
      </div>
    </div>
  );
}
