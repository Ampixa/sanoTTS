import { ARCH, LANES } from "../data/paperFacts";
import { useViz } from "../data/store";

export function MelCh() {
  const trace = useViz((s) => s.trace);
  return (
    <div>
      <div className="chapter-kicker">04 · mel contract</div>
      <h2>The 100-channel interface every stage must honor.</h2>

      <p>
        The acoustic model's output is a <b>log-mel spectrogram</b>: {ARCH.mel}{" "}
        perceptually-spaced frequency bands per frame. This is the fixed contract
        between "language thinking" and "waveform synthesis" — the decoder never
        sees phonemes, only this picture of what the sound should be.
      </p>

      <h3>How to read the picture</h3>
      <p>
        Time runs left to right; mel band 0 (low pitch) sits at the bottom, band{" "}
        {ARCH.mel - 1} at the top. Crimson is energy, white is silence. The
        horizontal stripes are the voice's <b>pitch harmonics</b>; the vertical
        blips are plosives like "t" and "k". This exact grid — not a
        training-target render — is what the decoder consumed to make the audio
        you can play on the right.
      </p>

      <h3>Why this contract</h3>
      <p>
        The {ARCH.mel}-channel mel at {ARCH.sampleRate / 1000} kHz was chosen
        empirically. A frozen Vocos decoder re-synthesizing the teacher's mel
        scores {LANES.rows[1].scoreq.toFixed(2)} SCOREQ against the teacher's own{" "}
        {LANES.rows[0].scoreq.toFixed(2)} — the interface itself costs almost
        nothing. That made mel-{ARCH.mel} the fixed boundary, and the frozen
        Vocos re-synthesis of the teacher's mel became the decoder's training
        target.
      </p>

      <h3>This sentence</h3>
      {trace && (
        <p className="small">
          [{ARCH.mel} × {trace.T}] over{" "}
          {(trace.T * ARCH.istft.hop / ARCH.sampleRate).toFixed(2)} seconds —{" "}
          {(ARCH.mel * trace.T).toLocaleString()} numbers, each a log-energy the
          decoder must turn into air pressure.
        </p>
      )}

      <div className="lookright">
        Press play: the crimson cursor sweeps the heatmap in sync with the audio.
        Hover any cell for its band, time, and exact log-mel value.
      </div>
      <p className="small faint">
        Provenance: the decoder's input tensor, recorded from the shipped model —
        the real [{ARCH.mel}×T] grid, not a re-synthesis.
      </p>
    </div>
  );
}
