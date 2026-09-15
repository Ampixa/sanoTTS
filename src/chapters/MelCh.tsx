import { ARCH } from "../data/paperFacts";
import { useViz } from "../data/store";

export function MelCh() {
  const trace = useViz((s) => s.trace);
  return (
    <div>
      <div className="chapter-kicker">04 · mel contract</div>
      <h2>The 100-channel interface every stage must honor.</h2>

      <p>
        The acoustic student's output is a <b>log-mel spectrogram</b>: 100
        perceptually-spaced frequency bands per frame. This is the fixed contract
        between "language thinking" and "waveform synthesis" — the decoder never
        sees phonemes, only this picture of what the sound should be.
      </p>

      <h3>how to read the picture</h3>
      <p>
        Time runs left to right; mel band 0 (low pitch) sits at the bottom, band 99
        at the top. Crimson is energy, white is silence. The horizontal stripes are
        the voice's <b>pitch harmonics</b>; the vertical blips are plosives like
        "t" and "k". This exact grid — not a training-target render — is what the
        decoder consumed to make the audio you can play on the right.
      </p>

      <h3>this sentence</h3>
      {trace && (
        <p className="small">
          [100 × {trace.T}] over{" "}
          {(trace.T * ARCH.istft.hop / ARCH.sampleRate).toFixed(2)} seconds —{" "}
          {(100 * trace.T).toLocaleString()} numbers, each a log-energy the decoder
          must turn into air pressure.
        </p>
      )}

      <div className="lookright">
        press play: the crimson cursor sweeps the heatmap in sync with the audio.
        Hover any cell for its band, time, and exact log-mel value.
      </div>
      <p className="small faint">
        provenance: decoder input tensor, dumped by the host tracer after the
        mel head — the real [100×T] grid, not a re-synthesis.
      </p>
    </div>
  );
}
