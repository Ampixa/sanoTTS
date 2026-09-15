import { ARCH } from "../data/paperFacts";
import { useViz } from "../data/store";

export function SpectrumCh() {
  const trace = useViz((s) => s.trace);
  const hzPerBin = ARCH.sampleRate / ARCH.istft.nFft;
  return (
    <div>
      <div className="chapter-kicker">06 · spectrum head</div>
      <h2>1,026 numbers per frame: magnitude <i>and phase</i>.</h2>

      <p>
        The final LayerNorm and a 62→1026 conv head turn the trunk's output into a
        full linear spectrum each frame: <b>513 log-magnitudes and 513 phases</b>,
        one per STFT bin. Bins are {hzPerBin.toFixed(1)} Hz apart, 0 Hz to{" "}
        {(512 * hzPerBin / 1000).toFixed(0)} kHz; DC and Nyquist magnitudes are
        pinned to zero by construction.
      </p>

      <h3>why phase is the hard part</h3>
      <p>
        Classical pipelines throw phase away and rebuild it with Griffin–Lim.
        sanoTTS <b>predicts phase directly</b> — the mel carries no phase, so the
        decoder's noise conditioning is what makes phase plausible at all. The
        phase heatmap looks like static; the structure lives in how it changes
        frame to frame.
      </p>

      <h3>this sentence</h3>
      {trace && (
        <p className="small">
          [513×{trace.T}] ×2 = {(1026 * trace.T).toLocaleString()} numbers per
          utterance. Drag across the log-magnitude heatmap: the line plot below
          shows that exact frame's spectrum — the narrow crimson peaks are pitch
          harmonics, spaced by the voice's fundamental.
        </p>
      )}

      <div className="lookright">
        two heatmaps (magnitude with a real Hz axis, phase), and a draggable frame
        cursor with a live per-frame spectrum plot. Press play and the cursor
        follows the audio.
      </div>
    </div>
  );
}
