import { PARAMS } from "../data/paperFacts";
import { useViz } from "../data/store";

export function AcousticCh() {
  const trace = useViz((s) => s.trace);
  return (
    <div>
      <div className="chapter-kicker">03 · acoustic student</div>
      <h2>Think at phoneme speed, speak at frame speed.</h2>

      <p>
        A phoneme sequence is a few dozen symbols; a mel spectrogram is hundreds
        of frames. Expanding one into the other with a big network would burn the
        whole parameter budget. Instead, the acoustic student is{" "}
        <b>dual-rate</b>: {PARAMS.acoustic.toLocaleString()} parameters split into
        a 3-block token encoder that reasons once per phoneme, and a 3-block frame
        encoder that refines once per output frame.
      </p>

      <h3>the mechanism</h3>
      <p>
        Three blocks over a 31-channel hidden state reason once per phoneme
        (embedding, projection, residual conv k={5} blocks); between the two rates
        sits the <b>length regulator</b> — the simplest possible aligner: copy
        token t's hidden vector into the next d<sub>t</sub> frame slots. No
        attention, no monotonic-alignment machinery, no extra parameters; the
        duration student's d<sub>t</sub> is the entire control signal. Three more
        blocks then refine the stretched stream frame by frame.
      </p>
      <div className="formula">
        frame_hidden[:, Σd<sub>&lt;t</sub> … Σd<sub>≤t</sub>) = token_hidden[:, t]
        <span className="note"> — then 3 frame-rate blocks + LayerNorm + mel head</span>
      </div>

      <h3>this sentence</h3>
      {trace && (
        <p className="small">
          [{31}×{trace.N}] in, [{31}×{trace.T}] out — a{" "}
          <b>{(trace.T / trace.N).toFixed(1)}× stretch</b>. The fan animation cycles
          through real tokens: watch a short stop consonant paint a sliver and a
          vowel paint a wide swath.
        </p>
      )}

      <div className="lookright">
        the cycling fan shows exactly which frame columns each token owns; the two
        heat strips are the real hidden states before and after the stretch.
      </div>
      <p className="small faint">
        the mel it converges toward is next — the contract everything downstream
        is bound to.
      </p>
    </div>
  );
}
