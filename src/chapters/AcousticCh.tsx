import { ARCH, PARAMS, TRAINING } from "../data/paperFacts";
import { useViz } from "../data/store";

export function AcousticCh() {
  const trace = useViz((s) => s.trace);
  return (
    <div>
      <div className="chapter-kicker">03 · acoustic model</div>
      <h2>Think at phoneme speed, speak at frame speed.</h2>

      <p>
        A phoneme sequence is a few dozen symbols; a mel spectrogram is hundreds
        of frames. Expanding one into the other with a single large network
        would burn the whole parameter budget. Instead, the acoustic model is{" "}
        <b>dual-rate</b>: {PARAMS.acoustic.toLocaleString()} parameters split
        into a {ARCH.ac.tokenDepth}-block token encoder that reasons once per
        phoneme, and a {ARCH.ac.frameDepth}-block frame encoder that refines
        once per output frame.
      </p>

      <h3>The mechanism</h3>
      <p>
        Three residual kernel-{ARCH.ac.kernel} blocks over a {ARCH.ac.hidden}-channel
        hidden state read the phoneme ids together with token-position and
        duration features. Between the two rates sits the <b>length regulator</b> —
        the simplest possible aligner: copy token t's hidden vector into the next
        d<sub>t</sub> frame slots. No attention, no monotonic-alignment machinery,
        no extra parameters; the duration predictor's d<sub>t</sub> is the entire
        control signal. Three more blocks then refine the stretched stream frame
        by frame, and a pointwise projection emits the 100-channel mel.
      </p>
      <div className="formula">
        frame_hidden[:, Σd<sub>&lt;t</sub> … Σd<sub>≤t</sub>) = token_hidden[:, t]
        <span className="note"> — then {ARCH.ac.frameDepth} frame-rate blocks + LayerNorm + mel head</span>
      </div>

      <h3>Why two rates</h3>
      <p>
        The two rates answer two different questions. Token-rate blocks ask{" "}
        <i>which sounds surround this phoneme?</i> — linguistic context, computed
        once per token. Frame-rate blocks ask <i>how does the sound move while it
        is being spoken?</i> — the within-phoneme trajectory, refined once per
        frame. Splitting the questions is what lets {PARAMS.acoustic.toLocaleString()}{" "}
        parameters cover both. The model was trained for{" "}
        {TRAINING.acoustic.updates.toLocaleString()} updates with int8
        quantization-aware training, regulated by the teacher's durations so it
        never had to learn around the duration student's mistakes; at inference
        it runs on predicted durations.
      </p>

      <h3>This sentence</h3>
      {trace && (
        <p className="small">
          [{ARCH.ac.hidden}×{trace.N}] in, [{ARCH.ac.hidden}×{trace.T}] out — a{" "}
          <b>{(trace.T / trace.N).toFixed(1)}× stretch</b>. The fan animation cycles
          through real tokens: watch a short stop consonant paint a sliver and a
          vowel paint a wide swath.
        </p>
      )}

      <div className="lookright">
        The cycling fan shows exactly which frame columns each token owns; the two
        heat strips are the real hidden states before and after the stretch.
      </div>
      <p className="small faint">
        The mel it converges toward is next — the contract everything downstream
        is bound to.
      </p>
    </div>
  );
}
