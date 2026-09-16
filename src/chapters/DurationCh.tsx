import { ARCH, PARAMS, TRAINING } from "../data/paperFacts";
import { useViz } from "../data/store";

const MS_PER_FRAME = (ARCH.istft.hop / ARCH.sampleRate) * 1000; // 10.67 ms

export function DurationCh() {
  const trace = useViz((s) => s.trace);
  const maxD = trace ? Math.max(...trace.durs) : 0;
  const ms = (d: number) => (d * MS_PER_FRAME).toFixed(0);
  return (
    <div>
      <div className="chapter-kicker">02 · duration predictor</div>
      <h2>How long should each phoneme last?</h2>

      <p>
        Speech is not uniform: an "m" hums for a hundred milliseconds, a "t" is
        a two-frame click. Before any audio is shaped, a{" "}
        {PARAMS.duration.toLocaleString()}-parameter network reads the{" "}
        {trace?.N ?? "…"} ids and emits one integer per phoneme — how many{" "}
        <b>frames</b> it occupies. One frame is {ARCH.istft.hop} samples, or{" "}
        {MS_PER_FRAME.toFixed(1)} ms.
      </p>

      <h3>The mechanism</h3>
      <p>
        A {ARCH.vocab}→{ARCH.dur.hidden} embedding is projected together with
        three deterministic features — normalized token position, an
        utterance-length hint, and a validity indicator — then{" "}
        {ARCH.dur.depth} residual convolution blocks (two kernel-
        {ARCH.dur.kernel} convolutions each, SiLU, learned residual scale)
        refine a {ARCH.dur.hidden}-channel hidden state. A final projection
        emits one log-duration per phoneme:
      </p>
      <div className="formula">
        d<sub>t</sub> = clamp( round( e<sup>logd<sub>t</sub></sup> ), 1, {ARCH.dur.maxDuration} )
        <span className="note"> — the exp head keeps counts positive; the clamp caps any phoneme at {ms(ARCH.dur.maxDuration)} ms</span>
      </div>

      <h3>Why log-durations</h3>
      <p>
        The student is trained on the teacher's token-level frame counts with a
        SmoothL1 loss in <b>log-duration space</b> — {TRAINING.duration.updates.toLocaleString()}{" "}
        updates, batch {TRAINING.duration.batch}. Log space makes errors
        proportional, so being one frame short on a two-frame stop costs as much
        as being many frames short on a long pause. That matters: many small
        underestimates would otherwise accumulate and leave every utterance too
        short.
      </p>

      <h3>This sentence</h3>
      {trace && (
        <p className="small">
          Longest phoneme: <b>{maxD} frames = {ms(maxD)} ms</b>. Sum over all{" "}
          {trace.N} tokens: <b>{trace.T} frames ={" "}
          {(trace.T * MS_PER_FRAME / 1000).toFixed(2)} s</b>. That sum fixes the
          length of every downstream tensor — [31×{trace.T}], [100×{trace.T}],
          and finally {trace.pcm.length.toLocaleString()} samples.
        </p>
      )}

      <div className="lookright">
        Every phoneme's bar is real — hover for the exact frame count and
        milliseconds; the heat strip below is the [{ARCH.dur.hidden}×{trace?.N}]{" "}
        hidden state that produced them.
      </div>
    </div>
  );
}
