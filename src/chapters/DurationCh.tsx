import { ARCH, PARAMS } from "../data/paperFacts";
import { useViz } from "../data/store";

const MS_PER_FRAME = (ARCH.istft.hop / ARCH.sampleRate) * 1000; // 10.67 ms

export function DurationCh() {
  const trace = useViz((s) => s.trace);
  const maxD = trace ? Math.max(...trace.durs) : 0;
  const ms = (d: number) => (d * MS_PER_FRAME).toFixed(0);
  return (
    <div>
      <div className="chapter-kicker">02 · duration student</div>
      <h2>How long should each phoneme last?</h2>

      <p>
        Speech is not uniform: "m" hums for a hundred milliseconds, "t" is a
        two-frame click. Before any audio is shaped, a {PARAMS.duration.toLocaleString()}-parameter
        network reads the {trace?.N ?? "…"} ids and emits one integer per token —
        how many <b>frames</b> that phoneme occupies. One frame = {ARCH.istft.hop}{" "}
        samples = {MS_PER_FRAME.toFixed(1)} ms.
      </p>

      <h3>the mechanism</h3>
      <p>
        A 62→26 embedding is projected with three extra features — including a{" "}
        <code>log(1+N)/log(1+207)</code> length hint — then three residual blocks
        (two conv k={ARCH.dur.kernel} each, learned residual scale) refine a
        26-channel hidden state, and a linear head emits a log-duration:
      </p>
      <div className="formula">
        d<sub>t</sub> = clamp( round( e<sup>logd<sub>t</sub></sup> ), 1, {ARCH.dur.maxDuration} )
        <span className="note"> — the exp head keeps counts positive; the clamp caps any phoneme at {ms(ARCH.dur.maxDuration)} ms</span>
      </div>

      <h3>this sentence</h3>
      {trace && (
        <p className="small">
          longest phoneme: <b>{maxD} frames = {ms(maxD)} ms</b>. Sum over all{" "}
          {trace.N} tokens: <b>{trace.T} frames ={" "}
          {(trace.T * MS_PER_FRAME / 1000).toFixed(2)} s</b>. That sum fixes the
          length of every downstream tensor — [31×{trace.T}], [100×{trace.T}],
          and finally {trace.pcm.length.toLocaleString()} samples.
        </p>
      )}

      <div className="lookright">
        every phoneme's bar is real — hover for the exact frame count and
        milliseconds; the heat strip below is the [26×{trace?.N}] hidden state
        that produced them.
      </div>
      <p className="small faint">
        scope: durations here come from the trained student; the golden fixture's
        frozen durations agree closely, and the firmware's 0.98-correlation gate
        absorbs any residual difference.
      </p>
    </div>
  );
}
