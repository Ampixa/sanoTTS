import { LANES, PARAMS } from "../data/paperFacts";

export function LanesCh() {
  const [teacher, target, oracle, full] = LANES.rows;
  return (
    <div>
      <div className="chapter-kicker">09 · four lanes</div>
      <h2>Where the quality is actually lost.</h2>

      <p>
        Distillation chains let you measure <i>who</i> loses the quality. Score each
        hand-off with SCOREQ on the same {LANES.n} held-out sentences and the blame
        is unambiguous:
      </p>

      <p>
        the 82M teacher scores <b>{teacher.scoreq.toFixed(2)}</b>; a frozen Vocos
        decoder fed teacher mels scores <b>{target.scoreq.toFixed(2)}</b> — so the
        100-channel mel contract itself is nearly free. Swap in the{" "}
        {PARAMS.decoder.toLocaleString()}-param student decoder and it drops to{" "}
        <b>{oracle.scoreq.toFixed(2)}</b>; the full student with predicted durations
        lands at <b>{full.scoreq.toFixed(2)}</b>.
      </p>

      <h3>the punchline</h3>
      <p>
        <b>{(LANES.decoderGapShare * 100).toFixed(0)}%</b> of the gap between the
        mel-target ceiling and the full student is the decoder. Not the rule
        frontend, not the 22,858-param duration student — the waveform decoder is
        where a sub-300K budget hurts, and where any future parameter should go.
      </p>

      <div className="lookright">
        the four lanes on a true 1–5 scale, each annotated with what it isolates.
      </div>
      <p className="small faint">
        source: <code>e12-nano-final-250k-20260822.json</code> — in-sample
        (single-speaker LJ-style domain); generalization claims are out of scope
        by design.
      </p>
    </div>
  );
}
