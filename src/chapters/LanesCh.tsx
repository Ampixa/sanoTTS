import { DESIGN, LANES, PARAMS } from "../data/paperFacts";

export function LanesCh() {
  const [teacher, target, oracle, full] = LANES.rows;
  return (
    <div>
      <div className="chapter-kicker">09 · four lanes</div>
      <h2>Where the quality is actually lost.</h2>

      <p>
        A distilled pipeline lets you measure <i>who</i> loses the quality.
        Score each hand-off with SCOREQ on the same {LANES.n} held-out sentences
        and the blame is unambiguous:
      </p>

      <p>
        The 82M teacher scores <b>{teacher.scoreq.toFixed(2)}</b>; a frozen Vocos
        decoder fed teacher mels scores <b>{target.scoreq.toFixed(2)}</b> — so the
        100-channel mel contract itself is nearly free. Swap in the{" "}
        {PARAMS.decoder.toLocaleString()}-parameter student decoder and it drops
        to <b>{oracle.scoreq.toFixed(2)}</b>; the full student with predicted
        durations lands at <b>{full.scoreq.toFixed(2)}</b>.
      </p>

      <h3>The punchline</h3>
      <p>
        <b>{(LANES.decoderGapShare * 100).toFixed(0)}%</b> of the gap between the
        mel-target ceiling and the full student is the decoder. Not the rule
        frontend, not the {PARAMS.duration.toLocaleString()}-parameter duration
        predictor — the waveform decoder is where a sub-300K budget hurts, and
        where any future parameter should go.
      </p>

      <h3>Why four stages at all</h3>
      <p>
        A merged text-to-waveform network was tried: one compact model doing
        everything. On {DESIGN.merged.n} held-out sentences it scored{" "}
        <b>{DESIGN.merged.scoreq} SCOREQ with {DESIGN.merged.wer} WER</b> — it fit
        the training distribution without recovering the teacher's
        generalization. The factored pipeline scored {DESIGN.factored.scoreq}{" "}
        with zero WER on the same set. Separate stages won on quality, and they
        make every failure attributable — which is exactly what the lanes above
        exploit.
      </p>

      <div className="lookright">
        The four lanes on a true 1–5 scale, each annotated with what it isolates.
      </div>
      <p className="small faint">
        Scope: {LANES.n} held-out English sentences (paper Table 5). The
        in-sample caveat in the next chapter applies to the listening pilot, not
        to these lanes.
      </p>
    </div>
  );
}
