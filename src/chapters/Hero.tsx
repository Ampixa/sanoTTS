import { Stat, StatRow } from "../components/Panel";
import { FRONTEND, OBJECTIVE, PARAMS, WER } from "../data/paperFacts";
import { useViz } from "../data/store";

export function Hero() {
  const trace = useViz((s) => s.trace);
  return (
    <div id="hero">
      <h1 className="wordmark">sano<span className="tts">TTS</span></h1>
      <p className="tagline">
        a complete neural text-to-speech system in <b>{PARAMS.total.toLocaleString()} parameters</b>,
        built to run on the class of chip inside a sensor node.
      </p>

      <div className="howto">
        <b>how to read this page.</b> The left column is the story — scrolling it is
        the only thing that moves the pipeline stage on the right. The right
        panel is a workbench: hover any tensor, drag cursors, press play, run the
        model — it will never scroll or switch stages on its own. The crimson bar
        up top always shows where you are in the chain.
      </div>

      <StatRow>
        <Stat label="total parameters" value={PARAMS.total.toLocaleString()} />
        <Stat label="frontend params" value="0" accent />
        <Stat label="fp32 size" value={`${(PARAMS.total * 4 / 1e6).toFixed(2)} MB`} />
        <Stat label="SCOREQ (n=249)" value={OBJECTIVE.scoreq.toFixed(2)} />
        <Stat label="WER vs recordings" value={`+${WER.delta.toFixed(2)} pt`} />
      </StatRow>

      <h3>what you are looking at</h3>
      <p>
        Everything drawn here is a <b>real intermediate tensor</b> from the shipped
        int8 model, traced while it synthesized the sentence selected above. No
        schematic mock-ups, no stand-in data: the C99 that produced these numbers
        is the same code the ESP32 firmware compiles.
      </p>
      {trace && (
        <p className="small">
          currently loaded: <b className="accent">{trace.info.row_id}</b> —{" "}
          "{trace.info.text.slice(0, 80)}{trace.info.text.length > 80 ? "…" : ""}"
          {" "}→ {trace.N} tokens, {trace.T} frames, {trace.pcm.length.toLocaleString()} samples.
          The rule frontend reproduces the reference phoneme ids bit-for-bit on{" "}
          {(FRONTEND.rulesFw.bitIdentical * 100).toFixed(1)}% of {FRONTEND.rulesFw.n}{" "}
          held-out rows (segmentally identical on{" "}
          {(FRONTEND.rulesFw.segmentalIdentical * 100).toFixed(1)}%).
        </p>
      )}
      <p className="small faint">scroll to begin the walkthrough ↓</p>
    </div>
  );
}
