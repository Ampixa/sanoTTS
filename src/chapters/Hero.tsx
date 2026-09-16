import { Stat, StatRow } from "../components/Panel";
import { FRONTEND, OBJECTIVE, PARAMS, WER } from "../data/paperFacts";
import { useViz } from "../data/store";

export function Hero() {
  const trace = useViz((s) => s.trace);
  return (
    <div id="hero">
      <h1 className="wordmark">sano<span className="tts">TTS</span></h1>
      <p className="tagline">
        A complete neural text-to-speech system in <b>{PARAMS.total.toLocaleString()} parameters</b> —
        small enough to run on the class of chip inside a sensor node.
      </p>

      <div className="howto">
        <b>How to read this page.</b> The left column is the story — scrolling it is
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
        <Stat label="WER gap vs teacher" value={`+${WER.delta.toFixed(2)} pt`} />
      </StatRow>

      <h3>What you are looking at</h3>
      <p>
        Every stage of a real TTS pipeline, drawn from the inside: text becomes
        phoneme ids, ids become durations, durations become a mel spectrogram,
        and the spectrogram becomes a 24&nbsp;kHz waveform. Everything rendered
        here is a <b>real intermediate tensor</b> from the shipped int8 model,
        captured while it synthesized the sentence selected above. No schematic
        mock-ups and no stand-in data — the C99 that produced these numbers is
        the same code the ESP32 firmware compiles.
      </p>
      {trace && (
        <p className="small">
          Currently loaded: <b className="accent">{trace.info.row_id}</b> —{" "}
          "{trace.info.text.slice(0, 80)}{trace.info.text.length > 80 ? "…" : ""}"
          {" "}→ {trace.N} tokens, {trace.T} frames, {trace.pcm.length.toLocaleString()} samples.
          The rule-based frontend replaced a{" "}
          {FRONTEND.neuralTagger.toLocaleString()}-parameter learned tagger with no
          measurable quality change (SCOREQ, {FRONTEND.swapN} held-out sentences).
        </p>
      )}
      <p className="small faint">Scroll to begin the walkthrough ↓</p>
    </div>
  );
}
