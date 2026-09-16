import { MCU } from "../data/paperFacts";
import { useViz } from "../data/store";

export function LiveCh() {
  const trace = useViz((s) => s.trace);
  return (
    <div>
      <div className="chapter-kicker">12 · live demo</div>
      <h2>Run the chip's brain in your browser.</h2>

      <p>
        The button on the right loads the product's own{" "}
        <code>snt_nano_heartnano.wasm</code> — compiled from the same int8 C99
        the ESP32 runs — feeds it this sentence's {trace?.N ?? "…"} ids{" "}
        <i>and its recorded seed</i>, and times the full synthesis: duration
        prediction, acoustic model, decoder, iSTFT.
      </p>

      <h3>What to watch</h3>
      <p>
        <b>Arena peak</b> — it should land near{" "}
        {(MCU.arenaPeakBytes / 1024).toFixed(0)} KiB, the number the SRAM budget
        on the previous page was drawn from. <b>Synthesis time</b> — the whole
        utterance, in a browser tab, on an ordinary laptop. And the{" "}
        <b>correlation badge</b>: because the PRNG is seeded and the run is
        deterministic, your browser should reproduce the shipped trace{" "}
        <b>sample-for-sample (corr 1.0000)</b>.
      </p>

      <div className="lookright">
        The WASM console: run, then play the result. Your machine just did the
        microcontroller's job.
      </div>
    </div>
  );
}
