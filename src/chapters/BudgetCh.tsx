import { MCU, PARAMS } from "../data/paperFacts";

const PCT = (x: number) => `${((x / PARAMS.total) * 100).toFixed(1)}%`;

export function BudgetCh() {
  return (
    <div>
      <div className="chapter-kicker">08 · parameter budget</div>
      <h2>Where the {PARAMS.total.toLocaleString()} parameters go.</h2>

      <p>
        <b>0%</b> frontend (rules), <b>{PCT(PARAMS.duration)}</b> duration student,{" "}
        <b>{PCT(PARAMS.acoustic)}</b> acoustic student, <b>{PCT(PARAMS.decoder)}</b>{" "}
        waveform decoder. The decoder dominates because magnitudes-and-phase at
        1,026 channels per frame is the widest interface in the system — its head
        alone is 64,638 parameters.
      </p>

      <h3>the 14.9M-parameter ghost</h3>
      <p>
        Training uses a <b>multi-period discriminator</b> of {PARAMS.mpd.toLocaleString()}{" "}
        parameters — {(PARAMS.mpd / PARAMS.total).toFixed(0)}× the entire product —
        plus an 82M-parameter teacher. Neither ships: they exist only to shape the
        student, and the deployed artifact is the {PARAMS.total.toLocaleString()}-param
        stack on the bar. The dashed outline on the right is drawn at 1/50 of its
        true width; at full width it would need fifty screens.
      </p>

      <h3>bytes on device</h3>
      <p className="small">
        fp32 the stack is {(PARAMS.total * 4 / 1e6).toFixed(2)} MB; the int8 export
        is <b>{(MCU.weightsBytes / 1024).toFixed(1)} KiB</b> of weights, read
        straight from memory-mapped flash on the ESP32. Counts are re-verified
        against <code>meta.json</code> and the blob byte offsets by{" "}
        <code>tools/verify_facts.py</code>.
      </p>

      <div className="lookright">
        the full stacked bar, each segment to scale, with the training-only
        discriminator as a dashed ghost.
      </div>
    </div>
  );
}
