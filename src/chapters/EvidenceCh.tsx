import { MOS, OBJECTIVE, WER } from "../data/paperFacts";

export function EvidenceCh() {
  const sano = MOS.rows.find((r) => r.label.includes("sanoTTS"))!;
  const vocos = MOS.rows[0];
  return (
    <div>
      <div className="chapter-kicker">10 · evidence</div>
      <h2>Does anyone understand it? Does anyone like it?</h2>

      <h3>intelligibility</h3>
      <p>
        Four recognizers transcribe both the real recordings and sanoTTS output on{" "}
        {WER.set} (n={WER.n}): mean WER <b>{WER.meanSanotts.toFixed(2)}%</b> vs{" "}
        {WER.meanTeacher.toFixed(2)}% on the recordings — a{" "}
        <span className="accent">+{WER.delta.toFixed(2)} point</span> gap. Speech
        that survives ASR this close to ground truth is, by construction, intelligible.
      </p>

      <h3>human preference, honestly framed</h3>
      <p>
        A screened MOS pilot (<b>{MOS.sessionsPassed} of {MOS.sessionsTotal}{" "}
        sessions</b> passed attention checks; incomplete sessions excluded, no
        partial data counted): sanoTTS <b>{sano.mos.toFixed(2)}</b> [
        {sano.ci[0].toFixed(2)}, {sano.ci[1].toFixed(2)}] against a Vocos-decoder
        target at {vocos.mos.toFixed(2)}. The gap is real and the paper reports it
        as exactly that — a small, in-sample pilot, not a claim of parity.
      </p>

      <div className="lookright">
        the full four-recognizer WER table, the MOS table with confidence
        intervals, and the objective metrics (SCOREQ {OBJECTIVE.scoreq.toFixed(2)},
        UTMOS {OBJECTIVE.utmos.toFixed(2)}, DNSMOS) side by side.
      </div>
      <p className="small faint">
        scope: all evaluation is in-sample (single-speaker, LJ-style). Claims about
        other voices, languages, or domains are explicitly future work.
      </p>
    </div>
  );
}
