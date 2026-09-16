import { MOS, OBJECTIVE, WER } from "../data/paperFacts";

export function EvidenceCh() {
  const sano = MOS.rows.find((r) => r.label.includes("sanoTTS"))!;
  const vocos = MOS.rows[0];
  const anchor = MOS.rows[1];
  return (
    <div>
      <div className="chapter-kicker">10 · evidence</div>
      <h2>Does anyone understand it? Does anyone like it?</h2>

      <h3>Intelligibility</h3>
      <p>
        Four independent recognizers — Whisper-small, Whisper-medium,
        wav2vec2-large, HuBERT-large — transcribe both the teacher's speech and
        sanoTTS output on {WER.set} (n={WER.n}). Mean word error rate:{" "}
        <b>{WER.meanSanotts.toFixed(2)}%</b> for sanoTTS against{" "}
        {WER.meanTeacher.toFixed(2)}% for the 82M teacher — a{" "}
        <span className="accent">+{WER.delta.toFixed(2)}-point</span> gap. Speech
        that stays this close to the teacher under four different ASR systems is,
        by construction, intelligible. That supports the intelligibility claim —
        it does not claim parity in naturalness.
      </p>

      <h3>Human preference, honestly framed</h3>
      <p>
        A screened listening pilot (<b>{MOS.sessionsPassed} of {MOS.sessionsTotal}{" "}
        sessions</b> passed calibration; a single 1–5 naturalness question, TTS
        identities hidden): sanoTTS <b>{sano.mos.toFixed(2)}</b> [
        {sano.ci[0].toFixed(2)}, {sano.ci[1].toFixed(2)}], against the
        Vocos-decoder target at {vocos.mos.toFixed(2)} and a 3.5 kHz low-pass
        anchor at {anchor.mos.toFixed(2)}. Listeners placed sanoTTS below the
        low-pass anchor — <b>the words survive, the naturalness does not</b>.
        Because the test sentences were part of the training pack, this is an
        in-sample pilot, reported as exactly that: compact, intelligible
        synthesis — not yet high-quality synthesis.
      </p>

      <div className="lookright">
        The full four-recognizer WER table, the MOS table with confidence
        intervals, and the objective metrics (SCOREQ {OBJECTIVE.scoreq.toFixed(2)},
        UTMOS {OBJECTIVE.utmos.toFixed(2)}, DNSMOS) side by side.
      </div>
      <p className="small faint">
        Scope: all evaluation is English, single-speaker in style. Claims about
        other voices, languages, or domains are explicitly future work.
      </p>
    </div>
  );
}
