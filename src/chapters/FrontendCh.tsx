import { FRONTEND } from "../data/paperFacts";
import { useViz } from "../data/store";

export function FrontendCh() {
  const trace = useViz((s) => s.trace);
  return (
    <div>
      <div className="chapter-kicker">01 · frontend</div>
      <h2>Text to phoneme ids, with <span className="accent">zero parameters</span>.</h2>

      <p>
        Before any neural network runs, raw text must become a sequence of
        sounds. This stage is where TTS systems quietly grow — tokenizers,
        embeddings, normalization networks. sanoTTS makes it <b>rules only</b>:
        a modified Misaki G2P frontend with zero learned parameters, so the same
        sentence always produces the same compact integer array, on any device.
      </p>

      <h3>The mechanism</h3>
      <p>
        A rule tokenizer splits the sentence and a rule-based tagger attaches
        coarse part-of-speech tags (<code>DT JJ NN MD VB RB IN VBG VBN CC XX</code>).
        Pronunciation comes from lexicon lookup — tag-keyed entries distinguish{" "}
        <i>record/NOUN</i> from <i>record/VERB</i> — with text normalization and an{" "}
        <b>eSpeak NG fallback</b> covering out-of-vocabulary words. The resulting
        IPA string is then encoded character by character into ids from a frozen{" "}
        <b>{FRONTEND.vocab}-symbol vocabulary</b>.
      </p>

      <div className="formula">
        text → tokens(+tags) → lexicon lookup → IPA phonemes → ids ∈ {"{0…61}"}
        <span className="note"> — the only thing the neural stages receive</span>
      </div>

      <h3>Why rules, not a network</h3>
      <p>
        The reference configuration used a learned tagger of{" "}
        {FRONTEND.neuralTagger.toLocaleString()} parameters — more than five times
        the entire shipping model. sanoTTS replaces it with deterministic context
        rules and keeps everything else: {FRONTEND.kept}. On {FRONTEND.swapN}{" "}
        held-out sentences the swap moved SCOREQ by exactly{" "}
        {FRONTEND.swapScoreqDelta.toFixed(1)} — the quality stayed, the parameters
        did not. Determinism here also buys determinism everywhere downstream.
      </p>

      {trace && (
        <p className="small">
          This sentence: <b>{trace.N}</b> ids. The phoneme string —{" "}
          <span className="accent">{trace.manifest.phonemes.slice(0, 60)}…</span> —
          becomes one integer per phoneme, ids 0…{FRONTEND.vocab - 1}.
        </p>
      )}

      <div className="lookright">
        The four-stage flow, with the real phoneme→id chain for this sentence.
        The tagged-word chips are the documented tag example; the phoneme string
        and the id strip are live from the trace.
      </div>
    </div>
  );
}
