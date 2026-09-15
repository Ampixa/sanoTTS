import { FRONTEND } from "../data/paperFacts";
import { useViz } from "../data/store";

export function FrontendCh() {
  const trace = useViz((s) => s.trace);
  return (
    <div>
      <div className="chapter-kicker">01 · frontend</div>
      <h2>Text to phoneme ids, with <span className="accent">zero parameters</span>.</h2>

      <p>
        TTS frontends are usually where models quietly grow: tokenizers, embeddings,
        normalization networks. sanoTTS makes the frontend <b>rules only</b> — zero
        learned parameters — so every downstream network sees a compact,
        deterministic integer array.
      </p>

      <h3>the mechanism</h3>
      <p>
        A rule tokenizer splits the sentence; a rule tagger attaches coarse POS tags
        (<code>DT JJ NN MD VB RB IN VBG VBN CC XX</code>). A <b>gold lexicon</b> of{" "}
        {FRONTEND.goldEntries.toLocaleString()} entries — {FRONTEND.goldTagKeyed} of
        them tag-keyed, so <i>record/NOUN</i> and <i>record/VERB</i> resolve to
        different pronunciations — plus a {FRONTEND.silverEntries.toLocaleString()}-entry
        silver lexicon do the grapheme-to-phoneme mapping, with an espeak-ng fallback
        for out-of-vocabulary words. The resulting IPA string is encoded with a
        frozen <b>62-symbol vocabulary</b> into ids.
      </p>

      <div className="formula">
        text → tokens(+tags) → lexicon lookup → IPA phonemes → ids ∈ {"{0…61}"}
        <span className="note"> — the only thing the neural stages receive</span>
      </div>

      <h3>why rules, not a network</h3>
      <p>
        The frontend this replaces used a {FRONTEND ? "1.57M-parameter" : ""} spaCy
        tagger. The rule rewrite reproduces its phoneme ids <b>bit-for-bit on{" "}
        {(FRONTEND.rulesFw.bitIdentical * 100).toFixed(1)}%</b> of the{" "}
        {FRONTEND.rulesFw.n} held-out rows and segmentally-identically on{" "}
        {(FRONTEND.rulesFw.segmentalIdentical * 100).toFixed(1)}% — and swapping it in
        moved SCOREQ by exactly {FRONTEND.swapScoreqDelta.toFixed(1)}. Determinism here
        buys determinism everywhere downstream.
      </p>

      {trace && (
        <p className="small">
          this sentence: <b>{trace.N}</b> ids. The phoneme string —{" "}
          <span className="accent">{trace.manifest.phonemes.slice(0, 60)}…</span> —
          becomes one integer per phoneme, ids 0…61.
        </p>
      )}

      <div className="lookright">
        the four-stage flow, with the real phoneme→id chain for this sentence.
        The tagged-word chips are the documented tag example; the phoneme string
        and the id strip are live from the trace.
      </div>
    </div>
  );
}
