/**
 * FrontendFlow — the rule-based frontend as a 2D flow: text → tokens(+tags)
 * → lexicon → IPA phonemes → ids. The token/tag chips are the documented
 * worked example; the phoneme string and ids are the real per-sentence trace.
 */
import { useViz } from "../data/store";
import { FRONTEND } from "../data/paperFacts";

const EXAMPLE_TOKENS: [string, string][] = [
  ["The", "DT"], ["acoustic", "JJ"], ["student", "NN"], ["should", "MD"],
  ["speak", "VB"], ["clearly", "RB"], ["without", "IN"], ["sounding", "VBG"],
  ["rushed", "VBN"], ["or", "CC"], ["clipped", "VBN"], [".", "XX"],
];

function Arrow({ label }: { label: string }) {
  return (
    <div style={{ textAlign: "center", margin: "2px 0", color: "#888", fontSize: 11 }}>
      <span style={{ color: "#DC143C", fontSize: 14 }}>↓</span> {label}
    </div>
  );
}

const box: React.CSSProperties = {
  border: "1px solid var(--rule)",
  background: "#fff",
  padding: "8px 10px",
  fontSize: 12.5,
};

export function FrontendFlow() {
  const trace = useViz((s) => s.trace);
  const vocab = useViz((s) => s.vocab);
  if (!trace) return <div className="stage-caption">loading trace…</div>;

  const ids = trace.ids;
  const shown = Math.min(20, ids.length);

  return (
    <div style={{ overflowY: "auto", padding: "10px 16px", flex: 1 }}>
      <div style={box}>
        <span className="faint">input text — </span>{trace.info.text}
      </div>

      <Arrow label="rule tokenizer + rule tagger (no learned weights)" />
      <div style={box}>
        <div className="token-strip" style={{ margin: 0 }}>
          {EXAMPLE_TOKENS.map(([w, tag], i) => (
            <span key={i} className={`token-chip ${tag === "XX" ? "xx" : ""}`}>
              {w}
              <span className="tag">{tag}</span>
            </span>
          ))}
        </div>
        <div className="faint" style={{ fontSize: 10.5, marginTop: 4 }}>
          documented tag example (sentence 1); tags pick lexicon entries — record/NOUN ≠ record/VERB
        </div>
      </div>

      <Arrow label={`gold lexicon ${FRONTEND.goldEntries.toLocaleString()} (790 tag-keyed) · silver ${FRONTEND.silverEntries.toLocaleString()} · espeak-ng fallback`} />
      <div style={{ ...box, wordBreak: "break-all", lineHeight: 1.9 }}>
        <span className="faint">phonemes — </span>
        <b className="accent">{trace.manifest.phonemes}</b>
      </div>

      <Arrow label="character → id, frozen 62-symbol vocabulary" />
      <div style={box}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
          {Array.from(ids.slice(0, shown)).map((id, i) => (
            <span
              key={i}
              style={{
                border: "1px solid var(--rule)",
                padding: "2px 5px",
                fontSize: 11,
                background: i === 0 || i === ids.length - 1 ? "#f5f5f5" : "#fff",
              }}
              title={`token ${i}`}
            >
              <span className="accent">{id}</span>
              <span className="faint"> {JSON.stringify(vocab[id] ?? "?")}</span>
            </span>
          ))}
          <span className="faint" style={{ alignSelf: "center" }}>… {ids.length - shown} more</span>
        </div>
        <div style={{ fontSize: 10.5, color: "#888", marginTop: 4 }}>
          this integer array is the ONLY thing the neural stages receive — {ids.length} ids, each 0…61
        </div>
      </div>

      <div className="stage-caption" style={{ border: "none", padding: "8px 0 0" }}>
        hover panels in later stages to follow these exact ids downstream
      </div>
    </div>
  );
}
