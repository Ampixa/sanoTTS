/**
 * DurationDetail — the duration student's real output: one bar per phoneme
 * (frame count + milliseconds), the closed-form head, and the [26×N] hidden
 * state that produced them.
 */
import { useState } from "react";
import { useViz } from "../data/store";
import { TensorCanvas } from "./TensorCanvas";
import { ARCH } from "../data/paperFacts";

const MS_PER_FRAME = (ARCH.istft.hop / ARCH.sampleRate) * 1000; // 10.67 ms

export function DurationDetail() {
  const trace = useViz((s) => s.trace);
  const vocab = useViz((s) => s.vocab);
  const [sel, setSel] = useState<number | null>(null);
  if (!trace) return <div className="stage-caption">loading trace…</div>;

  const { durs, ids, N } = trace;
  const maxD = Math.max(...durs);
  const W = 560, H = 110, padL = 4, padB = 14;
  const bw = (W - padL * 2) / N;

  const first = Array.from({ length: Math.min(12, N - 2) }, (_, i) => i + 1); // skip <bos>

  return (
    <div style={{ overflowY: "auto", flex: 1, display: "flex", flexDirection: "column" }}>
      <div className="formula" style={{ margin: "8px 16px 4px" }}>
        d<sub>t</sub> = clamp( round( e<sup>logd<sub>t</sub></sup> ), 1, {ARCH.dur.maxDuration} )
        &nbsp;&nbsp;<span className="note">logd from a 26-hidden × 3-block conv stack · 1 frame = {MS_PER_FRAME.toFixed(1)} ms</span>
      </div>

      <div style={{ padding: "0 16px" }}>
        <div className="tc-label">
          <span><b>durations per phoneme</b> <span className="dim">[{N}]</span></span>
          <span className="faint">hover bars · longest {maxD} fr = {(maxD * MS_PER_FRAME).toFixed(0)} ms</span>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block", border: "1px solid var(--rule)" }}>
          <line x1={padL} y1={H - padB} x2={W - padL} y2={H - padB} stroke="#111" strokeWidth={1} />
          {Array.from(durs).map((d, i) => {
            const h = (d / ARCH.dur.maxDuration) * (H - padB - 6);
            const isSel = sel === i;
            return (
              <rect
                key={i}
                x={padL + i * bw}
                y={H - padB - h}
                width={Math.max(1, bw - 0.6)}
                height={h}
                fill={isSel ? "#111" : "#DC143C"}
                onMouseEnter={() => setSel(i)}
                onMouseLeave={() => setSel(null)}
              >
                <title>{`#${i} ${JSON.stringify(vocab[ids[i]] ?? "?")} — ${d} frames (${(d * MS_PER_FRAME).toFixed(0)} ms)`}</title>
              </rect>
            );
          })}
          {sel != null && (
            <text
              x={Math.min(W - 150, padL + sel * bw)}
              y={12}
              fontSize={11}
              fontFamily="Lucida Console, monospace"
              fill="#111"
            >
              {`token ${sel + 1}/${N} ${JSON.stringify(vocab[ids[sel]] ?? "?")} = ${durs[sel]} fr (${(durs[sel] * MS_PER_FRAME).toFixed(0)} ms)`}
            </text>
          )}
        </svg>
      </div>

      <div style={{ padding: "6px 16px", fontSize: 11.5, color: "#555", columnWidth: 220, columnGap: 18 }}>
        {first.map((i) => (
          <div key={i} style={{ breakInside: "avoid", whiteSpace: "nowrap" }}>
            <span className="accent">{JSON.stringify(vocab[ids[i]] ?? "?")}</span>
            {" → "}{durs[i]} fr <span className="faint">({(durs[i] * MS_PER_FRAME).toFixed(0)} ms)</span>
          </div>
        ))}
        <div className="faint">… all {N} bars above are hoverable</div>
      </div>

      <TensorCanvas
        data={trace.durHidden}
        rows={26}
        cols={N}
        label="duration hidden state"
        height={110}
        colLabel="token"
        xAxis={{ label: "phoneme index" }}
      />
      <div className="stage-caption">
        Σ d = {trace.T} frames = {(trace.T * MS_PER_FRAME / 1000).toFixed(2)} s of audio — the budget every downstream stage must fill
      </div>
    </div>
  );
}
