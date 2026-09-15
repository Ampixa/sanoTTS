/**
 * RegulatorFan — the acoustic student's dual-rate trick made visible: a slow
 * animation highlights one token column and fans it out into its d_t frame
 * columns (real durations), above the two real hidden-state grids.
 */
import { useEffect, useMemo, useState } from "react";
import { useViz } from "../data/store";
import { TensorCanvas } from "./TensorCanvas";

export function RegulatorFan() {
  const trace = useViz((s) => s.trace);
  const vocab = useViz((s) => s.vocab);
  const [hi, setHi] = useState(2);

  const N = trace?.N ?? 0;
  const T = trace?.T ?? 0;

  /* cycle the highlighted token, but only while this stage is mounted */
  useEffect(() => {
    if (!N) return;
    const t = setInterval(() => setHi((h) => (h + 1) % Math.max(1, Math.min(N - 2, 24)) + 1 - 1), 1500);
    return () => clearInterval(t);
  }, [N]);

  const bounds = useMemo(() => trace?.tokenFrameBounds(), [trace]);
  if (!trace || !bounds) return <div className="stage-caption">loading trace…</div>;

  const W = 560, H = 120;
  const tx0 = 10, tw = W * 0.30;         // token strip
  const fx0 = W * 0.42, fw = W - fx0 - 10; // frame strip
  const cy = 54;
  const cell = Math.min(16, (tw - 8) / Math.min(N, 26));
  const shown = Math.min(N, 26);
  const hiClamped = Math.min(hi, shown - 1);

  const f0 = bounds[hiClamped], f1 = bounds[hiClamped + 1];
  const x0 = fx0 + (f0 / T) * fw, x1 = fx0 + (f1 / T) * fw;

  return (
    <div style={{ overflowY: "auto", flex: 1, display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "6px 16px 0" }}>
        <div className="tc-label">
          <span><b>length regulator</b> <span className="dim">token {hiClamped + 1} {JSON.stringify(vocab[trace.ids[hiClamped]] ?? "?")} → {trace.durs[hiClamped]} frames</span></span>
          <span className="faint">auto-cycling · real durations</span>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
          {/* token strip */}
          <text x={tx0} y={14} fontSize={10} fill="#888" fontFamily="Lucida Console, monospace">token-rate · {N} columns</text>
          {Array.from({ length: shown }, (_, i) => (
            <rect
              key={i}
              x={tx0 + i * cell} y={cy - 20}
              width={cell - 1.5} height={40}
              fill={i === hiClamped ? "#DC143C" : "#fff"}
              stroke={i === hiClamped ? "#DC143C" : "#ccc"}
              onMouseEnter={() => setHi(i)}
              style={{ cursor: "pointer" }}
            />
          ))}
          <text x={tx0} y={cy + 34} fontSize={9} fill="#888" fontFamily="Lucida Console, monospace">
            {shown < N ? `first ${shown} of ${N}` : `${N} tokens`}
          </text>

          {/* fan arrows */}
          {Array.from({ length: Math.min(f1 - f0, 12) }, (_, k) => {
            const fx = x0 + ((k + 0.5) / Math.min(f1 - f0, 12)) * (x1 - x0);
            return (
              <path
                key={k}
                d={`M ${tx0 + hiClamped * cell + cell / 2} ${cy + 20} Q ${(tx0 + hiClamped * cell + fx) / 2} ${cy + 44} ${fx} ${cy + 20}`}
                fill="none" stroke="#DC143C" strokeWidth={0.8} opacity={0.65}
              />
            );
          })}

          {/* frame strip */}
          <text x={fx0} y={14} fontSize={10} fill="#888" fontFamily="Lucida Console, monospace">frame-rate · {T} columns</text>
          <rect x={fx0} y={cy - 20} width={fw} height={40} fill="#fff" stroke="#ccc" />
          <rect x={x0} y={cy - 20} width={Math.max(2, x1 - x0)} height={40} fill="rgba(220,20,60,0.25)" stroke="#DC143C" />
          <text x={fx0} y={cy + 34} fontSize={9} fill="#888" fontFamily="Lucida Console, monospace">
            frames {f0}…{f1 - 1} come from token {hiClamped + 1}
          </text>
        </svg>
      </div>

      <TensorCanvas
        data={trace.tokHidden}
        rows={31}
        cols={N}
        label="before: token-rate hidden"
        height={100}
        colLabel="token"
        xAxis={{ label: "phoneme index" }}
      />
      <TensorCanvas
        data={trace.frameHidden}
        rows={31}
        cols={T}
        label="after: frame-rate hidden (3 more blocks refine)"
        height={130}
        playhead
        xAxis={{
          label: "time (s)",
          toUnit: (c) => `${(c * 256 / 24000).toFixed(2)} s`,
          ticks: [0, 0.25, 0.5, 0.75, 1].map((f) => ({ at: f * T, text: (f * T * 256 / 24000).toFixed(1) })),
        }}
      />
      <div className="stage-caption">
        the frame grid is the token grid stretched by d_t per column, then
        refined in place — {T / N > 0 ? `≈${(T / N).toFixed(1)} frames per token on this sentence` : ""}
      </div>
    </div>
  );
}
