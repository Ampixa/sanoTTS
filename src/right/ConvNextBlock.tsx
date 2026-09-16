/**
 * ConvNextBlock — one decoder block as a labeled flow diagram with per-op
 * parameter math verified against the shipped blob layout (nano_q8_meta.h
 * byte offsets); beside the real trunk activations (stem + 4 blocks) and the
 * 4-channel seeded noise that conditions them.
 */
import { useViz } from "../data/store";
import { TensorCanvas } from "./TensorCanvas";
import { ARCH, PARAMS } from "../data/paperFacts";

/* exact param counts from dims (62 / 186 / k7), cross-checked against the
 * blob byte offsets: stem 45,384 + 4×23,994 + fnorm 124 + head 64,638
 * = 206,122 — the shipped decoder total, exactly. */
const OPS: { name: string; math: string; params: number }[] = [
  { name: "depthwise conv k=7", math: "62×7 w + 62 b", params: 62 * 7 + 62 },
  { name: "LayerNorm", math: "γ,β per channel", params: 124 },
  { name: "1×1 conv 62→186", math: "62×186 + 186", params: 62 * 186 + 186 },
  { name: "GELU", math: "0 params", params: 0 },
  { name: "1×1 conv 186→62", math: "186×62 + 62", params: 186 * 62 + 62 },
  { name: "γ scale + residual add", math: "62", params: 62 },
];

export function ConvNextBlock() {
  const trace = useViz((s) => s.trace);
  if (!trace) return <div className="stage-caption">loading trace…</div>;
  const T = trace.T;

  const bw = 200, bh = 44, gap = 6;
  const H = OPS.length * (bh + gap) + 70;

  return (
    <div style={{ overflowY: "auto", flex: 1, display: "flex", minHeight: 0 }}>
      {/* mechanism diagram */}
      <div style={{ flex: "none", width: 250, padding: "8px 0 8px 12px" }}>
        <svg viewBox={`0 0 230 ${H}`} style={{ width: "100%", display: "block" }}>
          <text x={115} y={12} textAnchor="middle" fontSize={10} fill="#888" fontFamily="Lucida Console, monospace">
            one ConvNeXt1D block (×4)
          </text>
          {/* residual loop */}
          <path
            d={`M ${15 + bw / 2} 24 L 15 24 L 15 ${H - 30} L ${15 + bw / 2} ${H - 30}`}
            fill="none" stroke="#DC143C" strokeWidth={1} strokeDasharray="3 2"
          />
          {OPS.map((op, i) => {
            const y = 20 + i * (bh + gap);
            return (
              <g key={op.name}>
                <rect x={15} y={y} width={bw} height={bh} fill="#fff" stroke={op.params ? "#111" : "#ccc"} />
                <text x={22} y={y + 16} fontSize={10} fill="#111" fontFamily="Lucida Console, monospace">{op.name}</text>
                <text x={22} y={y + 29} fontSize={9} fill="#888" fontFamily="Lucida Console, monospace">{op.math}</text>
                <text x={208} y={y + 16} textAnchor="end" fontSize={9.5} fill="#DC143C" fontFamily="Lucida Console, monospace">
                  {op.params ? op.params.toLocaleString() : "—"}
                </text>
                {i < OPS.length - 1 && (
                  <line x1={15 + bw / 2} y1={y + bh} x2={15 + bw / 2} y2={y + bh + gap} stroke="#999" />
                )}
              </g>
            );
          })}
          <text x={115} y={H - 8} textAnchor="middle" fontSize={9} fill="#888" fontFamily="Lucida Console, monospace">
            23,994 per block — verified from blob offsets
          </text>
        </svg>
      </div>

      {/* real activations */}
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", justifyContent: "center" }}>
        <TensorCanvas
          data={trace.noise} rows={ARCH.dec.noiseCh} cols={T}
          label="seeded Gaussian noise (conditioning)" height={52}
          rowLabel="noise ch" xAxis={{ label: "" }}
        />
        <TensorCanvas
          data={trace.stem} rows={62} cols={T}
          label="stem: embed(mel) + adapter(noise) + LN — 45,384 p" height={86}
          xAxis={{ label: "" }}
        />
        {trace.trunk.map((t, b) => (
          <TensorCanvas
            key={b}
            data={t} rows={62} cols={T}
            label={`after block ${b + 1}/4`} height={72}
            playhead={b === 3}
            xAxis={b === 3 ? {
              label: "time (s)",
              toUnit: (c) => `${(c * 256 / 24000).toFixed(2)} s`,
              ticks: [0, 0.5, 1].map((f) => ({ at: f * T, text: (f * T * 256 / 24000).toFixed(1) })),
            } : { label: "" }}
          />
        ))}
        <div className="stage-caption">
          same [62×{T}] plane, written in place by each block — then final LN (124 p) + 62→1026 head (64,638 p):
          45,384 + 4×23,994 + 124 + 64,638 = {PARAMS.decoder.toLocaleString()}, exactly the shipped decoder
        </div>
      </div>
    </div>
  );
}
