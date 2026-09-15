/**
 * OlaDiagram + WaveformDetail — the iSTFT stage: an exact Hann-window OLA
 * schematic (real window math, real n_fft/hop ratio) above the traced PCM
 * with a time axis, playhead, and play button.
 */
import { useEffect, useRef } from "react";
import { useViz } from "../data/store";
import { ARCH } from "../data/paperFacts";
import { togglePlayback } from "../audio/player";

/* exact Hann window as the engine computes it: 0.5 - 0.5·cos(2πn/N) */
const hann = (n: number, N: number) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N);

export function OlaDiagram() {
  const W = 560, H = 170;
  const frameW = 150;                       // pixels per n_fft window
  const hopPx = frameW * (ARCH.istft.hop / ARCH.istft.nFft); // exact 1:4 ratio
  const baseY = 78;

  const win = (x: number, wpx: number) =>
    Array.from({ length: 40 }, (_, i) => {
      const t = i / 39;
      return `${x + t * wpx},${baseY - hann(t, 1) * 56}`;
    }).join(" ");

  const frameXs = [30, 30 + hopPx, 30 + 2 * hopPx, 30 + 3 * hopPx];

  return (
    <div style={{ padding: "8px 16px 0" }}>
      <div className="tc-label">
        <span><b>overlap-add</b> <span className="dim">windows drawn with the engine's own Hann formula</span></span>
        <span className="faint">n_fft {ARCH.istft.nFft} · hop {ARCH.istft.hop}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
        {frameXs.map((x, i) => (
          <g key={i}>
            <polygon
              points={`${x},${baseY} ${win(x, frameW)} ${x + frameW},${baseY}`}
              fill={i === 1 ? "rgba(220,20,60,0.10)" : "rgba(0,0,0,0.04)"}
              stroke="none"
            />
            <polyline points={win(x, frameW)} fill="none" stroke={i === 1 ? "#DC143C" : "#555"} strokeWidth={i === 1 ? 2.2 : 1.4} />
            <text x={x + frameW / 2} y={14} textAnchor="middle" fontSize={9.5} fill={i === 1 ? "#DC143C" : "#666"} fontFamily="Lucida Console, monospace">
              frame {i}
            </text>
            <line x1={x} y1={baseY} x2={x} y2={baseY + 8} stroke="#999" />
            <line x1={x + frameW} y1={baseY} x2={x + frameW} y2={baseY + 8} stroke="#999" />
          </g>
        ))}
        <line x1={30} y1={baseY + 8} x2={30 + hopPx} y2={baseY + 8} stroke="#111" strokeWidth={1.2} />
        <text x={30 + hopPx / 2} y={baseY + 20} textAnchor="middle" fontSize={9.5} fill="#111" fontFamily="Lucida Console, monospace">hop 256</text>
        <line x1={30} y1={baseY + 28} x2={30 + frameW} y2={baseY + 28} stroke="#111" strokeWidth={1.2} />
        <text x={30 + frameW / 2} y={baseY + 40} textAnchor="middle" fontSize={9.5} fill="#111" fontFamily="Lucida Console, monospace">window 1024</text>
        <text x={30 + 4 * hopPx + frameW + 8} y={baseY - 20} fontSize={9.5} fill="#666" fontFamily="Lucida Console, monospace">
          each output sample =
        </text>
        <text x={30 + 4 * hopPx + frameW + 8} y={baseY - 8} fontSize={9.5} fill="#666" fontFamily="Lucida Console, monospace">
          Σ of 4 overlapping frames
        </text>
        {/* sum */}
        <text x={30} y={H - 26} fontSize={10} fill="#888" fontFamily="Lucida Console, monospace">Σ frames = PCM</text>
        <polyline
          points={Array.from({ length: 200 }, (_, i) => {
            const t = i / 199;
            const y = 0.6 * Math.sin(t * 40) * hann(t * 4 % 1, 1) + 0.4 * Math.sin(t * 90 + 1);
            return `${30 + t * (W - 60)},${H - 40 + y * 10}`;
          }).join(" ")}
          fill="none" stroke="#111" strokeWidth={1}
        />
        <text x={W - 30} y={H - 26} textAnchor="end" fontSize={9} fill="#888" fontFamily="Lucida Console, monospace">
          then DC-block high-pass
        </text>
      </svg>
    </div>
  );
}

export function WaveformDetail() {
  const trace = useViz((s) => s.trace);
  const playing = useViz((s) => s.playing);
  const ref = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  /* paint waveform (+ optional playhead), rAF loop while playing */
  useEffect(() => {
    const cv = ref.current;
    const wrap = wrapRef.current;
    if (!cv || !wrap || !trace) return;
    const pcm = trace.pcm;
    const secs = pcm.length / ARCH.sampleRate;
    const dpr = Math.min(2, window.devicePixelRatio || 1);

    const paint = (ph: number | null) => {
      const w = wrap.clientWidth * dpr, h = cv.clientHeight * dpr || 150 * dpr;
      if (cv.width !== w) cv.width = w;
      if (cv.height !== h) cv.height = h;
      const ctx = cv.getContext("2d")!;
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, w, h);
      const padL = 40 * dpr, padB = 18 * dpr, padT = 8 * dpr, padR = 8 * dpr;
      const pw = w - padL - padR, phh = h - padT - padB;
      const midY = padT + phh / 2;
      ctx.strokeStyle = "#eee";
      ctx.beginPath(); ctx.moveTo(padL, midY); ctx.lineTo(padL + pw, midY); ctx.stroke();
      ctx.fillStyle = "#111";
      const cols = Math.floor(pw / dpr);
      for (let px = 0; px < cols; px++) {
        const s0 = Math.floor((px / cols) * pcm.length);
        const s1 = Math.max(s0 + 1, Math.floor(((px + 1) / cols) * pcm.length));
        let lo = Infinity, hi = -Infinity;
        for (let s = s0; s < s1; s++) {
          const v = pcm[s];
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
        ctx.fillRect(padL + px * dpr, midY - hi * (phh / 2) * 0.92, Math.max(1, dpr * 0.9), Math.max(1, (hi - lo) * (phh / 2) * 0.92));
      }
      ctx.strokeStyle = "#ccc";
      ctx.strokeRect(padL, padT, pw, phh);
      ctx.fillStyle = "#666";
      ctx.font = `${10 * dpr}px "Lucida Console", monospace`;
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      [0, 1, 2, 3, 4].forEach((f) => {
        const t = f * (secs / 4);
        ctx.fillText(`${t.toFixed(1)}s`, padL + f * 0.25 * pw, padT + phh + 4 * dpr);
      });
      if (ph != null) {
        ctx.strokeStyle = "#DC143C";
        ctx.lineWidth = 2 * dpr;
        const x = padL + ph * pw;
        ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + phh); ctx.stroke();
        ctx.lineWidth = 1;
      }
    };

    paint(null);
    if (!playing) return;
    let raf = 0;
    const tick = () => { paint(useViz.getState().playhead); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [trace, playing]);

  if (!trace) return <div className="stage-caption">loading trace…</div>;

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <div className="playbar" style={{ padding: "6px 16px 0" }}>
        <button className="playbtn" onClick={() => togglePlayback(trace.pcm, ARCH.sampleRate)}>
          {playing ? "■ stop" : "▶ play"}
        </button>
        <span className="playtime">
          {trace.pcm.length.toLocaleString()} samples @ {ARCH.sampleRate / 1000} kHz ·{" "}
          {(trace.pcm.length / ARCH.sampleRate).toFixed(2)} s · int8→int16, full scale ±32,767
        </span>
      </div>
      <div ref={wrapRef} style={{ flex: 1, minHeight: 110, padding: "0 16px" }}>
        <canvas ref={ref} style={{ width: "100%", height: "100%", display: "block" }} />
      </div>
      <div className="stage-caption">
        this exact bitstream is what the ESP32 DMAs to its I²S codec
      </div>
    </div>
  );
}
