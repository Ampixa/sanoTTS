/**
 * SpectrumDetail — the head's 513 log-magnitudes + 513 phases with a true Hz
 * axis, plus a single-frame magnitude line plot driven by a draggable cursor
 * (or by playback).
 */
import { useEffect, useRef, useState } from "react";
import { useViz } from "../data/store";
import { TensorCanvas } from "./TensorCanvas";
import { ARCH } from "../data/paperFacts";
import { togglePlayback } from "../audio/player";

const HZ_PER_BIN = ARCH.sampleRate / ARCH.istft.nFft; // 23.4375 Hz

function FramePlot({ frame }: { frame: number }) {
  const trace = useViz((s) => s.trace);
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !trace) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = cv.clientWidth * dpr, h = cv.clientHeight * dpr;
    cv.width = w; cv.height = h;
    const ctx = cv.getContext("2d")!;
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, w, h);

    const T = trace.T;
    const f = Math.max(0, Math.min(T - 1, frame));
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < 513; k++) {
      const v = trace.specLogmag[k * T + f];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const padL = 40 * dpr, padB = 16 * dpr, padT = 6 * dpr, padR = 8 * dpr;
    const pw = w - padL - padR, ph = h - padT - padB;
    ctx.strokeStyle = "#ccc";
    ctx.strokeRect(padL, padT, pw, ph);
    ctx.beginPath();
    for (let k = 0; k < 513; k++) {
      const v = trace.specLogmag[k * T + f];
      const x = padL + (k / 512) * pw;
      const y = padT + (1 - (v - lo) / (hi - lo + 1e-9)) * ph;
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#DC143C";
    ctx.lineWidth = 1.4 * dpr;
    ctx.stroke();
    ctx.lineWidth = 1;

    ctx.fillStyle = "#666";
    ctx.font = `${10 * dpr}px "Lucida Console", monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    [0, 3, 6, 9, 12].forEach((khz) => {
      const k = (khz * 1000) / HZ_PER_BIN;
      ctx.fillText(`${khz}k`, padL + (k / 512) * pw, padT + ph + 3 * dpr);
    });
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(hi.toFixed(1), padL - 4 * dpr, padT + 4 * dpr);
    ctx.fillText(lo.toFixed(1), padL - 4 * dpr, padT + ph - 4 * dpr);
    ctx.fillStyle = "#888"; ctx.textAlign = "left";
    ctx.fillText(`frame ${f + 1}/${T} · ${(f * 256 / 24000).toFixed(3)} s · one column of the upper heatmap`, padL, 2 * dpr);
  }, [frame, trace]);

  return <canvas ref={ref} style={{ width: "100%", height: 130, display: "block" }} />;
}

export function SpectrumDetail() {
  const trace = useViz((s) => s.trace);
  const playing = useViz((s) => s.playing);
  const [cursor, setCursor] = useState<number | null>(null);

  /* playback drives the frame cursor */
  const playhead = useViz((s) => s.playhead);
  useEffect(() => {
    if (playing && trace) setCursor(Math.floor(playhead * (trace.T - 1)));
  }, [playhead, playing, trace]);

  if (!trace) return <div className="stage-caption">loading trace…</div>;
  const T = trace.T;
  const frame = cursor ?? Math.floor(T / 2);

  const hzTicks = [0, 128, 256, 384, 512].map((k) => ({
    at: k,
    text: k === 512 ? "12k" : `${((k * HZ_PER_BIN) / 1000).toFixed(0)}k`,
  }));

  return (
    <div style={{ overflowY: "auto", flex: 1, display: "flex", flexDirection: "column" }}>
      <div className="playbar" style={{ padding: "6px 16px 0" }}>
        <button className="playbtn" onClick={() => togglePlayback(trace.pcm, ARCH.sampleRate)}>
          {playing ? "■ stop" : "▶ play"}
        </button>
        <span className="playtime">click/drag the heatmaps to pick a frame</span>
      </div>
      <TensorCanvas
        data={trace.specLogmag}
        rows={513}
        cols={T}
        label="log-magnitude"
        height={190}
        mode="hot"
        flipY
        playhead
        cursorCol={frame}
        onCursor={setCursor}
        rowLabel="bin"
        xAxis={{ label: "time (s)", toUnit: (c) => `${(c * 256 / 24000).toFixed(2)} s` }}
        yAxis={{ label: "Hz", toUnit: (r) => `${((r * HZ_PER_BIN) / 1000).toFixed(1)} kHz`, ticks: hzTicks }}
        valueFmt={(v) => `${v.toFixed(2)} ln·mag`}
      />
      <FramePlot frame={frame} />
      <TensorCanvas
        data={trace.specPhase}
        rows={513}
        cols={T}
        label="phase"
        height={130}
        mode="phase"
        flipY
        cursorCol={frame}
        onCursor={setCursor}
        rowLabel="bin"
        xAxis={{ label: "time (s)", toUnit: (c) => `${(c * 256 / 24000).toFixed(2)} s` }}
        yAxis={{ label: "Hz", ticks: hzTicks }}
        valueFmt={(v) => `${v.toFixed(2)} rad`}
      />
      <div className="stage-caption">
        1,026 numbers per frame: 513 magnitudes (crimson plot above) + 513
        phases · DC and Nyquist are pinned to zero by construction
      </div>
    </div>
  );
}
