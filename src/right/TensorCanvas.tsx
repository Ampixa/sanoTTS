/**
 * TensorCanvas — the workhorse tensor view: a [rows × cols] float tensor drawn
 * to a 2D canvas with real axes (time, Hz, band…), an optional colorbar, a
 * hover readout, an optional draggable column cursor, and an optional
 * playback playhead. All values shown are dequantized engine outputs.
 */
import { useEffect, useMemo, useRef } from "react";
import { minmax, rampHot, rampPhase, rampSigned } from "../three/ramp";
import { useViz } from "../data/store";

export interface AxisSpec {
  label: string;
  /** map a column/row index to a unit string for hover, e.g. frame→seconds */
  toUnit?: (i: number) => string;
  /** explicit tick positions in cell-index units with labels */
  ticks?: { at: number; text: string }[];
}

interface Props {
  data: Float32Array;
  rows: number;
  cols: number;
  label: string;
  height?: number;          // css px of the plot area
  mode?: "signed" | "hot" | "phase";
  xAxis?: AxisSpec;
  yAxis?: AxisSpec;
  colorbar?: boolean;
  valueFmt?: (v: number) => string;
  rowLabel?: string;        // hover axis name
  colLabel?: string;
  cursorCol?: number | null;
  onCursor?: (c: number) => void;
  playhead?: boolean;       // overlay store.playhead cursor
  flipY?: boolean;          // draw row 0 at the BOTTOM (spectrogram convention)
  maxDrawCols?: number;
}

const M = { l: 46, r: 8, t: 4, b: 20 };
const CB_W = 44;

export function TensorCanvas({
  data, rows, cols, label,
  height = 180,
  mode = "signed",
  xAxis, yAxis,
  colorbar = false,
  valueFmt = (v) => v.toFixed(3),
  rowLabel = "ch",
  colLabel = "frame",
  cursorCol = null,
  onCursor,
  playhead = false,
  flipY = false,
  maxDrawCols = 1400,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const setHover = useViz((s) => s.setHover);
  const playing = useViz((s) => s.playing);
  const stride = Math.max(1, Math.ceil(cols / maxDrawCols));

  const { lo, hi, absmax } = useMemo(() => {
    const [lo, hi] = minmax(data);
    return { lo, hi, absmax: Math.max(Math.abs(lo), Math.abs(hi)) || 1e-9 };
  }, [data]);

  /* offscreen image of the tensor itself (rebuilt on data change) */
  const img = useMemo(() => {
    const dcols = Math.ceil(cols / stride);
    const off = document.createElement("canvas");
    off.width = dcols; off.height = rows;
    const cx = off.getContext("2d")!;
    const im = cx.createImageData(dcols, rows);
    let i = 0;
    for (let r = 0; r < rows; r++) {
      for (let dc = 0; dc < dcols; dc++, i++) {
        const v = data[r * cols + dc * stride];
        let rgb: [number, number, number];
        if (mode === "phase") rgb = rampPhase(v);
        else if (mode === "hot") rgb = rampHot((v - lo) / (hi - lo));
        else rgb = rampSigned(v / absmax);
        im.data[i * 4] = rgb[0]; im.data[i * 4 + 1] = rgb[1];
        im.data[i * 4 + 2] = rgb[2]; im.data[i * 4 + 3] = 255;
      }
    }
    cx.putImageData(im, 0, 0);
    return off;
  }, [data, rows, cols, stride, mode, lo, hi, absmax]);

  const paint = (playheadT: number | null) => {
    const cv = canvasRef.current;
    if (!cv) return;
    const W = cv.width, H = cv.height;
    const ctx = cv.getContext("2d");
    if (!ctx || W === 0) return;
    const cbSpace = colorbar ? CB_W : 0;
    const pw = W - M.l - M.r - cbSpace;
    const ph = H - M.t - M.b;
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = false;
    if (flipY) {
      ctx.save();
      ctx.translate(0, M.t + ph);
      ctx.scale(1, -1);
      ctx.drawImage(img, M.l, 0, pw, ph);
      ctx.restore();
    } else {
      ctx.drawImage(img, M.l, M.t, pw, ph);
    }

    /* frame */
    ctx.strokeStyle = "#ccc";
    ctx.lineWidth = 1;
    ctx.strokeRect(M.l + 0.5, M.t + 0.5, pw - 1, ph - 1);

    /* axes */
    ctx.fillStyle = "#666";
    ctx.font = '10px "Lucida Console", monospace';
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    xAxis?.ticks?.forEach(({ at, text }) => {
      const x = M.l + (at / cols) * pw;
      ctx.fillText(text, x, H - M.b + 5);
      ctx.strokeStyle = "#eee";
      ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, M.t + ph); ctx.stroke();
    });
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    yAxis?.ticks?.forEach(({ at, text }) => {
      const y = flipY ? M.t + (1 - at / rows) * ph : M.t + (at / rows) * ph;
      ctx.fillText(text, M.l - 5, y);
      ctx.strokeStyle = "#eee";
      ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(M.l + pw, y); ctx.stroke();
    });
    if (xAxis) { ctx.textAlign = "center"; ctx.fillStyle = "#888"; ctx.fillText(xAxis.label, M.l + pw / 2, H - 11); }

    /* colorbar */
    if (colorbar) {
      const bx = W - CB_W + 12, bw = 10;
      for (let y = 0; y < ph; y++) {
        const t = 1 - y / ph;
        const rgb = mode === "phase" ? rampPhase((t * 2 - 1) * Math.PI)
          : mode === "hot" ? rampHot(t)
          : rampSigned(t * 2 - 1);
        ctx.fillStyle = `rgb(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0})`;
        ctx.fillRect(bx, M.t + y, bw, 1.5);
      }
      ctx.strokeStyle = "#ccc";
      ctx.strokeRect(bx + 0.5, M.t + 0.5, bw - 1, ph - 1);
      ctx.fillStyle = "#666"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      const fmt = (v: number) => Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1);
      if (mode === "phase") {
        ctx.fillText("π", bx + bw + 3, M.t);
        ctx.fillText("0", bx + bw + 3, M.t + ph / 2);
        ctx.fillText("-π", bx + bw + 3, M.t + ph);
      } else if (mode === "hot") {
        ctx.fillText(fmt(hi), bx + bw + 3, M.t + 4);
        ctx.fillText(fmt(lo), bx + bw + 3, M.t + ph - 4);
      } else {
        ctx.fillText(fmt(absmax), bx + bw + 3, M.t + 4);
        ctx.fillText("0", bx + bw + 3, M.t + ph / 2);
        ctx.fillText(fmt(-absmax), bx + bw + 3, M.t + ph - 4);
      }
    }

    /* cursors */
    const drawCursor = (t: number, color: string, wpx: number) => {
      const x = M.l + t * pw;
      ctx.strokeStyle = color;
      ctx.lineWidth = wpx;
      ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, M.t + ph); ctx.stroke();
      ctx.lineWidth = 1;
    };
    if (cursorCol != null) drawCursor(cursorCol / cols, "#111", 1);
    if (playheadT != null) drawCursor(playheadT, "#DC143C", 2);
  };

  /* size handling + redraw on data/cursor change */
  useEffect(() => {
    const cv = canvasRef.current!;
    const wrap = wrapRef.current!;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const resize = () => {
      const w = wrap.clientWidth;
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(height * dpr);
      cv.style.height = `${height}px`;
      paint(null);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [img, height, cursorCol, colorbar]);

  useEffect(() => { paint(null); }, [cursorCol, img]);

  /* playhead rAF */
  useEffect(() => {
    if (!playhead || !playing) { if (playhead) paint(null); return; }
    let raf = 0;
    const tick = () => {
      paint(useViz.getState().playhead);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, playhead, img, cursorCol]);

  const cellFromEvent = (e: React.MouseEvent) => {
    const cv = canvasRef.current!;
    const rect = cv.getBoundingClientRect();
    const cbSpace = colorbar ? CB_W : 0;
    const dpr = cv.width / rect.width;
    void dpr;
    const pwCss = rect.width - ((M.l + M.r + cbSpace) / cv.width) * rect.width;
    const x = e.clientX - rect.left - (M.l / cv.width) * rect.width;
    const y = e.clientY - rect.top - (M.t / cv.height) * rect.height;
    const phCss = rect.height - ((M.t + M.b) / cv.height) * rect.height;
    const c = Math.floor((x / pwCss) * cols);
    let r = Math.floor((y / phCss) * rows);
    if (flipY) r = rows - 1 - r;
    if (c < 0 || c >= cols || r < 0 || r >= rows) return null;
    return { r, c };
  };

  return (
    <div className="tc-block" ref={wrapRef}>
      <div className="tc-label">
        <span><b>{label}</b> <span className="dim">[{rows}×{cols}]</span></span>
        <span className="faint">{mode === "hot" ? `min ${lo.toFixed(2)} · max ${hi.toFixed(2)}` : mode === "signed" ? `±${absmax.toFixed(2)}` : "-π…π"}</span>
      </div>
      <canvas
        ref={canvasRef}
        onMouseMove={(e) => {
          const cell = cellFromEvent(e);
          if (!cell) { setHover(null); return; }
          const { r, c } = cell;
          const v = data[r * cols + c];
          const lines = [
            label,
            `${rowLabel} ${r + 1}/${rows}${yAxis?.toUnit ? ` · ${yAxis.toUnit(r)}` : ""} · ${colLabel} ${c + 1}/${cols}${xAxis?.toUnit ? ` · ${xAxis.toUnit(c)}` : ""}`,
            valueFmt(v),
          ];
          setHover({ x: e.clientX + 14, y: e.clientY + 10, lines });
          if (onCursor && (e.buttons & 1)) onCursor(c);
        }}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          const cell = cellFromEvent(e);
          if (cell && onCursor) onCursor(cell.c);
        }}
      />
    </div>
  );
}
