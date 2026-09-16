/**
 * SchematicBar — the always-visible abstract pipeline. Nine boxes mirror the
 * 13 story chapters; the active stage is crimson, completed stages get an ink
 * tick. Clicking a box jump-scrolls the left column (explicit navigation).
 */
import { useViz } from "../data/store";
import { PARAMS } from "../data/paperFacts";

export interface StageBox {
  label: string;
  sub: string;
  chapters: number[]; // story chapters that belong to this box
}

export const STAGE_BOXES: StageBox[] = [
  { label: "overview", sub: "", chapters: [0] },
  { label: "frontend", sub: "0", chapters: [1] },
  { label: "duration", sub: PARAMS.duration.toLocaleString(), chapters: [2] },
  { label: "acoustic", sub: PARAMS.acoustic.toLocaleString(), chapters: [3] },
  { label: "mel", sub: "100 ch", chapters: [4] },
  { label: "decoder", sub: PARAMS.decoder.toLocaleString(), chapters: [5] },
  { label: "spectrum", sub: "1026", chapters: [6] },
  { label: "waveform", sub: "24 kHz", chapters: [7] },
  { label: "evidence", sub: "& live", chapters: [8, 9, 10, 11, 12] },
];

export function chapterToBox(chapter: number): number {
  const i = STAGE_BOXES.findIndex((b) => b.chapters.includes(chapter));
  return i < 0 ? 0 : i;
}

export function SchematicBar({ onJump }: { onJump: (chapter: number) => void }) {
  const chapter = useViz((s) => s.chapter);
  const active = chapterToBox(chapter);

  const n = STAGE_BOXES.length;
  const bw = 86, bh = 40, gap = 14;
  const W = n * bw + (n - 1) * gap + 8;
  const H = 62;

  return (
    <div id="schematic">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
        {STAGE_BOXES.map((b, i) => {
          const x = 4 + i * (bw + gap);
          const isActive = i === active;
          const isDone = i < active;
          return (
            <g
              key={b.label}
              onClick={() => onJump(b.chapters[0])}
              style={{ cursor: "pointer" }}
            >
              {i < n - 1 && (
                <line
                  x1={x + bw} y1={H / 2 - 4} x2={x + bw + gap} y2={H / 2 - 4}
                  stroke={isDone || isActive ? "#DC143C" : "#ccc"}
                  strokeWidth={1.5}
                  markerEnd="none"
                />
              )}
              {i < n - 1 && (
                <polygon
                  points={`${x + bw + gap - 6},${H / 2 - 7} ${x + bw + gap},${H / 2 - 4} ${x + bw + gap - 6},${H / 2 - 1}`}
                  fill={isDone || isActive ? "#DC143C" : "#ccc"}
                />
              )}
              <rect
                x={x} y={H / 2 - bh / 2 - 2} width={bw} height={bh}
                fill={isActive ? "#DC143C" : "#fff"}
                stroke={isActive ? "#DC143C" : isDone ? "#111" : "#ccc"}
                strokeWidth={isActive ? 2 : 1}
              />
              <text
                x={x + bw / 2} y={H / 2 - (b.sub ? 6 : 0)}
                textAnchor="middle" fontSize={11}
                fontFamily="Lucida Console, monospace"
                fill={isActive ? "#fff" : "#111"}
              >
                {b.label}
              </text>
              {b.sub && (
                <text
                  x={x + bw / 2} y={H / 2 + 9}
                  textAnchor="middle" fontSize={9}
                  fontFamily="Lucida Console, monospace"
                  fill={isActive ? "rgba(255,255,255,0.85)" : "#888"}
                >
                  {b.sub}
                </text>
              )}
              {isDone && (
                <text x={x + bw - 8} y={H / 2 - bh / 2 + 10} fontSize={10} fill="#DC143C" fontFamily="Lucida Console, monospace">✓</text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
