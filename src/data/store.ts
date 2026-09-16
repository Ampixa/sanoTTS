import { create } from "zustand";
import type { RowInfo, RowTrace } from "./traces";

export interface HoverInfo {
  x: number;
  y: number;
  lines: string[];
}

interface VizState {
  chapter: number; // active chapter index
  chapterT: number; // 0..1 progress within chapter (scroll scrub)
  rowId: string;
  rows: RowInfo[];
  vocab: string[];
  trace: RowTrace | null;
  loading: boolean;
  playing: boolean;
  playhead: number; // 0..1 across the utterance
  hover: HoverInfo | null;
  /** optional 3D-overview overlay; local view state, NEVER touches chapter */
  overviewOpen: boolean;
  setChapter: (i: number, t: number) => void;
  setRow: (id: string) => void;
  setRows: (rows: RowInfo[], vocab: string[]) => void;
  setTrace: (t: RowTrace | null, loading: boolean) => void;
  setPlaying: (p: boolean) => void;
  setPlayhead: (t: number) => void;
  setHover: (h: HoverInfo | null) => void;
  setOverviewOpen: (o: boolean) => void;
}

export const useViz = create<VizState>((set) => ({
  chapter: 0,
  chapterT: 0,
  rowId: "",
  rows: [],
  vocab: [],
  trace: null,
  loading: true,
  playing: false,
  playhead: 0,
  hover: null,
  overviewOpen: false,
  setChapter: (chapter, chapterT) => set({ chapter, chapterT }),
  setRow: (rowId) => set({ rowId, trace: null, loading: true, playhead: 0, playing: false }),
  setRows: (rows, vocab) => set({ rows, vocab, rowId: rows[0]?.row_id ?? "" }),
  setTrace: (trace, loading) => set({ trace, loading }),
  setPlaying: (playing) => set({ playing }),
  setPlayhead: (playhead) => set({ playhead }),
  setHover: (hover) => set({ hover }),
  setOverviewOpen: (overviewOpen) => set({ overviewOpen }),
}));

export const CHAPTERS = [
  { id: "hero", label: "hero" },
  { id: "frontend", label: "01 frontend" },
  { id: "duration", label: "02 durations" },
  { id: "acoustic", label: "03 acoustic" },
  { id: "mel", label: "04 mel" },
  { id: "decoder", label: "05 decoder" },
  { id: "spectrum", label: "06 spectrum" },
  { id: "istft", label: "07 waveform" },
  { id: "budget", label: "08 budget" },
  { id: "lanes", label: "09 lanes" },
  { id: "evidence", label: "10 evidence" },
  { id: "deploy", label: "11 deployment" },
  { id: "live", label: "12 live demo" },
] as const;
