import { useEffect, useRef } from "react";
import { CHAPTERS, useViz } from "./data/store";
import { loadIndex, loadRow } from "./data/traces";
import { PARAMS } from "./data/paperFacts";
import { SchematicBar } from "./right/SchematicBar";
import { StageHeader } from "./right/StageHeader";
import { StageView } from "./right/StageView";
import { Hero } from "./chapters/Hero";
import { FrontendCh } from "./chapters/FrontendCh";
import { DurationCh } from "./chapters/DurationCh";
import { AcousticCh } from "./chapters/AcousticCh";
import { MelCh } from "./chapters/MelCh";
import { DecoderCh } from "./chapters/DecoderCh";
import { SpectrumCh } from "./chapters/SpectrumCh";
import { WaveformCh } from "./chapters/WaveformCh";
import { BudgetCh } from "./chapters/BudgetCh";
import { LanesCh } from "./chapters/LanesCh";
import { EvidenceCh } from "./chapters/EvidenceCh";
import { DeployCh } from "./chapters/DeployCh";
import { LiveCh } from "./chapters/LiveCh";

const CHAPTER_NODES = [
  Hero, FrontendCh, DurationCh, AcousticCh, MelCh, DecoderCh, SpectrumCh,
  WaveformCh, BudgetCh, LanesCh, EvidenceCh, DeployCh, LiveCh,
];

export default function App() {
  const rows = useViz((s) => s.rows);
  const rowId = useViz((s) => s.rowId);
  const setRows = useViz((s) => s.setRows);
  const setRow = useViz((s) => s.setRow);
  const setTrace = useViz((s) => s.setTrace);
  const hover = useViz((s) => s.hover);
  const explainRef = useRef<HTMLDivElement>(null);
  const sectionRefs = useRef<(HTMLElement | null)[]>([]);

  /* trace index → sentence list, then first row */
  useEffect(() => {
    loadIndex().then((ix) => {
      setRows(ix.rows, ix.vocab);
      loadRow(ix.rows[0].row_id, ix.rows[0]).then((t) => setTrace(t, false));
    }).catch((e) => console.error(e));
  }, [setRows, setTrace]);

  /* row switching */
  useEffect(() => {
    if (!rowId || !rows.length) return;
    const info = rows.find((r) => r.row_id === rowId);
    if (!info) return;
    let live = true;
    loadRow(rowId, info).then((t) => { if (live) setTrace(t, false); });
    return () => { live = false; };
  }, [rowId, rows, setTrace]);

  /* LEFT-column scroll is the ONLY driver of stage focus */
  useEffect(() => {
    const el = explainRef.current;
    if (!el) return;
    let raf = 0;
    const onScroll = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const mid = el.scrollTop + el.clientHeight * 0.42;
        let active = 0, t = 0;
        sectionRefs.current.forEach((sec, i) => {
          if (!sec) return;
          const top = sec.offsetTop, bottom = top + sec.offsetHeight;
          if (top <= mid && bottom >= mid) {
            active = i;
            t = (mid - top) / sec.offsetHeight;
          }
        });
        useViz.getState().setChapter(active, Math.max(0, Math.min(1, t)));
      });
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => { el.removeEventListener("scroll", onScroll); cancelAnimationFrame(raf); };
  }, []);

  /* schematic click → jump the left column (explicit navigation) */
  const jumpTo = (chapterIdx: number) => {
    const el = explainRef.current;
    const sec = sectionRefs.current[chapterIdx];
    if (el && sec) el.scrollTo({ top: sec.offsetTop + 8, behavior: "smooth" });
  };

  return (
    <div>
      <div id="topbar">
        <span className="wordmark">sano<span className="tts">TTS</span></span>
        <span className="dim">{PARAMS.total.toLocaleString()} params · traced live</span>
        <span className="spacer" />
        <label className="dim" htmlFor="rowsel">sentence</label>
        <select id="rowsel" value={rowId} onChange={(e) => setRow(e.target.value)}>
          {rows.map((r) => (
            <option key={r.row_id} value={r.row_id}>
              {r.text.length > 52 ? r.text.slice(0, 52) + "…" : r.text}
            </option>
          ))}
        </select>
      </div>

      <div id="app-shell">
        {/* LEFT */}
        <div id="explain" ref={explainRef}>
          {CHAPTER_NODES.map((Node, i) => (
            <section
              key={CHAPTERS[i].id}
              className="story-block"
              ref={(el) => { sectionRefs.current[i] = el; }}
            >
              <Node />
            </section>
          ))}
          <div id="footer" style={{ padding: "26px 7% 60px" }}>
            <p>
              <b className="accent">provenance.</b> Every tensor shown is a trace of
              the shipped int8 <b>en_us_e13b</b> weights running the same C99 the
              ESP32 executes, dumped stage-by-stage by the host tracer in this
              branch (<code>engine/trace/</code>) and validated against the
              repository's golden fixtures (min Pearson corr 0.981 over 8 rows,
              gate 0.98). The live demo runs the prebuilt WASM module from the
              main repository.
            </p>
            <p>
              evaluation numbers are reported in <i>sanoTTS: A Complete
              Sub-300K-Parameter Neural Text-to-Speech System for
              Microcontroller-Class Hardware</i> and cross-checked against the{" "}
              <code>saanotts</code> / <code>sanoTTS</code> /{" "}
              <code>saanotts-embedded</code> repositories. Sentences are the first
              eight rows of the held-out 249-sentence evaluation subset.
            </p>
            <p className="faint">ampixa · sanotts-viz branch · static build, no cookies, no tracking</p>
          </div>
        </div>

        {/* RIGHT */}
        <div
          id="stageviz"
          onWheel={(e) => {
            const el = explainRef.current;
            if (el) el.scrollTop += e.deltaY;
          }}
        >
          <SchematicBar onJump={jumpTo} />
          <StageHeader />
          <StageView />
        </div>
      </div>

      <div
        id="tooltip"
        style={hover ? { display: "block", left: hover.x, top: hover.y } : { display: "none" }}
      >
        {hover?.lines.map((l, i) => (
          <div key={i} style={i === 0 ? { color: "#DC143C" } : undefined}>{l}</div>
        ))}
      </div>
    </div>
  );
}
