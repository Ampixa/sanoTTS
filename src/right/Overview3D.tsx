/**
 * Overview3D — the whole-pipeline 3D scene as an interactive workbench:
 * zoom/pan/rotate (OrbitControls), play with the crimson playhead sweeping
 * the stations, and per-stage camera presets. Used as the hero stage view and
 * as the optional overlay every stage can summon. Opening it never changes
 * the story's chapter — it is pure view state.
 */
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Component, useEffect, useState, type ReactNode } from "react";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { PipelineModel, STAGE_X } from "../three/PipelineModel";
import { useViz } from "../data/store";
import { togglePlayback } from "../audio/player";
import { ARCH } from "../data/paperFacts";

/** full-chain view (what the hero used before controls existed) */
const FULL = { pos: [46, 11, 48], tgt: [46, 0, 0] } as const;

/** per-stage camera presets keyed by story chapter (stations sit on STAGE_X) */
const FOCUS: Record<number, { pos: [number, number, number]; tgt: [number, number, number] }> = {
  1: { pos: [STAGE_X.ids, 3, 14], tgt: [STAGE_X.ids, 0, 0] },
  2: { pos: [STAGE_X.durs, 3, 13], tgt: [STAGE_X.durs, -0.4, 0] },
  3: { pos: [30, 4, 18], tgt: [30, 0.3, 0] },          // tok (24) + frame (36)
  4: { pos: [STAGE_X.mel, 3, 15], tgt: [STAGE_X.mel, 0.2, 0] },
  5: { pos: [STAGE_X.dec, 2, 19], tgt: [STAGE_X.dec, 0.4, 0] },
  6: { pos: [STAGE_X.spec, 2, 18], tgt: [STAGE_X.spec, 0.5, 0] },
  7: { pos: [STAGE_X.wave, 2, 13], tgt: [STAGE_X.wave, 0, 0] },
};

/** WebGL can be unavailable (headless, locked-down browsers) — degrade, never crash. */
class GLBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(err: unknown) { console.warn("3D overview unavailable:", err); }
  render() {
    if (this.state.failed) {
      return (
        <div style={{ flex: 1, display: "grid", placeItems: "center", padding: 24 }}>
          <div style={{ maxWidth: 420, fontSize: 13, color: "#555", border: "1px dashed var(--rule)", padding: 18 }}>
            <b className="accent">3D overview unavailable.</b> This browser could not
            create a WebGL context. The full walkthrough still works — scroll the
            left column; every stage view below is a 2D canvas.
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

/** applies camera presets when `preset` changes (null = full-chain view) */
function CameraFocus({ preset }: { preset: number | null }) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null;
  useEffect(() => {
    const f = preset != null ? FOCUS[preset] : undefined;
    const pos = f ? f.pos : FULL.pos;
    const tgt = f ? f.tgt : FULL.tgt;
    camera.position.set(pos[0], pos[1], pos[2]);
    if (controls) {
      controls.target.set(tgt[0], tgt[1], tgt[2]);
      controls.update();
    }
  }, [preset, camera, controls]);
  return null;
}

export function Overview3D({
  focusChapter = null,
  onClose,
}: {
  /** stage to frame on open; null = whole chain */
  focusChapter?: number | null;
  /** when set, renders a close button (overlay mode) */
  onClose?: () => void;
}) {
  const trace = useViz((s) => s.trace);
  const playing = useViz((s) => s.playing);
  const [preset, setPreset] = useState<number | null>(focusChapter);

  /* re-frame if the overlay is reused for a different stage */
  useEffect(() => setPreset(focusChapter), [focusChapter]);

  return (
    <div style={{ flex: 1, position: "relative", minHeight: 0, display: "flex", flexDirection: "column" }}>
      <div className="ov-toolbar">
        <button
          className="playbtn"
          onClick={() => {
            if (trace) togglePlayback(trace.pcm, ARCH.sampleRate);
          }}
        >
          {playing ? "■ stop" : "▶ play"}
        </button>
        <button className="ovbtn" onClick={() => setPreset(null)}>full chain</button>
        {preset != null && (
          <button className="ovbtn" onClick={() => setPreset(focusChapter)}>this stage</button>
        )}
        <span className="ov-hint">drag rotate · wheel zoom · right-drag pan</span>
        <span style={{ flex: 1 }} />
        {onClose && <button className="ovbtn" onClick={onClose}>✕ close overview</button>}
      </div>
      <div
        style={{ flex: 1, position: "relative", minHeight: 0 }}
        /* zoom belongs to the 3D view — don't forward wheel into the story here */
        onWheel={(e) => e.stopPropagation()}
      >
        <GLBoundary>
          <Canvas dpr={[1, 1.5]} camera={{ position: [46, 11, 48], fov: 50 }}>
            <color attach="background" args={["#ffffff"]} />
            <ambientLight intensity={1.15} />
            <directionalLight position={[30, 60, 40]} intensity={0.7} />
            <PipelineModel />
            <OrbitControls
              makeDefault
              enableDamping
              dampingFactor={0.08}
              minDistance={0.5}
              maxDistance={180}
            />
            <CameraFocus preset={preset} />
          </Canvas>
        </GLBoundary>
        <div className="stage-caption" style={{ position: "absolute", bottom: 0, left: 0, right: 0, background: "rgba(255,255,255,0.92)" }}>
          the whole chain at once — text in (left), waveform out (right) · hover
          any tensor · press play and the crimson cursor sweeps the stations
        </div>
      </div>
    </div>
  );
}
