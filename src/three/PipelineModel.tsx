/**
 * PipelineModel — the whole E13b pipeline as a line of 3D stations, every one
 * fed by the selected sentence's real engine trace:
 *
 *   ids → durations → tok_hidden [31×N] → frame_hidden [31×T] → mel [100×T]
 *       → noise ⊕ stem → trunk ×4 [62×T] → spectrum [513×T]×2 → PCM
 */
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import type { ThreeEvent } from "@react-three/fiber";
import { useViz } from "../data/store";
import { TensorGrid } from "./TensorGrid";
import { Heatmap } from "./Heatmap";
import { Waveform } from "./Waveform";
import { Label3D } from "./Label3D";
import { rampSigned } from "./ramp";
import { ARCH } from "../data/paperFacts";

export const STAGE_X = {
  ids: 0,
  durs: 12,
  tok: 24,
  frame: 36,
  mel: 50,
  dec: 64,
  spec: 78,
  wave: 92,
} as const;

const W = 10; // world width of frame-rate stations

/* ---------------------------------------------------------------- ids --- */
function IdsStrip() {
  const trace = useViz((s) => s.trace);
  const vocab = useViz((s) => s.vocab);
  const setHover = useViz((s) => s.setHover);
  const mesh = useRef<THREE.InstancedMesh>(null);
  const N = trace?.N ?? 0;

  const data = useMemo(() => (trace ? Float32Array.from(trace.ids) : null), [trace]);

  useFrame(() => {
    const im = mesh.current;
    if (!im || !trace || !data) return;
    if (im.userData.row === trace.info.row_id) return;
    im.userData.row = trace.info.row_id;
    const m = new THREE.Matrix4();
    const c = new THREE.Color();
    const cell = Math.min(0.16, 9 / N);
    for (let i = 0; i < N; i++) {
      m.makeTranslation(i * cell - (N * cell) / 2, 0, 0);
      im.setMatrixAt(i, m);
      const rgb = rampSigned((trace.ids[i] - 31) / 31);
      c.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
      im.setColorAt(i, c);
    }
    im.count = N;
    im.instanceMatrix.needsUpdate = true;
    if (im.instanceColor) im.instanceColor.needsUpdate = true;
    im.computeBoundingSphere();
  });

  if (!trace) return null;
  return (
    <instancedMesh
      ref={mesh}
      args={[undefined, undefined, 220]}
      onPointerMove={(e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation();
        const i = e.instanceId ?? 0;
        setHover({
          x: e.clientX + 14,
          y: e.clientY + 10,
          lines: [
            "phoneme ids",
            `token ${i + 1}/${N} · id ${trace.ids[i]}`,
            JSON.stringify(vocab[trace.ids[i]] ?? "?"),
          ],
        });
      }}
      onPointerOut={() => setHover(null)}
    >
      <boxGeometry args={[0.12, 0.5, 0.12]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

/* --------------------------------------------------------------- durs --- */
function DurationBars() {
  const trace = useViz((s) => s.trace);
  const vocab = useViz((s) => s.vocab);
  const setHover = useViz((s) => s.setHover);
  const mesh = useRef<THREE.InstancedMesh>(null);
  const N = trace?.N ?? 0;

  useFrame(() => {
    const im = mesh.current;
    if (!im || !trace) return;
    if (im.userData.row === trace.info.row_id) return;
    im.userData.row = trace.info.row_id;
    const m = new THREE.Matrix4();
    const c = new THREE.Color();
    const cell = 9 / N;
    for (let i = 0; i < N; i++) {
      const h = (trace.durs[i] / ARCH.dur.maxDuration) * 4 + 0.05;
      m.makeScale(1, h, 1);
      m.setPosition(i * cell - (N * cell) / 2, h / 2 - 1.6, 0);
      im.setMatrixAt(i, m);
      c.set("#DC143C");
      im.setColorAt(i, c);
    }
    im.count = N;
    im.instanceMatrix.needsUpdate = true;
    if (im.instanceColor) im.instanceColor.needsUpdate = true;
    im.computeBoundingSphere();
  });

  if (!trace) return null;
  return (
    <instancedMesh
      ref={mesh}
      args={[undefined, undefined, 220]}
      onPointerMove={(e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation();
        const i = e.instanceId ?? 0;
        setHover({
          x: e.clientX + 14,
          y: e.clientY + 10,
          lines: [
            "duration student output",
            `token ${i + 1}/${N} · ${JSON.stringify(vocab[trace.ids[i]] ?? "?")}`,
            `${trace.durs[i]} frames (${((trace.durs[i] * 256) / 24).toFixed(0)} ms)`,
          ],
        });
      }}
      onPointerOut={() => setHover(null)}
    >
      <boxGeometry args={[0.08, 1, 0.08]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

/* -------------------------------------------------------- flow links ---- */
function FlowLink({ from, to }: { from: number; to: number }) {
  const a = useRef<THREE.Mesh>(null);
  const b = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    const t = (clock.elapsedTime * 0.5) % 1;
    if (a.current) a.current.position.x = from + (to - from - 2) * t + 1;
    if (b.current) b.current.position.x = from + (to - from - 2) * ((t + 0.5) % 1) + 1;
  });
  const y = -2.6;
  return (
    <group>
      <mesh position={[(from + to) / 2, y, 0]}>
        <planeGeometry args={[to - from - 2, 0.03]} />
        <meshBasicMaterial color="#bbbbbb" toneMapped={false} />
      </mesh>
      <mesh ref={a} position={[from, y, 0]}>
        <sphereGeometry args={[0.09, 10, 10]} />
        <meshBasicMaterial color="#DC143C" toneMapped={false} />
      </mesh>
      <mesh ref={b} position={[from, y, 0]}>
        <sphereGeometry args={[0.09, 10, 10]} />
        <meshBasicMaterial color="#DC143C" toneMapped={false} />
      </mesh>
    </group>
  );
}

/* ------------------------------------------------------------ playhead -- */
function Playhead({ x, width, height }: { x: number; width: number; height: number }) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(() => {
    if (!ref.current) return;
    const { playhead, playing } = useViz.getState();
    ref.current.visible = playing || playhead > 0;
    ref.current.position.x = x - width / 2 + playhead * width;
  });
  return (
    <mesh ref={ref} position={[x, 0, 0.4]}>
      <planeGeometry args={[0.1, height]} />
      <meshBasicMaterial color="#DC143C" transparent opacity={0.95} depthTest={false} toneMapped={false} />
    </mesh>
  );
}

/* ------------------------------------------------------------- stations - */
export function PipelineModel() {
  const trace = useViz((s) => s.trace);
  const N = trace?.N ?? 0;
  const T = trace?.T ?? 0;

  if (!trace) return null;

  return (
    <group position={[0, 0, 0]}>
      {/* 01 frontend output: phoneme ids */}
      <group position={[STAGE_X.ids, 0, 0]}>
        <IdsStrip />
        <Label3D text={`phoneme ids  [${N}]`} position={[0, 1.2, 0]} size={0.5} />
        <Label3D text="rule-based · 0 params" position={[0, 0.75, 0]} size={0.3} color="#555555" />
      </group>

      {/* 02 durations */}
      <group position={[STAGE_X.durs, 0, 0]}>
        <DurationBars />
        <mesh position={[0, -1.6, 0]}>
          <planeGeometry args={[9.6, 0.03]} />
          <meshBasicMaterial color="#111111" toneMapped={false} />
        </mesh>
        <Label3D text={`durations  [${N}]`} position={[0, 1.6, 0]} size={0.5} />
        <Label3D text="22,858 params · w26 ×3 blocks" position={[0, 1.1, 0]} size={0.3} color="#555555" />
      </group>

      {/* 03 acoustic token-rate */}
      <group position={[STAGE_X.tok, 0, 0]}>
        <TensorGrid data={trace.tokHidden} rows={31} cols={N} label="acoustic · token-rate" width={Math.min(9, N * 0.09)} />
        <Label3D text={`token-rate  [31×${N}]`} position={[0, 2.2, 0]} size={0.5} />
        <Label3D text="65,299 params · w31 · 3 blocks" position={[0, 1.7, 0]} size={0.3} color="#555555" />
      </group>

      {/* length regulator + frame-rate */}
      <group position={[STAGE_X.frame, 0, 0]}>
        <TensorGrid data={trace.frameHidden} rows={31} cols={T} label="acoustic · frame-rate" width={W} maxCells={31000} />
        <Label3D text={`frame-rate  [31×${T}]`} position={[0, 2.2, 0]} size={0.5} />
        <Label3D text="×3 blocks at frame rate" position={[0, 1.7, 0]} size={0.3} color="#555555" />
      </group>

      {/* 04 mel */}
      <group position={[STAGE_X.mel, 0, 0]}>
        <Heatmap data={trace.mel} rows={100} cols={T} label="mel spectrogram" width={W} rowLabel="mel" />
        <Label3D text={`mel  [100×${T}]`} position={[0, 2.4, 0]} size={0.5} />
        <Label3D text="the 100-band contract" position={[0, 1.9, 0]} size={0.3} color="#555555" />
        <Playhead x={0} width={W} height={4.4} />
      </group>

      {/* 05 decoder: noise + stem + 4 trunk blocks stacked vertically */}
      <group position={[STAGE_X.dec, 0, 0]}>
        <TensorGrid data={trace.stem} rows={62} cols={T} label="decoder stem (embed+noise+LN)" width={W} position={[0, 2.6, 0]} maxCells={32000} />
        {[0, 1, 2, 3].map((b) => (
          <TensorGrid
            key={b}
            data={trace.trunk[b]}
            rows={62}
            cols={T}
            label={`ConvNeXt block ${b + 1}/4 output`}
            width={W}
            position={[0, 1.3 - b * 1.3, 0]}
            maxCells={32000}
          />
        ))}
        <TensorGrid data={trace.noise} rows={4} cols={T} label="seeded Gaussian noise" width={W} position={[0, -3.0, 0]} rowLabel="noise ch" />
        <Label3D text={`decoder  [62×${T}] stem + 4 ConvNeXt blocks`} position={[0, 4.2, 0]} size={0.5} />
        <Label3D text="206,122 params · dim 62 · pw 186 · noise below" position={[0, 3.7, 0]} size={0.3} color="#555555" />
        <Playhead x={0} width={W} height={7.6} />
      </group>

      {/* 06 spectrum */}
      <group position={[STAGE_X.spec, 0, 0]}>
        <Heatmap data={trace.specLogmag} rows={513} cols={T} label="513 log-magnitudes" width={W} position={[0, 1.2, 0]} valueFmt={(v) => `${v.toFixed(2)} ln·mag`} />
        <Heatmap data={trace.specPhase} rows={513} cols={T} label="513 phases" width={W} mode="phase" position={[0, -3.2, 0]} valueFmt={(v) => `${v.toFixed(2)} rad`} />
        <Label3D text={`spectrum  [513+513×${T}]`} position={[0, 4.4, 0]} size={0.5} />
        <Label3D text="log-magnitude · phase" position={[0, 3.9, 0]} size={0.3} color="#555555" />
        <Playhead x={0} width={W} height={8.6} />
      </group>

      {/* 07 waveform */}
      <group position={[STAGE_X.wave, 0, 0]}>
        <Waveform pcm={trace.pcm} sampleRate={24000} width={W} />
        <Label3D text={`waveform  [${trace.pcm.length.toLocaleString()} samples]`} position={[0, 1.7, 0]} size={0.5} />
        <Label3D text="iSTFT · Hann OLA · DC block · 24 kHz" position={[0, 1.25, 0]} size={0.3} color="#555555" />
        <Playhead x={0} width={W} height={2.4} />
      </group>

      {/* flow links */}
      <FlowLink from={STAGE_X.ids + 4.5} to={STAGE_X.durs - 4.5} />
      <FlowLink from={STAGE_X.durs + 4.5} to={STAGE_X.tok - 4.5} />
      <FlowLink from={STAGE_X.tok + 4.5} to={STAGE_X.frame - 4.5} />
      <FlowLink from={STAGE_X.frame + 5} to={STAGE_X.mel - 5} />
      <FlowLink from={STAGE_X.mel + 5} to={STAGE_X.dec - 5} />
      <FlowLink from={STAGE_X.dec + 5} to={STAGE_X.spec - 5} />
      <FlowLink from={STAGE_X.spec + 5} to={STAGE_X.wave - 5} />
    </group>
  );
}
