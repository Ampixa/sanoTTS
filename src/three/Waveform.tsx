/**
 * Waveform — PCM as a mirrored envelope ribbon, with hover sample readout.
 */
import { useMemo } from "react";
import * as THREE from "three";
import type { ThreeEvent } from "@react-three/fiber";
import { useViz } from "../data/store";

interface Props {
  pcm: Float32Array;
  sampleRate: number;
  position?: [number, number, number];
  width?: number;
  height?: number;
}

export function Waveform({ pcm, sampleRate, position = [0, 0, 0], width = 10, height = 1.6 }: Props) {
  const shape = useMemo(() => {
    const nPts = Math.min(1600, pcm.length);
    const block = pcm.length / nPts;
    const s = new THREE.Shape();
    const peak = (i: number, sign: 1 | -1) => {
      let m = 0;
      const a = Math.floor(i * block), b = Math.min(pcm.length, Math.floor((i + 1) * block));
      for (let j = a; j < b; j++) {
        const v = pcm[j];
        if (sign > 0 ? v > m : v < m) m = v;
      }
      return m;
    };
    /* centered like the sibling stations: x spans [-width/2, +width/2] so the
     * Playhead sweep (x - width/2 .. x + width/2) tracks the signal exactly */
    s.moveTo(-width / 2, 0);
    for (let i = 0; i < nPts; i++) s.lineTo((i / nPts) * width - width / 2, (peak(i, 1) * height) / 2);
    for (let i = nPts - 1; i >= 0; i--) s.lineTo((i / nPts) * width - width / 2, (peak(i, -1) * height) / 2);
    s.closePath();
    return s;
  }, [pcm, width, height]);

  const setHover = useViz((s) => s.setHover);
  const onMove = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    /* uv.x of the raycast hit runs 0..1 across the plane — independent of
     * world/station offsets, so hover always maps to the right sample */
    const t = Math.min(0.9999, Math.max(0, e.uv?.x ?? 0));
    const i = Math.floor(t * pcm.length);
    setHover({
      x: e.clientX + 14,
      y: e.clientY + 10,
      lines: [
        "waveform (24 kHz PCM)",
        `sample ${i.toLocaleString()} · ${(i / sampleRate).toFixed(3)} s`,
        pcm[i]?.toFixed(4) ?? "",
      ],
    });
  };

  return (
    <group position={position}>
      <mesh position={[0, 0, 0]}>
        <shapeGeometry args={[shape]} />
        <meshBasicMaterial color="#DC143C" transparent opacity={0.9} toneMapped={false} />
      </mesh>
      {/* invisible fat hover target */}
      <mesh position={[0, 0, -0.01]} onPointerMove={onMove} onPointerOut={() => setHover(null)}>
        <planeGeometry args={[width, height]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>
      {/* zero line */}
      <mesh position={[0, 0, -0.02]}>
        <planeGeometry args={[width, 0.012]} />
        <meshBasicMaterial color="#cccccc" toneMapped={false} />
      </mesh>
    </group>
  );
}
