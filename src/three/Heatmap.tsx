/**
 * Heatmap — a [rows × cols] tensor on a DataTexture plane (mel, spectrum).
 * Hover maps the UV back to a cell and reports the dequantized value.
 */
import { useMemo } from "react";
import * as THREE from "three";
import type { ThreeEvent } from "@react-three/fiber";
import { minmax, rampHot, rampPhase, rampSigned } from "./ramp";
import { useViz } from "../data/store";

interface Props {
  data: Float32Array;
  rows: number;
  cols: number;
  label: string;
  position?: [number, number, number];
  width?: number;         // world width (default keeps square pixels)
  mode?: "hot" | "signed" | "phase";
  valueFmt?: (v: number) => string;
  rowLabel?: string;
  colLabel?: string;
  maxTexCols?: number;    // downsample columns for texture size
}

export function Heatmap({
  data, rows, cols, label,
  position = [0, 0, 0],
  width,
  mode = "hot",
  valueFmt = (v) => v.toFixed(3),
  rowLabel = "bin",
  colLabel = "frame",
  maxTexCols = 1024,
}: Props) {
  const stride = Math.max(1, Math.ceil(cols / maxTexCols));
  const tcols = Math.ceil(cols / stride);

  const { texture, lo, hi, absmax } = useMemo(() => {
    const [lo, hi] = minmax(data);
    const absmax = Math.max(Math.abs(lo), Math.abs(hi)) || 1e-9;
    const px = new Uint8Array(rows * tcols * 4);
    let i = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < tcols; c++, i++) {
        const v = data[r * cols + c * stride];
        let rgb: [number, number, number];
        if (mode === "phase") rgb = rampPhase(v);
        else if (mode === "signed") rgb = rampSigned(v / absmax);
        else rgb = rampHot((v - lo) / (hi - lo));
        px[i * 4] = rgb[0]; px[i * 4 + 1] = rgb[1]; px[i * 4 + 2] = rgb[2]; px[i * 4 + 3] = 255;
      }
    }
    const tex = new THREE.DataTexture(px, tcols, rows, THREE.RGBAFormat);
    tex.needsUpdate = true;
    tex.magFilter = THREE.NearestFilter;
    tex.minFilter = THREE.NearestFilter;
    return { texture: tex, lo, hi, absmax };
  }, [data, rows, cols, tcols, stride, mode]);

  const w = width ?? cols * 0.012;
  const h = (w * rows) / cols;

  const setHover = useViz((s) => s.setHover);
  const onMove = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    const uv = e.uv;
    if (!uv) return;
    const c = Math.min(cols - 1, Math.floor(uv.x * cols));
    const r = Math.min(rows - 1, Math.floor((1 - uv.y) * rows));
    const v = data[r * cols + c];
    setHover({
      x: e.clientX + 14,
      y: e.clientY + 10,
      lines: [label, `${rowLabel} ${r + 1}/${rows} · ${colLabel} ${c + 1}/${cols}`, valueFmt(v)],
    });
  };

  return (
    <mesh position={position} onPointerMove={onMove} onPointerOut={() => setHover(null)}>
      <planeGeometry args={[w, h]} />
      <meshBasicMaterial map={texture} toneMapped={false} />
    </mesh>
  );
}
