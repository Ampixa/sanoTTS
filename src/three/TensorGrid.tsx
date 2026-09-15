/**
 * TensorGrid — a [rows × cols] tensor rendered as instanced cells, colored by
 * value, with raycast hover that reports channel / frame / dequantized value.
 * Columns can be stride-downsampled for dense frame tensors.
 */
import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { ThreeEvent } from "@react-three/fiber";
import { minmax, rampHot, rampSigned } from "./ramp";
import { useViz } from "../data/store";

interface Props {
  data: Float32Array;
  rows: number;
  cols: number;
  label: string;
  position?: [number, number, number];
  width?: number;         // world width of the whole grid
  maxCells?: number;      // cap rows*cols by column stride
  signed?: boolean;       // diverging ink/white/crimson vs white→crimson
  valueFmt?: (v: number) => string;
  rowLabel?: string;      // axis name for rows (default "ch")
  colLabel?: string;      // axis name for cols (default "frame")
}

export function TensorGrid({
  data, rows, cols, label,
  position = [0, 0, 0],
  width = 10,
  maxCells = 60000,
  signed = true,
  valueFmt = (v) => v.toFixed(3),
  rowLabel = "ch",
  colLabel = "frame",
}: Props) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const stride = Math.max(1, Math.ceil((rows * cols) / maxCells));
  const dispCols = Math.ceil(cols / stride);
  const count = rows * dispCols;
  const cell = width / dispCols;

  const { lo, hi, absmax } = useMemo(() => {
    const [lo, hi] = minmax(data);
    return { lo, hi, absmax: Math.max(Math.abs(lo), Math.abs(hi)) || 1e-9 };
  }, [data]);

  useLayoutEffect(() => {
    const im = mesh.current;
    if (!im) return;
    const m = new THREE.Matrix4();
    const h = rows * cell;
    let i = 0;
    for (let r = 0; r < rows; r++) {
      for (let dc = 0; dc < dispCols; dc++, i++) {
        m.makeTranslation(dc * cell - width / 2 + cell / 2, h / 2 - r * cell - cell / 2, 0);
        im.setMatrixAt(i, m);
      }
    }
    im.instanceMatrix.needsUpdate = true;
    im.computeBoundingSphere();
  }, [rows, dispCols, cell, width, data]);

  useLayoutEffect(() => {
    const im = mesh.current;
    if (!im) return;
    const c3 = new THREE.Color();
    let i = 0;
    for (let r = 0; r < rows; r++) {
      for (let dc = 0; dc < dispCols; dc++, i++) {
        const v = data[r * cols + dc * stride];
        const rgb = signed ? rampSigned(v / absmax) : rampHot((v - lo) / (hi - lo));
        c3.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
        im.setColorAt(i, c3);
      }
    }
    if (im.instanceColor) im.instanceColor.needsUpdate = true;
  }, [data, rows, cols, dispCols, stride, lo, hi, absmax, signed]);

  const setHover = useViz((s) => s.setHover);

  const onMove = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    const id = e.instanceId ?? 0;
    const r = Math.floor(id / dispCols);
    const c = (id % dispCols) * stride;
    const v = data[r * cols + c];
    setHover({
      x: e.clientX + 14,
      y: e.clientY + 10,
      lines: [label, `${rowLabel} ${r + 1}/${rows} · ${colLabel} ${c + 1}/${cols}`, valueFmt(v)],
    });
  };

  return (
    <instancedMesh
      ref={mesh}
      args={[undefined, undefined, count]}
      position={position}
      onPointerMove={onMove}
      onPointerOut={() => setHover(null)}
    >
      <boxGeometry args={[cell * 0.92, cell * 0.92, cell * 0.3]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}
