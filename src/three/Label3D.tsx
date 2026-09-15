/**
 * Label3D — canvas-rendered sprite labels in Lucida Console, so the 3D scene
 * carries the same typeface as the DOM and needs no webfont downloads.
 */
import { useMemo } from "react";
import * as THREE from "three";

export function Label3D({
  text,
  position,
  size = 0.4,
  color = "#111111",
  align = "center",
}: {
  text: string;
  position: [number, number, number];
  size?: number;
  color?: string;
  align?: "center" | "left";
}) {
  const sprite = useMemo(() => {
    const pad = 12;
    const font = `${Math.round(44)}px "Lucida Console", Monaco, monospace`;
    const cv = document.createElement("canvas");
    const cx = cv.getContext("2d")!;
    cx.font = font;
    const w = Math.ceil(cx.measureText(text).width) + pad * 2;
    const h = 44 + pad * 2;
    cv.width = w; cv.height = h;
    const c2 = cv.getContext("2d")!;
    c2.font = font;
    c2.fillStyle = color;
    c2.textBaseline = "middle";
    c2.textAlign = align === "center" ? "center" : "left";
    c2.fillText(text, align === "center" ? w / 2 : pad, h / 2);
    const tex = new THREE.CanvasTexture(cv);
    tex.anisotropy = 4;
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
    const sp = new THREE.Sprite(mat);
    const aspect = w / h;
    sp.scale.set(size * aspect, size, 1);
    return sp;
  }, [text, size, color, align]);

  return <primitive object={sprite} position={position} />;
}
