/**
 * ramp.ts — colormaps in the ampixa/sanoTTS palette. White is silence,
 * crimson is energy; ink for signed negative, crimson for signed positive.
 */
export const INK = "#111111";
export const ACCENT = "#DC143C";

export function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }

function hexRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

const WHITE: [number, number, number] = [255, 255, 255];
const CRIM = hexRgb(ACCENT);
const INKR = hexRgb(INK);

/** 0..1 → white..crimson; gamma lifts the midrange so quiet structure shows */
export function rampHot(t: number): [number, number, number] {
  const c = Math.pow(Math.min(1, Math.max(0, t)), 0.6);
  return [lerp(WHITE[0], CRIM[0], c), lerp(WHITE[1], CRIM[1], c), lerp(WHITE[2], CRIM[2], c)];
}

/** signed -1..1 → ink..white..crimson */
export function rampSigned(t: number): [number, number, number] {
  const c = Math.min(1, Math.max(-1, t));
  if (c >= 0) return rampHot(c);
  const m = -c;
  return [lerp(WHITE[0], INKR[0], m), lerp(WHITE[1], INKR[1], m), lerp(WHITE[2], INKR[2], m)];
}

/** cyclic phase -π..π → crimson..white..ink (continuous at the wrap) */
export function rampPhase(t: number): [number, number, number] {
  const c = (Math.cos(t) + 1) / 2; // 1 at 0, 0 at ±π
  return [lerp(40, CRIM[0], c), lerp(40, CRIM[1], c), lerp(60, CRIM[2], c)];
}

export function minmax(data: ArrayLike<number>): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!(hi > lo)) { hi = lo + 1e-9; }
  return [lo, hi];
}
