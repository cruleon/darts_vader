import type { Mat3, Vec2 } from "../types";

/** Standard dartboard geometry (mm), mirroring darts_vader/board/geometry.py.
 *  Model coordinates: origin at the centre, y pointing down, angles clockwise from the top. */
export const SECTOR_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5] as const;

export const RADII = {
  bullseye: 6.35,
  bull: 15.9,
  trebleIn: 99,
  trebleOut: 107,
  doubleIn: 162,
  doubleOut: 170,
  board: 225,
} as const;

export function modelPoint(r: number, deg: number): Vec2 {
  const a = (deg * Math.PI) / 180;
  return [r * Math.sin(a), -r * Math.cos(a)];
}

/** Apply a 3x3 homography to a point. */
export function applyH(H: Mat3, [x, y]: Vec2): Vec2 {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w];
}

/** Annular sector between radii r1 < r2 and angles a0 < a1 (degrees), as an SVG path. */
export function annularSector(r1: number, r2: number, a0: number, a1: number): string {
  const [x1, y1] = modelPoint(r2, a0);
  const [x2, y2] = modelPoint(r2, a1);
  const [x3, y3] = modelPoint(r1, a1);
  const [x4, y4] = modelPoint(r1, a0);
  const f = (v: number) => v.toFixed(2);
  return `M${f(x1)} ${f(y1)}A${r2} ${r2} 0 0 1 ${f(x2)} ${f(y2)}L${f(x3)} ${f(y3)}A${r1} ${r1} 0 0 0 ${f(x4)} ${f(y4)}Z`;
}

export function remPx(): number {
  return parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(Math.max(v, lo), Math.max(lo, hi));
}
