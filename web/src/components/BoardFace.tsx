import { annularSector, modelPoint, RADII, SECTOR_ORDER } from "../lib/geometry";

export const BOARD_BLACK = "#12161f";
export const BOARD_CREAM = "#eadfc2";
export const BOARD_RED = "#e5484d";
export const BOARD_GREEN = "#1fa35c";

const FACE = (() => {
  const parts: { d: string; fill: string }[] = [];
  for (let k = 0; k < 20; k++) {
    const a0 = 18 * k - 9;
    const a1 = 18 * k + 9;
    const even = k % 2 === 0;
    const single = even ? BOARD_BLACK : BOARD_CREAM;
    const ring = even ? BOARD_RED : BOARD_GREEN;
    parts.push({ d: annularSector(RADII.bull, RADII.trebleIn, a0, a1), fill: single });
    parts.push({ d: annularSector(RADII.trebleIn, RADII.trebleOut, a0, a1), fill: ring });
    parts.push({ d: annularSector(RADII.trebleOut, RADII.doubleIn, a0, a1), fill: single });
    parts.push({ d: annularSector(RADII.doubleIn, RADII.doubleOut, a0, a1), fill: ring });
  }
  const wires = Array.from({ length: 20 }, (_, k) => {
    const [x1, y1] = modelPoint(RADII.bull, 9 + 18 * k);
    const [x2, y2] = modelPoint(RADII.doubleOut, 9 + 18 * k);
    return { x1, y1, x2, y2 };
  });
  const numbers = SECTOR_ORDER.map((n, k) => ({ n, p: modelPoint(197, 18 * k) }));
  return { parts, wires, numbers };
})();

/** Vector dartboard in model millimetres (use inside an SVG with viewBox "-242 -242 484 484").
 *  `idPrefix` keeps gradient ids unique when several boards are on screen; `dim` tones the board
 *  down so that markers drawn on top stand out. */
export function BoardFace({ idPrefix, dim = false }: { idPrefix: string; dim?: boolean }) {
  return (
    <g opacity={dim ? 0.6 : 1}>
      <defs>
        <radialGradient id={`${idPrefix}-halo`}>
          <stop offset="0.82" stopColor="#22d3ee" stopOpacity="0.28" />
          <stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
        </radialGradient>
        <radialGradient id={`${idPrefix}-face`} cx="0.5" cy="0.45" r="0.6">
          <stop offset="0" stopColor="#1a2033" />
          <stop offset="1" stopColor="#090c16" />
        </radialGradient>
      </defs>
      {!dim && <circle r={240} fill={`url(#${idPrefix}-halo)`} />}
      <circle r={228} fill={`url(#${idPrefix}-face)`} stroke="rgb(148 163 184 / 0.28)" strokeWidth={1.5} />
      {FACE.parts.map((p, i) => (
        <path key={i} d={p.d} fill={p.fill} />
      ))}
      <circle r={RADII.bull} fill={BOARD_GREEN} />
      <circle r={RADII.bullseye} fill={BOARD_RED} />
      <g stroke="rgb(203 213 225 / 0.45)" strokeWidth={0.8} fill="none">
        {[RADII.bull, RADII.trebleIn, RADII.trebleOut, RADII.doubleIn, RADII.doubleOut].map((r) => (
          <circle key={r} r={r} />
        ))}
        {FACE.wires.map((w, i) => (
          <line key={i} {...w} />
        ))}
      </g>
      {FACE.numbers.map(({ n, p }) => (
        <text
          key={n}
          x={p[0]}
          y={p[1]}
          textAnchor="middle"
          dominantBaseline="central"
          fontSize={19}
          fontWeight={700}
          fontFamily="Barlow Condensed, Bahnschrift, sans-serif"
          fill={n === 20 ? "#f9a8d4" : "#cbd5e1"}
        >
          {n}
        </text>
      ))}
    </g>
  );
}
