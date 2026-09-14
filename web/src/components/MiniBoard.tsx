import { motion } from "framer-motion";
import { useMemo, useRef, type MouseEvent } from "react";
import { annularSector, modelPoint, RADII, SECTOR_ORDER } from "../lib/geometry";
import { DART_COLORS } from "../lib/theme";
import type { EngineState, Send, Vec2 } from "../types";

const BLACK = "#12161f";
const CREAM = "#eadfc2";
const RED = "#e5484d";
const GREEN = "#1fa35c";

/** Vector dartboard with the darts of the turn, ignored zones and (optionally) raw detections.
 *  Click an empty spot to add an approximate dart; right-click a dart to remove it. */
export function MiniBoard({ state, send }: { state: EngineState; send: Send }) {
  const svgRef = useRef<SVGSVGElement>(null);

  const board = useMemo(() => {
    const parts: { d: string; fill: string }[] = [];
    for (let k = 0; k < 20; k++) {
      const a0 = 18 * k - 9;
      const a1 = 18 * k + 9;
      const even = k % 2 === 0;
      const single = even ? BLACK : CREAM;
      const ring = even ? RED : GREEN;
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
  }, []);

  const toMm = (e: MouseEvent): Vec2 | null => {
    const svg = svgRef.current;
    const ctm = svg?.getScreenCTM();
    if (!svg || !ctm) return null;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
    return [p.x, p.y];
  };

  const nearest = (p: Vec2) => {
    let best = -1;
    let bestDist = Infinity;
    for (const d of state.turn) {
      const dist = Math.hypot(d.tip_mm[0] - p[0], d.tip_mm[1] - p[1]);
      if (dist < bestDist) {
        bestDist = dist;
        best = d.index;
      }
    }
    return bestDist <= 14 ? best : -1;
  };

  const interactive = state.mode === "playing" || state.mode === "review";

  const onClick = (e: MouseEvent) => {
    if (!interactive) return;
    const p = toMm(e);
    if (!p || Math.hypot(p[0], p[1]) > RADII.doubleOut + 15 || nearest(p) >= 0) return;
    send({ type: "add_sim", x_mm: p[0], y_mm: p[1] });
  };

  const onContextMenu = (e: MouseEvent) => {
    e.preventDefault();
    if (!interactive) return;
    const p = toMm(e);
    const k = p ? nearest(p) : -1;
    if (k >= 0) send({ type: "remove", index: k });
  };

  return (
    <svg
      ref={svgRef}
      viewBox="-242 -242 484 484"
      className="aspect-square h-full max-h-full max-w-full"
      style={{ cursor: interactive ? "crosshair" : "default" }}
      onClick={onClick}
      onContextMenu={onContextMenu}
    >
      <defs>
        <radialGradient id="mb-halo">
          <stop offset="0.82" stopColor="#22d3ee" stopOpacity="0.28" />
          <stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="mb-face" cx="0.5" cy="0.45" r="0.6">
          <stop offset="0" stopColor="#1a2033" />
          <stop offset="1" stopColor="#090c16" />
        </radialGradient>
        <filter id="mb-glow" x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <circle r={240} fill="url(#mb-halo)" />
      <circle r={228} fill="url(#mb-face)" stroke="rgb(148 163 184 / 0.28)" strokeWidth={1.5} />
      {board.parts.map((p, i) => (
        <path key={i} d={p.d} fill={p.fill} />
      ))}
      <circle r={RADII.bull} fill={GREEN} />
      <circle r={RADII.bullseye} fill={RED} />
      <g stroke="rgb(203 213 225 / 0.45)" strokeWidth={0.8} fill="none">
        {[RADII.bull, RADII.trebleIn, RADII.trebleOut, RADII.doubleIn, RADII.doubleOut].map((r) => (
          <circle key={r} r={r} />
        ))}
        {board.wires.map((w, i) => (
          <line key={i} {...w} />
        ))}
      </g>
      {board.numbers.map(({ n, p }) => (
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

      {state.ignored.map(([x, y], i) => (
        <g key={`ig-${i}`} stroke="#94a3b8" strokeWidth={2.5} strokeLinecap="round" opacity={0.85}>
          <line x1={x - 7} y1={y - 7} x2={x + 7} y2={y + 7} />
          <line x1={x - 7} y1={y + 7} x2={x + 7} y2={y - 7} />
        </g>
      ))}
      {state.show_detections &&
        state.detections.map(([x, y], i) => <circle key={`det-${i}`} cx={x} cy={y} r={3} fill="#fde047" />)}

      {state.turn.map((d) => {
        const color = DART_COLORS[d.index % 3];
        return (
          <motion.g
            key={d.index}
            initial={{ opacity: 0, scale: 2.2 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ type: "spring", stiffness: 300, damping: 18 }}
            style={{ transformOrigin: `${d.tip_mm[0]}px ${d.tip_mm[1]}px`, transformBox: "view-box" }}
          >
            <circle cx={d.tip_mm[0]} cy={d.tip_mm[1]} r={14} fill={color} opacity={0.35} filter="url(#mb-glow)" />
            <circle cx={d.tip_mm[0]} cy={d.tip_mm[1]} r={9} fill={color} stroke="#070b18" strokeWidth={2.5} />
            <text
              x={d.tip_mm[0]}
              y={d.tip_mm[1]}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={11}
              fontWeight={800}
              fill="#070b18"
            >
              {d.index + 1}
            </text>
          </motion.g>
        );
      })}
    </svg>
  );
}
