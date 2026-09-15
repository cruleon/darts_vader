import { motion } from "framer-motion";
import { useRef, type MouseEvent } from "react";
import { annularSector, RADII, SECTOR_ORDER } from "../lib/geometry";
import { AMBER, DART_COLORS } from "../lib/theme";
import type { Dart, EngineState, Send, Vec2 } from "../types";
import { BoardFace } from "./BoardFace";

/** Vector dartboard with the darts of the turn, ignored zones and (optionally) raw detections.
 *  Click an empty spot to add an approximate dart; right-click a dart to remove it. */
export function MiniBoard({ state, send }: { state: EngineState; send: Send }) {
  const svgRef = useRef<SVGSVGElement>(null);

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

  const targets = new Set(state.turn.flatMap((d) => (d.finish ? [d.finish.target] : [])));
  if (state.finish_target) targets.add(state.finish_target);

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
        <filter id="mb-glow" x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <BoardFace idPrefix="mb" />

      {[...targets].map((label) => (
        <path
          key={`target-${label}`}
          d={targetPath(label)}
          fill="rgb(251 191 36 / 0.18)"
          stroke={AMBER}
          strokeWidth={2.5}
          className="soft-pulse"
          filter="url(#mb-glow)"
        />
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
            {d.finish && d.finish.distance_mm > 0 && (
              <line
                x1={d.tip_mm[0]}
                y1={d.tip_mm[1]}
                x2={d.finish.point_mm[0]}
                y2={d.finish.point_mm[1]}
                stroke={color}
                strokeWidth={2}
                strokeDasharray="5 4"
                strokeLinecap="round"
              />
            )}
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
            {d.finish && <DistanceLabel dart={d} color={color} />}
          </motion.g>
        );
      })}
    </svg>
  );
}

/** SVG outline of a finishing double ("D16") or of the bullseye ("BULL"). */
function targetPath(label: string): string {
  if (label === "BULL") {
    const r = RADII.bullseye;
    return `M${-r} 0a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0Z`;
  }
  const k = SECTOR_ORDER.indexOf(Number(label.slice(1)) as (typeof SECTOR_ORDER)[number]);
  return annularSector(RADII.doubleIn, RADII.doubleOut, 18 * k - 9, 18 * k + 9);
}

/** Distance of a dart from the finishing double, placed beside the dashed guide line. */
function DistanceLabel({ dart, color }: { dart: Dart; color: string }) {
  const finish = dart.finish!;
  const [tx, ty] = dart.tip_mm;
  const [px, py] = finish.point_mm;
  const length = Math.hypot(px - tx, py - ty);
  const [ux, uy] = length > 0.5 ? [(px - tx) / length, (py - ty) / length] : [0, 1];
  // long guide line: label beside its middle; short one: label on the far side of the dart, off the double
  const [x, y] = length > 30 ? [(tx + px) / 2 - uy * 14, (ty + py) / 2 + ux * 14] : [tx - ux * 26, ty - uy * 26];
  return (
    <text
      x={x}
      y={y}
      textAnchor="middle"
      dominantBaseline="central"
      fontSize={16}
      fontWeight={800}
      fontFamily="Barlow Condensed, Bahnschrift, sans-serif"
      fill={color}
      stroke="#070b18"
      strokeWidth={4}
      paintOrder="stroke"
    >
      {Math.round(finish.distance_mm)} mm
    </text>
  );
}
