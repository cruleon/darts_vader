import { motion } from "framer-motion";
import type { MouseEvent } from "react";
import { useElementSize } from "../lib/hooks";
import { ORIGIN_STYLE, DART_COLORS } from "../lib/theme";
import type { Dart, EngineState, Send } from "../types";
import { AnimatedNumber, Kbd } from "./ui";

const TILE_HALF = 70; // half side, in photo pixels, of each zoomed tile

const REASONS: Record<string, string> = {
  three_darts: "three darts thrown",
  darts_pulled: "darts pulled",
  manual: "opened manually",
};

export function ReviewPanel({ state, send }: { state: EngineState; send: Send }) {
  const review = state.review;
  if (!review) return null;
  const game = state.game;
  const player = game.players[game.current];

  return (
    <motion.div
      className="glass glass-review flex h-full min-h-0 flex-col px-[1.6rem] pb-[1.35rem] pt-[1.25rem]"
      initial={{ opacity: 0, x: 40 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ type: "spring", stiffness: 260, damping: 28 }}
    >
      <div>
        <div className="label !text-amber-300">Review · {REASONS[review.reason] ?? review.reason}</div>
        <h2 className="mt-[0.3rem] font-display text-[3.1rem] font-bold leading-none text-white">Confirm the turn</h2>
        <p className="mt-[0.35rem] text-[1rem] text-slate-400">
          <span className="font-semibold text-slate-200">{player?.name}</span> · click a tile to move the tip, right-click to remove it
        </p>
      </div>

      <div className="mt-[0.9rem] flex min-h-0 flex-1 flex-col justify-center gap-[0.7rem] overflow-hidden">
        {[0, 1, 2].map((k) => (
          <ReviewRow key={k} k={k} dart={state.turn[k]} review={review} send={send} />
        ))}
      </div>

      <div className="mt-[0.9rem] flex items-end justify-between border-t border-white/10 pt-[0.9rem]">
        <div>
          <div className="label">Total</div>
          <AnimatedNumber value={state.turn_total} className="font-display text-[3.6rem] font-bold leading-none text-white" />
        </div>
        <div className="text-right">
          <div className={`label ${state.bust ? "!text-rose-300" : ""}`}>{state.bust ? "Bust" : "Left"}</div>
          <div className={`font-display text-[3.6rem] font-bold leading-none ${state.bust ? "text-rose-400" : "text-emerald-300"}`}>
            {state.bust ? player?.score : state.remaining_after}
          </div>
        </div>
      </div>

      <div className="mt-[0.9rem] grid grid-cols-3 gap-[0.6rem]">
        <button type="button" className="btn btn-primary col-span-3" onClick={() => send({ type: "confirm", save: true })}>
          <Kbd>ENTER</Kbd> Confirm &amp; save
        </button>
        <button type="button" className="btn btn-secondary !h-[3.1rem] !gap-[0.5rem] !text-[0.98rem]" onClick={() => send({ type: "confirm", save: false })}>
          <Kbd>K</Kbd> Score only
        </button>
        <button type="button" className="btn btn-ghost !h-[3.1rem] !gap-[0.5rem] !text-[0.98rem]" onClick={() => send({ type: "undo" })}>
          <Kbd>U</Kbd> Undo
        </button>
        <button type="button" className="btn btn-ghost !h-[3.1rem] !gap-[0.5rem] !text-[0.98rem]" onClick={() => send({ type: "resume" })}>
          <Kbd>ESC</Kbd> Back
        </button>
      </div>
    </motion.div>
  );
}

interface RowProps {
  k: number;
  dart: Dart | undefined;
  review: { id: number; w: number; h: number };
  send: Send;
}

function ReviewRow({ k, dart, review, send }: RowProps) {
  const color = DART_COLORS[k];
  if (!dart || !dart.tip_img) {
    return (
      <div className="flex items-center gap-[1.2rem]">
        <div className="grid aspect-square h-[8.4rem] place-items-center rounded-[1.1rem] border-2 border-dashed border-slate-500/30 bg-white/[0.02]">
          <span className="font-display text-[3.4rem] text-slate-600">+</span>
        </div>
        <div>
          <div className="label" style={{ color: "#64748b" }}>
            Dart {k + 1}
          </div>
          <div className="mt-[0.2rem] text-[1.7rem] font-bold text-slate-500">None</div>
          <div className="text-[0.95rem] text-slate-500">Click the photo to add it</div>
        </div>
      </div>
    );
  }
  const origin = ORIGIN_STYLE[dart.origin] ?? { label: dart.origin, color: "#94a3b8" };
  return (
    <motion.div
      className="flex items-center gap-[1.2rem]"
      initial={{ opacity: 0, x: 24 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: k * 0.07, type: "spring", stiffness: 300, damping: 26 }}
    >
      <Tile dart={dart} review={review} send={send} color={color} />
      <div className="min-w-0">
        <div className="label" style={{ color }}>
          Dart {k + 1}
        </div>
        <div className="font-display text-[3.6rem] font-bold leading-[0.95] text-white">{dart.label}</div>
        <div className="flex items-center gap-[0.7rem]">
          <span className="text-[1.05rem] font-semibold text-slate-400">
            {dart.score} {dart.score === 1 ? "point" : "points"}
          </span>
          <span
            className="rounded-full px-[0.7rem] py-[0.1rem] text-[0.82rem] font-semibold"
            style={{ color: origin.color, background: `${origin.color}1f`, border: `1px solid ${origin.color}99` }}
          >
            {origin.label}
          </span>
        </div>
      </div>
    </motion.div>
  );
}

function Tile({ dart, review, send, color }: { dart: Dart; review: { id: number; w: number; h: number }; send: Send; color: string }) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const [tx, ty] = dart.tip_img!;
  const zoom = size.width > 0 ? size.width / (2 * TILE_HALF) : 1;

  const onClick = (e: MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const z = rect.width / (2 * TILE_HALF);
    send({ type: "move", index: dart.index, x: tx - TILE_HALF + (e.clientX - rect.left) / z, y: ty - TILE_HALF + (e.clientY - rect.top) / z });
  };

  return (
    <div
      ref={ref}
      className="group relative aspect-square h-[8.4rem] shrink-0 cursor-crosshair overflow-hidden rounded-[1.1rem] transition-shadow"
      style={{
        backgroundColor: "#000",
        backgroundImage: `url(/api/review.jpg?id=${review.id})`,
        backgroundRepeat: "no-repeat",
        backgroundSize: `${review.w * zoom}px ${review.h * zoom}px`,
        backgroundPosition: `${-(tx - TILE_HALF) * zoom}px ${-(ty - TILE_HALF) * zoom}px`,
        boxShadow: `0 0 0 2px ${color}, 0 0 1.6rem -0.4rem ${color}`,
      }}
      onClick={onClick}
      onContextMenu={(e) => {
        e.preventDefault();
        send({ type: "remove", index: dart.index });
      }}
    >
      <svg viewBox="-50 -50 100 100" className="pointer-events-none absolute inset-0 h-full w-full">
        <circle r={8} fill="none" stroke={color} strokeWidth={2} />
        <circle r={1.6} fill="#fff" />
        {[
          [0, -22, 0, -12],
          [0, 12, 0, 22],
          [-22, 0, -12, 0],
          [12, 0, 22, 0],
        ].map(([x1, y1, x2, y2], i) => (
          <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={2} strokeLinecap="round" />
        ))}
      </svg>
      <div className="pointer-events-none absolute inset-0 rounded-[1.1rem] opacity-0 ring-2 ring-white/70 transition-opacity group-hover:opacity-100" />
    </div>
  );
}
