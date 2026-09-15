import { AnimatePresence, motion } from "framer-motion";
import { playerColor } from "../lib/players";
import { colorVar, DART_COLORS } from "../lib/theme";
import type { EngineState, Send } from "../types";
import { MiniBoard } from "./MiniBoard";
import { Avatar } from "./PlayerPhoto";
import { AnimatedNumber } from "./ui";

export function GamePanel({ state, send }: { state: EngineState; send: Send }) {
  return (
    <div className="flex h-full min-h-0 flex-col gap-[0.95rem]">
      <Players state={state} />
      <TurnCard state={state} />
      <div className="glass flex min-h-0 flex-1 items-center justify-center p-[0.9rem]">
        <MiniBoard state={state} send={send} />
        <FinishLegend state={state} />
      </div>
    </div>
  );
}

function Players({ state }: { state: EngineState }) {
  const { game } = state;
  const count = game.players.length;
  const compact = count > 3;
  const legs = game.legs_to_win > 1;
  return (
    <div className="flex flex-col gap-[0.65rem]">
      {game.players.map((p, i) => {
        const active = i === game.current;
        const color = playerColor(i, count);
        return (
          <motion.div
            layout
            key={`${i}-${p.name}`}
            transition={{ type: "spring", stiffness: 320, damping: 32 }}
            className={active ? "glass glass-active overflow-hidden px-[1.6rem] pb-[1rem] pt-[0.95rem]" : "glass flex items-center justify-between px-[1.6rem] py-[0.55rem]"}
          >
            {active ? (
              <>
                <div className="flex items-center justify-between gap-[1rem]">
                  <span className="flex min-w-0 items-center gap-[0.8rem] text-[1.05rem] font-bold uppercase tracking-[0.2em] text-cyan-300">
                    <Avatar name={p.name} photo={p.photo} color={color} className="h-[2.8rem] w-[2.8rem] text-[1.05rem]" />
                    <span className="truncate">{p.name}</span>
                  </span>
                  <span className="shrink-0 text-[0.9rem] font-semibold tracking-wide text-slate-400">
                    LEGS {p.legs}
                    {legs ? `/${game.legs_to_win}` : ""} · AVG {p.average.toFixed(1)}
                  </span>
                </div>
                <div className="mt-[0.1rem] flex items-end justify-between">
                  <AnimatedNumber
                    value={state.bust ? p.score : state.remaining_after}
                    className={`font-display font-bold leading-[0.86] tracking-tight ${state.bust ? "text-rose-400" : "text-white"} ${compact ? "text-[5.6rem]" : "text-[7.4rem]"}`}
                  />
                  <div className="flex flex-col items-end gap-[0.5rem] pb-[0.55rem] text-right">
                    <div>
                      <div className="label">Turn start</div>
                      <div className="font-display text-[3.3rem] font-bold leading-none text-slate-400">{p.score}</div>
                    </div>
                    <DartsLeft thrown={state.turn.length} color={color} />
                  </div>
                </div>
              </>
            ) : (
              <>
                <span className="flex min-w-0 items-center gap-[0.7rem] text-[1.2rem] font-semibold text-slate-300">
                  <Avatar name={p.name} photo={p.photo} color={color} className="h-[2.2rem] w-[2.2rem] text-[0.85rem]" />
                  <span className="truncate">{p.name}</span>
                </span>
                <span className="flex items-baseline gap-[1rem]">
                  <span className="text-[0.85rem] font-medium text-slate-500">
                    LEGS {p.legs}
                    {legs ? `/${game.legs_to_win}` : ""}
                  </span>
                  <AnimatedNumber value={p.score} className="font-display text-[2.5rem] font-bold leading-none text-slate-100" />
                </span>
              </>
            )}
          </motion.div>
        );
      })}
    </div>
  );
}

/** Three dart icons: lit in the player's colour for each dart still to throw this turn. */
function DartsLeft({ thrown, color }: { thrown: number; color: string }) {
  return (
    <div className="flex gap-[0.35rem]">
      {[0, 1, 2].map((k) => (
        <DartIcon key={k} lit={k >= thrown} color={color} />
      ))}
    </div>
  );
}

function DartIcon({ lit, color }: { lit: boolean; color: string }) {
  return (
    <svg viewBox="0 0 24 24" width="1.15rem" height="1.15rem" style={{ opacity: lit ? 1 : 0.28 }}>
      <g transform="rotate(45 12 12)" fill={lit ? color : "#94a3b8"}>
        <path d="M11.2 2 L12.8 2 L13.6 13 L10.4 13 Z" />
        <path d="M12 13 L12 20" stroke={lit ? color : "#94a3b8"} strokeWidth="1.6" strokeLinecap="round" />
        <path d="M8 22 L12 19.3 L11 22 Z" />
        <path d="M16 22 L12 19.3 L13 22 Z" />
        <circle cx="12" cy="3.2" r="1.5" />
      </g>
    </svg>
  );
}

function TurnCard({ state }: { state: EngineState }) {
  return (
    <div className="glass px-[1.35rem] pb-[1.2rem] pt-[1rem]">
      <div className="flex items-center justify-between">
        <span className="label">Turn</span>
        <div className="flex items-center gap-[0.8rem]">
          <AnimatePresence>
            {state.bust && (
              <motion.span
                initial={{ scale: 0.6, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.6, opacity: 0 }}
                className="rounded-full border border-rose-400/60 bg-rose-500/15 px-[0.8rem] py-[0.15rem] text-[0.9rem] font-bold tracking-widest text-rose-300"
              >
                BUST
              </motion.span>
            )}
          </AnimatePresence>
          <AnimatedNumber value={state.turn_total} className="font-display text-[2.7rem] font-bold leading-none text-white" />
        </div>
      </div>

      <div className="relative mt-[0.75rem] grid grid-cols-3 gap-[0.75rem]">
        {[0, 1, 2].map((k) => {
          const d = state.turn[k];
          const color = DART_COLORS[k];
          return (
            <div key={k} className="relative h-[8.4rem]">
              <AnimatePresence mode="popLayout" initial={false}>
                {d ? (
                  <motion.div
                    key={`d-${k}-${d.label}`}
                    className="slot slot-filled"
                    style={colorVar(color)}
                    initial={{ scale: 0.5, opacity: 0, y: 14 }}
                    animate={{ scale: 1, opacity: 1, y: 0 }}
                    exit={{ scale: 0.85, opacity: 0 }}
                    transition={{ type: "spring", stiffness: 380, damping: 19 }}
                  >
                    <span className="slot-index">{k + 1}</span>
                    <span className="font-display text-[3.7rem] font-bold leading-none text-white">{d.label}</span>
                    <span className="mt-[0.3rem] text-[0.95rem] font-bold tracking-wide" style={{ color }}>
                      {d.score} PTS
                    </span>
                  </motion.div>
                ) : (
                  <motion.div
                    key={`e-${k}`}
                    className="slot slot-empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <span className="slot-index">{k + 1}</span>
                    <span className="font-display text-[3rem] leading-none text-slate-600">–</span>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}

        <AnimatePresence>
          {state.waiting && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 flex flex-col items-center justify-center rounded-[1.15rem] border border-amber-300/40 bg-ink/85 backdrop-blur-sm"
            >
              <span className="soft-pulse text-[1.9rem] font-bold text-amber-300">Pull your darts</span>
              <span className="mt-[0.2rem] text-[1rem] text-slate-400">the next turn starts with a clear board</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence initial={false}>
        {state.checkout && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="flex items-center gap-[0.6rem] pt-[0.9rem]">
              <span className="label mr-[0.4rem] !text-amber-300">Checkout</span>
              {state.checkout.split(" ").map((part, i) => (
                <motion.span
                  key={`${part}-${i}`}
                  className="checkout-pill"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.06 }}
                >
                  {part}
                </motion.span>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/** Double that finishes the leg and how far each dart landed from it (double-out games). */
function FinishLegend({ state }: { state: EngineState }) {
  const darts = state.turn.filter((d) => d.finish);
  if (!state.finish_target && darts.length === 0) return null;
  return (
    <div className="pointer-events-none absolute left-[1.2rem] top-[1.1rem] flex flex-col items-start gap-[0.45rem]">
      {state.finish_target && (
        <span className="flex items-center gap-[0.6rem]">
          <span className="label !text-amber-300">Finish on</span>
          <span className="checkout-pill">{state.finish_target}</span>
        </span>
      )}
      {darts.map((d) => (
        <span key={d.index} className="text-[1rem] font-semibold tabular-nums" style={{ color: DART_COLORS[d.index % 3] }}>
          Dart {d.index + 1} · {Math.round(d.finish!.distance_mm)} mm from {d.finish!.target}
        </span>
      ))}
    </div>
  );
}
