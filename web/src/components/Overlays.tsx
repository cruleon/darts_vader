import confetti from "canvas-confetti";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState, type KeyboardEvent, type ReactNode } from "react";
import type { AudioSnapshot } from "../lib/audio";
import { toggleFullscreen } from "../lib/hooks";
import { MODE_STYLE, TONE_COLORS } from "../lib/theme";
import type { EffectKind, EngineState, Mode, Tone } from "../types";
import { DartGlyph, Kbd, Logo } from "./ui";

/* ------------------------------------------------------------------ top bar */

interface TopBarProps {
  state: EngineState;
  audio: AudioSnapshot;
  onNewGame: () => void;
  onToggleSound: () => void;
}

export function TopBar({ state, audio, onNewGame, onToggleSound }: TopBarProps) {
  const { game } = state;
  const n = game.players.length;
  const status = MODE_STYLE[state.mode];
  return (
    <header className="relative flex h-[4.6rem] shrink-0 items-center justify-between">
      <div className="flex items-center gap-[1rem]">
        <Logo className="h-[3.7rem] w-[3.7rem] drop-shadow-[0_0_1rem_rgb(167_139_250_/_0.55)]" />
        <div>
          <h1 className="font-display text-[2.35rem] font-bold leading-none tracking-wide text-white">
            DARTS <span className="text-gradient">VADER</span>
          </h1>
          <p className="mt-[0.25rem] text-[0.95rem] font-medium text-slate-400">
            {game.start} · {game.double_out ? "double out" : "straight out"} · {n} {n === 1 ? "player" : "players"}
          </p>
        </div>
      </div>

      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
        <motion.div
          key={state.mode}
          initial={{ scale: 0.85, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="glass flex items-center gap-[0.8rem] rounded-full px-[1.5rem] py-[0.7rem]"
          style={{ borderColor: status.color, boxShadow: `0 0 2rem -0.6rem ${status.color}` }}
        >
          <span className="relative flex h-[0.75rem] w-[0.75rem]">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-70" style={{ background: status.color }} />
            <span className="relative inline-flex h-[0.75rem] w-[0.75rem] rounded-full" style={{ background: status.color }} />
          </span>
          <span className="text-[1.1rem] font-bold uppercase tracking-[0.22em]" style={{ color: status.color }}>
            {status.label}
          </span>
        </motion.div>
      </div>

      <div className="flex items-center gap-[0.7rem]">
        <StatChip label="AI" value={`${Math.round(state.fps_ai)} fps`} />
        <StatChip label="Saved" value={`${state.saved}`} />
        <button
          type="button"
          onClick={onToggleSound}
          title="Turn sound effects on or off (M)"
          aria-pressed={audio.sound}
          className={`btn !h-[2.9rem] rounded-full !gap-[0.55rem] px-[1rem] !text-[1rem] ${audio.sound ? "btn-secondary" : "btn-ghost !text-slate-400"}`}
        >
          <SpeakerIcon on={audio.sound} />
          {audio.sound ? "Sound on" : "Sound off"}
          <Kbd>M</Kbd>
        </button>
        <button type="button" onClick={onNewGame} className="btn btn-secondary !h-[2.9rem] rounded-full px-[1.2rem] !text-[1rem]">
          New game <Kbd>G</Kbd>
        </button>
        <button
          type="button"
          onClick={toggleFullscreen}
          title="Fullscreen (F)"
          className="btn btn-ghost !h-[2.9rem] w-[2.9rem] rounded-full !p-0"
        >
          <svg viewBox="0 0 24 24" className="h-[1.3rem] w-[1.3rem]" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round">
            <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
          </svg>
        </button>
      </div>
    </header>
  );
}

function SpeakerIcon({ on }: { on: boolean }) {
  return (
    <svg viewBox="0 0 24 24" className="h-[1.3rem] w-[1.3rem]" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 9.5h3.2L12 5.5v13l-4.8-4H4z" fill="currentColor" stroke="none" />
      {on ? <path d="M15.5 9a4.2 4.2 0 0 1 0 6M18.2 6.5a8 8 0 0 1 0 11" /> : <path d="M16 9.5l5 5M21 9.5l-5 5" />}
    </svg>
  );
}

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass flex h-[2.9rem] items-center gap-[0.6rem] rounded-full px-[1.1rem]">
      <span className="text-[0.75rem] font-bold uppercase tracking-[0.18em] text-slate-500">{label}</span>
      <span className="text-[1.05rem] font-semibold tabular-nums text-slate-200">{value}</span>
    </div>
  );
}

/* ------------------------------------------------------------------ keyboard hints */

const HINTS: Record<Mode, [string, string][]> = {
  searching: [["R", "recalibrate"], ["G", "new game"], ["M", "sound"], ["F", "fullscreen"]],
  calibrating: [["CLICK", "the 20 sector"], ["ENTER", "start playing"], ["R", "recalibrate"], ["M", "sound"], ["F", "fullscreen"]],
  playing: [
    ["SPACE", "review"],
    ["U", "undo"],
    ["RIGHT CLICK", "remove ghost"],
    ["CLICK BOARD", "add dart"],
    ["D", "detections"],
    ["C", "clear ignored"],
    ["R", "recalibrate"],
    ["M", "sound"],
    ["F", "fullscreen"],
  ],
  review: [
    ["CLICK TILE", "move tip"],
    ["RIGHT CLICK", "remove"],
    ["CLICK PHOTO", "magnifier"],
    ["ENTER", "confirm"],
    ["K", "score only"],
    ["U", "undo"],
    ["ESC", "back to game"],
  ],
};

export function HintBar({ mode }: { mode: Mode }) {
  return (
    <footer className="flex h-[2.4rem] shrink-0 items-center gap-[1.5rem] overflow-hidden">
      {HINTS[mode].map(([key, label]) => (
        <span key={key} className="flex items-center gap-[0.55rem] whitespace-nowrap text-[0.95rem] font-medium text-slate-400">
          <Kbd>{key}</Kbd>
          {label}
        </span>
      ))}
    </footer>
  );
}

/* ------------------------------------------------------------------ toasts */

export interface ToastItem {
  id: number;
  text: string;
  tone: Tone;
}

export function Toasts({ items }: { items: ToastItem[] }) {
  return (
    <div className="pointer-events-none fixed left-1/2 top-[6.8rem] z-40 flex -translate-x-1/2 flex-col items-center gap-[0.6rem]">
      <AnimatePresence initial={false}>
        {items.map((t) => (
          <motion.div
            key={t.id}
            layout
            initial={{ opacity: 0, y: -18, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -12, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 420, damping: 32 }}
            className="glass flex items-center gap-[0.85rem] rounded-full py-[0.75rem] pl-[1.1rem] pr-[1.6rem]"
            style={{ borderColor: `${TONE_COLORS[t.tone]}aa`, boxShadow: `0 0 2rem -0.8rem ${TONE_COLORS[t.tone]}, 0 1rem 2rem -1rem rgb(0 0 0 / 0.7)` }}
          >
            <span className="h-[0.7rem] w-[0.7rem] rounded-full" style={{ background: TONE_COLORS[t.tone], boxShadow: `0 0 0.7rem ${TONE_COLORS[t.tone]}` }} />
            <span className="whitespace-nowrap text-[1.2rem] font-semibold text-slate-100">{t.text}</span>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

/* ------------------------------------------------------------------ celebration effects */

export interface EffectItem {
  id: number;
  effect: EffectKind;
  text: string;
}

const EFFECT_MS: Record<EffectKind, number> = { bust: 1800, ton: 1900, "180": 3000, win: 3800 };
const CONFETTI = ["#ffd166", "#22d3ee", "#f472b6", "#a78bfa", "#34d399"];

export function EffectLayer({ effect, onDone }: { effect: EffectItem | null; onDone: () => void }) {
  useEffect(() => {
    if (!effect) return;
    const timers: number[] = [];
    const burst = (opts: confetti.Options) => void confetti({ zIndex: 45, colors: CONFETTI, disableForReducedMotion: true, ...opts });
    if (effect.effect === "180") {
      burst({ particleCount: 220, spread: 120, startVelocity: 62, scalar: 1.25, origin: { x: 0.5, y: 0.55 } });
      timers.push(window.setTimeout(() => burst({ particleCount: 120, spread: 160, startVelocity: 45, scalar: 1.1, origin: { x: 0.5, y: 0.5 } }), 450));
    } else if (effect.effect === "win") {
      for (let i = 0; i < 9; i++) {
        timers.push(
          window.setTimeout(() => {
            burst({ particleCount: 70, angle: 60, spread: 65, startVelocity: 70, origin: { x: 0, y: 0.9 } });
            burst({ particleCount: 70, angle: 120, spread: 65, startVelocity: 70, origin: { x: 1, y: 0.9 } });
          }, i * 280),
        );
      }
    } else if (effect.effect === "ton") {
      burst({ particleCount: 90, spread: 90, startVelocity: 45, origin: { x: 0.5, y: 0.55 } });
    }
    timers.push(window.setTimeout(onDone, EFFECT_MS[effect.effect]));
    return () => timers.forEach((t) => window.clearTimeout(t));
    // restart only for a new effect event
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effect?.id]);

  return (
    <AnimatePresence>
      {effect && (
        <motion.div
          key={effect.id}
          className="pointer-events-none fixed inset-0 z-50 grid place-items-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0, transition: { duration: 0.5 } }}
        >
          <div
            className="absolute inset-0"
            style={{
              background:
                effect.effect === "bust"
                  ? "radial-gradient(circle at 50% 50%, rgb(244 63 94 / 0.35), rgb(5 8 20 / 0.6) 70%)"
                  : "radial-gradient(circle at 50% 50%, rgb(5 8 20 / 0.35), rgb(5 8 20 / 0.78) 70%)",
            }}
          />
          <EffectContent effect={effect} />
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function Subtitle({ children, delay = 0.25, className = "" }: { children: ReactNode; delay?: number; className?: string }) {
  return (
    <motion.div
      className={`font-display font-bold uppercase text-white ${className}`}
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5 }}
    >
      {children}
    </motion.div>
  );
}

function EffectContent({ effect }: { effect: EffectItem }) {
  const pop = { initial: { scale: 0.3, opacity: 0 }, animate: { scale: 1, opacity: 1 }, transition: { type: "spring" as const, stiffness: 260, damping: 14 } };
  switch (effect.effect) {
    case "180":
      return (
        <motion.div className="relative text-center" {...pop}>
          <div className="fx-text fx-gold text-[24rem]">180</div>
          <Subtitle className="mt-[-1rem] text-[4.4rem] tracking-[0.5em]">Maximum</Subtitle>
        </motion.div>
      );
    case "bust":
      return (
        <div className="relative text-center">
          <motion.div
            className="fx-text fx-red text-[20rem]"
            initial={{ scale: 1.6, opacity: 0 }}
            animate={{ scale: 1, opacity: 1, x: [0, -32, 28, -20, 16, -8, 0] }}
            transition={{ duration: 0.6 }}
          >
            BUST
          </motion.div>
          <Subtitle delay={0.45} className="text-[2.6rem] tracking-[0.3em] !text-slate-300">
            The dark side of the double
          </Subtitle>
        </div>
      );
    case "win":
      return (
        <motion.div className="relative text-center" {...pop}>
          <div className="fx-text fx-gold text-[15rem]">GAME SHOT</div>
          <Subtitle delay={0.3} className="mt-[0.8rem] text-[3.6rem] tracking-[0.14em]">
            The Force is strong with {effect.text}
          </Subtitle>
        </motion.div>
      );
    default:
      return (
        <motion.div className="relative text-center" {...pop}>
          <div className="fx-text fx-cyan text-[20rem]">{effect.text}</div>
          <Subtitle className="mt-[-0.5rem] text-[3.4rem] tracking-[0.4em]">Impressive</Subtitle>
        </motion.div>
      );
  }
}

/* ------------------------------------------------------------------ new game dialog */

function Switch({ on }: { on: boolean }) {
  return (
    <span className={`relative h-[2rem] w-[3.6rem] shrink-0 rounded-full transition-colors ${on ? "bg-cyan-400" : "bg-slate-600"}`}>
      <motion.span
        className="absolute top-[0.25rem] h-[1.5rem] w-[1.5rem] rounded-full bg-white shadow"
        animate={{ left: on ? "1.85rem" : "0.25rem" }}
        transition={{ type: "spring", stiffness: 500, damping: 30 }}
      />
    </span>
  );
}

function ToggleRow({ on, title, subtitle, onClick, className = "" }: { on: boolean; title: string; subtitle: string; onClick: () => void; className?: string }) {
  return (
    <button
      type="button"
      className={`flex w-full items-center justify-between rounded-[1rem] border border-white/10 bg-white/[0.03] px-[1.2rem] py-[0.9rem] ${className}`}
      onClick={onClick}
    >
      <span className="text-left">
        <span className="block text-[1.15rem] font-semibold text-slate-100">{title}</span>
        <span className="block text-[0.95rem] text-slate-400">{subtitle}</span>
      </span>
      <Switch on={on} />
    </button>
  );
}

export function SetupDialog({
  open,
  state,
  audio,
  onToggleSound,
  onStart,
  onClose,
}: {
  open: boolean;
  state: EngineState;
  audio: AudioSnapshot;
  onToggleSound: () => void;
  onStart: (settings: { players: string[]; start: number; double_out: boolean }) => void;
  onClose: () => void;
}) {
  const [players, setPlayers] = useState<string[]>([]);
  const [start, setStart] = useState(301);
  const [doubleOut, setDoubleOut] = useState(false);

  useEffect(() => {
    if (!open) return;
    setPlayers(state.game.players.map((p) => p.name));
    setStart(state.game.start);
    setDoubleOut(state.game.double_out);
    // values are copied only when the dialog opens
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const names = players.map((p) => p.trim()).filter(Boolean);
  const submit = () => {
    if (names.length) onStart({ players: names, start, double_out: doubleOut });
  };
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      submit();
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[60] grid place-items-center bg-ink/70 backdrop-blur-md"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onKeyDown={onKeyDown}
        >
          <motion.div
            className="glass w-[42rem] px-[2.4rem] pb-[2.2rem] pt-[2rem]"
            initial={{ scale: 0.92, y: 24, opacity: 0 }}
            animate={{ scale: 1, y: 0, opacity: 1 }}
            exit={{ scale: 0.95, y: 12, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 28 }}
          >
            <div className="flex items-center gap-[1rem]">
              <Logo className="h-[3.4rem] w-[3.4rem] drop-shadow-[0_0_1rem_rgb(167_139_250_/_0.55)]" />
              <div>
                <div className="label">Darts Vader</div>
                <h2 className="font-display text-[3rem] font-bold leading-none text-white">New game</h2>
              </div>
            </div>

            <div className="mt-[1.8rem] label">Players</div>
            <div className="mt-[0.7rem] flex flex-col gap-[0.6rem]">
              {players.map((name, i) => (
                <div key={i} className="flex items-center gap-[0.6rem]">
                  <span className="w-[1.8rem] text-center font-display text-[1.6rem] font-bold text-slate-500">{i + 1}</span>
                  <input
                    className="field"
                    value={name}
                    maxLength={18}
                    autoFocus={i === 0}
                    placeholder={`Player ${i + 1}`}
                    onChange={(e) => setPlayers((ps) => ps.map((p, j) => (j === i ? e.target.value : p)))}
                  />
                  <button
                    type="button"
                    className="btn btn-ghost !h-[3.2rem] w-[3.2rem] shrink-0 !p-0 text-[1.4rem]"
                    disabled={players.length <= 1}
                    style={{ opacity: players.length <= 1 ? 0.35 : 1 }}
                    onClick={() => setPlayers((ps) => ps.filter((_, j) => j !== i))}
                  >
                    ×
                  </button>
                </div>
              ))}
              {players.length < 6 && (
                <button
                  type="button"
                  className="btn btn-ghost !h-[3rem] border-dashed !text-[1rem]"
                  onClick={() => setPlayers((ps) => [...ps, ""])}
                >
                  + Add player
                </button>
              )}
            </div>

            <div className="mt-[1.6rem] label">Starting score</div>
            <div className="mt-[0.7rem] grid grid-cols-4 gap-[0.6rem]">
              {[101, 301, 501, 701].map((v) => (
                <button
                  type="button"
                  key={v}
                  className={`btn !h-[3.6rem] font-display !text-[2rem] ${v === start ? "btn-secondary" : "btn-ghost"}`}
                  style={v === start ? { boxShadow: "0 0 1.6rem -0.4rem rgb(34 211 238 / 0.7)" } : undefined}
                  onClick={() => setStart(v)}
                >
                  {v}
                </button>
              ))}
            </div>

            <ToggleRow
              className="mt-[1.4rem]"
              on={doubleOut}
              title="Double out"
              subtitle="Finish on a double or the bull"
              onClick={() => setDoubleOut((d) => !d)}
            />
            <ToggleRow
              className="mt-[0.6rem]"
              on={audio.sound}
              title="Sound effects"
              subtitle="Hits, 180s, busts and wins · press M anytime"
              onClick={onToggleSound}
            />

            <div className="mt-[1.8rem] grid grid-cols-3 gap-[0.7rem]">
              <button type="button" className="btn btn-ghost" onClick={onClose}>
                <Kbd>ESC</Kbd> Close
              </button>
              <button
                type="button"
                className="btn btn-primary col-span-2"
                disabled={!names.length}
                style={{ opacity: names.length ? 1 : 0.5 }}
                onClick={submit}
              >
                <Kbd>ENTER</Kbd> Start game
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/* ------------------------------------------------------------------ splash screen and connection banner */

export function Splash({ message }: { message: string }) {
  return (
    <div className="relative grid h-full place-items-center">
      <motion.div className="flex flex-col items-center" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}>
        <div className="relative">
          <motion.div animate={{ y: [0, -8, 0] }} transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}>
            <Logo className="h-[13rem] w-[13rem] drop-shadow-[0_0_2.5rem_rgb(167_139_250_/_0.55)]" />
          </motion.div>
          <motion.div
            className="absolute left-[-6rem] top-[3rem] w-[9rem]"
            initial={{ x: "22rem", opacity: 0 }}
            animate={{ x: ["22rem", "-4rem"], opacity: [0, 1, 1, 0] }}
            transition={{ duration: 1.5, repeat: Infinity, repeatDelay: 0.9, ease: "easeOut" }}
          >
            <DartGlyph className="w-full" />
          </motion.div>
        </div>
        <h1 className="mt-[2rem] font-display text-[4.5rem] font-bold leading-none tracking-wide text-white">
          DARTS <span className="text-gradient">VADER</span>
        </h1>
        <p className="soft-pulse mt-[1rem] text-[1.3rem] text-slate-400">{message}</p>
      </motion.div>
    </div>
  );
}

export function ConnectionBanner({ connected }: { connected: boolean }) {
  return (
    <AnimatePresence>
      {!connected && (
        <motion.div
          className="fixed inset-0 z-[70] grid place-items-center bg-ink/75 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <div className="glass flex items-center gap-[1rem] px-[2rem] py-[1.3rem]" style={{ borderColor: "#fb7185aa" }}>
            <div className="h-[1.8rem] w-[1.8rem] animate-spin rounded-full border-2 border-rose-300/30 border-t-rose-300" />
            <span className="text-[1.3rem] font-semibold text-slate-100">Lost connection to the engine: reconnecting…</span>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
