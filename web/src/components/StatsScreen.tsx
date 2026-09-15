import confetti from "canvas-confetti";
import { animate, motion, useMotionValue, useTransform, type Variants } from "framer-motion";
import { useEffect, useState } from "react";
import { playerColor } from "../lib/players";
import type { EngineState, PlayerStats, ScoreBand } from "../types";
import { BoardFace } from "./BoardFace";
import { Avatar } from "./PlayerPhoto";
import { Kbd } from "./ui";

const EASE_OUT: [number, number, number, number] = [0.16, 1, 0.3, 1];
const CARD_DELAY = 0.45; // the sections start sliding in after the title has settled
const CARD_STAGGER = 0.14;

const gridVariants: Variants = {
  hidden: {},
  show: { transition: { delayChildren: CARD_DELAY, staggerChildren: CARD_STAGGER } },
};

const cardVariants: Variants = {
  hidden: { opacity: 0, y: 90, scale: 0.94, filter: "blur(14px)" },
  show: { opacity: 1, y: 0, scale: 1, filter: "blur(0px)", transition: { type: "spring", stiffness: 130, damping: 20 } },
};

/** End-of-match screen: one section per player with a scatter plot of every dart and all statistics. */
export function StatsScreen({
  state,
  onRematch,
  onNewGame,
  onEditLastTurn,
}: {
  state: EngineState;
  onRematch: () => void;
  onNewGame: () => void;
  onEditLastTurn: () => void;
}) {
  const summary = state.summary!;
  const count = summary.players.length;
  const columns = count <= 3 ? count : count === 4 ? 2 : 3;
  const winner = summary.players[summary.winner];
  const winnerColor = playerColor(summary.winner, count);
  const highestShare = Math.max(0, ...summary.players.flatMap((p) => p.score_bands.map((b) => b.pct)));
  const scaleMax = Math.min(100, Math.max(25, Math.ceil(highestShare / 25) * 25));

  useEffect(() => {
    const colors = [winnerColor, "#ffd166", "#ffffff"];
    const timer = window.setTimeout(() => {
      void confetti({ particleCount: 90, angle: 60, spread: 70, startVelocity: 60, origin: { x: 0, y: 0.75 }, colors, zIndex: 58 });
      void confetti({ particleCount: 90, angle: 120, spread: 70, startVelocity: 60, origin: { x: 1, y: 0.75 }, colors, zIndex: 58 });
    }, 650);
    return () => window.clearTimeout(timer);
  }, [winnerColor]);

  const legs = summary.legs_to_win > 1 ? `first to ${summary.legs_to_win} legs · ${summary.legs_played} played` : "single leg";

  return (
    <motion.div
      className="fixed inset-0 z-[55] overflow-hidden"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, transition: { duration: 0.35 } }}
      transition={{ duration: 0.6 }}
    >
      <div className="app-bg absolute inset-0" />
      <motion.div
        className="pointer-events-none absolute inset-0"
        style={{ background: `radial-gradient(80rem 38rem at 50% -8%, ${winnerColor}45, transparent 70%)` }}
        initial={{ opacity: 0, scale: 1.25 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1.6, ease: EASE_OUT }}
      />
      <div className="grid-overlay" />

      <div className="relative flex h-full flex-col px-[1.8rem] pb-[1.2rem] pt-[1.1rem]">
        <header className="flex items-center justify-between gap-[1.5rem]">
          <div className="flex min-w-0 items-center gap-[1.3rem]">
            <motion.div
              initial={{ scale: 0, rotate: -40 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ delay: 0.3, type: "spring", stiffness: 220, damping: 13 }}
            >
              <Avatar name={winner.name} photo={state.game.players[summary.winner]?.photo ?? null} color={winnerColor} className="h-[5.6rem] w-[5.6rem] text-[2.2rem]" />
            </motion.div>
            <div className="min-w-0">
              <motion.div layoutId="match-title" className="fx-text fx-gold w-fit text-[4.4rem]" transition={{ duration: 0.9, ease: EASE_OUT }}>
                GAME SHOT
              </motion.div>
              <motion.div
                className="mt-[0.35rem] truncate text-[1.35rem] font-semibold text-slate-300"
                initial={{ opacity: 0, x: -24 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.55, duration: 0.6, ease: EASE_OUT }}
              >
                <span className="font-bold" style={{ color: winnerColor }}>
                  {winner.name}
                </span>{" "}
                wins · {summary.start}
                {summary.double_out ? " double out" : ""} · {legs}
              </motion.div>
            </div>
          </div>
          <motion.div
            className="flex shrink-0 gap-[0.7rem]"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7, duration: 0.5, ease: EASE_OUT }}
          >
            <button type="button" className="btn btn-primary !h-[3.4rem] px-[1.6rem]" onClick={onRematch}>
              <Kbd>ENTER</Kbd> Rematch
            </button>
            <button type="button" className="btn btn-secondary !h-[3.4rem] px-[1.4rem]" onClick={onNewGame}>
              <Kbd>G</Kbd> New game
            </button>
            <button type="button" className="btn btn-ghost !h-[3.4rem] px-[1.4rem]" onClick={onEditLastTurn}>
              <Kbd>U</Kbd> Edit last turn
            </button>
          </motion.div>
        </header>

        <motion.div
          className="mt-[1.1rem] grid min-h-0 flex-1 gap-[1.1rem]"
          style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`, gridAutoRows: "minmax(0, 1fr)" }}
          variants={gridVariants}
          initial="hidden"
          animate="show"
        >
          {summary.players.map((stats, i) => (
            <PlayerSection
              key={stats.index}
              stats={stats}
              photo={state.game.players[i]?.photo ?? null}
              color={playerColor(i, count)}
              winner={i === summary.winner}
              count={count}
              doubleOut={summary.double_out}
              scaleMax={scaleMax}
              delay={CARD_DELAY + i * CARD_STAGGER}
            />
          ))}
        </motion.div>
      </div>
    </motion.div>
  );
}

interface SectionProps {
  stats: PlayerStats;
  photo: string | null;
  color: string;
  winner: boolean;
  count: number;
  doubleOut: boolean;
  scaleMax: number; // top of the score chart scale, shared by every player so the columns compare
  delay: number;
}

const BOARD_SIZE: Record<number, string> = { 1: "h-[25rem]", 2: "h-[21rem]", 3: "h-[19rem]" };
const DETAIL_COLUMNS: Record<number, string> = { 1: "grid-cols-6", 2: "grid-cols-4", 3: "grid-cols-3" };
const KPI_VALUE: Record<number, string> = { 1: "text-[4rem]", 2: "text-[4rem]", 3: "text-[2.8rem]" };
// shorter labels for the compact layout used with four or more players
const SHORT_LABELS: Record<string, string> = {
  "3-dart average": "Average",
  "First 9 average": "First 9",
  "Best turn": "Best",
  Checkout: "Out %",
  "Darts thrown": "Darts",
  "Legs won": "Legs",
  "High checkout": "High out",
  "Darts at double": "At double",
  "Avg to double": "Avg to dbl",
  "Closest to double": "Closest dbl",
};

function PlayerSection({ stats, photo, color, winner, count, doubleOut, scaleMax, delay }: SectionProps) {
  const compact = count >= 4;
  const label = (text: string) => (compact ? (SHORT_LABELS[text] ?? text) : text);
  const narrowKpis = count >= 3; // KPI tiles are too narrow for the long labels and decimals
  const kpiLabel = (text: string) => (narrowKpis ? (SHORT_LABELS[text] ?? text) : text);
  const dash = "—";
  const mm = (v: number | null) => (v == null ? dash : `${v.toFixed(1)} mm`);
  const share = (n: number, pct: number | null) => (pct == null ? String(n) : `${n} · ${pct.toFixed(0)}%`);
  const kpis: { label: string; value: number | null; decimals?: number; suffix?: string }[] = [
    { label: "3-dart average", value: stats.average, decimals: 1 },
    { label: "First 9 average", value: stats.first9_average, decimals: 1 },
    doubleOut
      ? { label: "Checkout", value: stats.checkout_pct, decimals: narrowKpis ? 0 : 1, suffix: "%" }
      : { label: "Points", value: stats.points },
    { label: "Best turn", value: stats.highest_turn },
  ];
  const details: [string, string][] = [
    ["Darts thrown", String(stats.darts_thrown)],
    ["Legs won", String(stats.legs_won)],
    ["High checkout", stats.highest_checkout == null ? dash : String(stats.highest_checkout)],
    ["Best leg", stats.best_leg_darts == null ? dash : `${stats.best_leg_darts} darts`],
    ["Busts", String(stats.busts)],
    ["Trebles", share(stats.trebles, stats.treble_pct)],
    ["Doubles", share(stats.doubles, stats.double_pct)],
    ["Bulls", String(stats.bulls)],
    ["Misses", share(stats.misses, stats.miss_pct)],
    ["Favourite", stats.favourite ? `${stats.favourite} ×${stats.favourite_hits}` : dash],
    ["Grouping", mm(stats.grouping_mm)],
    ...(doubleOut
      ? ([
          ["Darts at double", stats.darts_at_double == null ? dash : String(stats.darts_at_double)],
          ["Avg to double", mm(stats.avg_finish_distance_mm)],
          ["Closest to double", mm(stats.best_finish_distance_mm)],
        ] as [string, string][])
      : []),
  ];
  const appear = (extra: number) => ({ delay: delay + extra, duration: 0.45, ease: EASE_OUT });

  const board = (
    <div className={`relative shrink-0 ${compact ? "h-full w-[38%]" : `aspect-square ${BOARD_SIZE[count]}`}`}>
      <ScatterBoard stats={stats} color={color} delay={delay + 0.35} />
    </div>
  );

  const kpiGrid = (
    <div className={`grid min-w-0 gap-[0.55rem] ${compact ? "grid-cols-4" : "flex-1 grid-cols-2"}`}>
      {kpis.map((k, j) => (
        <motion.div
          key={k.label}
          className={`flex min-w-0 flex-col justify-center rounded-[1rem] border ${compact ? "px-[0.6rem] py-[0.4rem]" : "px-[0.9rem] py-[0.5rem]"}`}
          style={{ borderColor: `${color}40`, background: `linear-gradient(160deg, ${color}1f, transparent 70%)` }}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={appear(0.45 + j * 0.06)}
        >
          <span className={`truncate font-bold uppercase text-slate-400 ${compact ? "text-[0.62rem] tracking-[0.1em]" : "text-[0.72rem] tracking-[0.14em]"}`}>
            {kpiLabel(k.label)}
          </span>
          {k.value == null ? (
            <span className={`mt-[0.2rem] font-display font-bold leading-none text-slate-500 ${compact ? "text-[1.6rem]" : KPI_VALUE[count]}`}>—</span>
          ) : (
            <CountUp
              value={k.value}
              decimals={k.decimals ?? 0}
              suffix={k.suffix ?? ""}
              delay={delay + 0.5 + j * 0.06}
              className={`mt-[0.2rem] truncate font-display font-bold leading-none text-white ${compact ? "text-[1.6rem]" : KPI_VALUE[count]}`}
            />
          )}
        </motion.div>
      ))}
    </div>
  );

  const chart = (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={appear(0.6)}>
      <ScoreChart bands={stats.score_bands} color={color} scaleMax={scaleMax} compact={compact} delay={delay + 0.7} />
    </motion.div>
  );

  const detailGrid = compact ? (
    <div className={`grid min-h-0 flex-1 content-start gap-x-[1rem] ${count >= 5 ? "grid-cols-2" : "grid-cols-3"}`}>
      {details.map(([name, value], j) => (
        <motion.div
          key={name}
          className="flex min-w-0 items-baseline justify-between gap-[0.5rem] border-b border-white/[0.06] py-[0.08rem]"
          initial={{ opacity: 0, x: 10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={appear(0.7 + j * 0.02)}
        >
          <span className="truncate text-[0.76rem] font-semibold text-slate-400">{label(name)}</span>
          <span className="shrink-0 whitespace-nowrap font-display text-[1rem] font-bold text-slate-100">{value}</span>
        </motion.div>
      ))}
    </div>
  ) : (
    <div className={`grid min-h-0 flex-1 content-start gap-[0.45rem] ${DETAIL_COLUMNS[count]}`}>
      {details.map(([name, value], j) => (
        <motion.div
          key={name}
          className="rounded-[0.8rem] border border-white/10 bg-white/[0.035] px-[0.75rem] py-[0.4rem]"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={appear(0.7 + j * 0.025)}
        >
          <div className="truncate text-[0.68rem] font-bold uppercase tracking-[0.12em] text-slate-400">{name}</div>
          <div className="truncate font-display text-[1.35rem] font-bold leading-tight text-slate-100">{value}</div>
        </motion.div>
      ))}
    </div>
  );

  return (
    <motion.section
      variants={cardVariants}
      className={`glass relative flex min-h-0 flex-col overflow-hidden ${compact ? "px-[1.1rem] pb-[0.8rem] pt-[0.8rem]" : "px-[1.3rem] pb-[1rem] pt-[0.95rem]"}`}
      style={{ borderColor: `${color}80`, boxShadow: winner ? `0 0 0 1px ${color}60, 0 0 3.5rem -0.8rem ${color}` : undefined }}
    >
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[0.25rem]" style={{ background: `linear-gradient(90deg, transparent, ${color}, transparent)` }} />
      <div className="flex items-center gap-[0.9rem]">
        <Avatar name={stats.name} photo={photo} color={color} className={compact ? "h-[2.8rem] w-[2.8rem] text-[1.1rem]" : "h-[3.5rem] w-[3.5rem] text-[1.4rem]"} />
        <div className="min-w-0 flex-1">
          <div className={`truncate font-display font-bold leading-none text-white ${compact ? "text-[1.8rem]" : "text-[2.2rem]"}`}>{stats.name}</div>
          <div className="mt-[0.2rem] truncate text-[0.9rem] font-semibold text-slate-400">
            {stats.legs_won} {stats.legs_won === 1 ? "leg" : "legs"} · {stats.turns} {stats.turns === 1 ? "turn" : "turns"} · {stats.darts.length}{" "}
            {stats.darts.length === 1 ? "dart" : "darts"} on the board
          </div>
        </div>
        {winner && (
          <motion.span
            className="shrink-0 rounded-full px-[0.95rem] py-[0.3rem] text-[0.85rem] font-extrabold uppercase tracking-[0.2em] text-amber-950"
            style={{ background: "linear-gradient(90deg, #fde68a, #f59e0b)", boxShadow: "0 0 1.6rem -0.3rem #f59e0b" }}
            initial={{ scale: 0, rotate: -12 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ delay: delay + 0.8, type: "spring", stiffness: 320, damping: 12 }}
          >
            Winner
          </motion.span>
        )}
      </div>

      {compact ? (
        <div className="mt-[0.6rem] flex min-h-0 flex-1 gap-[0.9rem]">
          {board}
          <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-[0.5rem]">
            {kpiGrid}
            {chart}
            {detailGrid}
          </div>
        </div>
      ) : (
        <>
          <div className="mt-[0.9rem] flex min-w-0 gap-[0.9rem]">
            {board}
            {kpiGrid}
          </div>
          <div className="mt-[0.8rem]">{chart}</div>
          <div className="mt-[0.8rem] flex min-h-0 flex-1">{detailGrid}</div>
        </>
      )}
    </motion.section>
  );
}

/** Synthetic board with a dot for every dart of the player, plus the grouping circle. */
function ScatterBoard({ stats, color, delay }: { stats: PlayerStats; color: string; delay: number }) {
  const id = `scatter-${stats.index}`;
  const dotsDone = delay + 0.55 + Math.min(stats.darts.length * 0.045, 2.2);
  return (
    <svg viewBox="-242 -242 484 484" className="h-full w-full overflow-visible">
      <defs>
        <filter id={`${id}-glow`} x="-200%" y="-200%" width="500%" height="500%">
          <feGaussianBlur stdDeviation="4" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <motion.g
        initial={{ opacity: 0, scale: 0.55, rotate: -120 }}
        animate={{ opacity: 1, scale: 1, rotate: 0 }}
        transition={{ delay, duration: 1.2, ease: EASE_OUT }}
        style={{ transformBox: "view-box", transformOrigin: "0px 0px" }}
      >
        <BoardFace idPrefix={id} dim />
      </motion.g>
      {stats.darts.map((d, i) => (
        <motion.circle
          key={i}
          cx={d.x}
          cy={d.y}
          fill={color}
          stroke="#050814"
          strokeWidth={2}
          filter={`url(#${id}-glow)`}
          initial={{ r: 0, opacity: 0 }}
          animate={{ r: 7, opacity: 1 }}
          transition={{ delay: delay + 0.55 + Math.min(i * 0.045, 2.2), type: "spring", stiffness: 420, damping: 14 }}
        />
      ))}
      {stats.centroid_mm && stats.grouping_mm != null && (
        <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: dotsDone + 0.2, duration: 0.6 }}>
          <circle cx={stats.centroid_mm[0]} cy={stats.centroid_mm[1]} r={stats.grouping_mm} fill="none" stroke="#fff" strokeWidth={2} strokeDasharray="6 5" opacity={0.75} />
          <g stroke="#fff" strokeWidth={2.5} strokeLinecap="round">
            <line x1={stats.centroid_mm[0] - 9} y1={stats.centroid_mm[1]} x2={stats.centroid_mm[0] + 9} y2={stats.centroid_mm[1]} />
            <line x1={stats.centroid_mm[0]} y1={stats.centroid_mm[1] - 9} x2={stats.centroid_mm[0]} y2={stats.centroid_mm[1] + 9} />
          </g>
        </motion.g>
      )}
    </svg>
  );
}

// Score bands are ordered, so their columns step in lightness within the player's hue (low bands
// darker, high bands lighter). One ramp per player colour, each checked as an ordinal ramp against
// the card surface (#10152a): monotone lightness, visible steps, darkest step >= 2:1 contrast.
const BAND_RAMPS: Record<string, string[]> = {
  "#3b82f6": ["#3067c3", "#3879e6", "#4d8df7", "#6ea3f8", "#8fb8fa", "#b1cdfb"],
  "#ef4444": ["#802d37", "#ac363c", "#d93f41", "#f15757", "#f47c7c", "#f7a2a2"],
  "#22c55e": ["#186441", "#1c8e4e", "#21b95a", "#48cf79", "#7ddda0", "#b2ebc7"],
  "#f59e0b": ["#6c4c1e", "#875c1a", "#a36d16", "#be7d12", "#da8e0f", "#f59e0b"],
  "#a855f7": ["#8245c4", "#9d51e9", "#b268f8", "#c186f9", "#d1a5fb", "#e1c4fc"],
  "#ec4899": ["#893167", "#b53b7d", "#e14593", "#ef63a8", "#f388bd", "#f6add1"],
};

function bandColor(color: string, band: number): string {
  const ramp = BAND_RAMPS[color.toLowerCase()];
  return ramp ? ramp[Math.min(band, ramp.length - 1)] : color;
}

/** Share of the player's scoring turns in each score band (columns on a scale shared by all players). */
function ScoreChart({ bands, color, scaleMax, compact, delay }: { bands: ScoreBand[]; color: string; scaleMax: number; compact: boolean; delay: number }) {
  const [active, setActive] = useState<number | null>(null);
  const total = bands.reduce((n, b) => n + b.turns, 0);
  const last = bands.length - 1;
  return (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between gap-[0.5rem]">
        <span className={`font-bold uppercase text-slate-400 ${compact ? "text-[0.62rem] tracking-[0.1em]" : "text-[0.72rem] tracking-[0.14em]"}`}>Turn scores</span>
        <span className={`truncate font-semibold text-slate-500 ${compact ? "text-[0.66rem]" : "text-[0.74rem]"}`}>
          % of {total} scoring {total === 1 ? "turn" : "turns"}
        </span>
      </div>
      <div className={`relative flex border-b border-white/20 ${compact ? "mt-[0.1rem] h-[4rem] pt-[0.95rem]" : "mt-[0.2rem] h-[6.5rem] pt-[1.25rem]"}`}>
        {bands.map((b, k) => {
          const range = b.low === b.high ? String(b.low) : k === 0 ? `under ${b.high + 1}` : `${b.low}–${b.high}`;
          const align = k === 0 ? "left-0" : k === last ? "right-0" : "left-1/2 -translate-x-1/2";
          return (
            <div
              key={b.band}
              tabIndex={0}
              aria-label={`${b.band}: ${Math.round(b.pct)}% of scoring turns (${b.turns} of ${total})`}
              className="relative flex h-full min-w-0 flex-1 cursor-default flex-col items-center justify-end outline-none"
              onPointerEnter={() => setActive(k)}
              onPointerLeave={() => setActive(null)}
              onFocus={() => setActive(k)}
              onBlur={() => setActive(null)}
            >
              {b.turns > 0 && (
                <span className={`mb-[0.2rem] font-display font-bold leading-none tabular-nums text-slate-100 ${compact ? "text-[0.72rem]" : "text-[0.9rem]"}`}>
                  {Math.round(b.pct)}%
                </span>
              )}
              <motion.div
                className="w-[min(1.5rem,60%)] shrink-0 origin-bottom rounded-t-[0.25rem]"
                style={{ height: `${(b.pct / scaleMax) * 100}%`, background: bandColor(color, k), filter: active === k ? "brightness(1.3)" : undefined }}
                initial={{ scaleY: 0 }}
                animate={{ scaleY: 1 }}
                transition={{ delay: delay + k * 0.06, duration: 0.7, ease: EASE_OUT }}
              />
              {active === k && (
                <div className={`pointer-events-none absolute bottom-[calc(100%+0.3rem)] z-10 whitespace-nowrap rounded-[0.6rem] border border-white/15 bg-[#0b1122] px-[0.7rem] py-[0.4rem] shadow-xl ${align}`}>
                  <div className="font-display text-[1.05rem] font-bold leading-tight text-white">{b.pct.toFixed(1)}%</div>
                  <div className="text-[0.75rem] font-semibold text-slate-400">
                    {b.turns} of {total} turns scored {range}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="mt-[0.2rem] flex">
        {bands.map((b) => (
          <span key={b.band} className={`min-w-0 flex-1 text-center font-semibold text-slate-400 ${compact ? "text-[0.64rem]" : "text-[0.74rem]"}`}>
            {b.band}
          </span>
        ))}
      </div>
    </div>
  );
}

/** Number counting up from zero once the section appears. */
function CountUp({ value, decimals, suffix, delay, className }: { value: number; decimals: number; suffix: string; delay: number; className?: string }) {
  const mv = useMotionValue(0);
  const text = useTransform(mv, (v) => `${v.toFixed(decimals)}${suffix}`);
  useEffect(() => {
    const controls = animate(mv, value, { duration: 1.5, delay, ease: EASE_OUT });
    return () => controls.stop();
  }, [mv, value, delay]);
  return <motion.span className={className}>{text}</motion.span>;
}
