import { AnimatePresence, motion } from "framer-motion";
import { useMemo, useRef, useState, type MouseEvent } from "react";
import { applyH, clamp, modelPoint, RADII, remPx, SECTOR_ORDER } from "../lib/geometry";
import { useElementSize } from "../lib/hooks";
import { AMBER, colorVar, CYAN, DART_COLORS, ROSE, VIOLET } from "../lib/theme";
import type { EngineState, Send, Vec2 } from "../types";

const MAG_HALF = 55; // half side, in photo pixels, of the area shown by the magnifier

interface Props {
  state: EngineState;
  send: Send;
  lens: Vec2 | null;
  setLens: (lens: Vec2 | null) => void;
  streamKey: number;
}

export function CameraView({ state, send, lens, setLens, streamKey }: Props) {
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const boxRef = useRef<HTMLDivElement>(null);
  const review = state.mode === "review" ? state.review : null;
  const frame = review ? { w: review.w, h: review.h } : state.frame;

  const fit = useMemo(() => {
    if (!frame || size.width <= 0 || size.height <= 0) return null;
    const s = Math.min(size.width / frame.w, size.height / frame.h);
    return { w: frame.w * s, h: frame.h * s, s };
  }, [frame, size.width, size.height]);

  const overlay = useMemo(() => {
    const b = state.board;
    if (!b) return null;
    const H = b.H_inv;
    const rings = b.rings.map((r) => {
      const pts: string[] = [];
      for (let a = 0; a <= 360; a += 3) {
        const [x, y] = applyH(H, modelPoint(r, a));
        pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
      }
      return pts.join(" ");
    });
    const lines = Array.from({ length: 20 }, (_, k) => {
      const a = 9 + 18 * k;
      const [x1, y1] = applyH(H, modelPoint(b.rings[1], a));
      const [x2, y2] = applyH(H, modelPoint(b.rings[5], a));
      return { x1, y1, x2, y2 };
    });
    const numbers = SECTOR_ORDER.map((n, k) => ({ n, p: applyH(H, modelPoint(RADII.board + 24, 18 * k)) }));
    return { rings, lines, numbers };
  }, [state.board]);

  const mode = state.mode;
  const pct = (p: Vec2) => (frame ? { left: `${(p[0] / frame.w) * 100}%`, top: `${(p[1] / frame.h) * 100}%` } : {});

  const pointer = (e: MouseEvent) => {
    const rect = boxRef.current!.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    return { rect, sx, sy, img: [(sx / rect.width) * frame!.w, (sy / rect.height) * frame!.h] as Vec2 };
  };

  const nearestDart = (sx: number, sy: number, rect: DOMRect) => {
    let best = -1;
    let bestDist = Infinity;
    for (const d of state.turn) {
      if (!d.tip_img || !frame) continue;
      const dx = (d.tip_img[0] / frame.w) * rect.width - sx;
      const dy = (d.tip_img[1] / frame.h) * rect.height - sy;
      const dist = Math.hypot(dx, dy);
      if (dist < bestDist) {
        bestDist = dist;
        best = d.index;
      }
    }
    return bestDist <= remPx() * 2.4 ? best : -1;
  };

  const onClick = (e: MouseEvent) => {
    if (!frame || !boxRef.current) return;
    const { rect, sx, sy, img } = pointer(e);
    if (mode === "calibrating") {
      send({ type: "set20", x: img[0], y: img[1] });
    } else if (mode === "review") {
      if (lens) setLens(null);
      else if (nearestDart(sx, sy, rect) < 0) setLens(img);
    }
  };

  const onContextMenu = (e: MouseEvent) => {
    e.preventDefault();
    if (!frame || !boxRef.current || (mode !== "playing" && mode !== "review")) return;
    const { rect, sx, sy } = pointer(e);
    const k = nearestDart(sx, sy, rect);
    if (k >= 0) send({ type: "remove", index: k });
  };

  const borderColor = { searching: VIOLET, calibrating: VIOLET, playing: "rgb(148 163 184 / 0.22)", review: AMBER }[mode];
  const hint = cameraHint(state, lens !== null);

  return (
    <div className="glass relative h-full w-full overflow-hidden p-[0.85rem]">
      <div ref={wrapRef} className="relative flex h-full w-full items-center justify-center">
        {fit && frame ? (
          <div
            ref={boxRef}
            className="relative overflow-hidden rounded-[1.05rem]"
            style={{
              width: fit.w,
              height: fit.h,
              cursor: mode === "calibrating" || mode === "review" ? "crosshair" : "default",
              boxShadow: `0 0 0 2px ${borderColor}, 0 2rem 4rem -1.5rem rgb(0 0 0 / 0.85)`,
            }}
            onClick={onClick}
            onContextMenu={onContextMenu}
          >
            <img
              key={`live-${streamKey}`}
              src="/api/stream.mjpg"
              alt=""
              draggable={false}
              className="absolute inset-0 h-full w-full select-none"
              style={{ opacity: review ? 0 : 1, filter: mode === "searching" ? "brightness(0.7)" : undefined }}
            />
            {review && (
              <img
                key={`review-${review.id}`}
                src={`/api/review.jpg?id=${review.id}`}
                alt=""
                draggable={false}
                className="absolute inset-0 h-full w-full select-none"
              />
            )}

            {mode === "searching" && <ScanLine />}

            {overlay && (
              <svg
                viewBox={`0 0 ${frame.w} ${frame.h}`}
                preserveAspectRatio="none"
                className="pointer-events-none absolute inset-0 h-full w-full"
                style={{ opacity: mode === "calibrating" ? 0.95 : 0.6, filter: "drop-shadow(0 0 3px rgb(34 211 238 / 0.9))" }}
              >
                {overlay.rings.map((pts, i) => (
                  <polyline
                    key={i}
                    points={pts}
                    fill="none"
                    stroke={CYAN}
                    strokeWidth={i < 2 ? 1.2 : 1.8}
                    vectorEffect="non-scaling-stroke"
                  />
                ))}
                {overlay.lines.map((l, i) => (
                  <line key={i} {...l} stroke={CYAN} strokeWidth={1} strokeOpacity={0.75} vectorEffect="non-scaling-stroke" />
                ))}
              </svg>
            )}

            {mode === "playing" && state.show_detections && fit && (
              <svg viewBox={`0 0 ${frame.w} ${frame.h}`} preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full">
                {state.detections.map(([x, y], i) => {
                  const p = state.board ? applyH(state.board.H_inv, [x, y]) : null;
                  return p ? <circle key={i} cx={p[0]} cy={p[1]} r={(remPx() * 0.32) / fit.s} fill="#fde047" /> : null;
                })}
              </svg>
            )}

            {overlay && mode === "calibrating" &&
              overlay.numbers.map(({ n, p }) => (
                <div key={n} className="pointer-events-none absolute" style={pct(p)}>
                  <div
                    className="number-chip"
                    style={
                      n === 20
                        ? { color: "#fff", background: "#db2777", borderColor: "#f9a8d4", boxShadow: "0 0 1.2rem #ec4899", fontSize: "1.2rem" }
                        : undefined
                    }
                  >
                    {n}
                  </div>
                </div>
              ))}

            {(mode === "playing" || mode === "review") &&
              state.turn.map((d) =>
                d.tip_img ? (
                  <motion.div
                    key={d.index}
                    className="pointer-events-none absolute"
                    style={{ ...pct(d.tip_img), ...colorVar(DART_COLORS[d.index % 3]) }}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ type: "spring", stiffness: 420, damping: 20 }}
                  >
                    <div className="dart-marker" />
                    <div className={`dart-label dart-label-${d.index % 3}`}>
                      {d.index + 1} · {d.label}
                    </div>
                  </motion.div>
                ) : null,
              )}

            <AnimatePresence>
              {lens && review && fit && (
                <Lens key={`${lens[0]}-${lens[1]}`} lens={lens} review={review} fit={fit} send={send} close={() => setLens(null)} />
              )}
            </AnimatePresence>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 text-slate-400">
            <div className="h-10 w-10 animate-spin rounded-full border-2 border-cyan-400/30 border-t-cyan-300" />
            <span className="text-[1.2rem]">Starting camera…</span>
          </div>
        )}
      </div>

      <AnimatePresence mode="wait">
        {hint && (
          <motion.div
            key={hint.title}
            initial={{ opacity: 0, y: 16, x: "-50%" }}
            animate={{ opacity: 1, y: 0, x: "-50%" }}
            exit={{ opacity: 0, y: 16, x: "-50%" }}
            className="glass pointer-events-none absolute bottom-[2.2rem] left-1/2 px-[1.8rem] py-[1rem] text-center"
            style={{ borderColor: hint.color, boxShadow: `0 0 2.4rem -0.6rem ${hint.color}` }}
          >
            <div className="text-[1.45rem] font-bold text-white">{hint.title}</div>
            <div className="mt-[0.2rem] text-[1rem] text-slate-300">{hint.subtitle}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function cameraHint(state: EngineState, lensOpen: boolean) {
  if (state.source_error) return { title: "Camera unavailable", subtitle: state.source_error, color: ROSE };
  switch (state.mode) {
    case "searching":
      return { title: "Looking for the dartboard…", subtitle: "Keep it fully in frame and well lit", color: VIOLET };
    case "calibrating":
      return { title: "Is the 20 in the right place?", subtitle: "Click the 20 sector if needed · ENTER to start", color: VIOLET };
    case "review":
      return lensOpen
        ? { title: "Click where the tip enters the board", subtitle: "ESC to close the magnifier", color: AMBER }
        : { title: "Missing a dart?", subtitle: "Click near its tip on the photo to open the magnifier", color: AMBER };
    default:
      if (state.board_lost) return { title: "Board not visible", subtitle: "Check the webcam framing", color: ROSE };
      return null;
  }
}

function ScanLine() {
  return (
    <>
      <motion.div
        className="pointer-events-none absolute inset-x-0 h-[6rem]"
        style={{ background: "linear-gradient(180deg, transparent, rgb(34 211 238 / 0.28), transparent)" }}
        initial={{ top: "-6rem" }}
        animate={{ top: ["-6rem", "100%"] }}
        transition={{ duration: 2.6, repeat: Infinity, ease: "easeInOut", repeatType: "reverse" }}
      />
      <div className="pointer-events-none absolute inset-0" style={{ boxShadow: "inset 0 0 8rem rgb(5 8 20 / 0.9)" }} />
    </>
  );
}

interface LensProps {
  lens: Vec2;
  review: { id: number; w: number; h: number };
  fit: { w: number; h: number; s: number };
  send: Send;
  close: () => void;
}

function Lens({ lens, review, fit, send, close }: LensProps) {
  const [mouse, setMouse] = useState<Vec2 | null>(null);
  const size = Math.min(remPx() * 24, fit.w * 0.45, fit.h * 0.6);
  const zoom = size / (2 * MAG_HALF);
  const cx = (lens[0] / review.w) * fit.w;
  const cy = (lens[1] / review.h) * fit.h;
  const margin = remPx() * 0.6;
  const left = clamp(cx - size / 2, margin, fit.w - size - margin);
  const top = clamp(cy - size / 2, margin, fit.h - size - margin);

  const onClick = (e: MouseEvent<HTMLDivElement>) => {
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    const x = lens[0] - MAG_HALF + (e.clientX - rect.left) / zoom;
    const y = lens[1] - MAG_HALF + (e.clientY - rect.top) / zoom;
    send({ type: "add", x, y });
    close();
  };

  return (
    <motion.div
      className="absolute overflow-hidden rounded-[1.4rem]"
      style={{
        left,
        top,
        width: size,
        height: size,
        cursor: "none",
        backgroundColor: "#000",
        backgroundImage: `url(/api/review.jpg?id=${review.id})`,
        backgroundRepeat: "no-repeat",
        backgroundSize: `${review.w * zoom}px ${review.h * zoom}px`,
        backgroundPosition: `${-(lens[0] - MAG_HALF) * zoom}px ${-(lens[1] - MAG_HALF) * zoom}px`,
        boxShadow: `0 0 0 3px ${AMBER}, 0 0 3rem rgb(251 191 36 / 0.45), 0 2rem 4rem rgb(0 0 0 / 0.8)`,
      }}
      initial={{ scale: 0.6, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      exit={{ scale: 0.8, opacity: 0 }}
      transition={{ type: "spring", stiffness: 380, damping: 26 }}
      onClick={onClick}
      onContextMenu={(e) => {
        e.preventDefault();
        e.stopPropagation();
        close();
      }}
      onMouseMove={(e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        setMouse([e.clientX - rect.left, e.clientY - rect.top]);
      }}
      onMouseLeave={() => setMouse(null)}
    >
      {mouse && (
        <svg className="pointer-events-none absolute inset-0 h-full w-full">
          <line x1={mouse[0]} y1={0} x2={mouse[0]} y2={size} stroke={AMBER} strokeWidth={1.5} strokeOpacity={0.9} />
          <line x1={0} y1={mouse[1]} x2={size} y2={mouse[1]} stroke={AMBER} strokeWidth={1.5} strokeOpacity={0.9} />
          <circle cx={mouse[0]} cy={mouse[1]} r={remPx() * 0.9} fill="none" stroke="#fff" strokeWidth={2} />
        </svg>
      )}
      <button
        type="button"
        className="absolute right-[0.6rem] top-[0.6rem] grid h-[2.2rem] w-[2.2rem] place-items-center rounded-full bg-black/60 text-[1.2rem] font-bold text-white hover:bg-black/80"
        style={{ cursor: "pointer" }}
        onClick={(e) => {
          e.stopPropagation();
          close();
        }}
      >
        ×
      </button>
    </motion.div>
  );
}
