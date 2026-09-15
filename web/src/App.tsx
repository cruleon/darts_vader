import { AnimatePresence, LayoutGroup, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { CameraView } from "./components/CameraView";
import { GamePanel } from "./components/GamePanel";
import {
  ConnectionBanner,
  EffectLayer,
  HintBar,
  SetupDialog,
  Splash,
  Toasts,
  TopBar,
  type EffectItem,
  type ToastItem,
} from "./components/Overlays";
import { ReviewPanel } from "./components/ReviewPanel";
import { StatsScreen } from "./components/StatsScreen";
import { Background } from "./components/ui";
import { sound, type Sfx } from "./lib/audio";
import { toggleFullscreen, useEngine } from "./lib/hooks";
import type { EngineEvent, EngineState, GameSettings, Vec2 } from "./types";

const TOAST_MS = 3800;

function dartSound(number: number, multiplier: number): Sfx {
  if (multiplier === 0) return "miss";
  if (number === 25) return "bull";
  return multiplier === 3 ? "triple" : multiplier === 2 ? "double" : "dart";
}

/** Sound effect for each engine event. */
function playEvent(ev: EngineEvent): void {
  switch (ev.kind) {
    case "dart":
      sound.play(dartSound(ev.number, ev.multiplier));
      break;
    case "removed":
      sound.play("remove");
      break;
    case "moved":
    case "set20":
      sound.play("tick");
      break;
    case "review":
      sound.play("review");
      break;
    case "board_found":
      sound.play("lock");
      break;
    case "game_start":
    case "new_game":
      sound.play("start");
      break;
    case "effect":
      sound.play(ev.effect);
      break;
    case "turn":
      if (ev.outcome === "ok" && ev.points < 100) sound.play("confirm"); // busts, 180s, tons and wins have their own effect
      break;
    default:
      break;
  }
}

export default function App() {
  const { state, connected, generation, send } = useEngine();
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [effect, setEffect] = useState<EffectItem | null>(null);
  const [setupOpen, setSetupOpen] = useState(false);
  const [statsOpen, setStatsOpen] = useState(false);
  const [lens, setLens] = useState<Vec2 | null>(null);
  const lastEventId = useRef<number | null>(null);
  const effectRef = useRef<EffectItem | null>(null);
  const setupShown = useRef(false);
  const stateRef = useRef<EngineState | null>(null);
  const audio = useSyncExternalStore(sound.subscribe, sound.getSnapshot);
  stateRef.current = state;

  // after every (re)connection, events already in the buffer must not be replayed
  useEffect(() => {
    lastEventId.current = null;
  }, [generation]);

  const showEffect = useCallback((item: EffectItem | null) => {
    effectRef.current = item;
    setEffect(item);
  }, []);

  const handleEvent = useCallback(
    (ev: EngineEvent) => {
      playEvent(ev);
      if (ev.kind === "toast") {
        setToasts((ts) => [...ts.slice(-3), { id: ev.id, text: ev.text, tone: ev.tone }]);
        window.setTimeout(() => setToasts((ts) => ts.filter((t) => t.id !== ev.id)), TOAST_MS);
      } else if (ev.kind === "effect") {
        showEffect({ id: ev.id, effect: ev.effect, text: ev.text });
      } else if (ev.kind === "game_over") {
        const current = effectRef.current;
        // the statistics take over when the winning animation ends
        if (current?.effect === "win") showEffect({ ...current, final: true });
        else setStatsOpen(true);
      } else if (ev.kind === "review") {
        setLens(null);
      }
    },
    [showEffect],
  );

  const onEffectDone = useCallback(() => {
    const final = effectRef.current?.final;
    showEffect(null);
    if (final) setStatsOpen(true);
  }, [showEffect]);

  useEffect(() => {
    if (!state) return;
    const events = state.events;
    if (lastEventId.current === null) {
      lastEventId.current = events.length ? events[events.length - 1].id : 0;
      if (state.mode === "finished" && state.summary) setStatsOpen(true);
      if (!setupShown.current) {
        setupShown.current = true;
        if (state.game.history.length === 0) setSetupOpen(true);
      }
      return;
    }
    const fresh = events.filter((e) => e.id > lastEventId.current!);
    if (fresh.length) {
      lastEventId.current = fresh[fresh.length - 1].id;
      fresh.forEach(handleEvent);
    }
  }, [state, handleEvent]);

  const mode = state?.mode;
  useEffect(() => {
    if (mode !== "review") setLens(null);
    if (mode && mode !== "finished") setStatsOpen(false);
  }, [mode]);

  // browsers only allow audio after a user gesture
  useEffect(() => {
    const unlock = () => sound.unlock();
    window.addEventListener("pointerdown", unlock, true);
    window.addEventListener("keydown", unlock, true);
    return () => {
      window.removeEventListener("pointerdown", unlock, true);
      window.removeEventListener("keydown", unlock, true);
    };
  }, []);

  const startGame = useCallback(
    (settings: GameSettings) => {
      send({ type: "new_game", ...settings });
    },
    [send],
  );

  const rematch = useCallback(() => {
    const st = stateRef.current;
    if (!st) return;
    startGame({
      players: st.game.players.map((p) => p.name),
      start: st.game.start,
      double_out: st.game.double_out,
      legs_to_win: st.game.legs_to_win,
    });
  }, [startGame]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const st = stateRef.current;
      // keys already handled by a dialog (e.g. Enter that starts a new game) must not act on the game too
      if (!st || setupOpen || e.defaultPrevented || (e.target as HTMLElement | null)?.tagName === "INPUT") return;
      const key = e.key.toLowerCase();
      const current = st.mode;
      if (key === "f") toggleFullscreen();
      else if (key === "g") setSetupOpen(true);
      else if (key === "m") sound.toggleSound();
      else if (key === "enter") {
        if (current === "calibrating") send({ type: "confirm_orientation" });
        else if (current === "review") send({ type: "confirm", save: true });
        else if (current === "finished") rematch();
      } else if (current === "finished") return;
      else if (key === "k" && current === "review") send({ type: "confirm", save: false });
      else if (key === "u") send({ type: "undo" });
      else if (key === " ") {
        e.preventDefault();
        if (current === "playing") send({ type: "start_review" });
      } else if (key === "escape") {
        if (lens) setLens(null);
        else if (current === "review") send({ type: "resume" });
      } else if (key === "r") send({ type: "recalibrate" });
      else if (key === "d") send({ type: "toggle_detections" });
      else if (key === "c") send({ type: "clear_ignored" });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [send, setupOpen, lens, rematch]);

  if (!state) {
    return (
      <div className="app-bg relative h-screen w-screen overflow-hidden">
        <Background />
        <Splash message={connected ? "Starting the engine…" : "Connecting to the recognition engine…"} />
      </div>
    );
  }

  return (
    <div className="app-bg relative h-screen w-screen overflow-hidden text-slate-100" onContextMenu={(e) => e.preventDefault()}>
      <Background />
      <div className="relative z-10 flex h-full flex-col gap-[1rem] px-[1.8rem] pb-[0.9rem] pt-[1.1rem]">
        <TopBar
          state={state}
          audio={audio}
          onNewGame={() => setSetupOpen(true)}
          onToggleSound={() => sound.toggleSound()}
        />
        <main className="flex min-h-0 flex-1 gap-[1.2rem]">
          <section className="min-w-0 flex-1">
            <CameraView state={state} send={send} lens={lens} setLens={setLens} streamKey={generation} />
          </section>
          <aside className="flex min-h-0 w-[35%] min-w-[30rem] max-w-[50rem] flex-col">
            <AnimatePresence mode="wait">
              <motion.div
                key={state.mode === "review" ? "review" : "game"}
                className="h-full min-h-0"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.18 }}
              >
                {state.mode === "review" ? <ReviewPanel state={state} send={send} /> : <GamePanel state={state} send={send} />}
              </motion.div>
            </AnimatePresence>
          </aside>
        </main>
        <HintBar mode={state.mode} />
      </div>

      <Toasts items={toasts} />
      <LayoutGroup>
        <EffectLayer effect={effect} onDone={onEffectDone} />
        <AnimatePresence>
          {statsOpen && state.summary && (
            <StatsScreen key="stats" state={state} onRematch={rematch} onNewGame={() => setSetupOpen(true)} />
          )}
        </AnimatePresence>
      </LayoutGroup>
      <SetupDialog
        open={setupOpen}
        state={state}
        audio={audio}
        onToggleSound={() => sound.toggleSound()}
        onClose={() => setSetupOpen(false)}
        onStart={(settings) => {
          startGame(settings);
          setSetupOpen(false);
        }}
      />
      <ConnectionBanner connected={connected} />
    </div>
  );
}
