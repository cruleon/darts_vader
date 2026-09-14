import { useCallback, useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";
import type { Command, EngineState } from "../types";

/** WebSocket connection to the Python engine, with automatic reconnection. */
export function useEngine() {
  const [state, setState] = useState<EngineState | null>(null);
  const [connected, setConnected] = useState(false);
  const [generation, setGeneration] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let disposed = false;
    let retry: number | undefined;
    let delay = 500;

    const connect = () => {
      const proto = window.location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${window.location.host}/ws`);
      wsRef.current = ws;
      ws.onopen = () => {
        delay = 500;
        setConnected(true);
        setGeneration((g) => g + 1);
      };
      ws.onmessage = (event) => {
        try {
          setState(JSON.parse(event.data as string) as EngineState);
        } catch {
          /* malformed message: ignored */
        }
      };
      ws.onclose = () => {
        setConnected(false);
        if (!disposed) {
          retry = window.setTimeout(connect, delay);
          delay = Math.min(delay * 2, 4000);
        }
      };
      ws.onerror = () => ws.close();
    };

    connect();
    return () => {
      disposed = true;
      window.clearTimeout(retry);
      wsRef.current?.close();
    };
  }, []);

  const send = useCallback((command: Command) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(command));
  }, []);

  return { state, connected, generation, send };
}

/** Size of an element, updated whenever it is resized. */
export function useElementSize<T extends HTMLElement>(): [RefObject<T | null>, { width: number; height: number }] {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setSize((s) => (s.width === width && s.height === height ? s : { width, height }));
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, size];
}

export function toggleFullscreen(): void {
  if (document.fullscreenElement) void document.exitFullscreen();
  else void document.documentElement.requestFullscreen().catch(() => undefined);
}
