import type { CSSProperties } from "react";
import type { Mode, Tone } from "../types";

export const CYAN = "#22d3ee";
export const VIOLET = "#a78bfa";
export const PINK = "#f472b6";
export const AMBER = "#fbbf24";
export const EMERALD = "#34d399";
export const ROSE = "#fb7185";

export const DART_COLORS = [CYAN, PINK, AMBER] as const;

export const TONE_COLORS: Record<Tone, string> = {
  info: CYAN,
  success: EMERALD,
  warning: AMBER,
  danger: ROSE,
  violet: VIOLET,
  gold: "#ffd166",
};

export const MODE_STYLE: Record<Mode, { label: string; color: string }> = {
  searching: { label: "Scanning", color: VIOLET },
  calibrating: { label: "Calibration", color: VIOLET },
  playing: { label: "Live", color: EMERALD },
  review: { label: "Review", color: AMBER },
  finished: { label: "Game over", color: "#ffd166" },
};

export const ORIGIN_STYLE: Record<string, { label: string; color: string }> = {
  model: { label: "model", color: "#94a3b8" },
  corrected: { label: "corrected", color: EMERALD },
  click: { label: "added", color: EMERALD },
  approx: { label: "approx.", color: AMBER },
};

/** CSS variable --c used by coloured components. */
export function colorVar(color: string): CSSProperties {
  return { "--c": color } as CSSProperties;
}
