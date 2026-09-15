export type Mode = "searching" | "calibrating" | "playing" | "review" | "finished";
export type Tone = "info" | "success" | "warning" | "danger" | "violet" | "gold";
export type Mat3 = number[][];
export type Vec2 = [number, number];

export interface Dart {
  index: number;
  label: string;
  score: number;
  number: number;
  multiplier: number;
  tip_mm: Vec2;
  tip_img: Vec2 | null;
  origin: "model" | "corrected" | "click" | "approx" | string;
  confidence: number;
  /** Double-out games: distance from the double that would have finished the leg with this dart. */
  finish: { target: string; distance_mm: number; point_mm: Vec2 } | null;
}

export interface Player {
  name: string;
  score: number;
  legs: number;
  average: number;
  photo: string | null;
}

export interface HistoryItem {
  player: string;
  darts: string[];
  points: number;
  outcome: "ok" | "bust" | "win";
}

export interface PlayerStats {
  index: number;
  name: string;
  legs_won: number;
  turns: number;
  darts_thrown: number;
  points: number;
  average: number;
  first9_average: number;
  highest_turn: number;
  scores_180: number;
  scores_140: number;
  scores_100: number;
  scores_60: number;
  busts: number;
  highest_checkout: number | null;
  best_leg_darts: number | null;
  darts_at_double: number | null;
  checkout_pct: number | null;
  trebles: number;
  doubles: number;
  bulls: number;
  misses: number;
  singles: number;
  treble_pct: number | null;
  double_pct: number | null;
  miss_pct: number | null;
  favourite: string | null;
  favourite_hits: number;
  avg_finish_distance_mm: number | null;
  best_finish_distance_mm: number | null;
  centroid_mm: Vec2 | null;
  grouping_mm: number | null;
  darts: { x: number; y: number; label: string; score: number }[];
}

export interface MatchSummary {
  start: number;
  double_out: boolean;
  legs_to_win: number;
  winner: number;
  legs_played: number;
  players: PlayerStats[];
}

export type EngineEvent =
  | { id: number; kind: "toast"; text: string; tone: Tone }
  | { id: number; kind: "effect"; effect: EffectKind; text: string }
  | { id: number; kind: "dart"; index: number; label: string; number: number; multiplier: number; origin: string }
  | { id: number; kind: "removed"; label: string }
  | { id: number; kind: "moved"; index: number; label: string }
  | { id: number; kind: "review"; reason: string }
  | { id: number; kind: "turn"; player: string; darts: string[]; points: number; remaining: number; outcome: string }
  | { id: number; kind: "game_over"; winner: string }
  | { id: number; kind: "board_found" }
  | { id: number; kind: "set20" }
  | { id: number; kind: "game_start" }
  | { id: number; kind: "new_game" };

export type EffectKind = "180" | "bust" | "win" | "ton";

export interface EngineState {
  mode: Mode;
  frame: { w: number; h: number } | null;
  board: { H_inv: Mat3; rings: number[] } | null;
  game: {
    start: number;
    double_out: boolean;
    legs_to_win: number;
    current: number;
    players: Player[];
    history: HistoryItem[];
  };
  turn: Dart[];
  turn_total: number;
  remaining_after: number;
  bust: boolean;
  checkout: string | null;
  /** Double that finishes the leg with the next dart (double-out games only). */
  finish_target: string | null;
  waiting: boolean;
  board_lost: boolean;
  ignored: Vec2[];
  detections: Vec2[];
  show_detections: boolean;
  review: { id: number; reason: string; w: number; h: number } | null;
  /** Match statistics, available once the match is over. */
  summary: MatchSummary | null;
  fps_ai: number;
  saved: number;
  events: EngineEvent[];
  source_error: string | null;
}

export interface GameSettings {
  players: string[];
  start: number;
  double_out: boolean;
  legs_to_win: number;
}

export type Command =
  | { type: "set20"; x: number; y: number }
  | { type: "confirm_orientation" }
  | { type: "start_review" }
  | { type: "resume" }
  | { type: "confirm"; save: boolean }
  | { type: "undo" }
  | { type: "remove"; index: number }
  | { type: "move"; index: number; x: number; y: number }
  | { type: "add"; x: number; y: number }
  | { type: "add_sim"; x_mm: number; y_mm: number }
  | { type: "clear_ignored" }
  | { type: "toggle_detections" }
  | { type: "recalibrate" }
  | ({ type: "new_game" } & GameSettings);

export type Send = (command: Command) => void;
