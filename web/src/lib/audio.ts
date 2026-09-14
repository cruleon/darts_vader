/**
 * Sound effects synthesised on the fly with the Web Audio API (no audio files).
 */

export type Sfx =
  | "dart"
  | "double"
  | "triple"
  | "bull"
  | "miss"
  | "tick"
  | "review"
  | "confirm"
  | "remove"
  | "lock"
  | "start"
  | "bust"
  | "ton"
  | "180"
  | "win";

interface Bus {
  ctx: BaseAudioContext;
  out: AudioNode; // dry signal
  wet: AudioNode; // reverb send
  t: number; // start time
}

const hz = (midi: number) => 440 * 2 ** ((midi - 69) / 12);

/* ------------------------------------------------------------------ building blocks */

const noiseCache = new WeakMap<BaseAudioContext, AudioBuffer>();

function noiseBuffer(ctx: BaseAudioContext): AudioBuffer {
  let buf = noiseCache.get(ctx);
  if (!buf) {
    buf = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
    noiseCache.set(ctx, buf);
  }
  return buf;
}

function envelope(ctx: BaseAudioContext, t0: number, peak: number, attack: number, hold: number, decay: number): GainNode {
  const g = ctx.createGain();
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(peak, t0 + attack);
  if (hold > 0) g.gain.setValueAtTime(peak, t0 + attack + hold);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + attack + hold + decay);
  return g;
}

function route(b: Bus, node: AudioNode, wet = 0): void {
  node.connect(b.out);
  if (wet > 0) {
    const send = b.ctx.createGain();
    send.gain.value = wet;
    node.connect(send);
    send.connect(b.wet);
  }
}

interface ToneOpts {
  freq: number;
  to?: number; // glide towards this frequency
  type?: OscillatorType;
  at?: number;
  attack?: number;
  hold?: number;
  decay: number;
  gain: number;
  detune?: number;
  lowpass?: number;
  wet?: number;
}

function tone(b: Bus, o: ToneOpts): void {
  const { ctx } = b;
  const t0 = b.t + (o.at ?? 0);
  const attack = o.attack ?? 0.004;
  const hold = o.hold ?? 0;
  const osc = ctx.createOscillator();
  osc.type = o.type ?? "sine";
  osc.frequency.setValueAtTime(o.freq, t0);
  if (o.to) osc.frequency.exponentialRampToValueAtTime(o.to, t0 + attack + hold + o.decay * 0.8);
  if (o.detune) osc.detune.value = o.detune;
  const g = envelope(ctx, t0, o.gain, attack, hold, o.decay);
  if (o.lowpass) {
    const f = ctx.createBiquadFilter();
    f.type = "lowpass";
    f.frequency.value = o.lowpass;
    osc.connect(f).connect(g);
  } else {
    osc.connect(g);
  }
  route(b, g, o.wet);
  osc.start(t0);
  osc.stop(t0 + attack + hold + o.decay + 0.05);
}

interface NoiseOpts {
  freq: number;
  to?: number;
  filter?: BiquadFilterType;
  q?: number;
  at?: number;
  attack?: number;
  hold?: number;
  decay: number;
  gain: number;
  wet?: number;
}

function noise(b: Bus, o: NoiseOpts): void {
  const { ctx } = b;
  const t0 = b.t + (o.at ?? 0);
  const attack = o.attack ?? 0.002;
  const hold = o.hold ?? 0;
  const src = ctx.createBufferSource();
  src.buffer = noiseBuffer(ctx);
  src.loop = true;
  const f = ctx.createBiquadFilter();
  f.type = o.filter ?? "bandpass";
  f.frequency.setValueAtTime(o.freq, t0);
  if (o.to) f.frequency.exponentialRampToValueAtTime(o.to, t0 + attack + hold);
  f.Q.value = o.q ?? 0.8;
  const g = envelope(ctx, t0, o.gain, attack, hold, o.decay);
  src.connect(f).connect(g);
  route(b, g, o.wet);
  src.start(t0, Math.random() * 1.5);
  src.stop(t0 + attack + hold + o.decay + 0.05);
}

/** A dart hitting the board: sharp click plus a low thud. */
function impact(b: Bus): void {
  noise(b, { freq: 2800, q: 0.9, decay: 0.045, gain: 0.55 });
  noise(b, { filter: "lowpass", freq: 420, decay: 0.09, gain: 0.3 });
  tone(b, { freq: 180, to: 55, decay: 0.14, gain: 0.85 });
  tone(b, { type: "triangle", freq: 420, to: 220, decay: 0.06, gain: 0.18 });
}

/** Bell with inharmonic partials. */
function bell(b: Bus, f: number, at: number, gain: number): void {
  const partials: [number, number, number][] = [
    [1, 1, 1.1],
    [2.76, 0.35, 0.6],
    [5.4, 0.18, 0.35],
    [8.93, 0.08, 0.2],
  ];
  for (const [ratio, level, decay] of partials) tone(b, { freq: f * ratio, at, decay, gain: gain * level, wet: 0.7 });
}

/** Synth brass: two slightly detuned saw waves and a triangle, low-pass filtered. */
function brass(b: Bus, midi: number, at: number, hold: number, decay: number, gain: number): void {
  const freq = hz(midi);
  tone(b, { type: "sawtooth", freq, at, attack: 0.035, hold, decay, gain, detune: -7, lowpass: 2400, wet: 0.45 });
  tone(b, { type: "sawtooth", freq, at, attack: 0.035, hold, decay, gain, detune: 7, lowpass: 2400, wet: 0.45 });
  tone(b, { type: "triangle", freq, at, attack: 0.02, hold, decay, gain: gain * 1.2, wet: 0.3 });
}

function boom(b: Bus, at: number, gain: number): void {
  tone(b, { freq: 95, to: 32, at, decay: 0.9, gain });
  noise(b, { filter: "lowpass", freq: 260, at, decay: 0.5, gain: gain * 0.5 });
}

function crowd(b: Bus, at: number, hold: number, gain: number): void {
  noise(b, { freq: 1100, q: 0.45, at, attack: 0.5, hold, decay: 1.3, gain, wet: 0.35 });
  noise(b, { freq: 2600, q: 0.6, at: at + 0.1, attack: 0.6, hold, decay: 1.1, gain: gain * 0.45, wet: 0.35 });
}

function sparkles(b: Bus, at: number, count: number, spacing: number): void {
  const notes = [96, 100, 103, 108, 103, 105, 100, 107, 98, 104, 101, 106];
  for (let i = 0; i < count; i++) {
    tone(b, { type: "triangle", freq: hz(notes[i % notes.length]), at: at + i * spacing, decay: 0.45, gain: 0.07, wet: 0.8 });
  }
}

/* ------------------------------------------------------------------ catalogue */

const SFX: Record<Sfx, (b: Bus) => void> = {
  dart: (b) => impact(b),
  double: (b) => {
    impact(b);
    tone(b, { freq: hz(91), at: 0.03, decay: 0.35, gain: 0.16, wet: 0.5 });
    tone(b, { freq: hz(96), at: 0.09, decay: 0.45, gain: 0.13, wet: 0.6 });
  },
  triple: (b) => {
    impact(b);
    [88, 93, 100].forEach((m, i) => tone(b, { type: "triangle", freq: hz(m), at: 0.03 + i * 0.065, decay: 0.32, gain: 0.17, wet: 0.6 }));
  },
  bull: (b) => {
    impact(b);
    bell(b, hz(83), 0.03, 0.19);
    bell(b, hz(90), 0.16, 0.15);
  },
  miss: (b) => {
    tone(b, { freq: 120, to: 40, decay: 0.18, gain: 0.7 });
    noise(b, { filter: "lowpass", freq: 700, decay: 0.06, gain: 0.35 });
  },
  tick: (b) => {
    tone(b, { type: "triangle", freq: hz(88), decay: 0.07, gain: 0.26 });
    noise(b, { filter: "highpass", freq: 5000, decay: 0.025, gain: 0.14 });
  },
  review: (b) => {
    tone(b, { type: "triangle", freq: hz(76), decay: 0.25, gain: 0.18, wet: 0.4 });
    tone(b, { type: "triangle", freq: hz(83), at: 0.09, decay: 0.45, gain: 0.16, wet: 0.5 });
  },
  confirm: (b) => {
    [84, 88, 91, 96].forEach((m, i) => tone(b, { type: "triangle", freq: hz(m), at: i * 0.07, decay: 0.38, gain: 0.14, wet: 0.5 }));
    tone(b, { freq: hz(60), at: 0.21, decay: 0.3, gain: 0.25 });
  },
  remove: (b) => {
    tone(b, { freq: 760, to: 240, decay: 0.16, gain: 0.45 });
    tone(b, { type: "triangle", freq: hz(76), to: hz(64), at: 0.02, decay: 0.12, gain: 0.12 });
  },
  lock: (b) => {
    noise(b, { freq: 400, to: 4200, q: 1.2, attack: 0.25, decay: 0.12, gain: 0.12 });
    tone(b, { type: "triangle", freq: hz(86), at: 0.27, decay: 0.3, gain: 0.15, wet: 0.5 });
    tone(b, { type: "triangle", freq: hz(93), at: 0.34, decay: 0.55, gain: 0.14, wet: 0.6 });
  },
  start: (b) => {
    noise(b, { freq: 200, to: 6000, q: 0.7, attack: 0.35, decay: 0.3, gain: 0.16, wet: 0.3 });
    [48, 55, 60, 64, 67].forEach((m) => brass(b, m, 0.33, 0.25, 0.7, 0.035));
    boom(b, 0.33, 0.55);
  },
  bust: (b) => {
    noise(b, { filter: "lowpass", freq: 500, decay: 0.15, gain: 0.3 });
    tone(b, { type: "sawtooth", freq: hz(57), to: hz(55), attack: 0.02, hold: 0.16, decay: 0.12, gain: 0.2, lowpass: 900, wet: 0.2 });
    tone(b, { type: "square", freq: hz(45), to: hz(43), attack: 0.02, hold: 0.16, decay: 0.12, gain: 0.08, lowpass: 700 });
    tone(b, { type: "sawtooth", freq: hz(54), to: hz(50), at: 0.33, attack: 0.02, hold: 0.3, decay: 0.35, gain: 0.2, lowpass: 750, wet: 0.25 });
    tone(b, { type: "square", freq: hz(42), to: hz(38), at: 0.33, attack: 0.02, hold: 0.3, decay: 0.35, gain: 0.08, lowpass: 600 });
  },
  ton: (b) => {
    [55, 60, 64].forEach((m, i) => brass(b, m, i * 0.12, 0.06, 0.12, 0.05));
    [55, 60, 64, 67].forEach((m) => brass(b, m, 0.36, 0.35, 0.8, 0.04));
    boom(b, 0.36, 0.6);
    sparkles(b, 0.4, 5, 0.06);
  },
  "180": (b) => {
    boom(b, 0, 1);
    crowd(b, 0.05, 1.3, 0.2);
    [48, 55, 60, 64, 67, 72].forEach((m) => brass(b, m, 0.05, 0.9, 1.3, 0.032));
    [84, 88, 91, 96, 100, 103].forEach((m, i) => tone(b, { type: "triangle", freq: hz(m), at: 0.12 + i * 0.07, decay: 0.55, gain: 0.09, wet: 0.7 }));
    sparkles(b, 0.6, 10, 0.11);
    boom(b, 1.05, 0.5);
  },
  win: (b) => {
    [60, 60, 60].forEach((m, i) => brass(b, m, i * 0.13, 0.05, 0.08, 0.05));
    [60, 64, 67, 72].forEach((m) => brass(b, m, 0.42, 0.7, 1.4, 0.035));
    boom(b, 0.42, 0.9);
    crowd(b, 0.45, 1.8, 0.22);
    sparkles(b, 0.5, 18, 0.1);
    boom(b, 1.4, 0.45);
  },
};

export const SFX_NAMES = Object.keys(SFX) as Sfx[];

/** Shared output chain: dry signal + reverb -> compressor -> destination. */
function buildGraph(ctx: BaseAudioContext): { out: AudioNode; wet: AudioNode } {
  const master = ctx.createGain();
  master.gain.value = 0.8;
  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = -12;
  comp.knee.value = 8;
  comp.ratio.value = 5;
  comp.attack.value = 0.003;
  comp.release.value = 0.2;
  master.connect(comp).connect(ctx.destination);

  const seconds = 1.8;
  const impulse = ctx.createBuffer(2, Math.round(ctx.sampleRate * seconds), ctx.sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const data = impulse.getChannelData(ch);
    for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / data.length) ** 2.6;
  }
  const reverb = ctx.createConvolver();
  reverb.buffer = impulse;
  const wetReturn = ctx.createGain();
  wetReturn.gain.value = 0.3;
  const wet = ctx.createGain();
  wet.connect(reverb).connect(wetReturn).connect(master);
  return { out: master, wet };
}

/** Render an effect offline and measure its peak, RMS and length (used by tests). */
export async function renderSfx(name: Sfx, seconds = 4): Promise<{ peak: number; rms: number; tail: number }> {
  const rate = 48000;
  const ctx = new OfflineAudioContext(2, rate * seconds, rate);
  const graph = buildGraph(ctx);
  SFX[name]({ ctx, ...graph, t: 0.01 });
  const buf = await ctx.startRendering();
  let peak = 0;
  let sum = 0;
  let last = 0;
  for (let ch = 0; ch < buf.numberOfChannels; ch++) {
    const data = buf.getChannelData(ch);
    for (let i = 0; i < data.length; i++) {
      const v = Math.abs(data[i]);
      if (v > peak) peak = v;
      if (v > 0.002 && i > last) last = i;
      sum += v * v;
    }
  }
  return { peak, rms: Math.sqrt(sum / (buf.length * buf.numberOfChannels)), tail: last / rate };
}

/* ------------------------------------------------------------------ audio engine */

export interface AudioSnapshot {
  sound: boolean;
}

const STORE_KEY = "darts-vader-audio";

function loadSettings(): AudioSnapshot {
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    if (raw) return { sound: (JSON.parse(raw) as { sound?: unknown }).sound !== false };
  } catch {
    /* storage unavailable: defaults */
  }
  return { sound: true };
}

class SoundEngine {
  private ctx: AudioContext | null = null;
  private graph: { out: AudioNode; wet: AudioNode } | null = null;
  private listeners = new Set<() => void>();
  private snapshot: AudioSnapshot = loadSettings();
  readonly log: string[] = [];

  subscribe = (fn: () => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  getSnapshot = () => this.snapshot;

  toggleSound(): void {
    this.snapshot = { sound: !this.snapshot.sound };
    try {
      window.localStorage.setItem(STORE_KEY, JSON.stringify(this.snapshot));
    } catch {
      /* storage unavailable */
    }
    this.listeners.forEach((fn) => fn());
    if (this.snapshot.sound) this.play("tick");
  }

  private record(entry: string): void {
    this.log.push(entry);
    if (this.log.length > 200) this.log.splice(0, this.log.length - 200);
  }

  private context(): AudioContext | null {
    if (!this.ctx) {
      try {
        this.ctx = new AudioContext({ latencyHint: "interactive" });
        this.graph = buildGraph(this.ctx);
      } catch {
        return null;
      }
    }
    return this.ctx;
  }

  /** Call on every user interaction: browsers only allow audio after a user gesture. */
  unlock(): void {
    if (!this.snapshot.sound) return;
    const ctx = this.context();
    if (ctx && ctx.state === "suspended") void ctx.resume().catch(() => undefined);
  }

  play(name: Sfx): void {
    this.record(`sfx:${name}`);
    if (!this.snapshot.sound) return;
    const ctx = this.context();
    if (!ctx || !this.graph) return;
    if (ctx.state === "suspended") void ctx.resume().catch(() => undefined);
    try {
      SFX[name]({ ctx, ...this.graph, t: ctx.currentTime + 0.01 });
    } catch {
      /* a failed sound must never break the app */
    }
  }
}

export const sound = new SoundEngine();

// exposed for the end-to-end tests
(window as unknown as { __dartsAudio: unknown }).__dartsAudio = { log: sound.log, render: renderSfx, names: SFX_NAMES };
