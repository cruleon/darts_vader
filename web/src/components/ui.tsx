import { animate, motion, useMotionValue, useTransform } from "framer-motion";
import { useEffect, type ReactNode } from "react";

/** Number that animates towards its new value. */
export function AnimatedNumber({ value, className }: { value: number; className?: string }) {
  const mv = useMotionValue(value);
  const text = useTransform(mv, (v) => Math.round(v).toString());
  useEffect(() => {
    const controls = animate(mv, value, { duration: 0.9, ease: [0.16, 1, 0.3, 1] });
    return () => controls.stop();
  }, [mv, value]);
  return <motion.span className={className}>{text}</motion.span>;
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="kbd">{children}</kbd>;
}

/** Dark helmet logo with a neon outline and visor. */
export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 64" className={className} aria-hidden>
      <defs>
        <linearGradient id="logo-neon" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#22d3ee" />
          <stop offset="0.55" stopColor="#a78bfa" />
          <stop offset="1" stopColor="#f472b6" />
        </linearGradient>
        <linearGradient id="logo-shell" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#1e293b" />
          <stop offset="1" stopColor="#070b18" />
        </linearGradient>
      </defs>
      <path
        d="M32 5C19.5 5 12 14.5 12 27v9.5L6.5 50 3 58h19l4-6h12l4 6h19l-3.5-8L52 36.5V27C52 14.5 44.5 5 32 5Z"
        fill="url(#logo-shell)"
        stroke="url(#logo-neon)"
        strokeWidth="2.4"
        strokeLinejoin="round"
      />
      <path d="M14 26.5c5-2.2 11.5-3.3 18-3.3s13 1.1 18 3.3" fill="none" stroke="url(#logo-neon)" strokeWidth="1.6" opacity="0.8" />
      <path d="M21.5 11.8c3-2.3 6.6-3.4 10.5-3.4" fill="none" stroke="#e2e8f0" strokeWidth="1.6" strokeLinecap="round" opacity="0.35" />
      <path d="M16.5 30.5 29.5 32l-1.8 6.5-9.2-1.8z" fill="url(#logo-neon)" />
      <path d="M47.5 30.5 34.5 32l1.8 6.5 9.2-1.8z" fill="url(#logo-neon)" />
      <path d="M32 34l5.5 14h-11z" fill="#0b1122" stroke="url(#logo-neon)" strokeWidth="1" strokeLinejoin="round" opacity="0.9" />
      <path d="M29.6 44.2h4.8M28.6 46.6h6.8" stroke="#475569" strokeWidth="1" />
      <path d="M22 40l-6 9M42 40l6 9" stroke="url(#logo-neon)" strokeWidth="1.4" strokeLinecap="round" opacity="0.55" />
    </svg>
  );
}

/** Flying dart, tip pointing left. */
export function DartGlyph({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 100 24" className={className} aria-hidden>
      <path d="M1 12 15 10.6v2.8z" fill="#cbd5e1" />
      <rect x="15" y="9.3" width="30" height="5.4" rx="1.6" fill="#e2e8f0" />
      <path d="M21 9.3v5.4M25 9.3v5.4M29 9.3v5.4M33 9.3v5.4" stroke="#94a3b8" strokeWidth="1" />
      <rect x="45" y="10.8" width="25" height="2.4" rx="1" fill="#64748b" />
      <path d="M68 12 90 2.5l8 1.5-14 8 14 8-8 1.5z" fill="url(#logo-neon)" />
    </svg>
  );
}

export function Background() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <div className="aurora" style={{ width: "46rem", height: "46rem", left: "-12rem", top: "-18rem", background: "rgb(34 211 238 / 0.16)" }} />
      <div
        className="aurora"
        style={{ width: "52rem", height: "52rem", right: "-16rem", bottom: "-24rem", background: "rgb(244 114 182 / 0.13)", animationDelay: "-6s" }}
      />
      <div
        className="aurora"
        style={{ width: "34rem", height: "34rem", left: "48%", top: "-14rem", background: "rgb(167 139 250 / 0.09)", animationDelay: "-11s" }}
      />
      <div className="grid-overlay" />
    </div>
  );
}
