import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { fileToSquareDataUrl, savePlayerPhoto, squareDataUrl } from "../lib/players";
import { Kbd } from "./ui";

/** Round player picture, or initials on the player colour when there is no photo. */
export function Avatar({ name, photo, color, className = "" }: { name: string; photo: string | null; color: string; className?: string }) {
  const initials =
    name
      .split(/\s+/)
      .filter(Boolean)
      .map((word) => word[0])
      .join("")
      .slice(0, 2)
      .toUpperCase() || "?";
  return (
    <span
      className={`relative grid shrink-0 place-items-center overflow-hidden rounded-full font-display font-bold text-white ${className}`}
      style={{
        background: photo ? "#0b1122" : `linear-gradient(135deg, ${color}, ${color}55)`,
        boxShadow: `0 0 0 0.16rem ${color}, 0 0 1.4rem -0.3rem ${color}`,
      }}
    >
      {photo ? <img src={photo} alt="" draggable={false} className="h-full w-full object-cover" /> : initials}
    </span>
  );
}

const iconProps = { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" } as const;

/** Photo controls of a player row in the new game dialog: upload, camera capture, removal. */
export function PhotoPicker({ name, photo, color, onChange }: { name: string; photo: string | null; color: string; onChange: (url: string | null) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const player = name.trim();

  const store = async (image: string | null) => {
    try {
      setError(null);
      onChange(await savePlayerPhoto(player, image));
    } catch (err) {
      setError(err instanceof Error ? err.message : "photo not saved");
    }
  };

  return (
    <div className="relative flex shrink-0 items-center gap-[0.35rem]">
      <button
        type="button"
        title={player ? "Upload a photo" : "Type a name first"}
        disabled={!player}
        onClick={() => input.current?.click()}
        className="group relative rounded-full disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Avatar name={player || "?"} photo={photo} color={color} className="h-[3.2rem] w-[3.2rem] text-[1.25rem]" />
        <span className="absolute inset-0 grid place-items-center rounded-full bg-black/60 text-white opacity-0 transition-opacity group-hover:opacity-100">
          <svg {...iconProps} className="h-[1.3rem] w-[1.3rem]">
            <path d="M12 16V4M7 9l5-5 5 5M5 20h14" />
          </svg>
        </span>
      </button>
      <button
        type="button"
        title={player ? "Take a photo with a camera" : "Type a name first"}
        disabled={!player}
        onClick={() => setCameraOpen(true)}
        className="btn btn-ghost !h-[3.2rem] w-[3.2rem] !p-0 disabled:cursor-not-allowed disabled:opacity-40"
      >
        <svg {...iconProps} className="h-[1.35rem] w-[1.35rem]">
          <path d="M4 8h3l2-3h6l2 3h3v11H4z" />
          <circle cx="12" cy="13" r="3.5" />
        </svg>
      </button>
      {photo && (
        <button type="button" title="Remove the photo" onClick={() => void store(null)} className="btn btn-ghost !h-[3.2rem] w-[2.6rem] !p-0 text-slate-400">
          <svg {...iconProps} className="h-[1.2rem] w-[1.2rem]">
            <path d="M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13" />
          </svg>
        </button>
      )}
      <input
        ref={input}
        type="file"
        accept="image/*"
        hidden
        onChange={async (e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) await store(await fileToSquareDataUrl(file));
        }}
      />
      {error && <span className="absolute left-0 top-full z-10 mt-[0.2rem] whitespace-nowrap text-[0.8rem] text-rose-300">{error}</span>}
      <AnimatePresence>
        {cameraOpen && (
          <CameraCapture
            name={player}
            onCancel={() => setCameraOpen(false)}
            onCapture={async (image) => {
              setCameraOpen(false);
              await store(image);
            }}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

/** Take a player photo with any camera the browser can access. */
function CameraCapture({ name, onCapture, onCancel }: { name: string; onCapture: (image: string) => void; onCancel: () => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState("");

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;
    setReady(false);
    (async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser cannot access cameras here (use localhost or HTTPS).");
        stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId ? { deviceId: { exact: deviceId } } : { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (cancelled) return;
        if (video.current) {
          video.current.srcObject = stream;
          await video.current.play().catch(() => undefined);
        }
        setError(null);
        setReady(true);
        const all = await navigator.mediaDevices.enumerateDevices();
        if (!cancelled) setDevices(all.filter((d) => d.kind === "videoinput"));
      } catch (err) {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : String(err);
          setError(`${message}. The board webcam may be busy: pick another camera or upload a photo.`);
        }
      }
    })();
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, [deviceId]);

  const capture = () => {
    const v = video.current;
    if (v && v.videoWidth) onCapture(squareDataUrl(v, v.videoWidth, v.videoHeight));
  };

  const onKeyDown = (e: KeyboardEvent) => {
    // keep Enter/Escape away from the new game dialog underneath
    e.stopPropagation();
    if (e.key === "Enter") {
      e.preventDefault();
      capture();
    } else if (e.key === "Escape") {
      e.preventDefault();
      onCancel();
    }
  };

  return (
    <motion.div
      className="fixed inset-0 z-[80] grid place-items-center bg-ink/80 backdrop-blur-sm"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onKeyDown={onKeyDown}
      tabIndex={-1}
      ref={(el) => el?.focus()}
    >
      <motion.div
        className="glass w-[36rem] px-[2rem] pb-[1.8rem] pt-[1.6rem]"
        initial={{ scale: 0.9, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.95, y: 10 }}
        transition={{ type: "spring", stiffness: 320, damping: 26 }}
      >
        <div className="label">Player photo</div>
        <h3 className="font-display text-[2.4rem] font-bold leading-none text-white">{name}</h3>
        <div className="relative mt-[1rem] aspect-square w-full overflow-hidden rounded-[1.2rem] bg-black/60">
          <video ref={video} muted playsInline className="h-full w-full scale-x-[-1] object-cover" />
          {!ready && !error && <div className="absolute inset-0 grid place-items-center text-slate-400">Starting camera…</div>}
          {error && <div className="absolute inset-0 grid place-items-center px-[2rem] text-center text-[1rem] text-rose-300">{error}</div>}
          <div className="pointer-events-none absolute inset-[12%] rounded-full border-2 border-dashed border-white/40" />
        </div>
        {devices.length > 1 && (
          <select className="field mt-[0.8rem]" value={deviceId} onChange={(e) => setDeviceId(e.target.value)}>
            <option value="">Default camera</option>
            {devices.map((d, i) => (
              <option key={d.deviceId || i} value={d.deviceId}>
                {d.label || `Camera ${i + 1}`}
              </option>
            ))}
          </select>
        )}
        <div className="mt-[1rem] grid grid-cols-3 gap-[0.7rem]">
          <button type="button" className="btn btn-ghost" onClick={onCancel}>
            <Kbd>ESC</Kbd> Cancel
          </button>
          <button type="button" className="btn btn-primary col-span-2" disabled={!ready} style={{ opacity: ready ? 1 : 0.5 }} onClick={capture}>
            <Kbd>ENTER</Kbd> Take photo
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}
