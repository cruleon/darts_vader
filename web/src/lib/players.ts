/** Player colours and photo helpers. */

const TWO_PLAYERS = ["#3b82f6", "#ef4444"];
const PALETTE = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#a855f7", "#ec4899"];

/** Colour of a player: blue and red for two players, a distinct palette otherwise. */
export function playerColor(index: number, count: number): string {
  return count === 2 ? TWO_PLAYERS[index % 2] : PALETTE[index % PALETTE.length];
}

/** Centre square of an image source as a JPEG data URL. */
export function squareDataUrl(source: CanvasImageSource, width: number, height: number, size = 512): string {
  const side = Math.min(width, height);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  canvas.getContext("2d")!.drawImage(source, (width - side) / 2, (height - side) / 2, side, side, 0, 0, size, size);
  return canvas.toDataURL("image/jpeg", 0.9);
}

export async function fileToSquareDataUrl(file: File): Promise<string> {
  const bitmap = await createImageBitmap(file);
  try {
    return squareDataUrl(bitmap, bitmap.width, bitmap.height);
  } finally {
    bitmap.close();
  }
}

/** Store (or, with `image = null`, remove) the photo of a player. Returns the new photo URL. */
export async function savePlayerPhoto(name: string, image: string | null): Promise<string | null> {
  const response = await fetch("/api/player-photo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, image }),
  });
  const body = (await response.json().catch(() => ({}))) as { url?: string | null; error?: string };
  if (!response.ok) throw new Error(body.error ?? `photo upload failed (${response.status})`);
  return body.url ?? null;
}

export async function fetchPlayerPhoto(name: string): Promise<string | null> {
  const response = await fetch(`/api/player-photo-url?name=${encodeURIComponent(name)}`);
  return response.ok ? (((await response.json()) as { url: string | null }).url ?? null) : null;
}
