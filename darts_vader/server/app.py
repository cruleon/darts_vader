"""DARTS VADER web server: FastAPI backend for the React frontend.

    python -m darts_vader                                         # webcam 0, opens the browser
    python -m darts_vader --players Leo,Marco --start 501 --double-out
    python -m darts_vader --kiosk                                 # Microsoft Edge fullscreen, like an app
    python -m darts_vader --source turn.jpg --no-browser          # try it without a webcam

Endpoints:
  GET  /api/stream.mjpg   live camera video (MJPEG)
  GET  /api/review.jpg    still photo of the turn under review (full resolution)
  GET  /api/state         current engine state (JSON)
  WS   /ws                engine state about 20 times per second; commands from the frontend
  GET  /                  built frontend (web/dist)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import threading
import time
import traceback
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..camera import BACKENDS, FrameSource
from .engine import PROJECT_ROOT, REVIEW, START_SCORES, GameEngine

FRONTEND_DIST = PROJECT_ROOT / "web" / "dist"
PREVIEW_WIDTH = 1920  # the MJPEG preview is downscaled to this width
STATE_HZ = 20
STREAM_FPS = 30


def encode_preview(frame: np.ndarray) -> bytes:
    h, w = frame.shape[:2]
    if w > PREVIEW_WIDTH:
        frame = cv2.resize(frame, (PREVIEW_WIDTH, round(h * PREVIEW_WIDTH / w)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpg.tobytes() if ok else b""


def create_app(engine: GameEngine, source: FrameSource) -> FastAPI:
    stop = threading.Event()

    def analysis_loop() -> None:
        seq, t_last = 0, time.perf_counter()
        while not stop.is_set():
            item = source.latest(seq, 1.0)
            if item is None:
                continue
            frame, seq = item
            analysing = engine.mode != REVIEW  # detection is paused while a turn is reviewed
            try:
                engine.process_frame(frame, time.monotonic())
            except Exception:
                traceback.print_exc()
            t = time.perf_counter()
            if analysing:
                engine.fps_ai = 0.85 * engine.fps_ai + 0.15 / max(t - t_last, 1e-6)
            t_last = t

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        thread = threading.Thread(target=analysis_loop, name="analysis", daemon=True)
        thread.start()
        yield
        stop.set()
        source.stop()
        thread.join(timeout=2.0)

    app = FastAPI(title="DARTS VADER", lifespan=lifespan)

    def current_state() -> dict:
        state = engine.state()
        state["source_error"] = source.error
        return state

    @app.get("/api/state")
    async def get_state():
        return JSONResponse(await asyncio.to_thread(current_state))

    @app.get("/api/stream.mjpg")
    async def stream(request: Request):
        async def frames():
            seq = 0
            while not await request.is_disconnected():
                item = await asyncio.to_thread(source.latest, seq, 1.0)
                if item is None:
                    continue
                frame, seq = item
                jpg = await asyncio.to_thread(encode_preview, frame)
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(1 / STREAM_FPS)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/api/review.jpg")
    async def review_image():
        data = engine.review_jpeg
        if data is None:
            return Response(status_code=404)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        await ws.accept()

        async def sender():
            while True:
                state = await asyncio.to_thread(current_state)
                await ws.send_text(json.dumps(state, separators=(",", ":")))
                await asyncio.sleep(1 / STATE_HZ)

        task = asyncio.create_task(sender())
        try:
            while True:
                msg = await ws.receive_json()
                if isinstance(msg, dict):
                    try:
                        await asyncio.to_thread(engine.command, msg)
                    except Exception:
                        traceback.print_exc()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            task.cancel()

    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="web")
    else:
        @app.get("/")
        async def frontend_missing():
            return HTMLResponse("<h2>Frontend not built</h2><p>Run <code>npm install</code> and "
                                "<code>npm run build</code> in the <code>web</code> folder.</p>")
    return app


def open_browser(url: str, kiosk: bool) -> None:
    time.sleep(2.0)
    if kiosk:
        candidates = [shutil.which("msedge"), r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                      r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
        edge = next((c for c in candidates if c and Path(c).exists()), None)
        if edge:
            subprocess.Popen([edge, f"--app={url}", "--start-fullscreen", "--new-window", "--no-first-run",
                              "--autoplay-policy=no-user-gesture-required"])
            return
    webbrowser.open(url)


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m darts_vader", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    game = ap.add_argument_group("game")
    game.add_argument("--players", default="Player", help="comma-separated player names")
    game.add_argument("--start", type=int, default=301, choices=START_SCORES)
    game.add_argument("--double-out", action="store_true")
    camera = ap.add_argument_group("camera")
    camera.add_argument("--source", default="0", help="webcam index, stream URL, video file, image, or image folder/glob")
    camera.add_argument("--backend", choices=tuple(BACKENDS), default="msmf", help="OpenCV capture backend for webcams")
    camera.add_argument("--capture", default="3264x2448", help="resolution requested from the webcam")
    camera.add_argument("--hold", type=float, default=6.0, help="seconds per image when the source is images")
    model = ap.add_argument_group("model and data")
    model.add_argument("--model", default="models/tipnet.pt", help="TipNet checkpoint (see tools/train_tipnet.py)")
    model.add_argument("--threshold", type=float, default=0.4, help="minimum tip confidence")
    model.add_argument("--labels", default="webcam_labels", help="where confirmed turns are saved as training data")
    server = ap.add_argument_group("server")
    server.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open the app from a tablet or phone")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--no-browser", action="store_true")
    server.add_argument("--kiosk", action="store_true", help="open Microsoft Edge fullscreen, like an app")
    server.add_argument("--debug", action="store_true", help="enable test commands and never overwrite the saved calibration")
    args = ap.parse_args()

    model_path = resolve(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model weights not found: {model_path}\n"
                         "Train a model with tools/train_tipnet.py or pass --model (see README).")

    import torch

    from ..tips.model import load_tipnet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tip_model, view = load_tipnet(model_path, device)
    print(f"model {model_path.name} ({view} view, {device})")
    players = [p.strip() for p in args.players.split(",") if p.strip()]
    engine = GameEngine(tip_model, view, device, players, args.start, args.double_out, resolve(args.labels),
                        args.threshold, save_calibration=not args.debug, debug=args.debug)
    source = FrameSource(args.source, args.backend, args.capture, hold=args.hold)
    app = create_app(engine, source)

    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print(f"DARTS VADER running on {url}")
    if not args.no_browser:
        threading.Thread(target=open_browser, args=(url, args.kiosk), daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    saved = engine.labels.count()
    print(f"saved {saved} labelled turns" + (f" in {engine.session_dir}" if saved else ""))
