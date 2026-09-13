"""Esegue la pipeline su un video registrato.

Allo stato attuale implementa il Livello 1 (input + calibrazione ArUco):
per ogni frame rileva i marker, calcola l'homography e produce il
bersaglio raddrizzato. I livelli successivi (detection, scoring,
persistenza, dashboard) verranno agganciati qui via via che vengono
costruiti, senza cambiare l'interfaccia di questo script.

Uso:
    uv run python scripts/run_on_video.py --video data/videos/synthetic_calibration.mp4
    uv run python scripts/run_on_video.py --video data/videos/synthetic_calibration.mp4 \
        --output-rectified data/videos/out_rectified.mp4 \
        --output-annotated data/videos/out_annotated.mp4
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from dartvision.calibration.homography import Calibrator
from dartvision.config import load_config
from dartvision.input.frame_source import VideoFileSource


def draw_markers(frame: np.ndarray, detected) -> np.ndarray:
    annotated = frame.copy()
    if not detected.ids:
        return annotated
    corners_list = [
        c.reshape(1, 4, 2).astype(np.float32) for c in detected.corners_by_id.values()
    ]
    ids_arr = np.array(detected.ids, dtype=np.int32).reshape(-1, 1)
    cv2.aruco.drawDetectedMarkers(annotated, corners_list, ids_arr)
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/board_config.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-rectified", default=None)
    parser.add_argument("--output-annotated", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    calibrator = Calibrator(config)

    total = 0
    successes = 0
    markers_detected_sum = 0

    writer_rectified = None
    writer_annotated = None

    with VideoFileSource(args.video) as source:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = source.fps or 30.0

        for frame in source.frames():
            if args.max_frames is not None and total >= args.max_frames:
                break
            total += 1

            detected = calibrator.detect_markers(frame)
            result = calibrator.calibrate_from_detection(detected)
            markers_detected_sum += result.num_markers_detected

            if result.success:
                successes += 1
                rectified = calibrator.rectify(frame, result.homography)
                if args.output_rectified:
                    if writer_rectified is None:
                        h, w = rectified.shape[:2]
                        writer_rectified = cv2.VideoWriter(
                            args.output_rectified, fourcc, fps, (w, h)
                        )
                    writer_rectified.write(rectified)
            else:
                print(f"[frame {total}] calibrazione fallita: {result.message}")

            if args.output_annotated:
                annotated = draw_markers(frame, detected)
                if writer_annotated is None:
                    h, w = annotated.shape[:2]
                    writer_annotated = cv2.VideoWriter(
                        args.output_annotated, fourcc, fps, (w, h)
                    )
                writer_annotated.write(annotated)

    if writer_rectified is not None:
        writer_rectified.release()
    if writer_annotated is not None:
        writer_annotated.release()

    avg_markers = markers_detected_sum / total if total else 0.0
    print("\n--- Riepilogo calibrazione ---")
    print(f"Frame totali:              {total}")
    print(f"Frame calibrati con successo: {successes} ({successes / total:.1%})" if total else "n/a")
    print(f"Marker rilevati (media/frame): {avg_markers:.2f}")


if __name__ == "__main__":
    main()
