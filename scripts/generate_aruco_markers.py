"""Genera le immagini PNG dei marker ArUco da stampare e posizionare
fisicamente attorno al bersaglio.

Uso:
    uv run python scripts/generate_aruco_markers.py
    uv run python scripts/generate_aruco_markers.py --config config/board_config.yaml --dpi 300

Dopo la stampa: misurare la posizione reale del CENTRO di ciascun
marker rispetto al centro del bersaglio e aggiornare
``config/board_config.yaml`` (campi ``x_mm`` / ``y_mm``) di conseguenza.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from dartvision.calibration.aruco_detector import resolve_dictionary
from dartvision.config import load_config

MM_PER_INCH = 25.4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/board_config.yaml")
    parser.add_argument("--output-dir", default="docs/markers")
    parser.add_argument(
        "--dpi", type=int, default=300, help="Risoluzione di stampa target"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dictionary = resolve_dictionary(config.aruco.dictionary)

    marker_side_px = round(config.aruco.marker_length_mm / MM_PER_INCH * args.dpi)
    quiet_zone_px = marker_side_px // 4  # margine bianco consigliato per il rilevamento

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for marker in config.aruco.markers:
        marker_img = cv2.aruco.generateImageMarker(
            dictionary, marker.id, marker_side_px
        )
        padded = cv2.copyMakeBorder(
            marker_img,
            quiet_zone_px,
            quiet_zone_px,
            quiet_zone_px,
            quiet_zone_px,
            cv2.BORDER_CONSTANT,
            value=255,
        )

        label_height = 40
        canvas = cv2.copyMakeBorder(
            padded, 0, label_height, 0, 0, cv2.BORDER_CONSTANT, value=255
        )
        label = f"id={marker.id}  lato={config.aruco.marker_length_mm:.0f}mm  {args.dpi}dpi"
        cv2.putText(
            canvas,
            label,
            (10, canvas.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            0,
            1,
            cv2.LINE_AA,
        )

        out_path = output_dir / f"marker_{marker.id}.png"
        cv2.imwrite(str(out_path), canvas)
        print(
            f"Marker id={marker.id} -> {out_path} "
            f"({marker_side_px}px lato, stampare a {args.dpi} DPI per ottenere "
            f"esattamente {config.aruco.marker_length_mm:.0f} mm di lato)"
        )

    print(
        "\nIMPORTANTE: dopo aver stampato e posizionato i marker, misura la "
        "posizione reale del centro di ciascuno rispetto al centro del "
        "bersaglio e aggiorna 'x_mm'/'y_mm' in "
        f"{args.config}."
    )


if __name__ == "__main__":
    main()
