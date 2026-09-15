"""Export TipNet to ONNX and quantize it for CPU-only, torch-free inference.

This is a standalone experiment: it only reads the main project (never edits it) and writes
everything under lite/. Run with the project's normal venv (it needs torch to load the
checkpoint and trace the export); the resulting .onnx files then run with just onnxruntime,
no torch required.

    python lite/export_onnx.py [--checkpoint ../models/tipnet.pt] [--out model]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darts_vader.tips.model import INPUT_SIZE, TipNet  # noqa: E402


def export(checkpoint: Path, out_dir: Path) -> Path:
    device = "cpu"
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = TipNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "tipnet.onnx"
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model, (dummy,), str(onnx_path),
        input_names=["image"], output_names=["heat", "offset"],
        opset_version=17, dynamo=False,
    )
    (out_dir / "view.txt").write_text(ckpt.get("view", "rect"))
    print(f"view: {ckpt.get('view', 'rect')}")
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return onnx_path


def quantize(onnx_path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QUInt8)
    print(f"wrote {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB)")
    return int8_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="../models/tipnet.pt")
    ap.add_argument("--out", default="model")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    fp32 = export((here / args.checkpoint).resolve(), here / args.out)
    quantize(fp32)
