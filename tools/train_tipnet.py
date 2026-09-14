"""Train TipNet on dart tip annotations.

The dataset is a folder of ``*.jsonl`` files produced by tools/build_deepdarts.py,
tools/export_webcam_labels.py and tools/make_canonical_dataset.py. Every sample carries the
``px_to_mm`` matrix used to measure errors and scores in board millimetres.

    python tools/train_tipnet.py --data dataset_canon --out models_canon --epochs 60 --rot-deg 20
    python tools/train_tipnet.py --data dataset_canon --init models/tipnet.pt --epochs 15 --lr 3e-4
    python tools/train_tipnet.py --data dataset_canon --epochs 1 --limit 64        # quick smoke test

The checkpoint with the best validation score is saved to <out>/tipnet_best.pt and the final one
to <out>/tipnet_last.pt; copy one of them to models/tipnet.pt to use it in the app.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darts_vader.board import geometry as g  # noqa: E402
from darts_vader.tips.model import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, STRIDE, TipNet, decode  # noqa: E402
from darts_vader.tips.views import RECT_PX_TO_MM  # noqa: E402

MAX_TIPS = 8
MATCH_MM = 10.0  # a predicted tip counts as correct within this distance of a true tip


# ----------------------------------------------------------------------------- data

def load_records(data_dir: Path, split: str) -> list[dict]:
    records = []
    for path in sorted(data_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            records += [r for r in map(json.loads, f) if r["split"] == split]
    return records


def motion_blur(img: np.ndarray) -> np.ndarray:
    k = random.choice((5, 7, 9, 11))
    kernel = np.zeros((k, k), np.float32)
    angle = random.uniform(0, np.pi)
    c = k // 2
    dx, dy = int(round(c * np.cos(angle))), int(round(c * np.sin(angle)))
    cv2.line(kernel, (c - dx, c - dy), (c + dx, c + dy), 1.0, 1)
    return cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1.0))


def photometric(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= random.uniform(0.7, 1.3)
    hsv[..., 2] = hsv[..., 2] * random.uniform(0.6, 1.3) + random.uniform(-20, 20)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.3:
        img = motion_blur(img)
    elif random.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 1.5))
    if random.random() < 0.3:
        img = np.clip(img + np.random.normal(0, random.uniform(2, 8), img.shape), 0, 255).astype(np.uint8)
    return img


def make_targets(tips: np.ndarray, size: int, sigma: float = 1.5):
    heat = np.zeros((1, size, size), np.float32)
    offset = np.zeros((2, size, size), np.float32)
    mask = np.zeros((1, size, size), np.float32)
    ys, xs = np.mgrid[0:size, 0:size]
    for u, v in tips:
        gx, gy = u / STRIDE, v / STRIDE
        cx, cy = int(math.floor(gx)), int(math.floor(gy))
        if not (0 <= cx < size and 0 <= cy < size):
            continue
        heat[0] = np.maximum(heat[0], np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)))
        offset[:, cy, cx] = (gx - cx, gy - cy)
        mask[0, cy, cx] = 1.0
    return heat, offset, mask


class TipDataset(Dataset):
    def __init__(self, data_dir: Path, records: list[dict], augment: bool, perspective: float = 0.0,
                 rot_deg: float = 180.0):
        self.data_dir, self.records, self.augment = data_dir, records, augment
        self.perspective, self.rot_deg = perspective, rot_deg

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        img = cv2.imread(str(self.data_dir / r["image"]))
        size = img.shape[1]
        tips = np.array(r["tips_px"], np.float32).reshape(-1, 2)
        s = INPUT_SIZE / size
        A = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], np.float64)
        if self.augment:
            c = size / 2
            # normalised crops (camera always "from below") only need small rotations; almost frontal
            # views have no defined viewing direction and keep the full rotation range
            rot = self.rot_deg if r.get("tilt", 90.0) >= 12.0 else 180.0
            R = np.vstack([cv2.getRotationMatrix2D((c, c), random.uniform(-rot, rot), random.uniform(0.92, 1.08)), [0, 0, 1]])
            if random.random() < 0.5:
                R = np.array([[-1, 0, size], [0, 1, 0], [0, 0, 1]]) @ R
            j = 0.01 * size  # small registration error
            T = np.array([[1, 0, random.uniform(-j, j)], [0, 1, random.uniform(-j, j)], [0, 0, 1]])
            A = A @ T @ R
            if random.random() < self.perspective:
                # random perspective warp: simulates the board seen from another angle
                j = 0.12 * INPUT_SIZE
                src = np.float32([[0, 0], [INPUT_SIZE, 0], [INPUT_SIZE, INPUT_SIZE], [0, INPUT_SIZE]])
                dst = (src + np.random.uniform(-j, j, src.shape)).astype(np.float32)
                A = cv2.getPerspectiveTransform(src, dst).astype(np.float64) @ A
        x = cv2.warpPerspective(img, A, (INPUT_SIZE, INPUT_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if len(tips):
            tips = g.apply_homography(A, tips).astype(np.float32)
        if self.augment:
            x = photometric(x)
        x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        heat, offset, mask = make_targets(tips, INPUT_SIZE // STRIDE)
        padded = np.full((MAX_TIPS, 2), np.nan, np.float32)
        padded[:min(len(tips), MAX_TIPS)] = tips[:MAX_TIPS]
        input_to_mm = np.array(r.get("px_to_mm", RECT_PX_TO_MM), np.float64) @ np.linalg.inv(A)
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(heat), torch.from_numpy(offset),
                torch.from_numpy(mask), torch.from_numpy(padded), torch.from_numpy(input_to_mm))


def worker_init(_):
    cv2.setNumThreads(0)
    seed = torch.utils.data.get_worker_info().seed % 2**32
    random.seed(seed)
    np.random.seed(seed)


# ----------------------------------------------------------------------------- losses and metrics

def focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    pos_loss = torch.log(pred) * (1 - pred) ** 2 * pos
    neg_loss = torch.log(1 - pred) * pred ** 2 * (1 - target) ** 4 * (1 - pos)
    return -(pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)


def offset_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (torch.abs(pred.float() - target) * mask).sum() / mask.sum().clamp(min=1.0) / 2


def evaluate(model, loader, device, threshold=0.3) -> dict:
    """Tip recall/precision within MATCH_MM, mean error, and share of images scored entirely right."""
    model.eval()
    n_true = n_pred = n_match = images = images_ok = 0
    errors = []
    with torch.no_grad():
        for x, _, _, _, tips, to_mm in loader:
            heat, off = model(x.to(device, non_blocking=True))
            for pred, true, M in zip(decode(heat, off, threshold), tips.numpy(), to_mm.numpy()):
                true = true[~np.isnan(true[:, 0])]
                p_mm = g.apply_homography(M, [(u, v) for u, v, _ in pred]) if pred else np.zeros((0, 2))
                t_mm = g.apply_homography(M, true) if len(true) else np.zeros((0, 2))
                n_true, n_pred, images = n_true + len(t_mm), n_pred + len(p_mm), images + 1
                matched = []
                if len(p_mm) and len(t_mm):
                    d = np.linalg.norm(p_mm[:, None] - t_mm[None], axis=2)
                    used_p, used_t = set(), set()
                    for flat in np.argsort(d, axis=None):
                        i, j = divmod(int(flat), d.shape[1])
                        if d[i, j] > MATCH_MM:
                            break
                        if i in used_p or j in used_t:
                            continue
                        used_p.add(i)
                        used_t.add(j)
                        matched.append((i, j))
                        errors.append(d[i, j])
                n_match += len(matched)
                images_ok += len(matched) == len(t_mm) == len(p_mm) and all(
                    g.score_model_point(*p_mm[i]) == g.score_model_point(*t_mm[j]) for i, j in matched)
    return dict(recall=n_match / max(n_true, 1), precision=n_match / max(n_pred, 1),
                err_mm=float(np.mean(errors)) if errors else float("nan"),
                err_p90_mm=float(np.percentile(errors, 90)) if errors else float("nan"),
                images_ok=images_ok / max(images, 1))


def selection_key(metrics: dict) -> tuple[float, float]:
    return metrics["images_ok"], -metrics["err_mm"] if not math.isnan(metrics["err_mm"]) else -1e9


# ----------------------------------------------------------------------------- training

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_canon")
    ap.add_argument("--out", default="models_run")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--own-weight", type=float, default=3.0, help="sampling weight of non-DeepDarts samples")
    ap.add_argument("--webcam-weight", type=float, default=None, help="sampling weight of webcam photos (default: --own-weight)")
    ap.add_argument("--init", help="start from the weights of a trained checkpoint")
    ap.add_argument("--rot-deg", type=float, default=180.0,
                    help="random rotation range in degrees (about 20 for camera_canon datasets)")
    ap.add_argument("--perspective", type=float, default=0.5, help="probability of a random perspective warp per sample")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="use only this many samples (smoke test)")
    ap.add_argument("--no-pretrained", action="store_true", help="do not start from ImageNet weights")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_recs, val_recs = load_records(data_dir, "train"), load_records(data_dir, "val")
    if not train_recs:
        raise SystemExit(f"no training samples in {data_dir}")
    if args.limit:
        random.Random(0).shuffle(train_recs)
        train_recs, val_recs = train_recs[:args.limit], val_recs[:max(8, args.limit // 4)]
    views = Counter(r.get("view", "rect") for r in train_recs)
    webcam_weight = args.own_weight if args.webcam_weight is None else args.webcam_weight
    weights = [webcam_weight if r["source"] == "webcam" else 1.0 if r["source"].startswith("deepdarts") else args.own_weight
               for r in train_recs]
    print(f"training: {len(train_recs)} images ({sum(r['source'] == 'webcam' for r in train_recs)} webcam photos, "
          f"weight {webcam_weight}); validation: {len(val_recs)}; views {dict(views)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}{' - ' + torch.cuda.get_device_name(0) if device == 'cuda' else ''}")
    loader_kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=device == "cuda",
                     worker_init_fn=worker_init if args.workers else None, persistent_workers=args.workers > 0)
    train_loader = DataLoader(TipDataset(data_dir, train_recs, augment=True, perspective=args.perspective,
                                         rot_deg=args.rot_deg),
                              sampler=WeightedRandomSampler(weights, len(train_recs), replacement=True),
                              drop_last=len(train_recs) > args.batch, **loader_kw)
    val_loader = DataLoader(TipDataset(data_dir, val_recs, augment=False), shuffle=False, **loader_kw)

    model = TipNet(pretrained=not args.no_pretrained and not args.init).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=False)["model"])
        print(f"initial weights: {args.init}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(train_loader)),
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    view = views.most_common(1)[0][0]

    best_key, log = None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, count = time.perf_counter(), 0.0, 0
        for x, heat_t, off_t, mask, _, _ in train_loader:
            x, heat_t = x.to(device, non_blocking=True), heat_t.to(device, non_blocking=True)
            off_t, mask = off_t.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                heat, off = model(x)
            loss = focal_loss(heat, heat_t) + offset_loss(off, off_t, mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, count = total + loss.item() * len(x), count + len(x)
        metrics = evaluate(model, val_loader, device)
        metrics.update(epoch=epoch, loss=total / max(count, 1), seconds=time.perf_counter() - t0)
        log.append(metrics)
        print(f"epoch {epoch:3d}  loss {metrics['loss']:.3f}  | val: recall {100 * metrics['recall']:.1f}%  "
              f"precision {100 * metrics['precision']:.1f}%  error {metrics['err_mm']:.2f} mm (p90 {metrics['err_p90_mm']:.2f})  "
              f"images fully right {100 * metrics['images_ok']:.1f}%  ({metrics['seconds']:.0f} s)", flush=True)
        if best_key is None or selection_key(metrics) > best_key:
            best_key = selection_key(metrics)
            torch.save(dict(model=model.state_dict(), epoch=epoch, metrics=metrics, args=vars(args), view=view),
                       out_dir / "tipnet_best.pt")
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    torch.save(dict(model=model.state_dict(), epoch=args.epochs, metrics=log[-1], args=vars(args), view=view),
               out_dir / "tipnet_last.pt")
    best = max(log, key=selection_key)
    print(f"best: epoch {best['epoch']} -> {out_dir / 'tipnet_best.pt'}")


if __name__ == "__main__":
    main()
