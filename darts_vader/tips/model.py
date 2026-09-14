"""TipNet: a heatmap network that finds dart tips in a board-centred view.

CenterNet style: for every cell of a grid at 1/4 of the input resolution the network predicts
the probability that a tip lies there (heatmap) and the exact position of the tip inside the
cell (offset). The backbone is an ImageNet-pretrained ResNet-18 with a small FPN-like decoder
that brings the features back to the grid resolution.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..board.detector import BoardState
from .views import render_view

INPUT_SIZE = 512
STRIDE = 4
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class TipNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)  # 1/4
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4  # 1/4 ... 1/32
        self.lat = nn.ModuleList([nn.Conv2d(c, 128, 1) for c in (64, 128, 256, 512)])
        self.smooth = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.heat = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))
        self.offset = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 2, 1))
        nn.init.constant_(self.heat[-1].bias, -2.19)  # initial probability ~0.1, as in CenterNet

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        p = self.lat[3](c4)
        for feat, lat in ((c3, self.lat[2]), (c2, self.lat[1]), (c1, self.lat[0])):
            p = F.interpolate(p, size=feat.shape[-2:], mode="nearest") + lat(feat)
        p = self.smooth(p)
        return self.heat(p), self.offset(p)


def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    """Square BGR view -> normalised (3, INPUT_SIZE, INPUT_SIZE) tensor (same as in training)."""
    s = INPUT_SIZE / image_bgr.shape[1]
    x = cv2.warpAffine(image_bgr, np.float32([[s, 0, 0], [0, s, 0]]), (INPUT_SIZE, INPUT_SIZE), flags=cv2.INTER_LINEAR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy())


def load_tipnet(path, device: str = "cuda") -> tuple[TipNet, str]:
    """Load a checkpoint written by ``tools/train_tipnet.py``. Returns ``(model, view)``."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TipNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("view", "rect")


@torch.no_grad()
def decode(heat_logits: torch.Tensor, offset: torch.Tensor, threshold: float = 0.3, max_tips: int = 6):
    """For every image in the batch: a list of (x, y, confidence) in network input pixels."""
    heat = torch.sigmoid(heat_logits.float())
    peaks = heat * (F.max_pool2d(heat, 3, stride=1, padding=1) == heat)
    batch, _, _, width = heat.shape
    scores, idx = peaks.view(batch, -1).topk(max_tips)
    results = []
    for b in range(batch):
        tips = []
        for s, i in zip(scores[b].tolist(), idx[b].tolist()):
            if s < threshold:
                continue
            y, x = divmod(i, width)
            ox, oy = offset[b, 0, y, x].item(), offset[b, 1, y, x].item()
            tips.append(((x + ox) * STRIDE, (y + oy) * STRIDE, s))
        results.append(tips)
    return results


@torch.no_grad()
def predict_tips(model: TipNet, frame: np.ndarray, state: BoardState, view: str = "camera", threshold: float = 0.3,
                 max_tips: int = 6, device: str = "cuda") -> np.ndarray:
    """Tips found in an original BGR frame with board `state`: (N, 3) array of x mm, y mm, confidence."""
    image, to_mm = render_view(frame, state, view)
    s = INPUT_SIZE / image.shape[1]
    input_to_mm = to_mm @ np.diag([1 / s, 1 / s, 1.0])
    heat, offset = model(preprocess(image)[None].to(device))
    tips = decode(heat, offset, threshold, max_tips)[0]
    if not tips:
        return np.zeros((0, 3))
    q = np.array([(u, v, 1.0) for u, v, _ in tips]) @ input_to_mm.T
    return np.column_stack([q[:, :2] / q[:, 2:3], [c for _, _, c in tips]])
