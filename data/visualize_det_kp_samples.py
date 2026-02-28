from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from PIL import Image

from utils.config import load_config

N_SAMPLES = 10


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def draw_bbox(ax, bbox: List[float]) -> None:
    x1, y1, x2, y2 = bbox
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], linewidth=2)


def draw_keypoints(ax, kps: List[List[float]], names: List[str]) -> None:
    # kps: [[x,y,v], ...]
    for i, (x, y, v) in enumerate(kps):
        ax.scatter([x], [y], s=40)
        label = names[i] if i < len(names) else str(i)
        ax.text(x + 3, y + 3, f"{label}", fontsize=8)


def draw_skeleton(ax, kps: List[List[float]], names: List[str]) -> None:
    # Expected order: dial_max, dial_min, dial_center, dial_tip
    # Skeleton: center -> max, center -> min, center -> tip
    name_to_idx = {n: i for i, n in enumerate(names)}

    required = ["dial_center", "dial_max", "dial_min", "dial_tip"]
    if not all(n in name_to_idx for n in required):
        return

    c = name_to_idx["dial_center"]
    a = name_to_idx["dial_max"]
    b = name_to_idx["dial_min"]
    t = name_to_idx["dial_tip"]

    def line(i: int, j: int) -> None:
        x1, y1, v1 = kps[i]
        x2, y2, v2 = kps[j]
        if v1 <= 0 or v2 <= 0:
            return
        ax.plot([x1, x2], [y1, y2], linewidth=2)

    line(c, a)
    line(c, b)
    line(c, t)


def main() -> None:
    cfg = load_config("configs/config.yaml")

    idx_path = Path(cfg["paths"]["train_det_kp_output_json"]).resolve()
    if not idx_path.exists():
        raise FileNotFoundError(f"Index not found: {idx_path}")

    raw_root = Path(cfg["paths"]["raw_ds_path"]).resolve()
    # В jsonl мы сохраняли абсолютные пути; если вдруг у тебя относительные — поддержим и это:
    # image_path = raw_root / cfg["paths"]["images_path"] / rec["image_path"] (fallback)
    images_root = (raw_root / cfg["paths"]["images_path"]).resolve()

    kp_names = list(cfg["keypoints_target"]["names"])

    items = load_jsonl(idx_path)
    if len(items) == 0:
        raise RuntimeError("Index file is empty")

    chosen = random.sample(items, k=min(N_SAMPLES, len(items)))

    cols = 5
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = axes.flatten()

    for ax, rec in zip(axes, chosen):
        img_path = Path(rec["image_path"])
        if not img_path.is_absolute():
            img_path = (images_root / rec["image_path"]).resolve()

        bbox = rec["bbox"]
        kps = rec["keypoints"]

        with Image.open(img_path) as im:
            img = im.convert("RGB")

        ax.imshow(img)
        draw_bbox(ax, bbox)
        draw_skeleton(ax, kps, kp_names)
        draw_keypoints(ax, kps, kp_names)

        ax.set_title(f"id={rec.get('image_id', '?')}", fontsize=10)
        ax.axis("off")

    for ax in axes[len(chosen) :]:
        ax.axis("off")

    fig.suptitle("Det + Keypoints sanity check (raw annotations)", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
