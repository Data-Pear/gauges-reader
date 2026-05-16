from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Visualize v2.0 dataset samples with dial bbox, selected keypoints, and needle mask."
    )
    ap.add_argument("--config", type=str, default="configs/config_keypoints.yaml")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", type=str, default=None)
    return ap.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _bbox_xywh_to_xyxy(b: List[float]) -> List[float]:
    x, y, w, h = [float(v) for v in b]
    return [x, y, x + w, y + h]


def _draw_box(ax: plt.Axes, box: List[float], color: str, label: str) -> None:
    x1, y1, x2, y2 = box
    ax.add_patch(
        Rectangle(
            (x1, y1),
            max(1.0, x2 - x1),
            max(1.0, y2 - y1),
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
    )
    ax.text(x1, max(8.0, y1 - 2), label, color=color, fontsize=8)


def _draw_keypoints(
    ax: plt.Axes,
    keypoints: List[List[float]],
    names: List[str],
    color: str,
) -> None:
    for idx, kp in enumerate(keypoints):
        x, y, v = [float(value) for value in kp]
        if v <= 0:
            continue
        ax.scatter([x], [y], s=30, c=color)
        ax.text(x + 3, y + 3, names[idx], color=color, fontsize=7)


def _overlay_mask(ax: plt.Axes, mask: np.ndarray, color: tuple[float, float, float]) -> None:
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask, 0] = color[0]
    overlay[mask, 1] = color[1]
    overlay[mask, 2] = color[2]
    overlay[mask, 3] = 0.45
    ax.imshow(overlay)


def _find_category(coco: Dict[str, Any], category_name: str) -> Dict[str, Any]:
    for cat in coco.get("categories", []):
        if cat.get("name") == category_name:
            return cat
    cats = coco.get("categories", [])
    if len(cats) == 1:
        return cats[0]
    raise ValueError(f"Category '{category_name}' not found in COCO categories.")


def _selected_keypoints(
    ann: Dict[str, Any],
    category: Dict[str, Any],
    selected_names: List[str],
) -> Optional[List[List[float]]]:
    source_names = [str(v) for v in category.get("keypoints", [])]
    if not source_names:
        source_names = selected_names
    idx_by_name = {name: idx for idx, name in enumerate(source_names)}
    raw = ann.get("keypoints")
    if not isinstance(raw, list):
        return None

    out: List[List[float]] = []
    for name in selected_names:
        idx = idx_by_name.get(name)
        if idx is None or len(raw) < 3 * (idx + 1):
            return None
        out.append(
            [
                float(raw[3 * idx + 0]),
                float(raw[3 * idx + 1]),
                float(raw[3 * idx + 2]),
            ]
        )
    return out


def _build_records(
    coco: Dict[str, Any],
    raw_root: Path,
    split: str,
    category_name: str,
    keypoint_names: List[str],
) -> List[Dict[str, Any]]:
    category = _find_category(coco, category_name)
    target_cat_id = int(category["id"])

    image_by_id: Dict[int, Dict[str, Any]] = {}
    for image in coco.get("images", []):
        img_id = image.get("id")
        if isinstance(img_id, int):
            image_by_id[img_id] = image

    records: List[Dict[str, Any]] = []
    for ann in coco.get("annotations", []):
        if ann.get("category_id") != target_cat_id:
            continue
        img_id = ann.get("image_id")
        if not isinstance(img_id, int) or img_id not in image_by_id:
            continue
        bbox = ann.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        kps = _selected_keypoints(ann, category, keypoint_names)
        if kps is None:
            continue

        image_file = str(image_by_id[img_id]["file_name"])
        image_path = (raw_root / "images" / Path(image_file.replace("\\", "/"))).resolve()
        if not image_path.exists():
            image_path = (raw_root / Path(image_file.replace("\\", "/"))).resolve()
        mask_path = (raw_root / "segmentation" / split / Path(image_file).name).with_suffix(".png")
        if image_path.exists() and mask_path.exists():
            records.append(
                {
                    "image_id": img_id,
                    "image_path": image_path,
                    "mask_path": mask_path.resolve(),
                    "bbox": _bbox_xywh_to_xyxy(bbox),
                    "keypoints": kps,
                }
            )
    return records


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    cfg = load_config(args.config)
    raw_root = Path(cfg["paths"]["raw_ds_path"]).resolve()
    category_name = str(cfg.get("dataset", {}).get("category_name", "gauge"))
    keypoint_names = [
        str(v)
        for v in cfg.get("keypoints", {}).get(
            "names", ["center", "scale_start", "scale_end"]
        )
    ]

    coco_path = (raw_root / "annotations" / f"instances_{args.split}.json").resolve()
    coco = _load_json(coco_path)
    records = _build_records(
        coco=coco,
        raw_root=raw_root,
        split=args.split,
        category_name=category_name,
        keypoint_names=keypoint_names,
    )
    if not records:
        raise RuntimeError(f"No dataset records found for split={args.split}")

    chosen = random.sample(records, k=min(args.num_samples, len(records)))
    cols = 3
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2))
    axes_flat = np.array(axes, ndmin=1).reshape(-1)

    for ax, rec in zip(axes_flat, chosen):
        with Image.open(rec["image_path"]) as im:
            img = np.asarray(im.convert("RGB"))
        with Image.open(rec["mask_path"]) as im:
            mask = np.asarray(im) == 4

        ax.imshow(img)
        _overlay_mask(ax, mask, color=(1.0, 0.2, 0.1))
        _draw_box(ax, [float(v) for v in rec["bbox"]], color="lime", label="dial")
        _draw_keypoints(ax, rec["keypoints"], keypoint_names, color="cyan")
        ax.set_title(f"id={rec['image_id']} | {rec['image_path'].name}", fontsize=9)
        ax.axis("off")

    for ax in axes_flat[len(chosen) :]:
        ax.axis("off")

    fig.suptitle(f"Dataset v2.0 samples ({args.split}, n={len(chosen)})", fontsize=14)
    plt.tight_layout()
    if args.save:
        out_path = Path(args.save).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Saved figure: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
