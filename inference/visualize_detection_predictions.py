from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config
from utils.runtime import (
    find_weights_path,
    normalize_model_name,
    resolve_task_weights_dir,
    resolve_yolo_device,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Visualize detection predictions vs GT bounding boxes."
    )
    ap.add_argument("--config", type=str, default="configs/config_detection.yaml")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--score-thr", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="from-config")
    ap.add_argument("--save", type=str, default=None)
    return ap.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_device(requested: str, cfg_device: str) -> str:
    mode = cfg_device if requested == "from-config" else requested
    return resolve_yolo_device(str(mode))


def _resolve_weights_path(cfg: Dict[str, Any], weights_arg: Optional[str]) -> Path:
    model_name = normalize_model_name(str(cfg.get("model", {}).get("name", "yolo11n.pt")))
    weights_dir = resolve_task_weights_dir(
        cfg,
        weights_key="weights_dir_det",
        task_prefix="det",
        model_identifier=model_name,
    )
    return find_weights_path(
        explicit_path=weights_arg,
        weights_dir=weights_dir,
        include_nested_weights_dir=True,
    )


def _resolve_split_coco_path(cfg: Dict[str, Any], split: str) -> Path:
    key = f"{split}_inst_coco"
    rel = cfg.get("paths", {}).get(key)
    if not rel:
        raise KeyError(f"Missing path key in config: paths.{key}")
    dataset_root = Path(cfg["paths"]["raw_ds_path"]).resolve()
    p = (dataset_root / str(rel)).resolve()
    if not p.exists():
        raise FileNotFoundError(f"COCO file not found: {p}")
    return p


def _bbox_xywh_to_xyxy(b: List[float]) -> List[float]:
    x, y, w, h = [float(v) for v in b]
    return [x, y, x + w, y + h]


def _select_largest_ann(anns: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not anns:
        return None
    if len(anns) == 1:
        return anns[0]

    def _score(a: Dict[str, Any]) -> float:
        area = a.get("area")
        if isinstance(area, (int, float)):
            return float(area)
        bbox = a.get("bbox", [0.0, 0.0, 0.0, 0.0])
        if isinstance(bbox, list) and len(bbox) == 4:
            return float(bbox[2]) * float(bbox[3])
        return 0.0

    return max(anns, key=_score)


def _build_records(coco: Dict[str, Any], dataset_root: Path, category_name: str) -> List[Dict[str, Any]]:
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    target_cat_id: Optional[int] = None
    for cat in categories:
        if cat.get("name") == category_name and isinstance(cat.get("id"), int):
            target_cat_id = int(cat["id"])
            break
    if target_cat_id is None and len(categories) == 1 and isinstance(categories[0].get("id"), int):
        target_cat_id = int(categories[0]["id"])
    if target_cat_id is None:
        raise ValueError(f"Category '{category_name}' not found in COCO categories.")

    images_root = dataset_root / "images"
    if not images_root.exists():
        images_root = dataset_root

    img_by_id: Dict[int, Dict[str, Any]] = {}
    for img in images:
        img_id = img.get("id")
        if isinstance(img_id, int):
            img_by_id[img_id] = img

    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        if ann.get("category_id") != target_cat_id:
            continue
        img_id = ann.get("image_id")
        bbox = ann.get("bbox")
        if not isinstance(img_id, int):
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        anns_by_img.setdefault(img_id, []).append(ann)

    records: List[Dict[str, Any]] = []
    for img_id, anns in anns_by_img.items():
        img = img_by_id.get(img_id)
        if img is None:
            continue
        file_name = img.get("file_name")
        if not isinstance(file_name, str):
            continue

        image_path = (images_root / file_name).resolve()
        if not image_path.exists():
            continue

        ann = _select_largest_ann(anns)
        if ann is None:
            continue

        records.append(
            {
                "image_id": img_id,
                "image_path": str(image_path),
                "bbox": _bbox_xywh_to_xyxy(ann["bbox"]),
            }
        )
    return records


def _select_prediction(
    result: Any, score_thr: float
) -> Tuple[Optional[List[float]], Optional[float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None, None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    keep = np.where(conf >= score_thr)[0]
    if keep.size == 0:
        return None, None
    best = int(keep[np.argmax(conf[keep])])
    return [float(v) for v in xyxy[best].tolist()], float(conf[best])


def _draw_box(ax: plt.Axes, box: List[float], color: str, label: str) -> None:
    x1, y1, x2, y2 = box
    rect = Rectangle(
        (x1, y1),
        max(1.0, x2 - x1),
        max(1.0, y2 - y1),
        linewidth=2,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(x1, max(8.0, y1 - 2), label, color=color, fontsize=8)


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    tcfg = cfg.get("training", {})
    mcfg = cfg.get("model", {})
    ecfg = cfg.get("evaluation", {})
    category_name = str(cfg.get("dataset", {}).get("category_name", "gauge"))
    score_thr = (
        float(args.score_thr)
        if args.score_thr is not None
        else float(ecfg.get("score_thr", 0.25))
    )
    imgsz = int(args.imgsz) if args.imgsz is not None else int(mcfg.get("imgsz", 640))
    device = _resolve_device(args.device, str(tcfg.get("device", "auto")))

    split_coco_path = _resolve_split_coco_path(cfg, args.split)
    dataset_root = Path(cfg["paths"]["raw_ds_path"]).resolve()
    coco = _load_json(split_coco_path)
    records = _build_records(coco, dataset_root=dataset_root, category_name=category_name)
    if not records:
        raise RuntimeError(f"No records found for split={args.split} in {split_coco_path}")
    chosen = random.sample(records, k=min(args.num_samples, len(records)))

    weights_path = _resolve_weights_path(cfg, args.weights)
    model = YOLO(str(weights_path))

    cols = 3
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.0))
    axes_flat = np.array(axes, ndmin=1).reshape(-1)

    for ax, rec in zip(axes_flat, chosen):
        img_path = Path(rec["image_path"]).resolve()
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        np_img = np.asarray(img, dtype=np.uint8)

        pred = model.predict(
            source=np_img,
            conf=score_thr,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]
        pred_box, pred_score = _select_prediction(pred, score_thr)

        ax.imshow(np_img)
        _draw_box(ax, [float(v) for v in rec["bbox"]], color="lime", label="GT")
        if pred_box is not None:
            score_text = f"PRED s={pred_score:.2f}" if pred_score is not None else "PRED"
            _draw_box(ax, pred_box, color="red", label=score_text)
        else:
            ax.text(4, 14, "PRED: none", color="red", fontsize=9)

        ax.set_title(f"id={rec['image_id']} | {img_path.name}", fontsize=9)
        ax.axis("off")

    for ax in axes_flat[len(chosen) :]:
        ax.axis("off")

    fig.suptitle(
        f"Detection predictions ({args.split}, n={len(chosen)}, thr={score_thr:.2f})",
        fontsize=14,
    )
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
