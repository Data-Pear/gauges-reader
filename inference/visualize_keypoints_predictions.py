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
        description="Visualize keypoint predictions vs GT keypoints."
    )
    ap.add_argument("--config", type=str, default="configs/config_keypoints.yaml")
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
    model_name = normalize_model_name(
        str(cfg.get("model", {}).get("name", "yolo11n-pose.pt"))
    )
    weights_dir = resolve_task_weights_dir(
        cfg,
        weights_key="weights_dir_kp",
        task_prefix="kp",
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


def _extract_gt_keypoints(
    ann: Dict[str, Any],
    keypoint_indices: List[int],
) -> Optional[List[List[float]]]:
    kps = ann.get("keypoints")
    max_idx = max(keypoint_indices) if keypoint_indices else -1
    if not (
        isinstance(kps, list)
        and max_idx >= 0
        and len(kps) >= 3 * (max_idx + 1)
    ):
        return None
    out: List[List[float]] = []
    for i in keypoint_indices:
        x = float(kps[3 * i + 0])
        y = float(kps[3 * i + 1])
        v = float(kps[3 * i + 2])
        out.append([x, y, v])
    return out


def _build_records(
    coco: Dict[str, Any],
    dataset_root: Path,
    category_name: str,
    keypoint_names: List[str],
) -> List[Dict[str, Any]]:
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
    category = next(
        (cat for cat in categories if cat.get("id") == target_cat_id),
        categories[0] if len(categories) == 1 else {},
    )
    source_names = [str(v) for v in category.get("keypoints", [])]
    if source_names:
        idx_by_name = {name: idx for idx, name in enumerate(source_names)}
        missing = [name for name in keypoint_names if name not in idx_by_name]
        if missing:
            raise ValueError(
                "Configured keypoints are not present in COCO category keypoints: "
                + ", ".join(missing)
            )
        keypoint_indices = [idx_by_name[name] for name in keypoint_names]
    else:
        keypoint_indices = list(range(len(keypoint_names)))

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
        if not isinstance(img_id, int):
            continue
        bbox = ann.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        if _extract_gt_keypoints(ann, keypoint_indices=keypoint_indices) is None:
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
        gt_kps = _extract_gt_keypoints(ann, keypoint_indices=keypoint_indices)
        if gt_kps is None:
            continue

        records.append(
            {
                "image_id": img_id,
                "image_path": str(image_path),
                "bbox": _bbox_xywh_to_xyxy(ann["bbox"]),
                "keypoints": gt_kps,
            }
        )
    return records


def _iter_image_files(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not root.exists():
        return []
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    )


def _parse_yolo_pose_line(
    line: str,
    img_w: float,
    img_h: float,
    num_keypoints: int,
) -> Optional[Dict[str, Any]]:
    toks = line.strip().split()
    min_len = 1 + 4 + 3 * num_keypoints
    if len(toks) < min_len:
        return None

    try:
        cx = float(toks[1]) * img_w
        cy = float(toks[2]) * img_h
        bw = float(toks[3]) * img_w
        bh = float(toks[4]) * img_h
    except ValueError:
        return None

    bbox = [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]
    keypoints: List[List[float]] = []
    base = 5
    for i in range(num_keypoints):
        try:
            kx_n = float(toks[base + 3 * i + 0])
            ky_n = float(toks[base + 3 * i + 1])
            kv = float(toks[base + 3 * i + 2])
        except ValueError:
            return None
        keypoints.append([kx_n * img_w, ky_n * img_h, kv])

    return {"bbox": bbox, "keypoints": keypoints}


def _build_records_from_yolo_split(
    cfg: Dict[str, Any],
    split: str,
    num_keypoints: int,
) -> List[Dict[str, Any]]:
    yolo_root = Path(str(cfg.get("paths", {}).get("yolo_dataset_root", ""))).resolve()
    images_split = yolo_root / "images" / split
    labels_split = yolo_root / "labels" / split
    if not images_split.exists() or not labels_split.exists():
        return []

    records: List[Dict[str, Any]] = []
    for image_path in _iter_image_files(images_split):
        rel = image_path.relative_to(images_split).with_suffix(".txt")
        label_path = labels_split / rel
        if not label_path.exists():
            continue
        lines = [
            ln.strip()
            for ln in label_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if not lines:
            continue

        with Image.open(image_path) as im:
            w, h = im.size
        parsed = _parse_yolo_pose_line(lines[0], float(w), float(h), num_keypoints)
        if parsed is None:
            continue
        records.append(
            {
                "image_id": str(rel.with_suffix("")),
                "image_path": str(image_path.resolve()),
                "bbox": parsed["bbox"],
                "keypoints": parsed["keypoints"],
            }
        )

    return records


def _extract_pose_xy(result: Any, det_idx: int) -> Optional[np.ndarray]:
    kpts = getattr(result, "keypoints", None)
    if kpts is None:
        return None
    if hasattr(kpts, "xy") and kpts.xy is not None:
        xy = kpts.xy
        if len(xy) <= det_idx:
            return None
        return xy[det_idx].detach().cpu().numpy()
    data = getattr(kpts, "data", None)
    if data is None or len(data) <= det_idx:
        return None
    return data[det_idx, :, :2].detach().cpu().numpy()


def _select_prediction(
    result: Any, score_thr: float
) -> Tuple[Optional[List[float]], Optional[np.ndarray], Optional[float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None, None, None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    keep = np.where(conf >= score_thr)[0]
    if keep.size == 0:
        return None, None, None

    best = int(keep[np.argmax(conf[keep])])
    pred_box = [float(v) for v in xyxy[best].tolist()]
    pred_kps = _extract_pose_xy(result, det_idx=best)
    pred_score = float(conf[best])
    return pred_box, pred_kps, pred_score


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


def _draw_keypoints(
    ax: plt.Axes,
    keypoints: List[List[float]] | np.ndarray,
    names: List[str],
    color: str,
    with_visibility: bool,
) -> None:
    if isinstance(keypoints, np.ndarray):
        pts = keypoints.tolist()
    else:
        pts = keypoints

    for i, kp in enumerate(pts):
        x = float(kp[0])
        y = float(kp[1])
        v = float(kp[2]) if with_visibility and len(kp) > 2 else 2.0
        if with_visibility and v <= 0:
            continue
        ax.scatter([x], [y], s=24, c=color)
        if i < len(names):
            ax.text(x + 2, y + 2, names[i], color=color, fontsize=7)


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    tcfg = cfg.get("training", {})
    mcfg = cfg.get("model", {})
    ecfg = cfg.get("evaluation", {})
    category_name = str(cfg.get("dataset", {}).get("category_name", "gauge"))
    kp_cfg = cfg.get("keypoints", {})
    kp_names = [str(v) for v in kp_cfg.get("names", ["center", "scale_start", "scale_end"])]
    num_keypoints = int(kp_cfg.get("num_keypoints", len(kp_names)))

    score_thr = (
        float(args.score_thr)
        if args.score_thr is not None
        else float(ecfg.get("score_thr", 0.25))
    )
    imgsz = int(args.imgsz) if args.imgsz is not None else int(mcfg.get("imgsz", 640))
    device = _resolve_device(args.device, str(tcfg.get("device", "auto")))

    crop_mode = bool(kp_cfg.get("crop_dial", False))
    if crop_mode:
        records = _build_records_from_yolo_split(
            cfg=cfg,
            split=args.split,
            num_keypoints=num_keypoints,
        )
    else:
        split_coco_path = _resolve_split_coco_path(cfg, args.split)
        dataset_root = Path(cfg["paths"]["raw_ds_path"]).resolve()
        coco = _load_json(split_coco_path)
        records = _build_records(
            coco,
            dataset_root=dataset_root,
            category_name=category_name,
            keypoint_names=kp_names,
        )
    if not records:
        if crop_mode:
            raise RuntimeError(
                f"No crop records found for split={args.split}. "
                "Rebuild the keypoint dataset with "
                "`uv run --no-sync ./data/build_kp_yolo_pose_from_coco.py --config configs/config_keypoints.yaml`."
            )
        raise RuntimeError(f"No records found for split={args.split}")
    chosen = random.sample(records, k=min(args.num_samples, len(records)))

    weights_path = _resolve_weights_path(cfg, args.weights)
    model = YOLO(str(weights_path))

    cols = 3
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
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
        pred_box, pred_kps, pred_score = _select_prediction(pred, score_thr)

        ax.imshow(np_img)
        _draw_box(ax, [float(v) for v in rec["bbox"]], color="lime", label="GT")
        _draw_keypoints(ax, rec["keypoints"], kp_names, color="lime", with_visibility=True)

        if pred_box is not None:
            score_text = f"PRED s={pred_score:.2f}" if pred_score is not None else "PRED"
            _draw_box(ax, pred_box, color="red", label=score_text)
        if pred_kps is not None:
            _draw_keypoints(ax, pred_kps, kp_names, color="red", with_visibility=False)
        if pred_box is None and pred_kps is None:
            ax.text(4, 14, "PRED: none", color="red", fontsize=9)

        ax.set_title(f"id={rec['image_id']} | {img_path.name}", fontsize=9)
        ax.axis("off")

    for ax in axes_flat[len(chosen) :]:
        ax.axis("off")

    fig.suptitle(
        f"Keypoint predictions ({args.split}, n={len(chosen)}, thr={score_thr:.2f}, crop={crop_mode})",
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
