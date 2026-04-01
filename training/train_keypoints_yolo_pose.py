from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config_keypoints.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--prepare-data", action="store_true")
    ap.add_argument("--no-prepare-data", action="store_true")
    return ap.parse_args()


def _setup_logger(log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("train_keypoints_yolo_pose")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def _resolve_yolo_device(mode: str) -> str:
    m = str(mode).lower()
    if m == "cpu":
        return "cpu"
    if m == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return "0"
    if m == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return mode


def _normalize_model_name(model_name: str) -> str:
    name = model_name.strip()
    if name.endswith(".pt"):
        return name
    return f"{name}.pt"


def _model_tag(model_name: str) -> str:
    stem = Path(model_name).stem.lower()
    tag = re.sub(r"[^a-z0-9._-]+", "-", stem).strip("-")
    return tag or "model"


def _dataset_name(cfg: Dict[str, Any]) -> str:
    dataset_cfg = cfg.get("dataset", {})
    explicit = dataset_cfg.get("name")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    raw_ds = str(cfg.get("paths", {}).get("raw_ds_path", "dataset")).rstrip("/\\")
    return Path(raw_ds).name or "dataset"


def _resolve_weights_dir(cfg: Dict[str, Any], model_name: str) -> Path:
    paths = cfg.get("paths", {})
    explicit = paths.get("weights_dir_kp")
    if explicit:
        return Path(str(explicit)).resolve()

    dataset_name = _dataset_name(cfg)
    model_name_tag = _model_tag(model_name)
    return (Path("models/weights").resolve() / dataset_name / f"kp_{model_name_tag}")


def _ensure_data_yaml(
    cfg: Dict[str, Any],
    config_path: Path,
    prepare_data: bool,
    logger: logging.Logger,
) -> Path:
    data_yaml = Path(cfg["paths"]["yolo_data_yaml"]).resolve()
    if data_yaml.exists() and not prepare_data:
        return data_yaml

    from data.build_kp_yolo_pose_from_coco import build_from_config

    logger.info("Preparing keypoint labels from COCO annotations...")
    produced_yaml = build_from_config(config_path=config_path, out_yaml=data_yaml)
    if not produced_yaml.exists():
        raise FileNotFoundError(f"YOLO pose data yaml was not created: {data_yaml}")
    return produced_yaml


def _copy_best_last(weights_dir: Path) -> None:
    nested = weights_dir / "weights"
    if not nested.exists():
        return
    for fname in ["best.pt", "last.pt"]:
        src = nested / fname
        if src.exists():
            shutil.copy2(src, weights_dir / fname)


def _extract_pose_metrics(val_result: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}

    box = getattr(val_result, "box", None)
    if box is not None:
        mp = getattr(box, "mp", None)
        mr = getattr(box, "mr", None)
        map50 = getattr(box, "map50", None)
        map5095 = getattr(box, "map", None)
        if mp is not None:
            out["bbox_precision"] = float(mp)
        if mr is not None:
            out["bbox_recall"] = float(mr)
        if map50 is not None:
            out["bbox_mAP@0.5"] = float(map50)
        if map5095 is not None:
            out["bbox_mAP@0.5:0.95"] = float(map5095)

    pose = getattr(val_result, "pose", None)
    if pose is not None:
        mp = getattr(pose, "mp", None)
        mr = getattr(pose, "mr", None)
        map50 = getattr(pose, "map50", None)
        map5095 = getattr(pose, "map", None)
        if mp is not None:
            out["pose_precision"] = float(mp)
        if mr is not None:
            out["pose_recall"] = float(mr)
        if map50 is not None:
            out["pose_mAP@0.5"] = float(map50)
        if map5095 is not None:
            out["pose_mAP@0.5:0.95"] = float(map5095)

    return out


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _find_category_id(coco: Dict[str, Any], category_name: str) -> int:
    categories = coco.get("categories", [])
    for cat in categories:
        if cat.get("name") == category_name and isinstance(cat.get("id"), int):
            return int(cat["id"])

    if len(categories) == 1 and isinstance(categories[0].get("id"), int):
        return int(categories[0]["id"])

    raise ValueError(f"Category '{category_name}' not found in COCO categories.")


def _bbox_xywh_to_xyxy(b: List[float]) -> List[float]:
    x, y, w, h = [float(v) for v in b]
    return [x, y, x + w, y + h]


def _bbox_iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _select_largest_ann(anns: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not anns:
        return None
    if len(anns) == 1:
        return anns[0]

    def _score(ann: Dict[str, Any]) -> float:
        area = ann.get("area")
        if isinstance(area, (int, float)):
            return float(area)
        bbox = ann.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            return float(bbox[2]) * float(bbox[3])
        return 0.0

    return max(anns, key=_score)


def _extract_gt_keypoints(ann: Dict[str, Any], num_keypoints: int) -> Optional[List[List[float]]]:
    kps = ann.get("keypoints")
    if not (isinstance(kps, list) and len(kps) == 3 * num_keypoints):
        return None

    out: List[List[float]] = []
    for i in range(num_keypoints):
        out.append(
            [
                float(kps[3 * i + 0]),
                float(kps[3 * i + 1]),
                float(kps[3 * i + 2]),
            ]
        )
    return out


def _build_eval_records(
    coco: Dict[str, Any],
    dataset_root: Path,
    category_name: str,
    num_keypoints: int,
) -> List[Dict[str, Any]]:
    target_cat_id = _find_category_id(coco, category_name)

    images_root = dataset_root / "images"
    if not images_root.exists():
        images_root = dataset_root

    img_by_id: Dict[int, Dict[str, Any]] = {}
    for img in coco.get("images", []):
        img_id = img.get("id")
        if isinstance(img_id, int):
            img_by_id[img_id] = img

    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("category_id") != target_cat_id:
            continue
        img_id = ann.get("image_id")
        bbox = ann.get("bbox")
        if not isinstance(img_id, int):
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        if _extract_gt_keypoints(ann, num_keypoints) is None:
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

        gt_kps = _extract_gt_keypoints(ann, num_keypoints)
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


def _select_pose_prediction(
    result: Any,
    score_thr: float,
    gt_box: List[float],
) -> Tuple[Optional[List[float]], Optional[np.ndarray], Optional[float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None, None, None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()

    keep = np.where(conf >= score_thr)[0]
    if keep.size == 0:
        return None, None, None

    best_idx = int(keep[0])
    best_iou = -1.0
    best_score = -1.0
    for idx in keep:
        iou = _bbox_iou_xyxy([float(v) for v in xyxy[int(idx)].tolist()], gt_box)
        score = float(conf[int(idx)])
        if iou > best_iou + 1e-9 or (abs(iou - best_iou) <= 1e-9 and score > best_score):
            best_iou = iou
            best_score = score
            best_idx = int(idx)

    pred_box = [float(v) for v in xyxy[best_idx].tolist()]
    pred_kps = _extract_pose_xy(result, det_idx=best_idx)
    pred_score = float(conf[best_idx])
    return pred_box, pred_kps, pred_score


def _angle_deg_from_points(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _angular_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def _compute_pose_custom_metrics(
    model: YOLO,
    cfg: Dict[str, Any],
    split: str,
    imgsz: int,
    device: str,
    score_thr: float,
) -> Dict[str, float]:
    dataset_root = Path(cfg["paths"]["raw_ds_path"]).resolve()
    coco_path = _resolve_split_coco_path(cfg, split)
    coco = _load_json(coco_path)

    category_name = str(cfg.get("dataset", {}).get("category_name", "gauge"))
    kp_cfg = cfg.get("keypoints", {})
    kp_names = [str(v) for v in kp_cfg.get("names", ["center", "needle_tip", "scale_start", "scale_end"])]
    num_keypoints = int(kp_cfg.get("num_keypoints", len(kp_names)))

    records = _build_eval_records(
        coco=coco,
        dataset_root=dataset_root,
        category_name=category_name,
        num_keypoints=num_keypoints,
    )

    if not records:
        return {
            "PCK@0.05": float("nan"),
            "PCK@0.10": float("nan"),
            "mean_angular_error_deg": float("nan"),
        }

    center_idx = kp_names.index("center") if "center" in kp_names else 0
    tip_idx = kp_names.index("needle_tip") if "needle_tip" in kp_names else min(1, num_keypoints - 1)

    total_visible = 0
    correct_005 = 0
    correct_010 = 0
    angle_errors: List[float] = []
    images_with_prediction = 0

    for rec in records:
        img_path = Path(rec["image_path"]).resolve()
        gt_box = [float(v) for v in rec["bbox"]]
        gt_kps = rec["keypoints"]

        with Image.open(img_path) as im:
            np_img = np.asarray(im.convert("RGB"), dtype=np.uint8)

        pred_result = model.predict(
            source=np_img,
            conf=score_thr,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]

        _, pred_kps, _ = _select_pose_prediction(pred_result, score_thr=score_thr, gt_box=gt_box)

        bw = max(1.0, gt_box[2] - gt_box[0])
        bh = max(1.0, gt_box[3] - gt_box[1])
        scale = max(bw, bh)

        if pred_kps is not None and len(pred_kps) >= num_keypoints:
            images_with_prediction += 1
            for i in range(num_keypoints):
                gx, gy, gv = gt_kps[i]
                if gv <= 0:
                    continue

                total_visible += 1
                px = float(pred_kps[i][0])
                py = float(pred_kps[i][1])
                d = math.hypot(px - float(gx), py - float(gy))
                if d <= 0.05 * scale:
                    correct_005 += 1
                if d <= 0.10 * scale:
                    correct_010 += 1

            if (
                center_idx < num_keypoints
                and tip_idx < num_keypoints
                and float(gt_kps[center_idx][2]) > 0
                and float(gt_kps[tip_idx][2]) > 0
            ):
                gt_angle = _angle_deg_from_points(
                    float(gt_kps[center_idx][0]),
                    float(gt_kps[center_idx][1]),
                    float(gt_kps[tip_idx][0]),
                    float(gt_kps[tip_idx][1]),
                )
                pred_angle = _angle_deg_from_points(
                    float(pred_kps[center_idx][0]),
                    float(pred_kps[center_idx][1]),
                    float(pred_kps[tip_idx][0]),
                    float(pred_kps[tip_idx][1]),
                )
                angle_errors.append(_angular_diff_deg(pred_angle, gt_angle))
        else:
            for i in range(num_keypoints):
                if float(gt_kps[i][2]) > 0:
                    total_visible += 1

    pck_005 = float(correct_005 / total_visible) if total_visible > 0 else float("nan")
    pck_010 = float(correct_010 / total_visible) if total_visible > 0 else float("nan")
    mean_angular_error = (
        float(sum(angle_errors) / len(angle_errors)) if angle_errors else float("nan")
    )

    return {
        "PCK@0.05": pck_005,
        "PCK@0.10": pck_010,
        "mean_angular_error_deg": mean_angular_error,
        "pck_visible_points": float(total_visible),
        "angular_samples": float(len(angle_errors)),
        "pose_detection_rate": float(images_with_prediction / len(records)),
    }


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    paths = cfg.get("paths", {})
    tcfg = dict(cfg.get("training", {}))
    mcfg = dict(cfg.get("model", {}))

    if args.epochs is not None:
        tcfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        tcfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        tcfg["num_workers"] = int(args.num_workers)
    if args.lr is not None:
        tcfg["lr0"] = float(args.lr)
    if args.weight_decay is not None:
        tcfg["weight_decay"] = float(args.weight_decay)
    if args.imgsz is not None:
        mcfg["imgsz"] = int(args.imgsz)

    epochs = int(tcfg.get("epochs", 100))
    batch_size = int(tcfg.get("batch_size", 16))
    num_workers = int(tcfg.get("num_workers", 4))
    lr0 = float(tcfg.get("lr0", 1e-3))
    weight_decay = float(tcfg.get("weight_decay", 1e-4))
    imgsz = int(mcfg.get("imgsz", 640))
    optimizer = str(tcfg.get("optimizer", "AdamW"))
    cos_lr = str(tcfg.get("lr_scheduler", "cosine")).lower() == "cosine"
    seed = int(tcfg.get("seed", 42))
    device = _resolve_yolo_device(str(tcfg.get("device", "auto")))
    model_name = _normalize_model_name(str(mcfg.get("name", "yolov8n-pose.pt")))
    pretrained = bool(mcfg.get("pretrained", True))

    log_path = (
        Path(paths.get("processed_ds_path", "data/processed")).resolve()
        / "train_keypoints_yolo_pose.log"
    )
    logger = _setup_logger(log_path)

    prepare_data = True
    if args.no_prepare_data:
        prepare_data = False
    if args.prepare_data:
        prepare_data = True
    data_yaml = _ensure_data_yaml(cfg, cfg_path, prepare_data=prepare_data, logger=logger)

    weights_dir = _resolve_weights_dir(cfg, model_name=model_name)
    weights_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"model={model_name}")
    logger.info(f"data={data_yaml}")
    logger.info(f"weights_dir={weights_dir}")
    logger.info(
        "train args: "
        f"epochs={epochs} batch={batch_size} imgsz={imgsz} lr0={lr0} "
        f"optimizer={optimizer} cos_lr={cos_lr} device={device}"
    )

    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        lr0=lr0,
        optimizer=optimizer,
        weight_decay=weight_decay,
        cos_lr=cos_lr,
        workers=num_workers,
        seed=seed,
        device=device,
        project=str(weights_dir.parent),
        name=weights_dir.name,
        exist_ok=True,
        pretrained=pretrained,
    )

    _copy_best_last(weights_dir)
    logger.info("Training finished.")

    eval_cfg = cfg.get("evaluation", {})
    split = str(eval_cfg.get("split", "test"))
    score_thr = float(eval_cfg.get("score_thr", 0.25))

    logger.info(f"Running validation on split={split} ...")
    val_result = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
    )

    metrics = _extract_pose_metrics(val_result)

    logger.info("Running custom keypoint metrics (PCK + angular error) ...")
    custom_metrics = _compute_pose_custom_metrics(
        model=model,
        cfg=cfg,
        split=split,
        imgsz=imgsz,
        device=device,
        score_thr=score_thr,
    )
    metrics.update(custom_metrics)

    if metrics:
        logger.info(" ".join(f"{k}={v:.6f}" for k, v in metrics.items()))
        summary_path = (
            Path(paths.get("processed_ds_path", "data/processed")).resolve()
            / "keypoints_metrics.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info(f"Saved metrics: {summary_path}")


if __name__ == "__main__":
    main()
