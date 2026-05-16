from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
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
    setup_logger,
)


Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "End-to-end gauge reading evaluation: dial detection -> crop -> "
            "keypoints + needle segmentation -> angles -> normalized reading."
        )
    )
    ap.add_argument("--config", type=str, default="configs/config_full_pipeline.yaml")
    ap.add_argument("--det-config", type=str, default=None)
    ap.add_argument("--kp-config", type=str, default=None)
    ap.add_argument("--seg-config", type=str, default=None)
    ap.add_argument("--det-weights", type=str, default=None)
    ap.add_argument("--kp-weights", type=str, default=None)
    ap.add_argument("--seg-weights", type=str, default=None)
    ap.add_argument("--split", choices=["train", "val", "test"], default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default="from-config")
    ap.add_argument("--det-thr", type=float, default=None)
    ap.add_argument("--kp-thr", type=float, default=None)
    ap.add_argument("--seg-thr", type=float, default=None)
    ap.add_argument("--det-imgsz", type=int, default=None)
    ap.add_argument("--kp-imgsz", type=int, default=None)
    ap.add_argument("--seg-imgsz", type=int, default=None)
    ap.add_argument("--crop-pad-ratio", type=float, default=None)
    ap.add_argument("--acc-tolerance", type=float, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--predictions-out",
        type=str,
        default=None,
        help="Optional JSON file with per-image pipeline outputs.",
    )
    return ap.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_relative_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def _resolve_pipeline_config_path(args: argparse.Namespace, key: str, default: str) -> Path:
    explicit = getattr(args, f"{key}_config")
    if explicit:
        return _resolve_relative_path(explicit)

    pipeline_cfg = load_config(args.config)
    configured = pipeline_cfg.get("configs", {}).get(key, default)
    return _resolve_relative_path(str(configured))


def _resolve_device(requested: str, *cfg_devices: str) -> str:
    if requested != "from-config":
        return resolve_yolo_device(str(requested))
    for cfg_device in cfg_devices:
        if cfg_device:
            return resolve_yolo_device(str(cfg_device))
    return resolve_yolo_device("auto")


def _resolve_yolo_weights(
    cfg: Dict[str, Any],
    explicit: Optional[str],
    *,
    weights_key: str,
    task_prefix: str,
    default_model: str,
) -> Path:
    model_name = normalize_model_name(str(cfg.get("model", {}).get("name", default_model)))
    weights_dir = resolve_task_weights_dir(
        cfg,
        weights_key=weights_key,
        task_prefix=task_prefix,
        model_identifier=model_name,
    )
    return find_weights_path(
        explicit_path=explicit,
        weights_dir=weights_dir,
        include_nested_weights_dir=True,
    )


def _bbox_xywh_to_xyxy_float(bbox: Iterable[float]) -> List[float]:
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def _clip_bbox_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    w: int,
    h: int,
) -> BBox:
    x1i = int(max(0, min(w - 1, int(round(x1)))))
    y1i = int(max(0, min(h - 1, int(round(y1)))))
    x2i = int(max(x1i + 1, min(w, int(round(x2)))))
    y2i = int(max(y1i + 1, min(h, int(round(y2)))))
    return x1i, y1i, x2i, y2i


def _crop_box_from_detection(
    box_xyxy: List[float],
    img_w: int,
    img_h: int,
    pad_ratio: float,
) -> BBox:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh) * (1.0 + 2.0 * max(0.0, pad_ratio))
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return _clip_bbox_xyxy(
        cx - side / 2.0,
        cy - side / 2.0,
        cx + side / 2.0,
        cy + side / 2.0,
        w=img_w,
        h=img_h,
    )


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


def _find_category(coco: Dict[str, Any], category_name: str) -> Dict[str, Any]:
    categories = coco.get("categories", [])
    for cat in categories:
        if cat.get("name") == category_name and isinstance(cat.get("id"), int):
            return cat
    if len(categories) == 1 and isinstance(categories[0].get("id"), int):
        return categories[0]
    raise ValueError(f"Category '{category_name}' not found in COCO categories.")


def _resolve_image_path(dataset_root: Path, file_name: str) -> Optional[Path]:
    rel = Path(file_name.replace("\\", "/"))
    candidates = [
        dataset_root / "images" / rel,
        dataset_root / rel,
        dataset_root / "images" / rel.name,
    ]
    for candidate in candidates:
        p = candidate.resolve()
        if p.exists():
            return p
    return None


def _resolve_split_coco_path(cfg: Dict[str, Any], split: str) -> Path:
    rel = cfg.get("paths", {}).get(f"{split}_inst_coco")
    if not rel:
        raise KeyError(f"Missing config key: paths.{split}_inst_coco")
    dataset_root = Path(str(cfg["paths"]["raw_ds_path"])).resolve()
    path = (dataset_root / str(rel)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"COCO file not found: {path}")
    return path


def _resolve_semantic_json_path(cfg: Dict[str, Any], split: str) -> Path:
    rel = cfg.get("paths", {}).get(f"{split}_semantic_json", f"annotations/semantic_{split}.json")
    dataset_root = Path(str(cfg["paths"]["raw_ds_path"])).resolve()
    path = (dataset_root / str(rel)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Semantic JSON file not found: {path}")
    return path


def _resolve_target_class_id(cfg: Dict[str, Any], semantic: Dict[str, Any]) -> int:
    seg_cfg = cfg.get("segmentation", {})
    explicit = seg_cfg.get("target_class_id")
    if explicit is not None:
        return int(explicit)
    target_name = str(seg_cfg.get("target_class_name", "needle"))
    for cls in semantic.get("classes", []):
        if cls.get("name") == target_name:
            return int(cls["id"])
    raise ValueError(f"Semantic class '{target_name}' not found.")


def _build_keypoint_indices(category: Dict[str, Any], keypoint_names: List[str]) -> List[int]:
    source_names = [str(v) for v in category.get("keypoints", [])]
    if not source_names:
        return list(range(len(keypoint_names)))
    idx_by_name = {name: idx for idx, name in enumerate(source_names)}
    missing = [name for name in keypoint_names if name not in idx_by_name]
    if missing:
        raise ValueError(
            "Configured keypoints are not present in COCO category keypoints: "
            + ", ".join(missing)
        )
    return [idx_by_name[name] for name in keypoint_names]


def _extract_gt_keypoints(
    ann: Dict[str, Any],
    keypoint_names: List[str],
    keypoint_indices: List[int],
) -> Optional[Dict[str, Tuple[float, float, float]]]:
    kps = ann.get("keypoints")
    max_idx = max(keypoint_indices) if keypoint_indices else -1
    if not isinstance(kps, list) or len(kps) < 3 * (max_idx + 1):
        return None

    out: Dict[str, Tuple[float, float, float]] = {}
    for name, idx in zip(keypoint_names, keypoint_indices):
        out[name] = (
            float(kps[3 * idx + 0]),
            float(kps[3 * idx + 1]),
            float(kps[3 * idx + 2]),
        )
    return out


def _extract_reading(ann: Dict[str, Any]) -> Optional[float]:
    value = ann.get("reading_normalized")
    if isinstance(value, (int, float)):
        return float(value)
    attrs = ann.get("attributes")
    if isinstance(attrs, dict):
        value = attrs.get("reading_normalized")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_needle_angle(ann: Dict[str, Any]) -> Optional[float]:
    value = ann.get("needle_angle_cw_deg")
    if isinstance(value, (int, float)):
        return float(value) % 360.0
    attrs = ann.get("attributes")
    if isinstance(attrs, dict):
        value = attrs.get("needle_angle_cw_deg")
        if isinstance(value, (int, float)):
            return float(value) % 360.0
    return None


def _build_semantic_by_image_id(semantic: Dict[str, Any], dataset_root: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for rec in semantic.get("images", []):
        image_id = rec.get("id")
        mask_file = rec.get("mask_file")
        if not isinstance(image_id, int) or not isinstance(mask_file, str):
            continue
        mask_path = (dataset_root / Path(mask_file.replace("\\", "/"))).resolve()
        if mask_path.exists():
            out[image_id] = mask_path
    return out


def _bbox_center(ann: Dict[str, Any]) -> Optional[Point]:
    bbox = ann.get("bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0:
        return None
    return x + w / 2.0, y + h / 2.0


def _category_ids_by_name(coco: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for category in coco.get("categories", []):
        name = category.get("name")
        category_id = category.get("id")
        if isinstance(name, str) and isinstance(category_id, int):
            out[name] = int(category_id)
    return out


def _resolve_landmark_split_paths(
    pipeline_cfg: Dict[str, Any],
    split: str,
) -> Tuple[Path, Path]:
    dataset_cfg = pipeline_cfg.get("dataset", {})
    root_raw = dataset_cfg.get("raw_ds_path")
    if not root_raw:
        raise KeyError("Missing config key for landmark dataset: dataset.raw_ds_path")
    root = _resolve_relative_path(str(root_raw))
    split_dirs = dataset_cfg.get(
        "split_dirs",
        {"train": "train", "val": "valid", "test": "test"},
    )
    split_dir_name = str(split_dirs.get(split, split))
    split_dir = (root / split_dir_name).resolve()
    annotation_file = str(dataset_cfg.get("annotation_file", "_annotations.coco.json"))
    annotation_path = (split_dir / annotation_file).resolve()
    if not annotation_path.exists():
        raise FileNotFoundError(f"COCO landmark annotation file not found: {annotation_path}")
    return annotation_path, split_dir


def _build_landmark_eval_records(
    pipeline_cfg: Dict[str, Any],
    kp_cfg: Dict[str, Any],
    split: str,
) -> Tuple[List[Dict[str, Any]], int]:
    annotation_path, split_dir = _resolve_landmark_split_paths(pipeline_cfg, split)
    coco = _load_json(annotation_path)

    category_ids = _category_ids_by_name(coco)
    dataset_cfg = pipeline_cfg.get("dataset", {})
    category_map = dataset_cfg.get(
        "category_map",
        {
            "center": "center",
            "scale_start": "min",
            "scale_end": "max",
            "needle_tip": "pointer_tip",
            "meter": "meter",
        },
    )
    category_map = {str(k): str(v) for k, v in category_map.items()}
    required_roles = ["center", "scale_start", "scale_end", "needle_tip", "meter"]
    missing_categories = [
        category_map[role]
        for role in required_roles
        if category_map.get(role) not in category_ids
    ]
    if missing_categories:
        raise ValueError(
            "Missing landmark categories in COCO file: " + ", ".join(missing_categories)
        )

    role_to_cat_id = {role: category_ids[category_map[role]] for role in required_roles}
    keypoint_names = [
        str(v)
        for v in kp_cfg.get("keypoints", {}).get(
            "names",
            ["center", "scale_start", "scale_end"],
        )
    ]

    images_by_id: Dict[int, Dict[str, Any]] = {}
    for image in coco.get("images", []):
        image_id = image.get("id")
        if isinstance(image_id, int):
            images_by_id[image_id] = image

    anns_by_image: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    cat_id_to_role = {cat_id: role for role, cat_id in role_to_cat_id.items()}
    for ann in coco.get("annotations", []):
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        if not isinstance(image_id, int):
            continue
        role = cat_id_to_role.get(category_id)
        if role is None:
            continue
        anns_by_image.setdefault(image_id, {}).setdefault(role, []).append(ann)

    records: List[Dict[str, Any]] = []
    for image_id, image in images_by_id.items():
        file_name = image.get("file_name")
        if not isinstance(file_name, str):
            continue
        image_path = (split_dir / Path(file_name.replace("\\", "/"))).resolve()
        if not image_path.exists():
            continue

        grouped = anns_by_image.get(image_id, {})
        selected: Dict[str, Dict[str, Any]] = {}
        for role in required_roles:
            ann = _select_largest_ann(grouped.get(role, []))
            if ann is not None:
                selected[role] = ann
        if any(role not in selected for role in required_roles):
            continue

        center = _bbox_center(selected["center"])
        scale_start = _bbox_center(selected["scale_start"])
        scale_end = _bbox_center(selected["scale_end"])
        needle_tip = _bbox_center(selected["needle_tip"])
        if center is None or scale_start is None or scale_end is None or needle_tip is None:
            continue

        start_angle = _angle_cw_deg(center, scale_start)
        end_angle = _angle_cw_deg(center, scale_end)
        needle_angle = _angle_cw_deg(center, needle_tip)
        reading = _normalized_reading_from_angles(
            start_angle=start_angle,
            end_angle=end_angle,
            needle_angle=needle_angle,
        )
        if reading is None:
            continue

        gt_keypoints = {
            "center": (center[0], center[1], 2.0),
            "scale_start": (scale_start[0], scale_start[1], 2.0),
            "scale_end": (scale_end[0], scale_end[1], 2.0),
        }
        records.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "mask_path": None,
                "bbox_xyxy": _bbox_xywh_to_xyxy_float(selected["meter"]["bbox"]),
                "keypoints": {
                    name: gt_keypoints[name]
                    for name in keypoint_names
                    if name in gt_keypoints
                },
                "needle_tip_xy": [needle_tip[0], needle_tip[1]],
                "reading_normalized": reading,
                "needle_angle_cw_deg": needle_angle,
                "target_source": "derived_from_landmarks",
            }
        )

    # Landmark COCO has no semantic class ids or usable mask annotations.
    return records, -1


def _build_synthetic_eval_records(
    det_cfg: Dict[str, Any],
    kp_cfg: Dict[str, Any],
    seg_cfg: Dict[str, Any],
    split: str,
) -> Tuple[List[Dict[str, Any]], int]:
    dataset_root = Path(str(det_cfg["paths"]["raw_ds_path"])).resolve()
    coco = _load_json(_resolve_split_coco_path(det_cfg, split))
    semantic = _load_json(_resolve_semantic_json_path(seg_cfg, split))
    target_class_id = _resolve_target_class_id(seg_cfg, semantic)
    mask_by_image_id = _build_semantic_by_image_id(semantic, dataset_root)

    category_name = str(det_cfg.get("dataset", {}).get("category_name", "gauge"))
    category = _find_category(coco, category_name)
    target_cat_id = int(category["id"])

    keypoint_names = [
        str(v)
        for v in kp_cfg.get("keypoints", {}).get(
            "names",
            ["center", "scale_start", "scale_end"],
        )
    ]
    keypoint_indices = _build_keypoint_indices(category, keypoint_names)

    images_by_id: Dict[int, Dict[str, Any]] = {}
    for image in coco.get("images", []):
        image_id = image.get("id")
        if isinstance(image_id, int):
            images_by_id[image_id] = image

    anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("category_id") != target_cat_id:
            continue
        image_id = ann.get("image_id")
        bbox = ann.get("bbox")
        if not isinstance(image_id, int):
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        if _extract_reading(ann) is None:
            continue
        if _extract_gt_keypoints(ann, keypoint_names, keypoint_indices) is None:
            continue
        anns_by_image.setdefault(image_id, []).append(ann)

    records: List[Dict[str, Any]] = []
    for image_id, anns in anns_by_image.items():
        image = images_by_id.get(image_id)
        if image is None:
            continue
        file_name = image.get("file_name")
        if not isinstance(file_name, str):
            continue
        image_path = _resolve_image_path(dataset_root, file_name)
        if image_path is None:
            continue
        ann = _select_largest_ann(anns)
        if ann is None:
            continue
        gt_keypoints = _extract_gt_keypoints(ann, keypoint_names, keypoint_indices)
        gt_reading = _extract_reading(ann)
        if gt_keypoints is None or gt_reading is None:
            continue
        records.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "mask_path": mask_by_image_id.get(image_id),
                "bbox_xyxy": _bbox_xywh_to_xyxy_float(ann["bbox"]),
                "keypoints": gt_keypoints,
                "reading_normalized": gt_reading,
                "needle_angle_cw_deg": _extract_needle_angle(ann),
            }
        )

    return records, target_class_id


def _build_eval_records(
    pipeline_cfg: Dict[str, Any],
    det_cfg: Dict[str, Any],
    kp_cfg: Dict[str, Any],
    seg_cfg: Dict[str, Any],
    split: str,
) -> Tuple[List[Dict[str, Any]], int]:
    dataset_format = str(pipeline_cfg.get("dataset", {}).get("format", "synthetic")).lower()
    if dataset_format in {"coco_landmarks", "roboflow_landmarks", "landmarks"}:
        return _build_landmark_eval_records(pipeline_cfg, kp_cfg, split)
    return _build_synthetic_eval_records(det_cfg, kp_cfg, seg_cfg, split)


def _sample_records(
    records: List[Dict[str, Any]],
    *,
    max_samples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(records):
        return records
    rng = random.Random(seed)
    return rng.sample(records, k=max_samples)


def _best_box(result: Any, score_thr: float) -> Optional[Tuple[List[float], float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()
    keep = [
        idx
        for idx in range(len(xyxy))
        if float(conf[idx]) >= score_thr and int(cls[idx]) == 0
    ]
    if not keep:
        return None
    best = max(keep, key=lambda idx: float(conf[idx]))
    return [float(v) for v in xyxy[best].tolist()], float(conf[best])


def _extract_pose_prediction(
    result: Any,
    score_thr: float,
) -> Tuple[Optional[List[float]], Optional[np.ndarray], Optional[float]]:
    chosen = _best_box(result, score_thr)
    if chosen is None:
        return None, None, None

    box, score = chosen
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return box, None, score

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    best_idx = None
    for idx in range(len(xyxy)):
        if abs(float(conf[idx]) - score) < 1e-9 and np.allclose(xyxy[idx], np.asarray(box)):
            best_idx = idx
            break
    if best_idx is None:
        best_idx = int(np.argmax(conf))

    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return box, None, score
    if hasattr(keypoints, "xy") and keypoints.xy is not None and len(keypoints.xy) > best_idx:
        return box, keypoints.xy[best_idx].detach().cpu().numpy(), score
    data = getattr(keypoints, "data", None)
    if data is not None and len(data) > best_idx:
        return box, data[best_idx, :, :2].detach().cpu().numpy(), score
    return box, None, score


def _resize_binary_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    pil = pil.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(pil) > 0


def _prediction_mask(result: Any, width: int, height: int, score_thr: float) -> np.ndarray:
    pred = np.zeros((height, width), dtype=bool)
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None or len(boxes) == 0:
        return pred

    data = getattr(masks, "data", None)
    if data is None or len(data) == 0:
        return pred

    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()
    mask_data = data.detach().cpu().numpy()
    for idx in range(mask_data.shape[0]):
        if float(conf[idx]) < score_thr or int(cls[idx]) != 0:
            continue
        mask = mask_data[idx] > 0.5
        if mask.shape != (height, width):
            mask = _resize_binary_mask(mask, width=width, height=height)
        pred |= mask
    return pred


def _load_gt_crop_mask(
    mask_path: Optional[Path],
    *,
    target_class_id: int,
    crop_box: BBox,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    if mask_path is None or not Path(mask_path).exists():
        return None
    x1, y1, x2, y2 = crop_box
    with Image.open(mask_path) as im:
        mask_arr = np.asarray(im)
    gt_mask = mask_arr == target_class_id
    gt_crop = gt_mask[y1:y2, x1:x2]
    if gt_crop.shape != (height, width):
        gt_crop = _resize_binary_mask(gt_crop, width=width, height=height)
    return gt_crop


def _mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 1.0
    return float(inter / union)


def _angle_cw_deg(center: Point, point: Point) -> float:
    cx, cy = center
    px, py = point
    return (math.degrees(math.atan2(px - cx, cy - py)) + 360.0) % 360.0


def _angular_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _normalized_reading_from_angles(
    start_angle: float,
    end_angle: float,
    needle_angle: float,
) -> Optional[float]:
    sweep = (end_angle - start_angle) % 360.0
    if sweep <= 1e-6:
        return None

    progress = (needle_angle - start_angle) % 360.0
    if progress <= sweep:
        return float(max(0.0, min(1.0, progress / sweep)))

    # If the angle falls just outside the visible scale arc, snap to the
    # nearest scale endpoint instead of wrapping it across the whole gauge.
    before_start = (start_angle - needle_angle) % 360.0
    after_end = (needle_angle - end_angle) % 360.0
    return 0.0 if before_start <= after_end else 1.0


def _needle_tip_from_mask(mask: np.ndarray, center: Point) -> Optional[Point]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    cx, cy = center
    dist2 = (xs.astype(float) - cx) ** 2 + (ys.astype(float) - cy) ** 2
    top_n = max(1, int(round(0.01 * len(xs))))
    if top_n >= len(xs):
        idx = np.arange(len(xs))
    else:
        idx = np.argpartition(dist2, -top_n)[-top_n:]
    return float(xs[idx].mean()), float(ys[idx].mean())


def _compute_kpt_errors(
    pred_kps: Optional[np.ndarray],
    gt_kps: Dict[str, Tuple[float, float, float]],
    keypoint_names: List[str],
    crop_box: BBox,
) -> Tuple[List[float], List[float]]:
    if pred_kps is None or len(pred_kps) < len(keypoint_names):
        return [], []

    x1, y1, x2, y2 = crop_box
    scale = max(1.0, float(max(x2 - x1, y2 - y1)))
    px_errors: List[float] = []
    norm_errors: List[float] = []
    for idx, name in enumerate(keypoint_names):
        gt = gt_kps.get(name)
        if gt is None or float(gt[2]) <= 0:
            continue
        gx = float(gt[0]) - float(x1)
        gy = float(gt[1]) - float(y1)
        px = float(pred_kps[idx][0])
        py = float(pred_kps[idx][1])
        err = math.hypot(px - gx, py - gy)
        px_errors.append(float(err))
        norm_errors.append(float(err / scale))
    return px_errors, norm_errors


def _nanmean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _evaluate_records(
    *,
    records: List[Dict[str, Any]],
    target_class_id: int,
    det_model: YOLO,
    kp_model: YOLO,
    seg_model: YOLO,
    keypoint_names: List[str],
    det_thr: float,
    kp_thr: float,
    seg_thr: float,
    det_imgsz: int,
    kp_imgsz: int,
    seg_imgsz: int,
    crop_pad_ratio: float,
    acc_tolerance: float,
    device: str,
    logger,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    reading_errors: List[float] = []
    angle_errors: List[float] = []
    kpt_errors_px: List[float] = []
    kpt_errors_norm: List[float] = []
    seg_ious: List[float] = []
    predictions: List[Dict[str, Any]] = []

    detected = 0
    keypoint_ok = 0
    mask_positive = 0
    reading_ok = 0
    acc_correct = 0

    for idx, rec in enumerate(records, start=1):
        image_path = Path(rec["image_path"])
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            full_np = np.asarray(rgb, dtype=np.uint8)
        full_h, full_w = full_np.shape[:2]

        sample: Dict[str, Any] = {
            "image_id": rec["image_id"],
            "image_path": str(image_path),
            "target_reading": float(rec["reading_normalized"]),
        }

        det_result = det_model.predict(
            source=full_np,
            conf=det_thr,
            imgsz=det_imgsz,
            device=device,
            verbose=False,
        )[0]
        det = _best_box(det_result, det_thr)
        if det is None:
            gt_mask = None
            if rec.get("mask_path") is not None:
                with Image.open(Path(rec["mask_path"])) as mask_im:
                    gt_mask = np.asarray(mask_im) == target_class_id
            if gt_mask is not None and int(gt_mask.sum()) > 0:
                seg_ious.append(0.0)
            predictions.append({**sample, "status": "no_detection"})
            if idx % 50 == 0:
                logger.info(f"processed={idx}/{len(records)}")
            continue

        detected += 1
        det_box, det_score = det
        crop_box = _crop_box_from_detection(
            det_box,
            img_w=full_w,
            img_h=full_h,
            pad_ratio=crop_pad_ratio,
        )
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        crop_np = full_np[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        crop_h, crop_w = crop_np.shape[:2]

        kp_result = kp_model.predict(
            source=crop_np,
            conf=kp_thr,
            imgsz=kp_imgsz,
            device=device,
            verbose=False,
        )[0]
        _, pred_kps, kp_score = _extract_pose_prediction(kp_result, kp_thr)

        px_errors, norm_errors = _compute_kpt_errors(
            pred_kps,
            rec["keypoints"],
            keypoint_names,
            crop_box,
        )
        kpt_errors_px.extend(px_errors)
        kpt_errors_norm.extend(norm_errors)
        if pred_kps is not None and len(pred_kps) >= len(keypoint_names):
            keypoint_ok += 1

        seg_result = seg_model.predict(
            source=crop_np,
            conf=seg_thr,
            imgsz=seg_imgsz,
            device=device,
            verbose=False,
        )[0]
        pred_mask = _prediction_mask(seg_result, width=crop_w, height=crop_h, score_thr=seg_thr)
        if int(pred_mask.sum()) > 0:
            mask_positive += 1

        gt_crop_mask = _load_gt_crop_mask(
            rec.get("mask_path"),
            target_class_id=target_class_id,
            crop_box=crop_box,
            width=crop_w,
            height=crop_h,
        )
        seg_iou = None
        if gt_crop_mask is not None:
            seg_iou = _mask_iou(pred_mask, gt_crop_mask)
            seg_ious.append(seg_iou)

        pred_reading = None
        pred_needle_angle = None
        if pred_kps is not None and len(pred_kps) >= len(keypoint_names):
            pred_by_name = {
                name: (float(pred_kps[i][0]), float(pred_kps[i][1]))
                for i, name in enumerate(keypoint_names)
            }
            if {"center", "scale_start", "scale_end"}.issubset(pred_by_name):
                center = pred_by_name["center"]
                tip = _needle_tip_from_mask(pred_mask, center)
                if tip is not None:
                    start_angle = _angle_cw_deg(center, pred_by_name["scale_start"])
                    end_angle = _angle_cw_deg(center, pred_by_name["scale_end"])
                    pred_needle_angle = _angle_cw_deg(center, tip)
                    pred_reading = _normalized_reading_from_angles(
                        start_angle=start_angle,
                        end_angle=end_angle,
                        needle_angle=pred_needle_angle,
                    )

        target_reading = float(rec["reading_normalized"])
        abs_error = None
        if pred_reading is not None:
            reading_ok += 1
            abs_error = abs(float(pred_reading) - target_reading)
            reading_errors.append(abs_error)
            if abs_error <= acc_tolerance:
                acc_correct += 1

        gt_needle_angle = rec.get("needle_angle_cw_deg")
        if pred_needle_angle is not None and gt_needle_angle is not None:
            angle_errors.append(_angular_diff_deg(pred_needle_angle, float(gt_needle_angle)))

        predictions.append(
            {
                **sample,
                "status": "ok" if pred_reading is not None else "no_reading",
                "det_score": float(det_score),
                "kp_score": float(kp_score) if kp_score is not None else None,
                "crop_xyxy": [int(v) for v in crop_box],
                "prediction_reading": float(pred_reading) if pred_reading is not None else None,
                "abs_reading_error": float(abs_error) if abs_error is not None else None,
                "segmentation_iou": float(seg_iou) if seg_iou is not None else None,
            }
        )

        if idx % 50 == 0:
            logger.info(f"processed={idx}/{len(records)}")

    total = len(records)
    metrics = {
        "reading_mae": _nanmean(reading_errors),
        "reading_acc@5%": float(acc_correct / total) if total > 0 else float("nan"),
        "kpt_error": _nanmean(kpt_errors_norm),
        "kpt_error_px": _nanmean(kpt_errors_px),
        "segmentation_iou": _nanmean(seg_ious),
        "needle_angle_mae_deg": _nanmean(angle_errors),
        "num_samples": float(total),
        "reading_valid_samples": float(reading_ok),
        "reading_coverage": float(reading_ok / total) if total > 0 else float("nan"),
        "detection_rate": float(detected / total) if total > 0 else float("nan"),
        "keypoint_detection_rate": float(keypoint_ok / total) if total > 0 else float("nan"),
        "needle_mask_detection_rate": float(mask_positive / total) if total > 0 else float("nan"),
        "segmentation_eval_samples": float(len(seg_ious)),
    }
    return metrics, predictions


def main() -> None:
    args = _parse_args()
    pipeline_cfg_path = _resolve_relative_path(args.config)
    pipeline_cfg = load_config(pipeline_cfg_path)

    det_cfg_path = _resolve_pipeline_config_path(args, "det", "configs/config_detection.yaml")
    kp_cfg_path = _resolve_pipeline_config_path(args, "kp", "configs/config_keypoints.yaml")
    seg_cfg_path = _resolve_pipeline_config_path(args, "seg", "configs/config_segmentation.yaml")
    det_cfg = load_config(det_cfg_path)
    kp_cfg = load_config(kp_cfg_path)
    seg_cfg = load_config(seg_cfg_path)

    eval_cfg = pipeline_cfg.get("evaluation", {})
    split = args.split or str(eval_cfg.get("split", "test"))
    seed = int(args.seed if args.seed is not None else eval_cfg.get("seed", 42))
    max_samples = args.max_samples
    if max_samples is None:
        max_samples = args.num_samples
    if max_samples is None:
        configured_max = eval_cfg.get("max_samples")
        max_samples = int(configured_max) if configured_max is not None else None

    det_eval = det_cfg.get("evaluation", {})
    kp_eval = kp_cfg.get("evaluation", {})
    seg_eval = seg_cfg.get("evaluation", {})
    det_model_cfg = det_cfg.get("model", {})
    kp_model_cfg = kp_cfg.get("model", {})
    seg_model_cfg = seg_cfg.get("model", {})

    det_thr = float(args.det_thr if args.det_thr is not None else det_eval.get("score_thr", 0.25))
    kp_thr = float(args.kp_thr if args.kp_thr is not None else kp_eval.get("score_thr", 0.25))
    seg_thr = float(args.seg_thr if args.seg_thr is not None else seg_eval.get("score_thr", 0.25))
    det_imgsz = int(args.det_imgsz if args.det_imgsz is not None else det_model_cfg.get("imgsz", 640))
    kp_imgsz = int(args.kp_imgsz if args.kp_imgsz is not None else kp_model_cfg.get("imgsz", 960))
    seg_imgsz = int(args.seg_imgsz if args.seg_imgsz is not None else seg_model_cfg.get("imgsz", 640))
    crop_pad_ratio = float(
        args.crop_pad_ratio
        if args.crop_pad_ratio is not None
        else eval_cfg.get(
            "crop_pad_ratio",
            kp_cfg.get("keypoints", {}).get("crop_pad_ratio", 0.08),
        )
    )
    acc_tolerance = float(
        args.acc_tolerance
        if args.acc_tolerance is not None
        else eval_cfg.get("acc_tolerance", 0.05)
    )
    device = _resolve_device(
        args.device,
        str(det_cfg.get("training", {}).get("device", "")),
        str(kp_cfg.get("training", {}).get("device", "")),
        str(seg_cfg.get("training", {}).get("device", "")),
    )

    processed_root = Path(
        pipeline_cfg.get("paths", {}).get(
            "processed_ds_path",
            det_cfg.get("paths", {}).get("processed_ds_path", "data/processed"),
        )
    ).resolve()
    logger = setup_logger("eval_full_pipeline", processed_root / "eval_full_pipeline.log")

    det_weights = _resolve_yolo_weights(
        det_cfg,
        args.det_weights,
        weights_key="weights_dir_det",
        task_prefix="det",
        default_model="yolo11n.pt",
    )
    kp_weights = _resolve_yolo_weights(
        kp_cfg,
        args.kp_weights,
        weights_key="weights_dir_kp",
        task_prefix="kp",
        default_model="yolo11n-pose.pt",
    )
    seg_weights = _resolve_yolo_weights(
        seg_cfg,
        args.seg_weights,
        weights_key="weights_dir_seg",
        task_prefix="seg",
        default_model="yolo11n-seg.pt",
    )

    logger.info(f"pipeline_config={pipeline_cfg_path}")
    logger.info(f"det_config={det_cfg_path} weights={det_weights}")
    logger.info(f"kp_config={kp_cfg_path} weights={kp_weights}")
    logger.info(f"seg_config={seg_cfg_path} weights={seg_weights}")
    logger.info(
        f"eval args: split={split} max_samples={max_samples} "
        f"det_thr={det_thr} kp_thr={kp_thr} seg_thr={seg_thr} "
        f"det_imgsz={det_imgsz} kp_imgsz={kp_imgsz} seg_imgsz={seg_imgsz} "
        f"crop_pad_ratio={crop_pad_ratio} acc_tolerance={acc_tolerance} device={device}"
    )

    records, target_class_id = _build_eval_records(
        pipeline_cfg,
        det_cfg,
        kp_cfg,
        seg_cfg,
        split,
    )
    records = _sample_records(records, max_samples=max_samples, seed=seed)
    if not records:
        raise RuntimeError(f"No evaluation records found for split={split}")
    logger.info(f"records={len(records)} target_class_id={target_class_id}")

    det_model = YOLO(str(det_weights))
    kp_model = YOLO(str(kp_weights))
    seg_model = YOLO(str(seg_weights))
    keypoint_names = [
        str(v)
        for v in kp_cfg.get("keypoints", {}).get(
            "names",
            ["center", "scale_start", "scale_end"],
        )
    ]

    metrics, predictions = _evaluate_records(
        records=records,
        target_class_id=target_class_id,
        det_model=det_model,
        kp_model=kp_model,
        seg_model=seg_model,
        keypoint_names=keypoint_names,
        det_thr=det_thr,
        kp_thr=kp_thr,
        seg_thr=seg_thr,
        det_imgsz=det_imgsz,
        kp_imgsz=kp_imgsz,
        seg_imgsz=seg_imgsz,
        crop_pad_ratio=crop_pad_ratio,
        acc_tolerance=acc_tolerance,
        device=device,
        logger=logger,
    )
    metrics["split"] = split
    metrics["det_weights_path"] = str(det_weights)
    metrics["kp_weights_path"] = str(kp_weights)
    metrics["seg_weights_path"] = str(seg_weights)
    if records and records[0].get("target_source"):
        metrics["target_reading_source"] = str(records[0]["target_source"])
    if target_class_id < 0:
        metrics["segmentation_iou_available"] = 0.0

    out_path = Path(args.out).resolve() if args.out else processed_root / "full_pipeline_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.predictions_out:
        pred_path = Path(args.predictions_out).resolve()
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
        logger.info(f"Saved predictions: {pred_path}")

    logger.info(" ".join(f"{k}={v}" for k, v in metrics.items()))
    logger.info(f"Saved metrics: {out_path}")


if __name__ == "__main__":
    main()
