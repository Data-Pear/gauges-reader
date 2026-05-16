from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms import build_transforms
from models.model import GaugeRegressor, ModelConfig
from utils.config import load_config
from utils.runtime import (
    find_weights_path,
    normalize_model_name,
    resolve_task_weights_dir,
    resolve_torch_device,
    resolve_yolo_device,
)

TASKS = ("detection", "keypoints", "segmentation", "needle_segmentation", "regression")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Unified inference entrypoint for detection, keypoints, segmentation, and regression."
    )
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--score-thr", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", type=str, default="from-config")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    return ap.parse_args()


def _default_config(task: str) -> Path:
    if task == "detection":
        return (PROJECT_ROOT / "configs" / "config_detection.yaml").resolve()
    if task == "keypoints":
        return (PROJECT_ROOT / "configs" / "config_keypoints.yaml").resolve()
    if task in {"segmentation", "needle_segmentation"}:
        return (PROJECT_ROOT / "configs" / "config_segmentation.yaml").resolve()
    return (PROJECT_ROOT / "configs" / "config_regression.yaml").resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_detection_or_keypoints_images(
    cfg: Dict[str, Any],
    split: str,
    *,
    use_yolo_split: bool,
) -> List[Path]:
    if use_yolo_split:
        yolo_root = Path(str(cfg.get("paths", {}).get("yolo_dataset_root", ""))).resolve()
        images_root = yolo_root / "images" / split
        if images_root.exists():
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            return sorted(
                [
                    p.resolve()
                    for p in images_root.rglob("*")
                    if p.is_file() and p.suffix.lower() in exts
                ]
            )

    key = f"{split}_inst_coco"
    rel = cfg.get("paths", {}).get(key)
    if not rel:
        raise KeyError(f"Missing config key: paths.{key}")
    raw_root = Path(str(cfg.get("paths", {}).get("raw_ds_path", ""))).resolve()
    coco_path = (raw_root / str(rel)).resolve()
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO file not found: {coco_path}")

    coco = _load_json(coco_path)
    images_root = raw_root / "images"
    if not images_root.exists():
        images_root = raw_root

    out: List[Path] = []
    for image in coco.get("images", []):
        file_name = image.get("file_name")
        if not isinstance(file_name, str):
            continue
        p = (images_root / file_name).resolve()
        if p.exists():
            out.append(p)
    return out


def _resolve_segmentation_images(cfg: Dict[str, Any], split: str) -> List[Path]:
    yolo_root = Path(str(cfg.get("paths", {}).get("yolo_dataset_root", ""))).resolve()
    images_root = yolo_root / "images" / split
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if images_root.exists():
        return sorted(
            [
                p.resolve()
                for p in images_root.rglob("*")
                if p.is_file() and p.suffix.lower() in exts
            ]
        )

    raw_root = Path(str(cfg.get("paths", {}).get("raw_ds_path", ""))).resolve()
    semantic_rel = cfg.get("paths", {}).get(
        f"{split}_semantic_json", f"annotations/semantic_{split}.json"
    )
    semantic_path = (raw_root / str(semantic_rel)).resolve()
    if not semantic_path.exists():
        raise FileNotFoundError(f"Semantic JSON file not found: {semantic_path}")

    semantic = _load_json(semantic_path)
    out: List[Path] = []
    for rec in semantic.get("images", []):
        image_file = rec.get("image_file")
        if not isinstance(image_file, str):
            continue
        p = (raw_root / Path(image_file.replace("\\", "/"))).resolve()
        if p.exists():
            out.append(p)
    return out


def _resolve_regression_index(cfg: Dict[str, Any], split: str) -> Path:
    paths = cfg.get("paths", {})
    if split == "train":
        keys = ("train_reg_output_json", "train_output_json", "train_inst_json")
    elif split == "val":
        keys = ("val_reg_output_json", "val_output_json", "val_inst_json")
    else:
        keys = (
            "test_reg_output_json",
            "test_output_json",
            "test_inst_json",
            "val_reg_output_json",
            "val_output_json",
            "val_inst_json",
        )
    for key in keys:
        value = paths.get(key)
        if not value:
            continue
        p = Path(str(value)).resolve()
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not resolve regression index for split='{split}'. Checked keys: {', '.join(keys)}"
    )


def _sample_images(image_paths: List[Path], num_samples: int, seed: int) -> List[Path]:
    if not image_paths:
        return []
    k = min(int(num_samples), len(image_paths))
    random.seed(seed)
    return random.sample(image_paths, k=k)


def _resolve_yolo_weights(cfg: Dict[str, Any], task: str, explicit: Optional[str]) -> Path:
    model_name_default = {
        "detection": "yolo11n.pt",
        "keypoints": "yolo11n-pose.pt",
        "segmentation": "yolo11n-seg.pt",
        "needle_segmentation": "yolo11n-seg.pt",
    }[task]
    model_name = normalize_model_name(str(cfg.get("model", {}).get("name", model_name_default)))
    if task == "detection":
        weights_key = "weights_dir_det"
        task_prefix = "det"
    elif task == "keypoints":
        weights_key = "weights_dir_kp"
        task_prefix = "kp"
    else:
        weights_key = "weights_dir_seg"
        task_prefix = "seg"
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


def _resolve_regression_weights(cfg: Dict[str, Any], explicit: Optional[str]) -> Path:
    model_identifier = str(cfg.get("model", {}).get("backbone", "resnet18")).lower()
    weights_dir = resolve_task_weights_dir(
        cfg,
        weights_key="weights_dir_reg",
        task_prefix="reg",
        model_identifier=model_identifier,
    )
    return find_weights_path(
        explicit_path=explicit,
        weights_dir=weights_dir,
        include_nested_weights_dir=False,
    )


def _predict_detection_on_image(
    model: YOLO,
    image_path: Path,
    *,
    imgsz: int,
    score_thr: float,
    device: str,
) -> Dict[str, Any]:
    result = model.predict(
        source=str(image_path),
        conf=score_thr,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]

    detections: List[Dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy()
        for i in range(len(xyxy)):
            detections.append(
                {
                    "bbox_xyxy": [float(v) for v in xyxy[i].tolist()],
                    "score": float(conf[i]),
                    "class_id": int(cls[i]),
                }
            )
    return {"image_path": str(image_path), "detections": detections}


def _extract_pose_xy(result: Any, det_idx: int) -> Optional[List[List[float]]]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return None
    if hasattr(keypoints, "xy") and keypoints.xy is not None:
        xy = keypoints.xy
        if len(xy) <= det_idx:
            return None
        return [[float(x), float(y)] for x, y in xy[det_idx].detach().cpu().numpy().tolist()]
    data = getattr(keypoints, "data", None)
    if data is None or len(data) <= det_idx:
        return None
    xy = data[det_idx, :, :2].detach().cpu().numpy()
    return [[float(x), float(y)] for x, y in xy.tolist()]


def _predict_keypoints_on_image(
    model: YOLO,
    image_path: Path,
    *,
    imgsz: int,
    score_thr: float,
    device: str,
) -> Dict[str, Any]:
    result = model.predict(
        source=str(image_path),
        conf=score_thr,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]

    detections: List[Dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy()
        for i in range(len(xyxy)):
            detections.append(
                {
                    "bbox_xyxy": [float(v) for v in xyxy[i].tolist()],
                    "score": float(conf[i]),
                    "class_id": int(cls[i]),
                    "keypoints_xy": _extract_pose_xy(result, i),
                }
            )
    return {"image_path": str(image_path), "detections": detections}


def _extract_mask_polygons(result: Any, det_idx: int) -> Optional[List[List[List[float]]]]:
    masks = getattr(result, "masks", None)
    if masks is None:
        return None
    xy = getattr(masks, "xy", None)
    if xy is None or len(xy) <= det_idx:
        return None
    polygons: List[List[List[float]]] = []
    for poly in [xy[det_idx]]:
        polygons.append([[float(x), float(y)] for x, y in poly.tolist()])
    return polygons


def _predict_segmentation_on_image(
    model: YOLO,
    image_path: Path,
    *,
    imgsz: int,
    score_thr: float,
    device: str,
) -> Dict[str, Any]:
    result = model.predict(
        source=str(image_path),
        conf=score_thr,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]

    detections: List[Dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy()
        for i in range(len(xyxy)):
            detections.append(
                {
                    "bbox_xyxy": [float(v) for v in xyxy[i].tolist()],
                    "score": float(conf[i]),
                    "class_id": int(cls[i]),
                    "mask_polygons_xy": _extract_mask_polygons(result, i),
                }
            )
    return {"image_path": str(image_path), "detections": detections}


def _load_regression_model(cfg: Dict[str, Any], weights_path: Path, device: torch.device) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    model = GaugeRegressor(
        ModelConfig(
            backbone=str(model_cfg.get("backbone", "resnet18")),
            pretrained=bool(model_cfg.get("pretrained", True)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    )
    checkpoint = torch.load(weights_path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _predict_regression_on_image(
    model: torch.nn.Module,
    transform: Any,
    image_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    x = transform(rgb).unsqueeze(0).to(device)
    y_pred = model(x).view(-1)[0].item()
    return {"image_path": str(image_path), "prediction": float(y_pred)}


def main() -> None:
    args = _parse_args()

    cfg_path = Path(args.config).resolve() if args.config else _default_config(args.task)
    cfg = load_config(cfg_path)

    if args.image is None and args.split is None:
        raise ValueError("Provide either --image for single inference or --split for dataset sampling.")

    if args.image is not None and args.split is not None:
        raise ValueError("Use either --image or --split, not both.")

    result: Dict[str, Any] = {
        "task": args.task,
        "config_path": str(cfg_path),
    }

    task = "segmentation" if args.task == "needle_segmentation" else args.task

    if task in ("detection", "keypoints", "segmentation"):
        training_cfg = cfg.get("training", {})
        model_cfg = cfg.get("model", {})
        eval_cfg = cfg.get("evaluation", {})
        score_thr = (
            float(args.score_thr)
            if args.score_thr is not None
            else float(eval_cfg.get("score_thr", 0.25))
        )
        imgsz_default = 960 if task == "keypoints" else 640
        imgsz = int(args.imgsz) if args.imgsz is not None else int(model_cfg.get("imgsz", imgsz_default))
        requested_device = training_cfg.get("device", "auto") if args.device == "from-config" else args.device
        device = resolve_yolo_device(str(requested_device))
        weights_path = _resolve_yolo_weights(cfg, task, args.weights)
        model = YOLO(str(weights_path))

        if args.image is not None:
            image_paths = [Path(args.image).resolve()]
            mode = "single_image"
            split = None
        else:
            split = str(args.split)
            if task == "segmentation":
                image_paths = _resolve_segmentation_images(cfg, split=split)
            else:
                use_yolo_split = bool(
                    task == "keypoints"
                    and cfg.get("keypoints", {}).get("crop_dial", False)
                )
                image_paths = _resolve_detection_or_keypoints_images(
                    cfg,
                    split=split,
                    use_yolo_split=use_yolo_split,
                )
            image_paths = _sample_images(image_paths, num_samples=args.num_samples, seed=args.seed)
            mode = "dataset_split"

        if not image_paths:
            raise RuntimeError("No images found for inference.")

        predictions: List[Dict[str, Any]] = []
        for image_path in image_paths:
            if task == "detection":
                predictions.append(
                    _predict_detection_on_image(
                        model,
                        image_path,
                        imgsz=imgsz,
                        score_thr=score_thr,
                        device=device,
                    )
                )
            elif task == "keypoints":
                predictions.append(
                    _predict_keypoints_on_image(
                        model,
                        image_path,
                        imgsz=imgsz,
                        score_thr=score_thr,
                        device=device,
                    )
                )
            else:
                predictions.append(
                    _predict_segmentation_on_image(
                        model,
                        image_path,
                        imgsz=imgsz,
                        score_thr=score_thr,
                        device=device,
                    )
                )

        result.update(
            {
                "mode": mode,
                "split": split,
                "weights_path": str(weights_path),
                "device": device,
                "score_thr": score_thr,
                "imgsz": imgsz,
                "num_predictions": len(predictions),
                "predictions": predictions,
            }
        )
    else:
        training_cfg = cfg.get("training", {})
        requested_device = training_cfg.get("device", "auto") if args.device == "from-config" else args.device
        device = resolve_torch_device(str(requested_device))
        weights_path = _resolve_regression_weights(cfg, args.weights)
        model = _load_regression_model(cfg, weights_path, device=device)
        transform = build_transforms(cfg, split="val")

        if args.image is not None:
            image_paths = [Path(args.image).resolve()]
            mode = "single_image"
            split = None
            gt_values: Dict[str, float] = {}
        else:
            split = str(args.split)
            index_path = _resolve_regression_index(cfg, split)
            rows = []
            with index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line))
            if not rows:
                raise RuntimeError(f"Regression index is empty: {index_path}")
            random.seed(args.seed)
            chosen = random.sample(rows, k=min(args.num_samples, len(rows)))
            image_paths = [Path(r["image_path"]).resolve() for r in chosen]
            gt_values = {str(Path(r["image_path"]).resolve()): float(r["value"]) for r in chosen}
            mode = "dataset_split"

        predictions = []
        for image_path in image_paths:
            pred = _predict_regression_on_image(model, transform, image_path, device)
            image_key = str(image_path)
            if image_key in gt_values:
                pred["target"] = gt_values[image_key]
                pred["abs_error"] = abs(pred["prediction"] - gt_values[image_key])
            predictions.append(pred)

        result.update(
            {
                "mode": mode,
                "split": split,
                "weights_path": str(weights_path),
                "device": str(device),
                "num_predictions": len(predictions),
                "predictions": predictions,
            }
        )

    payload = json.dumps(result, indent=2)
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"[OK] Saved predictions: {out_path}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
