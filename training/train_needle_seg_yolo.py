from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config
from utils.runtime import (
    copy_best_last_weights,
    normalize_model_name,
    resolve_task_weights_dir,
    resolve_yolo_device,
    setup_logger,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config_segmentation.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--prepare-data", action="store_true")
    ap.add_argument("--no-prepare-data", action="store_true")
    return ap.parse_args()


def _ensure_data_yaml(
    cfg: Dict[str, Any],
    config_path: Path,
    prepare_data: bool,
    logger,
) -> Path:
    data_yaml = Path(cfg["paths"]["yolo_data_yaml"]).resolve()
    if data_yaml.exists() and not prepare_data:
        return data_yaml

    from data.build_needle_seg_yolo import build_from_config

    logger.info("Preparing needle segmentation labels from semantic masks...")
    produced_yaml = build_from_config(config_path=config_path, out_yaml=data_yaml)
    if not produced_yaml.exists():
        raise FileNotFoundError(f"YOLO segmentation data yaml was not created: {data_yaml}")
    return produced_yaml


def _extract_seg_metrics(val_result: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}

    box = getattr(val_result, "box", None)
    if box is not None:
        for attr, name in [
            ("mp", "bbox_precision"),
            ("mr", "bbox_recall"),
            ("map50", "bbox_mAP@0.5"),
            ("map", "bbox_mAP@0.5:0.95"),
        ]:
            value = getattr(box, attr, None)
            if value is not None:
                out[name] = float(value)

    seg = getattr(val_result, "seg", None)
    if seg is None:
        seg = getattr(val_result, "mask", None)
    if seg is not None:
        for attr, name in [
            ("mp", "mask_precision"),
            ("mr", "mask_recall"),
            ("map50", "mask_mAP@0.5"),
            ("map", "mask_mAP@0.5:0.95"),
        ]:
            value = getattr(seg, attr, None)
            if value is not None:
                out[name] = float(value)

    return out


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_semantic_json_path(cfg: Dict[str, Any], split: str) -> Path:
    key = f"{split}_semantic_json"
    rel = cfg.get("paths", {}).get(key, f"annotations/semantic_{split}.json")
    dataset_root = Path(cfg["paths"]["raw_ds_path"]).resolve()
    p = (dataset_root / str(rel)).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Semantic JSON file not found: {p}")
    return p


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


def _rel_after_anchor(path_value: str, anchor: str, split: str) -> Path:
    parts = Path(path_value.replace("\\", "/")).parts
    if anchor in parts:
        parts = parts[parts.index(anchor) + 1 :]
    if parts and parts[0] == split:
        parts = parts[1:]
    if not parts:
        return Path(Path(path_value).name)
    return Path(*parts)


def _resolve_raw_rel_path(dataset_root: Path, value: str) -> Path:
    return (dataset_root / Path(value.replace("\\", "/"))).resolve()


def _build_seg_eval_records(cfg: Dict[str, Any], split: str) -> tuple[List[Dict[str, Any]], int]:
    dataset_root = Path(cfg["paths"]["raw_ds_path"]).resolve()
    semantic_path = _resolve_semantic_json_path(cfg, split)
    semantic = _load_json(semantic_path)
    target_class_id = _resolve_target_class_id(cfg, semantic)

    yolo_root = Path(str(cfg.get("paths", {}).get("yolo_dataset_root", ""))).resolve()
    yolo_images_split = yolo_root / "images" / split
    yolo_masks_split = yolo_root / "masks" / split
    crop_dial = bool(cfg.get("segmentation", {}).get("crop_dial", True))

    if yolo_images_split.exists() and yolo_masks_split.exists():
        records: List[Dict[str, Any]] = []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for image_path in sorted(
            p for p in yolo_images_split.rglob("*") if p.is_file() and p.suffix.lower() in exts
        ):
            rel = image_path.relative_to(yolo_images_split).with_suffix(".png")
            mask_path = yolo_masks_split / rel
            if mask_path.exists():
                records.append(
                    {
                        "image_path": image_path.resolve(),
                        "mask_path": mask_path.resolve(),
                        "mask_mode": "binary",
                    }
                )
        if records:
            return records, target_class_id

    if crop_dial:
        return [], target_class_id

    records_json = semantic.get("images", [])
    if not isinstance(records_json, list):
        return [], target_class_id

    records = []
    for rec in records_json:
        image_file = rec.get("image_file")
        mask_file = rec.get("mask_file")
        if not isinstance(image_file, str) or not isinstance(mask_file, str):
            continue

        image_rel = _rel_after_anchor(image_file, anchor="images", split=split)
        yolo_image_path = (yolo_images_split / image_rel).resolve()
        raw_image_path = _resolve_raw_rel_path(dataset_root, image_file)
        image_path = yolo_image_path if yolo_image_path.exists() else raw_image_path
        mask_path = _resolve_raw_rel_path(dataset_root, mask_file)
        if image_path.exists() and mask_path.exists():
            records.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "mask_mode": "indexed",
                }
            )

    return records, target_class_id


def _resize_binary_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    pil = pil.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(pil) > 0


def _load_gt_mask(rec: Dict[str, Any], target_class_id: int) -> np.ndarray:
    with Image.open(Path(rec["mask_path"])) as im:
        mask_arr = np.asarray(im)
    if rec.get("mask_mode") == "binary":
        return mask_arr > 0
    return mask_arr == target_class_id


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


def _compute_seg_custom_metrics(
    model: YOLO,
    cfg: Dict[str, Any],
    split: str,
    imgsz: int,
    device: str,
    score_thr: float,
) -> Dict[str, float]:
    records, target_class_id = _build_seg_eval_records(cfg, split)
    if not records:
        return {
            "needle_iou": float("nan"),
            "needle_dice": float("nan"),
            "needle_pixel_precision": float("nan"),
            "needle_pixel_recall": float("nan"),
        }

    tp = 0
    fp = 0
    fn = 0
    pred_positive_images = 0
    gt_positive_images = 0
    area_abs_rel_errors: List[float] = []

    for rec in records:
        image_path = rec["image_path"]
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size
            np_img = np.asarray(rgb, dtype=np.uint8)
        gt_mask = _load_gt_mask(rec, target_class_id)
        if gt_mask.shape != (height, width):
            gt_mask = _resize_binary_mask(gt_mask, width=width, height=height)

        result = model.predict(
            source=np_img,
            conf=score_thr,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]
        pred_mask = _prediction_mask(result, width=width, height=height, score_thr=score_thr)

        gt_count = int(gt_mask.sum())
        pred_count = int(pred_mask.sum())
        if gt_count > 0:
            gt_positive_images += 1
        if pred_count > 0:
            pred_positive_images += 1
        area_abs_rel_errors.append(abs(pred_count - gt_count) / max(1.0, float(gt_count)))

        tp += int(np.logical_and(pred_mask, gt_mask).sum())
        fp += int(np.logical_and(pred_mask, ~gt_mask).sum())
        fn += int(np.logical_and(~pred_mask, gt_mask).sum())

    denom_iou = tp + fp + fn
    denom_dice = 2 * tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn

    return {
        "needle_iou": float(tp / denom_iou) if denom_iou > 0 else float("nan"),
        "needle_dice": float((2 * tp) / denom_dice) if denom_dice > 0 else float("nan"),
        "needle_pixel_precision": float(tp / precision_den) if precision_den > 0 else float("nan"),
        "needle_pixel_recall": float(tp / recall_den) if recall_den > 0 else float("nan"),
        "needle_area_mae_ratio": float(sum(area_abs_rel_errors) / len(area_abs_rel_errors)),
        "needle_mask_detection_rate": float(pred_positive_images / max(1, gt_positive_images)),
        "segmentation_eval_images": float(len(records)),
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
    device = resolve_yolo_device(str(tcfg.get("device", "auto")))
    model_name = normalize_model_name(str(mcfg.get("name", "yolo11n-seg.pt")))
    pretrained = bool(mcfg.get("pretrained", True))

    log_path = (
        Path(paths.get("processed_ds_path", "data/processed")).resolve()
        / "train_needle_seg_yolo.log"
    )
    logger = setup_logger("train_needle_seg_yolo", log_path)

    prepare_data = True
    if args.no_prepare_data:
        prepare_data = False
    if args.prepare_data:
        prepare_data = True
    data_yaml = _ensure_data_yaml(cfg, cfg_path, prepare_data=prepare_data, logger=logger)

    weights_dir = resolve_task_weights_dir(
        cfg,
        weights_key="weights_dir_seg",
        task_prefix="seg",
        model_identifier=model_name,
    )
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

    copy_best_last_weights(weights_dir)
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

    metrics = _extract_seg_metrics(val_result)
    logger.info("Running custom needle segmentation metrics (IoU/Dice/pixel P/R) ...")
    metrics.update(
        _compute_seg_custom_metrics(
            model=model,
            cfg=cfg,
            split=split,
            imgsz=imgsz,
            device=device,
            score_thr=score_thr,
        )
    )

    if metrics:
        logger.info(" ".join(f"{k}={v:.6f}" for k, v in metrics.items()))
        summary_path = (
            Path(paths.get("processed_ds_path", "data/processed")).resolve()
            / "needle_segmentation_metrics.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info(f"Saved metrics: {summary_path}")


if __name__ == "__main__":
    main()
