from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_keypoints_yolo_pose import (
    _compute_pose_custom_metrics,
    _ensure_data_yaml,
    _extract_pose_metrics,
    _normalize_model_name,
    _resolve_weights_dir,
    _resolve_yolo_device,
)
from utils.config import load_config


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config_keypoints.yaml")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--score-thr", type=float, default=None)
    ap.add_argument("--prepare-data", action="store_true")
    ap.add_argument("--no-prepare-data", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    return ap.parse_args()


def _setup_logger(log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("eval_keypoints_yolo_pose")
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


def _resolve_weights_path(cfg: Dict[str, Any], explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if not p.exists():
            alt = (p.parent / "weights" / p.name).resolve()
            if alt.exists():
                return alt
            raise FileNotFoundError(
                f"Weights file not found: {p}\n"
                f"Also checked: {alt}"
            )
        return p

    model_name = _normalize_model_name(
        str(cfg.get("model", {}).get("name", "yolov8n-pose.pt"))
    )
    weights_dir = _resolve_weights_dir(cfg, model_name=model_name)
    candidates = [
        weights_dir / "best.pt",
        weights_dir / "last.pt",
        weights_dir / "weights" / "best.pt",
        weights_dir / "weights" / "last.pt",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        "No keypoints weights found. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    paths = cfg.get("paths", {})
    tcfg = cfg.get("training", {})
    mcfg = cfg.get("model", {})
    ecfg = cfg.get("evaluation", {})

    split = args.split or str(ecfg.get("split", "test"))
    batch_size = int(args.batch_size or tcfg.get("batch_size", 16))
    num_workers = int(args.num_workers or tcfg.get("num_workers", 4))
    imgsz = int(args.imgsz or mcfg.get("imgsz", 640))
    score_thr = float(args.score_thr or ecfg.get("score_thr", 0.25))
    device = _resolve_yolo_device(args.device or str(tcfg.get("device", "auto")))

    processed_root = Path(paths.get("processed_ds_path", "data/processed")).resolve()
    log_path = processed_root / "eval_keypoints_yolo_pose.log"
    logger = _setup_logger(log_path)

    prepare_data = False
    if args.no_prepare_data:
        prepare_data = False
    if args.prepare_data:
        prepare_data = True

    data_yaml = _ensure_data_yaml(cfg, cfg_path, prepare_data=prepare_data, logger=logger)
    weights_path = _resolve_weights_path(cfg, args.weights)

    logger.info(f"weights={weights_path}")
    logger.info(f"data={data_yaml}")
    logger.info(
        f"eval args: split={split} batch={batch_size} imgsz={imgsz} "
        f"workers={num_workers} score_thr={score_thr} device={device}"
    )

    model = YOLO(str(weights_path))
    val_result = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch_size,
        workers=num_workers,
        device=device,
    )

    metrics = _extract_pose_metrics(val_result)
    custom_metrics = _compute_pose_custom_metrics(
        model=model,
        cfg=cfg,
        split=split,
        imgsz=imgsz,
        device=device,
        score_thr=score_thr,
    )
    metrics.update(custom_metrics)
    metrics["split"] = split
    metrics["weights_path"] = str(weights_path)

    out_path = (
        Path(args.out).resolve()
        if args.out
        else processed_root / "keypoints_metrics.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info(" ".join(f"{k}={v}" for k, v in metrics.items()))
    logger.info(f"Saved metrics: {out_path}")


if __name__ == "__main__":
    main()

