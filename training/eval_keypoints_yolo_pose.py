from __future__ import annotations

import argparse
import json
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
)
from utils.config import load_config
from utils.runtime import (
    find_weights_path,
    normalize_model_name,
    resolve_task_weights_dir,
    resolve_yolo_device,
    setup_logger,
)


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


def _resolve_weights_path(cfg: Dict[str, Any], explicit: Optional[str]) -> Path:
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
        explicit_path=explicit,
        weights_dir=weights_dir,
        include_nested_weights_dir=True,
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
    device = resolve_yolo_device(args.device or str(tcfg.get("device", "auto")))

    processed_root = Path(paths.get("processed_ds_path", "data/processed")).resolve()
    log_path = processed_root / "eval_keypoints_yolo_pose.log"
    logger = setup_logger("eval_keypoints_yolo_pose", log_path)

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
