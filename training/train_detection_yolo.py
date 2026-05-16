from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

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
    ap.add_argument("--config", type=str, default="configs/config_detection.yaml")
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

    from data.build_det_yolo_from_coco import build_from_config

    logger.info("Preparing detection labels from COCO annotations...")
    produced_yaml = build_from_config(config_path=config_path, out_yaml=data_yaml)
    if not produced_yaml.exists():
        raise FileNotFoundError(f"YOLO data yaml was not created: {data_yaml}")
    return produced_yaml


def _extract_map_metrics(val_result: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    box = getattr(val_result, "box", None)
    if box is not None:
        mp = getattr(box, "mp", None)
        mr = getattr(box, "mr", None)
        map50 = getattr(box, "map50", None)
        map5095 = getattr(box, "map", None)
        if mp is not None:
            out["precision"] = float(mp)
        if mr is not None:
            out["recall"] = float(mr)
        if map50 is not None:
            out["mAP@0.5"] = float(map50)
        if map5095 is not None:
            out["mAP@0.5:0.95"] = float(map5095)
    return out


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
    model_name = normalize_model_name(str(mcfg.get("name", "yolo11n.pt")))
    pretrained = bool(mcfg.get("pretrained", True))

    log_path = (
        Path(paths.get("processed_ds_path", "data/processed")).resolve()
        / "train_detection_yolo.log"
    )
    logger = setup_logger("train_detection_yolo", log_path)

    prepare_data = True
    if args.no_prepare_data:
        prepare_data = False
    if args.prepare_data:
        prepare_data = True
    data_yaml = _ensure_data_yaml(cfg, cfg_path, prepare_data=prepare_data, logger=logger)

    weights_dir = resolve_task_weights_dir(
        cfg,
        weights_key="weights_dir_det",
        task_prefix="det",
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
    logger.info(f"Running validation on split={split} ...")
    val_result = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
    )
    metrics = _extract_map_metrics(val_result)
    if metrics:
        logger.info(" ".join(f"{k}={v:.6f}" for k, v in metrics.items()))
        summary_path = (
            Path(paths.get("processed_ds_path", "data/processed")).resolve()
            / "detection_metrics.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info(f"Saved metrics: {summary_path}")


if __name__ == "__main__":
    main()
