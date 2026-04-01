from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import GaugeValueDataset
from data.transforms import build_transforms
from models.model import GaugeRegressor, ModelConfig
from training.train_regression import (
    _get_device,
    _log_device_info,
    _resolve_weights_dir,
    evaluate,
)
from utils.config import load_config
from utils.metrics import format_metrics


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config_regression.yaml")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--out", type=str, default=None)
    return ap.parse_args()


def _setup_logger(log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("eval_regression")
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _resolve_split_index(paths: Dict[str, Any], split: str) -> Path:
    split = split.lower()
    if split == "train":
        keys = ("train_reg_output_json", "train_output_json", "train_inst_json")
    elif split == "val":
        keys = ("val_reg_output_json", "val_output_json", "val_inst_json")
    elif split == "test":
        keys = (
            "test_reg_output_json",
            "test_output_json",
            "test_inst_json",
            "val_reg_output_json",
            "val_output_json",
            "val_inst_json",
        )
    else:
        raise ValueError(f"Unsupported split: {split}")

    for key in keys:
        value = paths.get(key)
        if not value:
            continue
        p = Path(str(value)).resolve()
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not resolve index jsonl for split='{split}'. "
        f"Checked keys: {', '.join(keys)}"
    )


def _resolve_weights_path(cfg: Dict[str, Any], explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Weights file not found: {p}")
        return p

    weights_dir = _resolve_weights_dir(cfg)
    candidates = [weights_dir / "best.pt", weights_dir / "last.pt"]
    for p in candidates:
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        "No regression weights found. Checked: " + ", ".join(str(p) for p in candidates)
    )


def _load_model(
    cfg: Dict[str, Any],
    weights_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    mcfg = cfg.get("model", {})
    model = GaugeRegressor(
        ModelConfig(
            backbone=mcfg.get("backbone", "resnet18"),
            pretrained=bool(mcfg.get("pretrained", True)),
            dropout=float(mcfg.get("dropout", 0.0)),
        )
    )
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    paths = cfg.get("paths", {})
    tcfg = cfg.get("training", {})
    ecfg = cfg.get("evaluation", {})

    split = args.split or str(ecfg.get("split", "test"))
    batch_size = int(args.batch_size or tcfg.get("batch_size", 32))
    num_workers = int(args.num_workers or tcfg.get("num_workers", 4))
    seed = int(tcfg.get("seed", 42))
    _set_seed(seed)

    processed_root = Path(paths.get("processed_ds_path", "data/processed")).resolve()
    log_path = processed_root / "eval_regression.log"
    logger = _setup_logger(log_path)

    device_cfg = args.device or tcfg.get("device", "auto")
    device = _get_device(device_cfg, logger=logger)
    _log_device_info(logger, device)

    amp_cfg = bool(tcfg.get("amp", True))
    amp = amp_cfg and device.type == "cuda"
    tol = (
        float(args.tol)
        if args.tol is not None
        else float(ecfg.get("drr_tolerance", tcfg.get("tol", 0.02)))
    )

    index_path = _resolve_split_index(paths, split=split)
    weights_path = _resolve_weights_path(cfg, args.weights)

    logger.info(f"weights={weights_path}")
    logger.info(f"split={split} index={index_path}")
    logger.info(
        f"eval args: batch={batch_size} workers={num_workers} "
        f"device={device} amp={amp} tol={tol}"
    )

    tf_eval = build_transforms(cfg, split="val")
    ds = GaugeValueDataset(index_path, transform=tf_eval)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = _load_model(cfg, weights_path, device=device)
    metrics = evaluate(model, loader, device=device, tol=tol, amp=amp)
    logger.info(format_metrics(metrics, prefix="eval/"))

    metrics["split"] = split
    metrics["weights_path"] = str(weights_path)

    out_path = (
        Path(args.out).resolve()
        if args.out
        else processed_root / "regression_metrics.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info(f"Saved metrics: {out_path}")


if __name__ == "__main__":
    main()

