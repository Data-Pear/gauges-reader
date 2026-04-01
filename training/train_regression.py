from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import GaugeValueDataset
from data.transforms import build_transforms
from models.model import GaugeRegressor, ModelConfig
from utils.config import load_config
from utils.metrics import RegressionMeter, format_metrics


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config_regression.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--max-train-steps", type=int, default=None)
    return ap.parse_args()


def _setup_logger(log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("train_regression")
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


def _get_device(device_cfg: str, logger: logging.Logger) -> torch.device:
    mode = str(device_cfg).lower()

    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if mode == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS requested but unavailable; fallback to CPU.")
        return torch.device("cpu")

    # auto: cuda -> mps -> cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _log_device_info(logger: logging.Logger, device: torch.device) -> None:
    if device.type != "cuda":
        logger.warning(f"Training is running on {device.type}, not CUDA.")
        return

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    total_mem_gb = props.total_memory / (1024**3)
    logger.info(
        "cuda_device="
        f"index={idx} "
        f"name={torch.cuda.get_device_name(idx)} "
        f"capability={props.major}.{props.minor} "
        f"vram={total_mem_gb:.2f}GB"
    )


def _build_loss(tcfg: Dict[str, Any]) -> nn.Module:
    loss_name = str(tcfg.get("loss", "huber")).lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "l1":
        return nn.L1Loss()
    delta = float(tcfg.get("huber_delta", 0.05))
    return nn.HuberLoss(delta=delta)


def _metric_value(metrics: Dict[str, float], key: str) -> float:
    return metrics.get(key, float("nan"))


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Dict[str, Any],
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tol: Optional[float],
    amp: bool,
) -> Dict[str, float]:
    meter = RegressionMeter(tol=tol)
    meter.reset()
    model.eval()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            y_pred = model(x)

        meter.update(y_pred, y)

    return meter.compute()


def _resolve_path(paths: Dict[str, Any], primary: str, *fallbacks: str) -> Path:
    for key in (primary, *fallbacks):
        if key in paths and paths[key]:
            return Path(paths[key]).resolve()
    raise KeyError(f"Missing path key in config: {primary}")


def _dataset_name(cfg: Dict[str, Any]) -> str:
    dataset_cfg = cfg.get("dataset", {})
    explicit = dataset_cfg.get("name")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    raw_ds = str(cfg.get("paths", {}).get("raw_ds_path", "dataset")).rstrip("/\\")
    return Path(raw_ds).name or "dataset"


def _resolve_weights_dir(cfg: Dict[str, Any]) -> Path:
    paths = cfg.get("paths", {})
    explicit = paths.get("weights_dir_reg")
    if explicit:
        return Path(str(explicit)).resolve()

    dataset_name = _dataset_name(cfg)
    model_name = str(cfg.get("model", {}).get("backbone", "resnet18")).lower()
    return Path("models/weights").resolve() / dataset_name / f"reg_{model_name}"


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    tcfg = dict(cfg.get("training", {}))
    paths = cfg.get("paths", {})

    if args.epochs is not None:
        tcfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        tcfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        tcfg["num_workers"] = int(args.num_workers)
    if args.lr is not None:
        tcfg["lr"] = float(args.lr)
    if args.weight_decay is not None:
        tcfg["weight_decay"] = float(args.weight_decay)

    log_path = (
        Path(paths.get("processed_ds_path", "data/processed")).resolve()
        / "train_regression.log"
    )
    logger = _setup_logger(log_path)

    seed = int(tcfg.get("seed", 42))
    _set_seed(seed)

    device = _get_device(tcfg.get("device", "cuda"), logger=logger)
    amp_cfg = bool(tcfg.get("amp", True))
    amp = amp_cfg and device.type == "cuda"
    _log_device_info(logger, device)

    epochs = int(tcfg.get("epochs", 30))
    batch_size = int(tcfg.get("batch_size", 32))
    num_workers = int(tcfg.get("num_workers", 4))
    lr = float(tcfg.get("lr", 3e-4))
    weight_decay = float(tcfg.get("weight_decay", 1e-4))
    log_every = int(tcfg.get("log_every", 50))
    max_train_steps = (
        int(args.max_train_steps) if args.max_train_steps is not None else None
    )

    tol = tcfg.get("tol", None)
    tol = float(tol) if tol is not None else None
    best_metric = str(tcfg.get("best_metric", "mae"))
    minimize = best_metric in ("mae", "rmse")

    train_index = _resolve_path(
        paths, "train_reg_output_json", "train_output_json", "train_inst_json"
    )
    val_index = _resolve_path(
        paths, "val_reg_output_json", "val_output_json", "val_inst_json"
    )
    if not train_index.exists():
        raise FileNotFoundError(f"train index not found: {train_index}")
    if not val_index.exists():
        raise FileNotFoundError(f"val index not found: {val_index}")

    weights_dir = _resolve_weights_dir(cfg)

    logger.info(f"device={device} amp={amp}")
    logger.info(f"train_index={train_index}")
    logger.info(f"val_index={val_index}")
    logger.info(f"weights_dir={weights_dir}")

    tf_train = build_transforms(cfg, split="train")
    tf_val = build_transforms(cfg, split="val")
    train_ds = GaugeValueDataset(train_index, transform=tf_train)
    val_ds = GaugeValueDataset(val_index, transform=tf_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    mcfg = cfg.get("model", {})
    model = GaugeRegressor(
        ModelConfig(
            backbone=mcfg.get("backbone", "convnext_tiny"),
            pretrained=bool(mcfg.get("pretrained", True)),
            dropout=float(mcfg.get("dropout", 0.0)),
        )
    ).to(device)

    criterion = _build_loss(tcfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_score = float("inf") if minimize else -float("inf")
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=amp and device.type == "cuda",
            ):
                y_pred = model(x).view(-1)
                loss = criterion(y_pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            global_step += 1

            if global_step % log_every == 0:
                logger.info(
                    f"epoch={epoch} step={global_step} "
                    f"train/loss={running_loss / batch_idx:.6f}"
                )

            if max_train_steps is not None and batch_idx >= max_train_steps:
                logger.info(
                    f"epoch={epoch} reached max_train_steps={max_train_steps}; "
                    "stopping epoch early."
                )
                break

        val_metrics = evaluate(model, val_loader, device=device, tol=tol, amp=amp)
        logger.info(f"epoch={epoch} " + format_metrics(val_metrics, prefix="val/"))

        _save_checkpoint(weights_dir / "last.pt", model, optimizer, epoch, cfg, val_metrics)

        score = _metric_value(val_metrics, best_metric)
        if score != score:
            logger.warning(
                f"Metric '{best_metric}' is NaN or missing; skip best checkpoint update."
            )
            continue

        improved = (score < best_score) if minimize else (score > best_score)
        if improved:
            best_score = score
            best_path = weights_dir / "best.pt"
            epoch_best_path = weights_dir / f"best_epoch_{epoch:03d}.pt"
            _save_checkpoint(best_path, model, optimizer, epoch, cfg, val_metrics)
            _save_checkpoint(epoch_best_path, model, optimizer, epoch, cfg, val_metrics)
            logger.info(
                f"[BEST] epoch={epoch} {best_metric}={score:.6f} -> "
                f"saved {best_path.name} and {epoch_best_path.name}"
            )

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
