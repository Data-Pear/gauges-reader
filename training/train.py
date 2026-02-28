from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.datasets import GaugeValueDataset
from data.transforms import build_transforms
from models.model import GaugeRegressor, ModelConfig
from utils.config import load_config
from utils.metrics import RegressionMeter, format_metrics


def _setup_logger(log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("train")
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
    torch.backends.cudnn.benchmark = (
        True  # быстрее; для строгой детерминированности выключай
    )
    # torch.use_deterministic_algorithms(True)  # если нужна строгая воспроизводимость


def _get_device(device_cfg: str) -> torch.device:
    device_cfg = str(device_cfg).lower()

    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda")
    if device_cfg == "mps":
        # MPS доступен только на macOS и если torch собран с поддержкой
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # auto: cuda -> mps -> cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_loss(cfg: Dict[str, Any]) -> nn.Module:
    tcfg = cfg.get("training", {})
    loss_name = str(tcfg.get("loss", "huber")).lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "l1":
        return nn.L1Loss()
    # huber default
    delta = float(tcfg.get("huber_delta", 0.05))
    return nn.HuberLoss(delta=delta)


def _metric_value(metrics: Dict[str, float], key: str) -> float:
    v = metrics.get(key, float("nan"))
    return v


def save_checkpoint(
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
            device_type=device.type, enabled=amp and device.type == "cuda"
        ):
            y_pred = model(x)  # [B]

        meter.update(y_pred, y)

    return meter.compute()


def main() -> None:
    cfg = load_config("configs/config.yaml")
    tcfg = cfg.get("training", {})

    seed = int(tcfg.get("seed", 42))
    _set_seed(seed)

    device = _get_device(tcfg.get("device", "auto"))
    amp = bool(tcfg.get("amp", True))
    epochs = int(tcfg.get("epochs", 30))
    batch_size = int(tcfg.get("batch_size", 32))
    num_workers = int(tcfg.get("num_workers", 4))
    lr = float(tcfg.get("lr", 3e-4))
    wd = float(tcfg.get("weight_decay", 1e-4))
    log_every = int(tcfg.get("log_every", 50))

    tol = tcfg.get("tol", None)
    tol = float(tol) if tol is not None else None

    best_metric = str(tcfg.get("best_metric", "mae"))
    minimize = best_metric in ("mae", "rmse")  # r2/acc@tol - maximize

    train_index = Path(cfg["paths"]["train_output_json"]).resolve()
    val_index = Path(cfg["paths"]["val_output_json"]).resolve()
    if not train_index.exists():
        raise FileNotFoundError(f"train index not found: {train_index}")
    if not val_index.exists():
        raise FileNotFoundError(f"val index not found: {val_index}")

    weights_dir = Path(
        cfg.get("paths", {}).get("weights_dir", "models/weights")
    ).resolve()
    log_path = (
        Path(cfg.get("paths", {}).get("processed_ds_path", "data/processed")).resolve()
        / "train.log"
    )
    logger = _setup_logger(log_path)

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

    criterion = _build_loss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_score = float("inf") if minimize else -float("inf")
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1)  # ensure [B]

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=amp):
                y_pred = model(x).view(-1)  # [B]
                loss = criterion(y_pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = running_loss / max(1, i)
                logger.info(
                    f"epoch={epoch} step={global_step} train/loss={avg_loss:.6f}"
                )

        # --- val ---
        val_metrics = evaluate(model, val_loader, device=device, tol=tol, amp=amp)
        logger.info(f"epoch={epoch} " + format_metrics(val_metrics, prefix="val/"))

        # save last
        save_checkpoint(
            weights_dir / "last.pt", model, optimizer, epoch, cfg, val_metrics
        )

        # save best
        score = _metric_value(val_metrics, best_metric)
        improved = (score < best_score) if minimize else (score > best_score)
        if improved:
            best_score = score
            save_checkpoint(
                weights_dir / "best.pt", model, optimizer, epoch, cfg, val_metrics
            )
            logger.info(
                f"[BEST] epoch={epoch} {best_metric}={score:.6f} -> saved best.pt"
            )

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
