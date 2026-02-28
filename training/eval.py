from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from data.datasets import GaugeValueDataset
from data.transforms import build_transforms
from models.model import GaugeRegressor, ModelConfig
from utils.config import load_config
from utils.metrics import RegressionMeter, format_metrics


def _get_device(device_cfg: str) -> torch.device:
    device_cfg = str(device_cfg).lower()
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(h)
    return logger


def load_model(
    cfg: Dict[str, Any], weights_path: Path, device: torch.device
) -> torch.nn.Module:
    mcfg = cfg.get("model", {})
    model = GaugeRegressor(
        ModelConfig(
            backbone=mcfg.get("backbone", "convnext_tiny"),
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

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=amp):
            y_pred = model(x)  # [B]

        meter.update(y_pred, y)

    return meter.compute()


def main() -> None:
    cfg = load_config("configs/config.yaml")
    logger = _setup_logger()

    tcfg = cfg.get("training", {})
    device = _get_device(tcfg.get("device", "auto"))
    amp_cfg = bool(tcfg.get("amp", True))
    amp = amp_cfg and device.type == "cuda"
    tol = tcfg.get("tol", None)
    tol = float(tol) if tol is not None else None

    val_index = Path(cfg["paths"]["val_output_json"]).resolve()
    if not val_index.exists():
        raise FileNotFoundError(f"val index not found: {val_index}")

    weights_dir = Path(
        cfg.get("paths", {}).get("weights_dir", "models/weights")
    ).resolve()
    weights_path = weights_dir / "best.pt"
    if not weights_path.exists():
        weights_path = weights_dir / "last.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No weights found in: {weights_dir} (expected best.pt or last.pt)"
        )

    tf_val = build_transforms(cfg, split="val")
    val_ds = GaugeValueDataset(val_index, transform=tf_val)

    loader = DataLoader(
        val_ds,
        batch_size=int(tcfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(tcfg.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )

    model = load_model(cfg, weights_path, device)
    metrics = evaluate(model, loader, device=device, tol=tol, amp=amp)

    logger.info(f"weights: {weights_path}")
    logger.info(format_metrics(metrics, prefix="val/"))


if __name__ == "__main__":
    main()
