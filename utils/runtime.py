from __future__ import annotations

import logging
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def setup_logger(name: str, log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_torch_device(device_cfg: str, logger: Optional[logging.Logger] = None) -> torch.device:
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
        if logger is not None:
            logger.warning("MPS requested but unavailable; fallback to CPU.")
        return torch.device("cpu")
    if mode == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(mode)


def resolve_yolo_device(device_cfg: str) -> str:
    mode = str(device_cfg).lower()
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return "0"
    if mode == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return device_cfg


def log_torch_device_info(logger: logging.Logger, device: torch.device) -> None:
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


def normalize_model_name(model_name: str) -> str:
    name = model_name.strip()
    if name.endswith(".pt"):
        return name
    return f"{name}.pt"


def model_tag(model_name: str) -> str:
    stem = Path(model_name).stem.lower()
    tag = re.sub(r"[^a-z0-9._-]+", "-", stem).strip("-")
    return tag or "model"


def dataset_name(cfg: Dict[str, Any]) -> str:
    dataset_cfg = cfg.get("dataset", {})
    explicit = dataset_cfg.get("name")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    raw_ds = str(cfg.get("paths", {}).get("raw_ds_path", "dataset")).rstrip("/\\")
    return Path(raw_ds).name or "dataset"


def resolve_task_weights_dir(
    cfg: Dict[str, Any],
    *,
    weights_key: str,
    task_prefix: str,
    model_identifier: str,
) -> Path:
    paths = cfg.get("paths", {})
    explicit = paths.get(weights_key)
    if explicit:
        return Path(str(explicit)).resolve()

    ds_name = dataset_name(cfg)
    tag = model_tag(model_identifier)
    return Path("models/weights").resolve() / ds_name / f"{task_prefix}_{tag}"


def find_weights_path(
    *,
    explicit_path: Optional[str],
    weights_dir: Path,
    include_nested_weights_dir: bool = True,
) -> Path:
    if explicit_path:
        p = Path(explicit_path).resolve()
        if p.exists():
            return p
        if include_nested_weights_dir:
            alt = (p.parent / "weights" / p.name).resolve()
            if alt.exists():
                return alt
            raise FileNotFoundError(f"Weights file not found: {p}\nAlso checked: {alt}")
        raise FileNotFoundError(f"Weights file not found: {p}")

    best_candidates = [weights_dir / "best.pt"]
    last_candidates = [weights_dir / "last.pt"]
    if include_nested_weights_dir:
        best_candidates.append(weights_dir / "weights" / "best.pt")
        last_candidates.append(weights_dir / "weights" / "last.pt")

    for group in [best_candidates, last_candidates]:
        existing = [p for p in group if p.exists()]
        if existing:
            return max(existing, key=lambda p: p.stat().st_mtime).resolve()

    raise FileNotFoundError(
        "No weights found. Checked: "
        + ", ".join(str(p) for p in best_candidates + last_candidates)
    )


def copy_best_last_weights(weights_dir: Path) -> None:
    nested = weights_dir / "weights"
    if not nested.exists():
        return
    for name in ["best.pt", "last.pt"]:
        src = nested / name
        if src.exists():
            dst = weights_dir / name
            shutil.copy2(src, dst)
