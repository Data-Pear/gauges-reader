from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms import build_transforms
from models.model import GaugeRegressor, ModelConfig
from utils.config import load_config


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Visualize regression predictions vs GT values."
    )
    ap.add_argument("--config", type=str, default="configs/config_regression.yaml")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="from-config")
    ap.add_argument("--save", type=str, default=None)
    return ap.parse_args()


def _resolve_device(requested: str, cfg_device: str) -> torch.device:
    mode = cfg_device if requested == "from-config" else requested
    mode = str(mode).lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(mode)


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


def _resolve_weights_path(cfg: Dict[str, Any], weights_arg: Optional[str]) -> Path:
    if weights_arg:
        p = Path(weights_arg).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Weights not found: {p}")
        return p

    weights_dir = _resolve_weights_dir(cfg)
    candidates = [weights_dir / "best.pt", weights_dir / "last.pt"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No regression weights found under {weights_dir} (expected best.pt or last.pt)."
    )


def _resolve_index_path(cfg: Dict[str, Any], split: str) -> Path:
    key_map = {
        "train": "train_reg_output_json",
        "val": "val_reg_output_json",
        "test": "test_reg_output_json",
    }
    paths = cfg.get("paths", {})
    key = key_map[split]
    rel = paths.get(key)
    if not rel:
        raise KeyError(f"Missing path key in config: paths.{key}")
    p = Path(str(rel)).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Index file not found: {p}")
    return p


def _load_index(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"Index file is empty: {path}")
    return rows


def _load_regression_model(cfg: Dict[str, Any], weights_path: Path, device: torch.device) -> torch.nn.Module:
    mcfg = cfg.get("model", {})
    model = GaugeRegressor(
        ModelConfig(
            backbone=str(mcfg.get("backbone", "resnet18")),
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
def _predict_value(
    model: torch.nn.Module,
    transform: Any,
    img: Image.Image,
    device: torch.device,
) -> float:
    x = transform(img).unsqueeze(0).to(device)
    pred = model(x)
    return float(pred.view(-1)[0].item())


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    cfg = load_config(Path(args.config).resolve())
    tcfg = cfg.get("training", {})
    device = _resolve_device(args.device, str(tcfg.get("device", "auto")))

    index_path = _resolve_index_path(cfg, args.split)
    records = _load_index(index_path)
    chosen = random.sample(records, k=min(args.num_samples, len(records)))

    weights_path = _resolve_weights_path(cfg, args.weights)
    model = _load_regression_model(cfg, weights_path=weights_path, device=device)
    transform = build_transforms(cfg, split="val")

    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
    axes_flat = np.array(axes, ndmin=1).reshape(-1)

    for ax, rec in zip(axes_flat, chosen):
        img_path = Path(rec["image_path"]).resolve()
        gt_value = float(rec["value"])
        with Image.open(img_path) as im:
            img = im.convert("RGB")
            pred_value = _predict_value(model, transform, img, device=device)
            vis_img = np.asarray(img, dtype=np.uint8)

        abs_err = abs(pred_value - gt_value)
        ax.imshow(vis_img)
        ax.set_title(
            f"{img_path.name}\nGT={gt_value:.4f} | Pred={pred_value:.4f}\n|err|={abs_err:.4f}",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes_flat[len(chosen) :]:
        ax.axis("off")

    fig.suptitle(
        f"Regression predictions ({args.split}, n={len(chosen)})",
        fontsize=14,
    )
    plt.tight_layout()

    if args.save:
        out_path = Path(args.save).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Saved figure: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
