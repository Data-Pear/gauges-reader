from __future__ import annotations

import argparse
from contextlib import nullcontext
import logging
import random
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import (
    KeypointRCNNPredictor,
    KeypointRCNN_ResNet50_FPN_Weights,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import DetKpDataset
from data.transforms import build_transforms_det_kp
from utils.config import load_config
from utils.io import det_collate_fn
from utils.metrics import KeypointMeter, format_metrics


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--max-train-steps", type=int, default=None)
    ap.add_argument("--max-val-batches", type=int, default=None)
    return ap.parse_args()


def _setup_logger(log_path: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("train_det_kp")
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


def _get_device(device_cfg: str, logger: logging.Logger) -> torch.device:
    mode = str(device_cfg).lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; fallback to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if mode == "mps":
        logger.warning("MPS is not used for torchvision detection ops; fallback to CPU.")
        return torch.device("cpu")

    # auto: prefer CUDA, otherwise CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(
    num_classes: int,
    num_keypoints: int,
    pretrained_coco: bool,
) -> torch.nn.Module:
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_coco else None
    model = keypointrcnn_resnet50_fpn(weights=weights, weights_backbone=None)

    # Replace detection head for our number of classes (incl. background).
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace keypoint head for our number of keypoints.
    in_features_kp = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
        in_features_kp,
        num_keypoints,
    )
    return model


def _move_targets_to_device(
    targets: list[Dict[str, torch.Tensor]], device: torch.device
) -> list[Dict[str, torch.Tensor]]:
    out: list[Dict[str, torch.Tensor]] = []
    for t in targets:
        out.append({k: v.to(device) for k, v in t.items()})
    return out


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


def _resolve_pck_thr(best_metric: str, cfg_pck_thr: float) -> float:
    if best_metric.startswith("kpt_pck@"):
        raw = best_metric.split("@", maxsplit=1)[1]
        try:
            return float(raw)
        except ValueError:
            return cfg_pck_thr
    return cfg_pck_thr


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pck_thr: float,
    score_thr: float,
    max_batches: int | None = None,
) -> Dict[str, float]:
    meter = KeypointMeter(pck_thr=pck_thr, score_thr=score_thr)
    meter.reset()
    model.eval()

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)

        outputs_cpu: list[Dict[str, torch.Tensor]] = []
        for o in outputs:
            outputs_cpu.append({k: v.detach().cpu() for k, v in o.items()})

        targets_cpu: list[Dict[str, torch.Tensor]] = []
        for t in targets:
            targets_cpu.append({k: v.detach().cpu() for k, v in t.items()})

        meter.update_batch(outputs_cpu, targets_cpu)

        if max_batches is not None and batch_idx >= max_batches:
            break

    return meter.compute()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    tcfg = dict(cfg.get("training_det_kp", {}))
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
        / "train_det_kp.log"
    )
    logger = _setup_logger(log_path)

    seed = int(tcfg.get("seed", 42))
    _set_seed(seed)

    device = _get_device(tcfg.get("device", "auto"), logger=logger)
    amp_cfg = bool(tcfg.get("amp", True))
    amp = amp_cfg and device.type == "cuda"

    epochs = int(tcfg.get("epochs", 50))
    batch_size = int(tcfg.get("batch_size", 8))
    num_workers = int(tcfg.get("num_workers", 4))
    lr = float(tcfg.get("lr", 1e-4))
    weight_decay = float(tcfg.get("weight_decay", 1e-4))
    log_every = int(tcfg.get("log_every", 50))
    max_train_steps = (
        int(args.max_train_steps) if args.max_train_steps is not None else None
    )
    max_val_batches = (
        int(args.max_val_batches) if args.max_val_batches is not None else None
    )

    best_metric = str(tcfg.get("best_metric", "kpt_mae"))
    pck_thr = _resolve_pck_thr(best_metric, float(tcfg.get("pck_thr", 0.05)))
    score_thr = float(tcfg.get("score_thr", 0.3))

    if best_metric == "kpt_pck":
        best_metric = f"kpt_pck@{pck_thr:.2f}"

    minimize = best_metric == "kpt_mae"
    pretrained_coco = bool(tcfg.get("pretrained_coco", False))

    train_index = Path(paths["train_det_kp_output_json"]).resolve()
    val_index = Path(paths["val_det_kp_output_json"]).resolve()
    if not train_index.exists():
        raise FileNotFoundError(f"train index not found: {train_index}")
    if not val_index.exists():
        raise FileNotFoundError(f"val index not found: {val_index}")

    weights_dir = Path(paths.get("weights_dir_det_kp", "models/weights/det_kp")).resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    num_keypoints = int(cfg.get("keypoints_target", {}).get("num_keypoints", 4))
    num_obj_classes = int(cfg.get("detector_target", {}).get("num_classes", 1))
    num_classes = num_obj_classes + 1  # + background

    logger.info(f"device={device} amp={amp}")
    logger.info(f"train_index={train_index}")
    logger.info(f"val_index={val_index}")
    logger.info(f"weights_dir={weights_dir}")
    logger.info(
        f"num_classes={num_classes} num_keypoints={num_keypoints} "
        f"pretrained_coco={pretrained_coco}"
    )

    tf_train = build_transforms_det_kp(cfg, split="train")
    tf_val = build_transforms_det_kp(cfg, split="val")
    train_ds = DetKpDataset(train_index, transform=tf_train)
    val_ds = DetKpDataset(val_index, transform=tf_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=det_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=det_collate_fn,
    )

    model = _build_model(
        num_classes=num_classes,
        num_keypoints=num_keypoints,
        pretrained_coco=pretrained_coco,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_score = float("inf") if minimize else -float("inf")
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images = [img.to(device, non_blocking=True) for img in images]
            targets = _move_targets_to_device(targets, device)

            optimizer.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.autocast(device_type="cuda", enabled=True)
                if amp
                else nullcontext()
            )
            with amp_ctx:
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step={global_step + 1}: {loss}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            global_step += 1

            if global_step % log_every == 0:
                parts = [f"{k}={float(v.item()):.4f}" for k, v in loss_dict.items()]
                logger.info(
                    f"epoch={epoch} step={global_step} train/loss={running_loss / batch_idx:.6f} "
                    + " ".join(parts)
                )

            if max_train_steps is not None and batch_idx >= max_train_steps:
                logger.info(
                    f"epoch={epoch} reached max_train_steps={max_train_steps}; "
                    "stopping epoch early."
                )
                break

        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            pck_thr=pck_thr,
            score_thr=score_thr,
            max_batches=max_val_batches,
        )
        logger.info(f"epoch={epoch} " + format_metrics(val_metrics, prefix="val/"))

        _save_checkpoint(
            weights_dir / "last.pt",
            model,
            optimizer,
            epoch,
            cfg,
            val_metrics,
        )

        score = val_metrics.get(best_metric, float("nan"))
        if score != score:
            logger.warning(
                f"Metric '{best_metric}' is NaN or missing; skip best checkpoint update."
            )
            continue

        improved = (score < best_score) if minimize else (score > best_score)
        if improved:
            best_score = score
            _save_checkpoint(
                weights_dir / "best.pt",
                model,
                optimizer,
                epoch,
                cfg,
                val_metrics,
            )
            logger.info(
                f"[BEST] epoch={epoch} {best_metric}={score:.6f} -> saved best.pt"
            )

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
