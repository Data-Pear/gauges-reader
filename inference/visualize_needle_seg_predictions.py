from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_needle_seg_yolo import (
    _build_seg_eval_records,
    _load_gt_mask,
    _prediction_mask,
)
from utils.config import load_config
from utils.runtime import (
    find_weights_path,
    normalize_model_name,
    resolve_task_weights_dir,
    resolve_yolo_device,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Visualize needle segmentation predictions on dial crops."
    )
    ap.add_argument("--config", type=str, default="configs/config_segmentation.yaml")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--score-thr", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument(
        "--display-size",
        type=int,
        default=None,
        help="Upscale visualization so the longest crop side reaches this size. Defaults to imgsz. Use 0 to keep original crop size.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="from-config")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument(
        "--hide-boxes",
        action="store_true",
        help="Do not draw YOLO predicted bounding boxes.",
    )
    ap.add_argument(
        "--show-gt",
        action="store_true",
        help="Overlay the crop GT mask in green for debugging.",
    )
    return ap.parse_args()


def _resolve_device(requested: str, cfg_device: str) -> str:
    mode = cfg_device if requested == "from-config" else requested
    return resolve_yolo_device(str(mode))


def _resolve_weights_path(cfg: Dict[str, Any], weights_arg: Optional[str]) -> Path:
    model_name = normalize_model_name(
        str(cfg.get("model", {}).get("name", "yolo11n-seg.pt"))
    )
    weights_dir = resolve_task_weights_dir(
        cfg,
        weights_key="weights_dir_seg",
        task_prefix="seg",
        model_identifier=model_name,
    )
    return find_weights_path(
        explicit_path=weights_arg,
        weights_dir=weights_dir,
        include_nested_weights_dir=True,
    )


def _overlay_mask(
    ax: plt.Axes,
    mask: np.ndarray,
    color: tuple[float, float, float],
    alpha: float,
) -> None:
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask, 0] = color[0]
    overlay[mask, 1] = color[1]
    overlay[mask, 2] = color[2]
    overlay[mask, 3] = alpha
    ax.imshow(overlay)


def _resize_for_display(
    image: np.ndarray,
    masks: list[np.ndarray],
    display_size: int,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    if display_size <= 0:
        return image, masks, 1.0

    height, width = image.shape[:2]
    longest = max(width, height)
    if longest <= 0 or longest >= display_size:
        return image, masks, 1.0

    scale = float(display_size) / float(longest)
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))

    upscaled_image = np.asarray(
        Image.fromarray(image).resize((out_w, out_h), Image.Resampling.BICUBIC)
    )
    upscaled_masks = [
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (out_w, out_h),
                Image.Resampling.NEAREST,
            )
        )
        > 0
        for mask in masks
    ]
    return upscaled_image, upscaled_masks, scale


def _prediction_boxes(
    result: Any,
    score_thr: float,
) -> list[tuple[list[float], float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()

    out: list[tuple[list[float], float]] = []
    for idx in range(len(xyxy)):
        if float(conf[idx]) < score_thr or int(cls[idx]) != 0:
            continue
        out.append(([float(v) for v in xyxy[idx].tolist()], float(conf[idx])))
    return out


def _draw_prediction_box(
    ax: plt.Axes,
    box: list[float],
    score: float,
    scale: float,
) -> None:
    x1, y1, x2, y2 = [float(v) * scale for v in box]
    rect = Rectangle(
        (x1, y1),
        max(1.0, x2 - x1),
        max(1.0, y2 - y1),
        linewidth=1.8,
        edgecolor="yellow",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(
        x1,
        max(8.0, y1 - 3.0),
        f"bbox {score:.2f}",
        color="yellow",
        fontsize=8,
    )


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    tcfg = cfg.get("training", {})
    mcfg = cfg.get("model", {})
    ecfg = cfg.get("evaluation", {})

    score_thr = (
        float(args.score_thr)
        if args.score_thr is not None
        else float(ecfg.get("score_thr", 0.25))
    )
    imgsz = int(args.imgsz) if args.imgsz is not None else int(mcfg.get("imgsz", 640))
    display_size = int(args.display_size) if args.display_size is not None else imgsz
    device = _resolve_device(args.device, str(tcfg.get("device", "auto")))

    records, target_class_id = _build_seg_eval_records(cfg, args.split)
    if not records:
        raise RuntimeError(
            f"No crop records found for split={args.split}. "
            "Rebuild the segmentation dataset with "
            "`uv run --no-sync ./data/build_needle_seg_yolo.py --config configs/config_segmentation.yaml`."
        )
    chosen = random.sample(records, k=min(args.num_samples, len(records)))

    weights_path = _resolve_weights_path(cfg, args.weights)
    model = YOLO(str(weights_path))

    cols = 3
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2))
    axes_flat = np.array(axes, ndmin=1).reshape(-1)

    for ax, rec in zip(axes_flat, chosen):
        image_path = Path(rec["image_path"]).resolve()
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size
            np_img = np.asarray(rgb, dtype=np.uint8)

        result = model.predict(
            source=np_img,
            conf=score_thr,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]
        pred_mask = _prediction_mask(
            result,
            width=width,
            height=height,
            score_thr=score_thr,
        )
        pred_boxes = _prediction_boxes(result, score_thr=score_thr)

        masks_to_resize = [pred_mask]
        if args.show_gt:
            gt_mask = _load_gt_mask(rec, target_class_id)
            masks_to_resize.append(gt_mask)

        display_img, display_masks, scale = _resize_for_display(
            np_img,
            masks_to_resize,
            display_size=display_size,
        )
        display_pred_mask = display_masks[0]
        display_gt_mask = display_masks[1] if args.show_gt else None

        ax.imshow(display_img)
        if args.show_gt and display_gt_mask is not None:
            _overlay_mask(ax, display_gt_mask, color=(0.0, 1.0, 0.0), alpha=0.32)
            ax.text(4, 14, "GT", color="lime", fontsize=8)
        _overlay_mask(ax, display_pred_mask, color=(1.0, 0.0, 0.0), alpha=0.38)
        if not args.hide_boxes:
            for box, score in pred_boxes:
                _draw_prediction_box(ax, box, score, scale=scale)
        ax.text(4, 28 if args.show_gt else 14, "PRED", color="red", fontsize=8)
        if scale > 1.0:
            ax.set_title(
                f"{image_path.name} | {width}x{height}->{display_img.shape[1]}x{display_img.shape[0]}",
                fontsize=9,
            )
        else:
            ax.set_title(f"{image_path.name} | {width}x{height}", fontsize=9)
        ax.axis("off")

    for ax in axes_flat[len(chosen) :]:
        ax.axis("off")

    fig.suptitle(
        f"Needle segmentation predictions ({args.split}, n={len(chosen)}, thr={score_thr:.2f}, display={display_size})",
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
