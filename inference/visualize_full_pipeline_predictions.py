from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config
from utils.runtime import (
    find_weights_path,
    normalize_model_name,
    resolve_task_weights_dir,
    resolve_yolo_device,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Visualize full gauge pipeline: dial detection -> crop -> "
            "crop keypoints + needle segmentation."
        )
    )
    ap.add_argument("--det-config", type=str, default="configs/config_detection.yaml")
    ap.add_argument("--kp-config", type=str, default="configs/config_keypoints.yaml")
    ap.add_argument("--seg-config", type=str, default="configs/config_segmentation.yaml")
    ap.add_argument("--det-weights", type=str, default=None)
    ap.add_argument("--kp-weights", type=str, default=None)
    ap.add_argument("--seg-weights", type=str, default=None)
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--det-thr", type=float, default=None)
    ap.add_argument("--kp-thr", type=float, default=None)
    ap.add_argument("--seg-thr", type=float, default=None)
    ap.add_argument("--det-imgsz", type=int, default=None)
    ap.add_argument("--kp-imgsz", type=int, default=None)
    ap.add_argument("--seg-imgsz", type=int, default=None)
    ap.add_argument(
        "--crop-pad-ratio",
        type=float,
        default=None,
        help="Padding around detected dial bbox. Defaults to keypoint crop_pad_ratio.",
    )
    ap.add_argument(
        "--display-size",
        type=int,
        default=None,
        help="Upscale crop visualization so the longest side reaches this size. Defaults to segmentation imgsz. Use 0 for original crop size.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="from-config")
    ap.add_argument("--save", type=str, default=None)
    return ap.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_device(requested: str, *cfg_devices: str) -> str:
    if requested != "from-config":
        return resolve_yolo_device(str(requested))
    for cfg_device in cfg_devices:
        if cfg_device:
            return resolve_yolo_device(str(cfg_device))
    return resolve_yolo_device("auto")


def _resolve_yolo_weights(
    cfg: Dict[str, Any],
    explicit: Optional[str],
    *,
    weights_key: str,
    task_prefix: str,
    default_model: str,
) -> Path:
    model_name = normalize_model_name(str(cfg.get("model", {}).get("name", default_model)))
    weights_dir = resolve_task_weights_dir(
        cfg,
        weights_key=weights_key,
        task_prefix=task_prefix,
        model_identifier=model_name,
    )
    return find_weights_path(
        explicit_path=explicit,
        weights_dir=weights_dir,
        include_nested_weights_dir=True,
    )


def _resolve_split_images(cfg: Dict[str, Any], split: str) -> List[Path]:
    raw_root = Path(str(cfg.get("paths", {}).get("raw_ds_path", ""))).resolve()
    rel = cfg.get("paths", {}).get(f"{split}_inst_coco")
    if not rel:
        raise KeyError(f"Missing config key: paths.{split}_inst_coco")
    coco_path = (raw_root / str(rel)).resolve()
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO file not found: {coco_path}")

    coco = _load_json(coco_path)
    images_root = raw_root / "images"
    out: List[Path] = []
    for image in coco.get("images", []):
        file_name = image.get("file_name")
        if not isinstance(file_name, str):
            continue
        image_path = (images_root / Path(file_name.replace("\\", "/"))).resolve()
        if image_path.exists():
            out.append(image_path)
    return out


def _sample_images(
    cfg: Dict[str, Any],
    image_arg: Optional[str],
    split: str,
    num_samples: int,
    seed: int,
) -> List[Path]:
    if image_arg:
        return [Path(image_arg).resolve()]

    image_paths = _resolve_split_images(cfg, split)
    if not image_paths:
        raise RuntimeError(f"No images found for split={split}")
    random.seed(seed)
    return random.sample(image_paths, k=min(num_samples, len(image_paths)))


def _best_box(result: Any, score_thr: float) -> Optional[tuple[list[float], float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()
    keep = [idx for idx in range(len(xyxy)) if float(conf[idx]) >= score_thr and int(cls[idx]) == 0]
    if not keep:
        return None
    best = max(keep, key=lambda idx: float(conf[idx]))
    return [float(v) for v in xyxy[best].tolist()], float(conf[best])


def _clip_bbox_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    x1i = int(max(0, min(w - 1, int(x1))))
    y1i = int(max(0, min(h - 1, int(y1))))
    x2i = int(max(x1i + 1, min(w, int(x2))))
    y2i = int(max(y1i + 1, min(h, int(y2))))
    return x1i, y1i, x2i, y2i


def _crop_box_from_detection(
    box_xyxy: list[float],
    img_w: int,
    img_h: int,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh) * (1.0 + 2.0 * max(0.0, pad_ratio))
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return _clip_bbox_xyxy(
        cx - side / 2.0,
        cy - side / 2.0,
        cx + side / 2.0,
        cy + side / 2.0,
        w=img_w,
        h=img_h,
    )


def _extract_pose_prediction(
    result: Any,
    score_thr: float,
) -> tuple[Optional[list[float]], Optional[np.ndarray], Optional[float]]:
    chosen = _best_box(result, score_thr)
    if chosen is None:
        return None, None, None

    box, score = chosen
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return box, None, score

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    best_idx = None
    for idx in range(len(xyxy)):
        if abs(float(conf[idx]) - score) < 1e-9 and np.allclose(xyxy[idx], np.asarray(box)):
            best_idx = idx
            break
    if best_idx is None:
        best_idx = int(np.argmax(conf))

    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return box, None, score
    if hasattr(keypoints, "xy") and keypoints.xy is not None and len(keypoints.xy) > best_idx:
        return box, keypoints.xy[best_idx].detach().cpu().numpy(), score
    data = getattr(keypoints, "data", None)
    if data is not None and len(data) > best_idx:
        return box, data[best_idx, :, :2].detach().cpu().numpy(), score
    return box, None, score


def _prediction_mask(result: Any, width: int, height: int, score_thr: float) -> np.ndarray:
    pred = np.zeros((height, width), dtype=bool)
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None or len(boxes) == 0:
        return pred

    data = getattr(masks, "data", None)
    if data is None or len(data) == 0:
        return pred

    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()
    mask_data = data.detach().cpu().numpy()
    for idx in range(mask_data.shape[0]):
        if float(conf[idx]) < score_thr or int(cls[idx]) != 0:
            continue
        mask = mask_data[idx] > 0.5
        if mask.shape != (height, width):
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (width, height),
                    Image.Resampling.NEAREST,
                )
            ) > 0
        pred |= mask
    return pred


def _prediction_boxes(result: Any, score_thr: float) -> list[tuple[list[float], float]]:
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


def _display_resize(
    image: np.ndarray,
    mask: np.ndarray,
    display_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if display_size <= 0:
        return image, mask, 1.0
    height, width = image.shape[:2]
    longest = max(width, height)
    if longest <= 0 or longest >= display_size:
        return image, mask, 1.0
    scale = float(display_size) / float(longest)
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    image_out = np.asarray(
        Image.fromarray(image).resize((out_w, out_h), Image.Resampling.BICUBIC)
    )
    mask_out = (
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (out_w, out_h),
                Image.Resampling.NEAREST,
            )
        )
        > 0
    )
    return image_out, mask_out, scale


def _draw_box(
    ax: plt.Axes,
    box: list[float],
    *,
    color: str,
    label: str,
    scale: float = 1.0,
    linewidth: float = 2.0,
) -> None:
    x1, y1, x2, y2 = [float(v) * scale for v in box]
    ax.add_patch(
        Rectangle(
            (x1, y1),
            max(1.0, x2 - x1),
            max(1.0, y2 - y1),
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
    )
    ax.text(x1, max(8.0, y1 - 3.0), label, color=color, fontsize=8)


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


def _draw_keypoints(
    ax: plt.Axes,
    keypoints_xy: Optional[np.ndarray],
    names: list[str],
    scale: float,
) -> None:
    if keypoints_xy is None:
        ax.text(4, 42, "KEYPOINTS: none", color="cyan", fontsize=8)
        return
    for idx, point in enumerate(keypoints_xy.tolist()):
        x = float(point[0]) * scale
        y = float(point[1]) * scale
        ax.scatter([x], [y], s=28, c="cyan")
        label = names[idx] if idx < len(names) else f"kp{idx}"
        ax.text(x + 3, y + 3, label, color="cyan", fontsize=7)


def main() -> None:
    args = _parse_args()

    det_cfg = load_config(args.det_config)
    kp_cfg = load_config(args.kp_config)
    seg_cfg = load_config(args.seg_config)

    det_eval = det_cfg.get("evaluation", {})
    kp_eval = kp_cfg.get("evaluation", {})
    seg_eval = seg_cfg.get("evaluation", {})
    det_model_cfg = det_cfg.get("model", {})
    kp_model_cfg = kp_cfg.get("model", {})
    seg_model_cfg = seg_cfg.get("model", {})

    det_thr = float(args.det_thr if args.det_thr is not None else det_eval.get("score_thr", 0.25))
    kp_thr = float(args.kp_thr if args.kp_thr is not None else kp_eval.get("score_thr", 0.25))
    seg_thr = float(args.seg_thr if args.seg_thr is not None else seg_eval.get("score_thr", 0.25))
    det_imgsz = int(args.det_imgsz if args.det_imgsz is not None else det_model_cfg.get("imgsz", 640))
    kp_imgsz = int(args.kp_imgsz if args.kp_imgsz is not None else kp_model_cfg.get("imgsz", 960))
    seg_imgsz = int(args.seg_imgsz if args.seg_imgsz is not None else seg_model_cfg.get("imgsz", 640))
    crop_pad_ratio = float(
        args.crop_pad_ratio
        if args.crop_pad_ratio is not None
        else kp_cfg.get("keypoints", {}).get("crop_pad_ratio", 0.08)
    )
    display_size = int(args.display_size) if args.display_size is not None else seg_imgsz
    device = _resolve_device(
        args.device,
        str(det_cfg.get("training", {}).get("device", "")),
        str(kp_cfg.get("training", {}).get("device", "")),
        str(seg_cfg.get("training", {}).get("device", "")),
    )

    det_weights = _resolve_yolo_weights(
        det_cfg,
        args.det_weights,
        weights_key="weights_dir_det",
        task_prefix="det",
        default_model="yolo11n.pt",
    )
    kp_weights = _resolve_yolo_weights(
        kp_cfg,
        args.kp_weights,
        weights_key="weights_dir_kp",
        task_prefix="kp",
        default_model="yolo11n-pose.pt",
    )
    seg_weights = _resolve_yolo_weights(
        seg_cfg,
        args.seg_weights,
        weights_key="weights_dir_seg",
        task_prefix="seg",
        default_model="yolo11n-seg.pt",
    )

    det_model = YOLO(str(det_weights))
    kp_model = YOLO(str(kp_weights))
    seg_model = YOLO(str(seg_weights))

    image_paths = _sample_images(
        det_cfg,
        image_arg=args.image,
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    kp_names = [str(v) for v in kp_cfg.get("keypoints", {}).get("names", ["center", "scale_start", "scale_end"])]

    rows = len(image_paths)
    fig, axes = plt.subplots(rows, 2, figsize=(10.0, max(4.0, rows * 4.2)))
    axes_arr = np.array(axes, ndmin=2)

    for row_idx, image_path in enumerate(image_paths):
        full_ax = axes_arr[row_idx, 0]
        crop_ax = axes_arr[row_idx, 1]

        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            full_np = np.asarray(rgb, dtype=np.uint8)
        full_h, full_w = full_np.shape[:2]

        det_result = det_model.predict(
            source=full_np,
            conf=det_thr,
            imgsz=det_imgsz,
            device=device,
            verbose=False,
        )[0]
        det = _best_box(det_result, det_thr)

        full_ax.imshow(full_np)
        full_ax.set_title(f"{image_path.name} | detection", fontsize=10)
        full_ax.axis("off")

        if det is None:
            full_ax.text(4, 18, "DIAL: none", color="red", fontsize=9)
            crop_ax.axis("off")
            crop_ax.set_title("crop skipped", fontsize=10)
            continue

        det_box, det_score = det
        _draw_box(
            full_ax,
            det_box,
            color="lime",
            label=f"dial {det_score:.2f}",
            linewidth=2.0,
        )

        crop_x1, crop_y1, crop_x2, crop_y2 = _crop_box_from_detection(
            det_box,
            img_w=full_w,
            img_h=full_h,
            pad_ratio=crop_pad_ratio,
        )
        crop_np = full_np[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        crop_h, crop_w = crop_np.shape[:2]

        kp_result = kp_model.predict(
            source=crop_np,
            conf=kp_thr,
            imgsz=kp_imgsz,
            device=device,
            verbose=False,
        )[0]
        _, keypoints_xy, kp_score = _extract_pose_prediction(kp_result, kp_thr)

        seg_result = seg_model.predict(
            source=crop_np,
            conf=seg_thr,
            imgsz=seg_imgsz,
            device=device,
            verbose=False,
        )[0]
        pred_mask = _prediction_mask(
            seg_result,
            width=crop_w,
            height=crop_h,
            score_thr=seg_thr,
        )
        needle_boxes = _prediction_boxes(seg_result, seg_thr)

        display_crop, display_mask, scale = _display_resize(
            crop_np,
            pred_mask,
            display_size=display_size,
        )

        crop_ax.imshow(display_crop)
        _overlay_mask(crop_ax, display_mask, color=(1.0, 0.0, 0.0), alpha=0.38)
        for box, score in needle_boxes:
            _draw_box(
                crop_ax,
                box,
                color="yellow",
                label=f"needle {score:.2f}",
                scale=scale,
                linewidth=1.8,
            )
        _draw_keypoints(crop_ax, keypoints_xy, kp_names, scale=scale)
        if kp_score is not None:
            crop_ax.text(4, 18, f"kp {kp_score:.2f}", color="cyan", fontsize=8)
        crop_ax.text(4, 32, "mask", color="red", fontsize=8)
        crop_ax.set_title(
            f"crop {crop_w}x{crop_h}->{display_crop.shape[1]}x{display_crop.shape[0]}",
            fontsize=10,
        )
        crop_ax.axis("off")

    fig.suptitle(
        (
            f"Full pipeline ({args.split}, n={len(image_paths)}, "
            f"det={det_thr:.2f}, kp={kp_thr:.2f}, seg={seg_thr:.2f})"
        ),
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
