from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.eval_full_pipeline import (
    _angle_cw_deg,
    _best_box,
    _build_eval_records,
    _crop_box_from_detection,
    _extract_pose_prediction,
    _needle_tip_from_mask,
    _normalized_reading_from_angles,
    _prediction_mask,
    _resolve_device,
    _resolve_pipeline_config_path,
    _resolve_yolo_weights,
)
from utils.config import load_config


Point = Tuple[float, float]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Visualize real-image full gauge pipeline on the Roboflow "
            "center/min/max/pointer_tip landmark dataset."
        )
    )
    ap.add_argument("--config", type=str, default="configs/config_full_pipeline_real_landmarks.yaml")
    ap.add_argument("--det-config", type=str, default=None)
    ap.add_argument("--kp-config", type=str, default=None)
    ap.add_argument("--seg-config", type=str, default=None)
    ap.add_argument("--det-weights", type=str, default=None)
    ap.add_argument("--kp-weights", type=str, default=None)
    ap.add_argument("--seg-weights", type=str, default=None)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="from-config")
    ap.add_argument("--det-thr", type=float, default=None)
    ap.add_argument("--kp-thr", type=float, default=None)
    ap.add_argument("--seg-thr", type=float, default=None)
    ap.add_argument("--det-imgsz", type=int, default=None)
    ap.add_argument("--kp-imgsz", type=int, default=None)
    ap.add_argument("--seg-imgsz", type=int, default=None)
    ap.add_argument("--crop-pad-ratio", type=float, default=None)
    ap.add_argument("--display-size", type=int, default=760)
    ap.add_argument(
        "--save",
        type=str,
        default="data/processed/full_pipeline_real_landmarks_samples.png",
    )
    return ap.parse_args()


def _sample_records(records: List[Dict[str, Any]], num_samples: int, seed: int) -> List[Dict[str, Any]]:
    if num_samples <= 0 or num_samples >= len(records):
        return records
    rng = random.Random(seed)
    return rng.sample(records, k=num_samples)


def _draw_box(
    ax: plt.Axes,
    box: List[float],
    *,
    color: str,
    label: str,
    scale: float = 1.0,
    linewidth: float = 2.0,
    linestyle: str = "-",
) -> None:
    x1, y1, x2, y2 = [float(v) * scale for v in box]
    ax.add_patch(
        Rectangle(
            (x1, y1),
            max(1.0, x2 - x1),
            max(1.0, y2 - y1),
            linewidth=linewidth,
            linestyle=linestyle,
            edgecolor=color,
            facecolor="none",
        )
    )
    ax.text(x1, max(10.0, y1 - 5.0), label, color=color, fontsize=8)


def _overlay_mask(
    ax: plt.Axes,
    mask: np.ndarray,
    *,
    color: Tuple[float, float, float],
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
    mask: np.ndarray,
    display_size: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if display_size <= 0:
        return image, mask, 1.0
    height, width = image.shape[:2]
    longest = max(width, height)
    if longest <= 0:
        return image, mask, 1.0
    scale = float(display_size) / float(longest)
    if abs(scale - 1.0) < 1e-6:
        return image, mask, 1.0
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


def _prediction_boxes(result: Any, score_thr: float) -> List[Tuple[List[float], float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()
    out: List[Tuple[List[float], float]] = []
    for idx in range(len(xyxy)):
        if float(conf[idx]) < score_thr or int(cls[idx]) != 0:
            continue
        out.append(([float(v) for v in xyxy[idx].tolist()], float(conf[idx])))
    return out


def _draw_point(
    ax: plt.Axes,
    point: Point,
    *,
    label: str,
    color: str,
    scale: float = 1.0,
    size: float = 34.0,
) -> None:
    x, y = float(point[0]) * scale, float(point[1]) * scale
    ax.scatter([x], [y], s=size, c=color, edgecolors="black", linewidths=0.5)
    ax.text(x + 4, y + 4, label, color=color, fontsize=7)


def _draw_line(
    ax: plt.Axes,
    start: Point,
    end: Point,
    *,
    color: str,
    scale: float = 1.0,
    linewidth: float = 1.8,
    linestyle: str = "-",
) -> None:
    ax.plot(
        [float(start[0]) * scale, float(end[0]) * scale],
        [float(start[1]) * scale, float(end[1]) * scale],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )


def _gt_points_full(rec: Dict[str, Any]) -> Dict[str, Point]:
    kps = rec["keypoints"]
    out = {
        "center": (float(kps["center"][0]), float(kps["center"][1])),
        "min": (float(kps["scale_start"][0]), float(kps["scale_start"][1])),
        "max": (float(kps["scale_end"][0]), float(kps["scale_end"][1])),
    }
    tip = rec.get("needle_tip_xy")
    if isinstance(tip, list) and len(tip) == 2:
        out["tip"] = (float(tip[0]), float(tip[1]))
    return out


def _shift_points(points: Dict[str, Point], dx: float, dy: float) -> Dict[str, Point]:
    return {name: (xy[0] - dx, xy[1] - dy) for name, xy in points.items()}


def _draw_gt_landmarks(
    ax: plt.Axes,
    points: Dict[str, Point],
    *,
    scale: float = 1.0,
) -> None:
    colors = {
        "center": "white",
        "min": "deepskyblue",
        "max": "magenta",
        "tip": "orange",
    }
    for name in ["center", "min", "max", "tip"]:
        if name in points:
            _draw_point(ax, points[name], label=f"gt {name}", color=colors[name], scale=scale)
    if "center" in points and "tip" in points:
        _draw_line(ax, points["center"], points["tip"], color="orange", scale=scale)
    if "center" in points and "min" in points:
        _draw_line(ax, points["center"], points["min"], color="deepskyblue", scale=scale, linestyle="--")
    if "center" in points and "max" in points:
        _draw_line(ax, points["center"], points["max"], color="magenta", scale=scale, linestyle="--")


def _draw_pred_keypoints(
    ax: plt.Axes,
    pred_kps: Optional[np.ndarray],
    kp_names: List[str],
    *,
    scale: float,
) -> None:
    if pred_kps is None:
        ax.text(6, 44, "pred keypoints: none", color="cyan", fontsize=8)
        return
    for idx, point in enumerate(pred_kps.tolist()):
        if idx >= len(kp_names):
            continue
        _draw_point(
            ax,
            (float(point[0]), float(point[1])),
            label=f"pred {kp_names[idx]}",
            color="cyan",
            scale=scale,
            size=26.0,
        )


def _predict_reading(
    pred_kps: Optional[np.ndarray],
    pred_mask: np.ndarray,
    kp_names: List[str],
) -> Tuple[Optional[float], Optional[Point]]:
    if pred_kps is None or len(pred_kps) < len(kp_names):
        return None, None
    pred_by_name = {
        name: (float(pred_kps[idx][0]), float(pred_kps[idx][1]))
        for idx, name in enumerate(kp_names)
    }
    if not {"center", "scale_start", "scale_end"}.issubset(pred_by_name):
        return None, None
    center = pred_by_name["center"]
    tip = _needle_tip_from_mask(pred_mask, center)
    if tip is None:
        return None, None
    start_angle = _angle_cw_deg(center, pred_by_name["scale_start"])
    end_angle = _angle_cw_deg(center, pred_by_name["scale_end"])
    needle_angle = _angle_cw_deg(center, tip)
    return (
        _normalized_reading_from_angles(
            start_angle=start_angle,
            end_angle=end_angle,
            needle_angle=needle_angle,
        ),
        tip,
    )


def main() -> None:
    args = _parse_args()

    pipeline_cfg = load_config(args.config)
    det_cfg_path = _resolve_pipeline_config_path(args, "det", "configs/config_detection.yaml")
    kp_cfg_path = _resolve_pipeline_config_path(args, "kp", "configs/config_keypoints.yaml")
    seg_cfg_path = _resolve_pipeline_config_path(args, "seg", "configs/config_segmentation.yaml")
    det_cfg = load_config(det_cfg_path)
    kp_cfg = load_config(kp_cfg_path)
    seg_cfg = load_config(seg_cfg_path)

    records, _ = _build_eval_records(pipeline_cfg, det_cfg, kp_cfg, seg_cfg, args.split)
    records = _sample_records(records, args.num_samples, args.seed)
    if not records:
        raise RuntimeError(f"No records found for split={args.split}")

    det_eval = det_cfg.get("evaluation", {})
    kp_eval = kp_cfg.get("evaluation", {})
    seg_eval = seg_cfg.get("evaluation", {})
    det_model_cfg = det_cfg.get("model", {})
    kp_model_cfg = kp_cfg.get("model", {})
    seg_model_cfg = seg_cfg.get("model", {})
    pipe_eval = pipeline_cfg.get("evaluation", {})

    det_thr = float(args.det_thr if args.det_thr is not None else det_eval.get("score_thr", 0.25))
    kp_thr = float(args.kp_thr if args.kp_thr is not None else kp_eval.get("score_thr", 0.25))
    seg_thr = float(args.seg_thr if args.seg_thr is not None else seg_eval.get("score_thr", 0.25))
    det_imgsz = int(args.det_imgsz if args.det_imgsz is not None else det_model_cfg.get("imgsz", 640))
    kp_imgsz = int(args.kp_imgsz if args.kp_imgsz is not None else kp_model_cfg.get("imgsz", 960))
    seg_imgsz = int(args.seg_imgsz if args.seg_imgsz is not None else seg_model_cfg.get("imgsz", 640))
    crop_pad_ratio = float(
        args.crop_pad_ratio
        if args.crop_pad_ratio is not None
        else pipe_eval.get("crop_pad_ratio", kp_cfg.get("keypoints", {}).get("crop_pad_ratio", 0.08))
    )
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
    kp_names = [
        str(v)
        for v in kp_cfg.get("keypoints", {}).get(
            "names",
            ["center", "scale_start", "scale_end"],
        )
    ]

    rows = len(records)
    fig, axes = plt.subplots(rows, 2, figsize=(13.0, max(4.5, rows * 4.2)))
    axes_arr = np.array(axes, ndmin=2)

    for row_idx, rec in enumerate(records):
        full_ax = axes_arr[row_idx, 0]
        crop_ax = axes_arr[row_idx, 1]
        image_path = Path(rec["image_path"])
        target_reading = float(rec["reading_normalized"])
        gt_points = _gt_points_full(rec)

        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            full_np = np.asarray(rgb, dtype=np.uint8)
        full_h, full_w = full_np.shape[:2]

        full_ax.imshow(full_np)
        full_ax.axis("off")
        full_ax.set_title(f"{image_path.name}\ntarget={target_reading:.3f}", fontsize=9)
        _draw_box(
            full_ax,
            [float(v) for v in rec["bbox_xyxy"]],
            color="orange",
            label="gt meter",
            linewidth=1.8,
            linestyle="--",
        )
        _draw_gt_landmarks(full_ax, gt_points)

        det_result = det_model.predict(
            source=full_np,
            conf=det_thr,
            imgsz=det_imgsz,
            device=device,
            verbose=False,
        )[0]
        det = _best_box(det_result, det_thr)
        if det is None:
            full_ax.text(8, 28, "pred dial: none", color="red", fontsize=9)
            crop_ax.axis("off")
            crop_ax.set_title("crop skipped", fontsize=10)
            continue

        det_box, det_score = det
        _draw_box(full_ax, det_box, color="lime", label=f"pred dial {det_score:.2f}")

        crop_x1, crop_y1, crop_x2, crop_y2 = _crop_box_from_detection(
            det_box,
            img_w=full_w,
            img_h=full_h,
            pad_ratio=crop_pad_ratio,
        )
        crop_np = full_np[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        crop_h, crop_w = crop_np.shape[:2]
        shifted_gt = _shift_points(gt_points, float(crop_x1), float(crop_y1))

        kp_result = kp_model.predict(
            source=crop_np,
            conf=kp_thr,
            imgsz=kp_imgsz,
            device=device,
            verbose=False,
        )[0]
        _, pred_kps, kp_score = _extract_pose_prediction(kp_result, kp_thr)

        seg_result = seg_model.predict(
            source=crop_np,
            conf=seg_thr,
            imgsz=seg_imgsz,
            device=device,
            verbose=False,
        )[0]
        pred_mask = _prediction_mask(seg_result, width=crop_w, height=crop_h, score_thr=seg_thr)
        pred_reading, pred_tip = _predict_reading(pred_kps, pred_mask, kp_names)
        display_crop, display_mask, scale = _resize_for_display(
            crop_np,
            pred_mask,
            display_size=args.display_size,
        )

        crop_ax.imshow(display_crop)
        _overlay_mask(crop_ax, display_mask, color=(1.0, 0.0, 0.0), alpha=0.34)
        _draw_gt_landmarks(crop_ax, shifted_gt, scale=scale)
        _draw_pred_keypoints(crop_ax, pred_kps, kp_names, scale=scale)
        for box, score in _prediction_boxes(seg_result, seg_thr):
            _draw_box(
                crop_ax,
                box,
                color="yellow",
                label=f"needle {score:.2f}",
                scale=scale,
                linewidth=1.6,
            )
        if pred_tip is not None:
            _draw_point(crop_ax, pred_tip, label="pred tip", color="red", scale=scale, size=30.0)
            if pred_kps is not None and "center" in kp_names:
                center_idx = kp_names.index("center")
                pred_center = (float(pred_kps[center_idx][0]), float(pred_kps[center_idx][1]))
                _draw_line(crop_ax, pred_center, pred_tip, color="red", scale=scale, linewidth=2.0)

        if pred_reading is None:
            pred_text = "pred=none"
        else:
            pred_text = f"pred={pred_reading:.3f} err={abs(pred_reading - target_reading):.3f}"
        kp_text = f"kp={kp_score:.2f}" if kp_score is not None else "kp=none"
        crop_ax.set_title(
            (
                f"det crop {crop_w}x{crop_h} | {kp_text}\n"
                f"{pred_text} | mask_px={int(pred_mask.sum())}"
            ),
            fontsize=9,
        )
        crop_ax.axis("off")

    fig.suptitle(
        (
            f"Real landmarks full pipeline ({args.split}, n={len(records)}, "
            f"det={det_thr:.2f}, kp={kp_thr:.2f}, seg={seg_thr:.2f})"
        ),
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.985), h_pad=2.0, w_pad=1.5)

    out_path = Path(args.save).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved figure: {out_path}")


if __name__ == "__main__":
    main()
