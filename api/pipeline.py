from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.geometry import (  # noqa: E402
    angle_cw_deg,
    needle_tip_from_mask,
    normalized_reading_from_angles,
)
from utils.config import load_config  # noqa: E402
from utils.runtime import (  # noqa: E402
    find_weights_path,
    normalize_model_name,
    resolve_task_weights_dir,
    resolve_yolo_device,
)

BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class GaugeReaderSettings:
    pipeline_config: Path = PROJECT_ROOT / "configs" / "config_full_pipeline.yaml"
    det_config: Optional[Path] = None
    kp_config: Optional[Path] = None
    seg_config: Optional[Path] = None
    det_weights: Optional[str] = None
    kp_weights: Optional[str] = None
    seg_weights: Optional[str] = None
    device: str = "auto"
    det_thr: Optional[float] = None
    kp_thr: Optional[float] = None
    seg_thr: Optional[float] = None
    det_imgsz: Optional[int] = None
    kp_imgsz: Optional[int] = None
    seg_imgsz: Optional[int] = None
    crop_pad_ratio: Optional[float] = None


def settings_from_env() -> GaugeReaderSettings:
    return GaugeReaderSettings(
        pipeline_config=_path_env(
            "GAUGE_PIPELINE_CONFIG",
            PROJECT_ROOT / "configs" / "config_full_pipeline.yaml",
        ),
        det_config=_optional_path_env("GAUGE_DET_CONFIG"),
        kp_config=_optional_path_env("GAUGE_KP_CONFIG"),
        seg_config=_optional_path_env("GAUGE_SEG_CONFIG"),
        det_weights=os.getenv("GAUGE_DET_WEIGHTS"),
        kp_weights=os.getenv("GAUGE_KP_WEIGHTS"),
        seg_weights=os.getenv("GAUGE_SEG_WEIGHTS"),
        device=os.getenv("GAUGE_DEVICE", "auto"),
        det_thr=_optional_float_env("GAUGE_DET_THR"),
        kp_thr=_optional_float_env("GAUGE_KP_THR"),
        seg_thr=_optional_float_env("GAUGE_SEG_THR"),
        det_imgsz=_optional_int_env("GAUGE_DET_IMGSZ"),
        kp_imgsz=_optional_int_env("GAUGE_KP_IMGSZ"),
        seg_imgsz=_optional_int_env("GAUGE_SEG_IMGSZ"),
        crop_pad_ratio=_optional_float_env("GAUGE_CROP_PAD_RATIO"),
    )


def resolve_model_locations(
    settings: Optional[GaugeReaderSettings] = None,
) -> Dict[str, Any]:
    settings = settings or settings_from_env()
    pipeline_cfg, det_cfg_path, kp_cfg_path, seg_cfg_path, det_cfg, kp_cfg, seg_cfg = (
        _load_configs(settings)
    )

    return {
        "pipeline_config": str(_resolve_path(settings.pipeline_config)),
        "device": _resolve_device(settings.device, det_cfg, kp_cfg, seg_cfg),
        "stages": {
            "detection": _stage_location(
                det_cfg,
                explicit=settings.det_weights,
                config_path=det_cfg_path,
                weights_key="weights_dir_det",
                task_prefix="det",
                default_model="yolo11n.pt",
            ),
            "keypoints": _stage_location(
                kp_cfg,
                explicit=settings.kp_weights,
                config_path=kp_cfg_path,
                weights_key="weights_dir_kp",
                task_prefix="kp",
                default_model="yolo11n-pose.pt",
            ),
            "segmentation": _stage_location(
                seg_cfg,
                explicit=settings.seg_weights,
                config_path=seg_cfg_path,
                weights_key="weights_dir_seg",
                task_prefix="seg",
                default_model="yolo11n-seg.pt",
            ),
        },
        "configs": pipeline_cfg.get("configs", {}),
    }


class GaugeReader:
    def __init__(self, settings: Optional[GaugeReaderSettings] = None) -> None:
        self.settings = settings or settings_from_env()
        _, det_cfg_path, kp_cfg_path, seg_cfg_path, self.det_cfg, self.kp_cfg, self.seg_cfg = (
            _load_configs(self.settings)
        )
        self.det_cfg_path = det_cfg_path
        self.kp_cfg_path = kp_cfg_path
        self.seg_cfg_path = seg_cfg_path

        self.device = _resolve_device(
            self.settings.device,
            self.det_cfg,
            self.kp_cfg,
            self.seg_cfg,
        )
        self.det_thr = _score_thr(self.settings.det_thr, self.det_cfg)
        self.kp_thr = _score_thr(self.settings.kp_thr, self.kp_cfg)
        self.seg_thr = _score_thr(self.settings.seg_thr, self.seg_cfg)
        self.det_imgsz = _imgsz(self.settings.det_imgsz, self.det_cfg, 640)
        self.kp_imgsz = _imgsz(self.settings.kp_imgsz, self.kp_cfg, 960)
        self.seg_imgsz = _imgsz(self.settings.seg_imgsz, self.seg_cfg, 640)
        self.crop_pad_ratio = _crop_pad_ratio(self.settings.crop_pad_ratio, self.kp_cfg)
        self.keypoint_names = [
            str(v)
            for v in self.kp_cfg.get("keypoints", {}).get(
                "names",
                ["center", "scale_start", "scale_end"],
            )
        ]

        self.det_weights = _resolve_yolo_weights(
            self.det_cfg,
            self.settings.det_weights,
            weights_key="weights_dir_det",
            task_prefix="det",
            default_model="yolo11n.pt",
        )
        self.kp_weights = _resolve_yolo_weights(
            self.kp_cfg,
            self.settings.kp_weights,
            weights_key="weights_dir_kp",
            task_prefix="kp",
            default_model="yolo11n-pose.pt",
        )
        self.seg_weights = _resolve_yolo_weights(
            self.seg_cfg,
            self.settings.seg_weights,
            weights_key="weights_dir_seg",
            task_prefix="seg",
            default_model="yolo11n-seg.pt",
        )

        self.det_model = YOLO(str(self.det_weights))
        self.kp_model = YOLO(str(self.kp_weights))
        self.seg_model = YOLO(str(self.seg_weights))

    def describe(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "thresholds": {
                "detection": self.det_thr,
                "keypoints": self.kp_thr,
                "segmentation": self.seg_thr,
            },
            "image_sizes": {
                "detection": self.det_imgsz,
                "keypoints": self.kp_imgsz,
                "segmentation": self.seg_imgsz,
            },
            "crop_pad_ratio": self.crop_pad_ratio,
            "keypoints": self.keypoint_names,
            "weights": {
                "detection": str(self.det_weights),
                "keypoints": str(self.kp_weights),
                "segmentation": str(self.seg_weights),
            },
        }

    def predict_bytes(
        self,
        image_bytes: bytes,
        *,
        image_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")
        image_np = np.asarray(rgb, dtype=np.uint8)
        return self.predict_array(image_np, image_name=image_name)

    def predict_array(
        self,
        image_np: np.ndarray,
        *,
        image_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError("Expected an RGB image array with shape HxWx3.")

        full_h, full_w = image_np.shape[:2]
        base: Dict[str, Any] = {
            "image_name": image_name,
            "image_size": {"width": int(full_w), "height": int(full_h)},
            "reading_normalized": None,
        }

        det_result = self.det_model.predict(
            source=image_np,
            conf=self.det_thr,
            imgsz=self.det_imgsz,
            device=self.device,
            verbose=False,
        )[0]
        det = _best_box(det_result, self.det_thr)
        if det is None:
            return {**base, "status": "no_detection"}

        det_box, det_score = det
        crop_box = _crop_box_from_detection(
            det_box,
            img_w=full_w,
            img_h=full_h,
            pad_ratio=self.crop_pad_ratio,
        )
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        crop_np = image_np[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        crop_h, crop_w = crop_np.shape[:2]

        kp_result = self.kp_model.predict(
            source=crop_np,
            conf=self.kp_thr,
            imgsz=self.kp_imgsz,
            device=self.device,
            verbose=False,
        )[0]
        _, pred_kps, kp_score = _extract_pose_prediction(kp_result, self.kp_thr)
        if pred_kps is None or len(pred_kps) < len(self.keypoint_names):
            return {
                **base,
                "status": "no_keypoints",
                "dial": _dial_payload(det_box, det_score),
                "crop_xyxy": [int(v) for v in crop_box],
            }

        keypoints = _keypoints_payload(pred_kps, self.keypoint_names, crop_box)
        pred_by_name = {
            name: (float(pred_kps[idx][0]), float(pred_kps[idx][1]))
            for idx, name in enumerate(self.keypoint_names)
        }
        if not {"center", "scale_start", "scale_end"}.issubset(pred_by_name):
            return {
                **base,
                "status": "missing_required_keypoints",
                "dial": _dial_payload(det_box, det_score),
                "crop_xyxy": [int(v) for v in crop_box],
                "keypoints": keypoints,
            }

        seg_result = self.seg_model.predict(
            source=crop_np,
            conf=self.seg_thr,
            imgsz=self.seg_imgsz,
            device=self.device,
            verbose=False,
        )[0]
        pred_mask, needle_detections = _prediction_mask_and_detections(
            seg_result,
            width=crop_w,
            height=crop_h,
            score_thr=self.seg_thr,
        )
        if int(pred_mask.sum()) == 0:
            return {
                **base,
                "status": "no_needle_mask",
                "dial": _dial_payload(det_box, det_score),
                "crop_xyxy": [int(v) for v in crop_box],
                "keypoints": keypoints,
                "keypoint_score": float(kp_score) if kp_score is not None else None,
            }

        center = pred_by_name["center"]
        tip = needle_tip_from_mask(pred_mask, center)
        if tip is None:
            return {
                **base,
                "status": "no_needle_tip",
                "dial": _dial_payload(det_box, det_score),
                "crop_xyxy": [int(v) for v in crop_box],
                "keypoints": keypoints,
            }

        start_angle = angle_cw_deg(center, pred_by_name["scale_start"])
        end_angle = angle_cw_deg(center, pred_by_name["scale_end"])
        needle_angle = angle_cw_deg(center, tip)
        reading = normalized_reading_from_angles(
            start_angle=start_angle,
            end_angle=end_angle,
            needle_angle=needle_angle,
        )
        if reading is None:
            status = "invalid_scale"
        else:
            status = "ok"

        return {
            **base,
            "status": status,
            "reading_normalized": float(reading) if reading is not None else None,
            "dial": _dial_payload(det_box, det_score),
            "crop_xyxy": [int(v) for v in crop_box],
            "keypoint_score": float(kp_score) if kp_score is not None else None,
            "keypoints": keypoints,
            "scale": {
                "start_angle_cw_deg": float(start_angle),
                "end_angle_cw_deg": float(end_angle),
                "sweep_cw_deg": float((end_angle - start_angle) % 360.0),
            },
            "needle": {
                "tip_crop_xy": [float(tip[0]), float(tip[1])],
                "tip_image_xy": [float(tip[0] + crop_x1), float(tip[1] + crop_y1)],
                "angle_cw_deg": float(needle_angle),
                "mask_area_px": int(pred_mask.sum()),
                "detections": needle_detections,
            },
        }


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return _resolve_path(value) if value else default.resolve()


def _optional_path_env(name: str) -> Optional[Path]:
    value = os.getenv(name)
    return _resolve_path(value) if value else None


def _optional_float_env(name: str) -> Optional[float]:
    value = os.getenv(name)
    return float(value) if value else None


def _optional_int_env(name: str) -> Optional[int]:
    value = os.getenv(name)
    return int(value) if value else None


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _load_configs(
    settings: GaugeReaderSettings,
) -> Tuple[Dict[str, Any], Path, Path, Path, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    pipeline_path = _resolve_path(settings.pipeline_config)
    pipeline_cfg = load_config(pipeline_path)
    det_cfg_path = settings.det_config or _configured_path(
        pipeline_cfg,
        "det",
        "configs/config_detection.yaml",
    )
    kp_cfg_path = settings.kp_config or _configured_path(
        pipeline_cfg,
        "kp",
        "configs/config_keypoints.yaml",
    )
    seg_cfg_path = settings.seg_config or _configured_path(
        pipeline_cfg,
        "seg",
        "configs/config_segmentation.yaml",
    )

    det_cfg_path = _resolve_path(det_cfg_path)
    kp_cfg_path = _resolve_path(kp_cfg_path)
    seg_cfg_path = _resolve_path(seg_cfg_path)
    return (
        pipeline_cfg,
        det_cfg_path,
        kp_cfg_path,
        seg_cfg_path,
        load_config(det_cfg_path),
        load_config(kp_cfg_path),
        load_config(seg_cfg_path),
    )


def _configured_path(pipeline_cfg: Dict[str, Any], key: str, default: str) -> Path:
    return _resolve_path(str(pipeline_cfg.get("configs", {}).get(key, default)))


def _resolve_device(
    requested: str,
    det_cfg: Dict[str, Any],
    kp_cfg: Dict[str, Any],
    seg_cfg: Dict[str, Any],
) -> str:
    if requested != "from-config":
        return resolve_yolo_device(str(requested))
    for cfg in (det_cfg, kp_cfg, seg_cfg):
        device = str(cfg.get("training", {}).get("device", ""))
        if device:
            return resolve_yolo_device(device)
    return resolve_yolo_device("auto")


def _score_thr(override: Optional[float], cfg: Dict[str, Any]) -> float:
    if override is not None:
        return float(override)
    return float(cfg.get("evaluation", {}).get("score_thr", 0.25))


def _imgsz(override: Optional[int], cfg: Dict[str, Any], default: int) -> int:
    if override is not None:
        return int(override)
    return int(cfg.get("model", {}).get("imgsz", default))


def _crop_pad_ratio(override: Optional[float], kp_cfg: Dict[str, Any]) -> float:
    if override is not None:
        return float(override)
    return float(kp_cfg.get("keypoints", {}).get("crop_pad_ratio", 0.08))


def _stage_location(
    cfg: Dict[str, Any],
    *,
    explicit: Optional[str],
    config_path: Path,
    weights_key: str,
    task_prefix: str,
    default_model: str,
) -> Dict[str, Any]:
    model_name = normalize_model_name(str(cfg.get("model", {}).get("name", default_model)))
    weights_dir = resolve_task_weights_dir(
        cfg,
        weights_key=weights_key,
        task_prefix=task_prefix,
        model_identifier=model_name,
    )
    payload: Dict[str, Any] = {
        "config_path": str(config_path),
        "model_name": model_name,
        "weights_dir": str(weights_dir),
    }
    try:
        weights_path = find_weights_path(
            explicit_path=explicit,
            weights_dir=weights_dir,
            include_nested_weights_dir=True,
        )
    except FileNotFoundError as exc:
        payload.update({"weights_path": None, "available": False, "error": str(exc)})
        return payload
    payload.update({"weights_path": str(weights_path), "available": True})
    return payload


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


def _best_box(result: Any, score_thr: float) -> Optional[Tuple[List[float], float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()
    keep = [
        idx
        for idx in range(len(xyxy))
        if float(conf[idx]) >= score_thr and int(cls[idx]) == 0
    ]
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
) -> BBox:
    x1i = int(max(0, min(w - 1, int(round(x1)))))
    y1i = int(max(0, min(h - 1, int(round(y1)))))
    x2i = int(max(x1i + 1, min(w, int(round(x2)))))
    y2i = int(max(y1i + 1, min(h, int(round(y2)))))
    return x1i, y1i, x2i, y2i


def _crop_box_from_detection(
    box_xyxy: List[float],
    img_w: int,
    img_h: int,
    pad_ratio: float,
) -> BBox:
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
) -> Tuple[Optional[List[float]], Optional[np.ndarray], Optional[float]]:
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


def _resize_binary_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    pil = pil.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(pil) > 0


def _prediction_mask_and_detections(
    result: Any,
    *,
    width: int,
    height: int,
    score_thr: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    pred = np.zeros((height, width), dtype=bool)
    detections: List[Dict[str, Any]] = []
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None or len(boxes) == 0:
        return pred, detections

    data = getattr(masks, "data", None)
    if data is None or len(data) == 0:
        return pred, detections

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()
    mask_data = data.detach().cpu().numpy()
    for idx in range(mask_data.shape[0]):
        if float(conf[idx]) < score_thr or int(cls[idx]) != 0:
            continue
        mask = mask_data[idx] > 0.5
        if mask.shape != (height, width):
            mask = _resize_binary_mask(mask, width=width, height=height)
        pred |= mask
        detections.append(
            {
                "bbox_crop_xyxy": [float(v) for v in xyxy[idx].tolist()],
                "score": float(conf[idx]),
                "mask_area_px": int(mask.sum()),
            }
        )
    return pred, detections


def _dial_payload(box_xyxy: List[float], score: float) -> Dict[str, Any]:
    return {"bbox_xyxy": [float(v) for v in box_xyxy], "score": float(score)}


def _keypoints_payload(
    pred_kps: np.ndarray,
    keypoint_names: List[str],
    crop_box: BBox,
) -> Dict[str, Dict[str, List[float]]]:
    crop_x1, crop_y1, _, _ = crop_box
    out: Dict[str, Dict[str, List[float]]] = {}
    for idx, name in enumerate(keypoint_names):
        x = float(pred_kps[idx][0])
        y = float(pred_kps[idx][1])
        out[name] = {
            "crop_xy": [x, y],
            "image_xy": [x + crop_x1, y + crop_y1],
        }
    return out
