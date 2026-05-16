from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build YOLO segmentation labels for the needle class from v2.0 semantic masks."
    )
    ap.add_argument("--config", type=str, default="configs/config_segmentation.yaml")
    ap.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Override paths.raw_ds_path from config.",
    )
    ap.add_argument(
        "--out-yaml",
        type=str,
        default=None,
        help="Override paths.yolo_data_yaml from config.",
    )
    return ap.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_cfg_path(config_arg: str) -> Path:
    p = Path(config_arg)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def _resolve_dataset_root(cfg: Dict[str, Any], raw_root_arg: Optional[str]) -> Path:
    if raw_root_arg:
        return Path(raw_root_arg).resolve()
    paths = cfg.get("paths", {})
    return Path(paths.get("raw_ds_path", paths.get("dataset_root", ""))).resolve()


def _resolve_yolo_root(cfg: Dict[str, Any], dataset_root: Path) -> Path:
    yolo_root_cfg = cfg.get("paths", {}).get("yolo_dataset_root")
    if not yolo_root_cfg:
        raise KeyError("Missing config key: paths.yolo_dataset_root")
    yolo_root = Path(str(yolo_root_cfg)).resolve()
    if yolo_root.resolve() == dataset_root.resolve():
        raise ValueError(
            "paths.yolo_dataset_root must point to a processed directory, not raw_ds_path."
        )
    yolo_root.mkdir(parents=True, exist_ok=True)
    return yolo_root


def _resolve_split_semantic_paths(cfg: Dict[str, Any], dataset_root: Path) -> Dict[str, Path]:
    paths = cfg["paths"]
    split_to_key = {
        "train": "train_semantic_json",
        "val": "val_semantic_json",
        "test": "test_semantic_json",
    }
    out: Dict[str, Path] = {}
    for split, key in split_to_key.items():
        rel = paths.get(key, f"annotations/semantic_{split}.json")
        p = (dataset_root / str(rel)).resolve()
        if p.exists():
            out[split] = p
    if "train" not in out or "val" not in out:
        raise FileNotFoundError("Both train and val semantic JSON files are required.")
    return out


def _resolve_split_coco_paths(cfg: Dict[str, Any], dataset_root: Path) -> Dict[str, Path]:
    paths = cfg["paths"]
    split_to_key = {
        "train": "train_inst_coco",
        "val": "val_inst_coco",
        "test": "test_inst_coco",
    }
    out: Dict[str, Path] = {}
    for split, key in split_to_key.items():
        rel = paths.get(key, f"annotations/instances_{split}.json")
        p = (dataset_root / str(rel)).resolve()
        if p.exists():
            out[split] = p
    return out


def _class_id_from_semantic_json(
    semantic_json: Dict[str, Any],
    target_class_name: str,
    explicit_class_id: Optional[int],
) -> int:
    if explicit_class_id is not None:
        return int(explicit_class_id)

    for cls in semantic_json.get("classes", []):
        if cls.get("name") == target_class_name:
            return int(cls["id"])
    raise ValueError(f"Semantic class '{target_class_name}' not found.")


def _rel_after_anchor(path_value: str, anchor: str, split: str) -> Path:
    parts = Path(path_value.replace("\\", "/")).parts
    if anchor in parts:
        parts = parts[parts.index(anchor) + 1 :]
    if parts and parts[0] == split:
        parts = parts[1:]
    if not parts:
        return Path(Path(path_value).name)
    return Path(*parts)


def _resolve_raw_rel_path(dataset_root: Path, value: str) -> Path:
    return (dataset_root / Path(value.replace("\\", "/"))).resolve()


def _rel_key(path_value: Path | str) -> str:
    return Path(str(path_value).replace("\\", "/")).as_posix().lower()


def _select_largest_ann(anns: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not anns:
        return None
    if len(anns) == 1:
        return anns[0]

    def _score(ann: Dict[str, Any]) -> float:
        area = ann.get("area")
        if isinstance(area, (int, float)):
            return float(area)
        bbox = ann.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            return float(bbox[2]) * float(bbox[3])
        return 0.0

    return max(anns, key=_score)


def _target_category_id(coco: Dict[str, Any], category_name: str) -> int:
    categories = coco.get("categories", [])
    for cat in categories:
        if cat.get("name") == category_name and isinstance(cat.get("id"), int):
            return int(cat["id"])
    if len(categories) == 1 and isinstance(categories[0].get("id"), int):
        return int(categories[0]["id"])
    raise ValueError(f"Category '{category_name}' not found in COCO categories.")


def _build_dial_annotations_by_rel(
    coco_path: Path,
    split_name: str,
    category_name: str,
) -> Dict[str, Dict[str, Any]]:
    coco = _read_json(coco_path)
    target_cat_id = _target_category_id(coco, category_name)

    image_by_id: Dict[int, Dict[str, Any]] = {}
    for image in coco.get("images", []):
        image_id = image.get("id")
        if isinstance(image_id, int):
            image_by_id[image_id] = image

    anns_by_image: Dict[int, list[Dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("category_id") != target_cat_id:
            continue
        image_id = ann.get("image_id")
        bbox = ann.get("bbox")
        if not isinstance(image_id, int):
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        anns_by_image.setdefault(image_id, []).append(ann)

    out: Dict[str, Dict[str, Any]] = {}
    for image_id, anns in anns_by_image.items():
        image = image_by_id.get(image_id)
        if image is None:
            continue
        file_name = image.get("file_name")
        if not isinstance(file_name, str):
            continue
        rel = _rel_after_anchor(file_name, anchor="images", split=split_name)
        ann = _select_largest_ann(anns)
        if ann is not None:
            out[_rel_key(rel)] = ann
    return out


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


def _crop_box_from_bbox(
    x: float,
    y: float,
    bw: float,
    bh: float,
    img_w: int,
    img_h: int,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    side = max(1.0, max(bw, bh))
    side = side * (1.0 + 2.0 * max(0.0, pad_ratio))
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    x1 = cx - side / 2.0
    y1 = cy - side / 2.0
    x2 = cx + side / 2.0
    y2 = cy + side / 2.0
    return _clip_bbox_xyxy(x1, y1, x2, y2, w=img_w, h=img_h)


def _mask_to_polygons(
    mask: np.ndarray,
    *,
    min_area_px: float,
    epsilon_ratio: float,
) -> list[np.ndarray]:
    binary = np.asarray(mask > 0, dtype=np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        area = float(cv2.contourArea(contour))
        if area < min_area_px:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        epsilon = max(0.0, epsilon_ratio) * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = approx.reshape(-1, 2).astype(float)
        if pts.shape[0] < 3:
            x, y, w, h = cv2.boundingRect(contour)
            pts = np.asarray(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                dtype=float,
            )
        polygons.append((area, pts))

    polygons.sort(key=lambda item: item[0], reverse=True)
    return [pts for _, pts in polygons]


def _format_yolo_seg_line(points: np.ndarray, width: int, height: int) -> str:
    tokens = ["0"]
    for x, y in points.tolist():
        x_n = max(0.0, min(1.0, float(x) / float(width)))
        y_n = max(0.0, min(1.0, float(y) / float(height)))
        tokens.extend([f"{x_n:.6f}", f"{y_n:.6f}"])
    return " ".join(tokens)


def _write_split_labels(
    split_name: str,
    semantic_path: Path,
    coco_path: Path,
    dataset_root: Path,
    yolo_root: Path,
    category_name: str,
    target_class_id: int,
    crop_dial: bool,
    crop_pad_ratio: float,
    min_area_px: float,
    epsilon_ratio: float,
) -> tuple[int, int]:
    semantic = _read_json(semantic_path)
    records = semantic.get("images", [])
    if not isinstance(records, list):
        raise ValueError(f"Expected semantic JSON images list: {semantic_path}")

    images_split_root = yolo_root / "images" / split_name
    labels_split_root = yolo_root / "labels" / split_name
    masks_split_root = yolo_root / "masks" / split_name

    if images_split_root.exists():
        shutil.rmtree(images_split_root)
    if labels_split_root.exists():
        shutil.rmtree(labels_split_root)
    if masks_split_root.exists():
        shutil.rmtree(masks_split_root)
    images_split_root.mkdir(parents=True, exist_ok=True)
    labels_split_root.mkdir(parents=True, exist_ok=True)
    masks_split_root.mkdir(parents=True, exist_ok=True)

    written_images = 0
    written_objects = 0
    dial_ann_by_rel = _build_dial_annotations_by_rel(
        coco_path=coco_path,
        split_name=split_name,
        category_name=category_name,
    )

    for rec in records:
        image_file = rec.get("image_file")
        mask_file = rec.get("mask_file")
        if not isinstance(image_file, str) or not isinstance(mask_file, str):
            continue

        image_src = _resolve_raw_rel_path(dataset_root, image_file)
        mask_src = _resolve_raw_rel_path(dataset_root, mask_file)
        if not image_src.exists() or not mask_src.exists():
            continue

        image_rel = _rel_after_anchor(image_file, anchor="images", split=split_name)
        image_dst = images_split_root / image_rel
        label_dst = (labels_split_root / image_rel).with_suffix(".txt")
        mask_dst = (masks_split_root / image_rel).with_suffix(".png")
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        mask_dst.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(mask_src) as im:
            mask_arr = np.asarray(im)
        with Image.open(image_src) as im:
            rgb = im.convert("RGB")

        if crop_dial:
            ann = dial_ann_by_rel.get(_rel_key(image_rel))
            if ann is None:
                continue
            x, y, bw, bh = [float(v) for v in ann["bbox"]]
            if bw <= 0 or bh <= 0:
                continue
            crop_x1, crop_y1, crop_x2, crop_y2 = _crop_box_from_bbox(
                x=x,
                y=y,
                bw=bw,
                bh=bh,
                img_w=rgb.size[0],
                img_h=rgb.size[1],
                pad_ratio=crop_pad_ratio,
            )
            rgb = rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            target_mask = (mask_arr == target_class_id)[crop_y1:crop_y2, crop_x1:crop_x2]
        else:
            target_mask = mask_arr == target_class_id

        height, width = target_mask.shape[:2]
        polygons = _mask_to_polygons(
            target_mask,
            min_area_px=min_area_px,
            epsilon_ratio=epsilon_ratio,
        )

        rgb.save(image_dst)
        Image.fromarray(target_mask.astype(np.uint8)).save(mask_dst)
        lines = [_format_yolo_seg_line(poly, width, height) for poly in polygons]
        label_dst.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

        written_images += 1
        written_objects += len(lines)

    return written_images, written_objects


def _write_data_yaml(
    out_yaml: Path,
    dataset_root: Path,
    available_splits: Iterable[str],
    target_class_name: str,
) -> None:
    images_root = dataset_root / "images"
    payload: Dict[str, Any] = {
        "path": str(images_root),
        "train": "train",
        "val": "val",
        "names": {0: target_class_name},
    }
    if "test" in set(available_splits):
        payload["test"] = "test"

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def build_from_config(
    config_path: str | Path,
    raw_root: Optional[str | Path] = None,
    out_yaml: Optional[str | Path] = None,
) -> Path:
    cfg_path = _resolve_cfg_path(str(config_path))
    cfg = _load_yaml(cfg_path)

    dataset_root = _resolve_dataset_root(cfg, str(raw_root) if raw_root else None)
    yolo_root = _resolve_yolo_root(cfg, dataset_root)
    split_semantic = _resolve_split_semantic_paths(cfg, dataset_root)
    split_coco = _resolve_split_coco_paths(cfg, dataset_root)

    seg_cfg = cfg.get("segmentation", {})
    category_name = str(cfg.get("dataset", {}).get("category_name", "gauge"))
    target_class_name = str(seg_cfg.get("target_class_name", "needle"))
    explicit_class_id = seg_cfg.get("target_class_id")
    train_semantic = _read_json(split_semantic["train"])
    target_class_id = _class_id_from_semantic_json(
        train_semantic,
        target_class_name=target_class_name,
        explicit_class_id=int(explicit_class_id) if explicit_class_id is not None else None,
    )
    crop_dial = bool(seg_cfg.get("crop_dial", True))
    crop_pad_ratio = float(seg_cfg.get("crop_pad_ratio", 0.08))
    min_area_px = float(seg_cfg.get("min_area_px", 4))
    epsilon_ratio = float(seg_cfg.get("approx_epsilon_ratio", 0.002))

    split_written: Dict[str, tuple[int, int]] = {}
    for split, semantic_path in split_semantic.items():
        coco_path = split_coco.get(split)
        if coco_path is None:
            raise FileNotFoundError(f"Missing COCO instances file for split={split}")
        split_written[split] = _write_split_labels(
            split_name=split,
            semantic_path=semantic_path,
            coco_path=coco_path,
            dataset_root=dataset_root,
            yolo_root=yolo_root,
            category_name=category_name,
            target_class_id=target_class_id,
            crop_dial=crop_dial,
            crop_pad_ratio=crop_pad_ratio,
            min_area_px=min_area_px,
            epsilon_ratio=epsilon_ratio,
        )

    cfg_out_yaml = cfg.get("paths", {}).get(
        "yolo_data_yaml", "configs/synthgauge_v2_needle_seg_yolo_data.yaml"
    )
    out_yaml_path = (
        Path(out_yaml).resolve()
        if out_yaml
        else (PROJECT_ROOT / str(cfg_out_yaml)).resolve()
    )
    _write_data_yaml(
        out_yaml=out_yaml_path,
        dataset_root=yolo_root,
        available_splits=split_semantic.keys(),
        target_class_name=target_class_name,
    )

    print(f"[OK] dataset_root: {dataset_root}")
    print(f"[OK] yolo_root:    {yolo_root}")
    print(f"[OK] class:        {target_class_name} (mask id={target_class_id})")
    print(f"[OK] crop_dial:    {crop_dial} (pad_ratio={crop_pad_ratio:.3f})")
    for split in ["train", "val", "test"]:
        if split in split_written:
            n_images, n_objects = split_written[split]
            print(
                f"[OK] {split} labels written for {n_images} images, {n_objects} mask objects"
            )
    print(f"[OK] YOLO segmentation data yaml: {out_yaml_path}")
    return out_yaml_path


def main() -> None:
    args = _parse_args()
    build_from_config(args.config, raw_root=args.raw_root, out_yaml=args.out_yaml)


if __name__ == "__main__":
    main()
