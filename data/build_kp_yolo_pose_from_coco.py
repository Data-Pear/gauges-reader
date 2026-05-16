from __future__ import annotations

import argparse
import json
from collections import defaultdict
import os
from pathlib import Path
import stat
import shutil
from typing import Any, Dict, Iterable, Optional

from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert COCO keypoints annotations to YOLO pose labels."
    )
    ap.add_argument("--config", type=str, default="configs/config_keypoints.yaml")
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


def _is_linked_directory(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_symlink():
        return True
    if os.name == "nt":
        try:
            attrs = os.lstat(path).st_file_attributes
            if attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                return True
        except Exception:
            pass
    try:
        return os.path.normcase(str(path.resolve())) != os.path.normcase(str(path))
    except Exception:
        return False


def _remove_linked_directory(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
        return
    # For Windows junctions/mount points use rmdir to remove only the link entry.
    os.rmdir(path)


def _ensure_images_view(dataset_root: Path, yolo_root: Path) -> None:
    if yolo_root.resolve() == dataset_root.resolve():
        return

    src_images = (dataset_root / "images").resolve()
    if not src_images.exists():
        raise FileNotFoundError(
            f"Expected images directory for yolo_dataset_root setup: {src_images}"
        )

    # Keep unresolved path; resolving a junction points to source dir and hides links.
    dst_images = yolo_root / "images"
    if dst_images.exists():
        # Ultralytics resolves image paths; if images is a link to raw/,
        # label lookup falls back to raw/labels and mixes task-specific labels.
        if _is_linked_directory(dst_images):
            _remove_linked_directory(dst_images)
            shutil.copytree(src_images, dst_images, dirs_exist_ok=True)
        return

    yolo_root.mkdir(parents=True, exist_ok=True)
    # Always use a physical copy (not links) to keep labels resolution stable.
    shutil.copytree(src_images, dst_images, dirs_exist_ok=True)


def _resolve_yolo_root(
    cfg: Dict[str, Any],
    dataset_root: Path,
    prepare_images_view: bool,
) -> Path:
    yolo_root_cfg = cfg.get("paths", {}).get("yolo_dataset_root")
    if not yolo_root_cfg:
        return dataset_root
    yolo_root = Path(str(yolo_root_cfg)).resolve()
    if prepare_images_view:
        _ensure_images_view(dataset_root=dataset_root, yolo_root=yolo_root)
    else:
        yolo_root.mkdir(parents=True, exist_ok=True)
    return yolo_root


def _resolve_split_coco_paths(cfg: Dict[str, Any], dataset_root: Path) -> Dict[str, Path]:
    paths = cfg["paths"]
    split_to_key = {
        "train": "train_inst_coco",
        "val": "val_inst_coco",
        "test": "test_inst_coco",
    }
    out: Dict[str, Path] = {}
    for split, key in split_to_key.items():
        rel = paths.get(key)
        if rel:
            p = (dataset_root / str(rel)).resolve()
            if p.exists():
                out[split] = p
    if "train" not in out or "val" not in out:
        raise FileNotFoundError("Both train and val COCO files are required.")
    return out


def _resolve_target_category_name(cfg: Dict[str, Any], train_coco: Dict[str, Any]) -> str:
    dataset_cfg = cfg.get("dataset", {})
    explicit = dataset_cfg.get("category_name")
    if isinstance(explicit, str) and explicit:
        return explicit

    cats = train_coco.get("categories", [])
    if len(cats) == 1 and isinstance(cats[0].get("name"), str):
        return str(cats[0]["name"])
    raise ValueError(
        "Target category name is not provided and cannot be inferred. "
        "Set dataset.category_name in config."
    )


def _find_category(categories: Iterable[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for cat in categories:
        if cat.get("name") == name:
            return cat
    raise ValueError(f"Category '{name}' not found in COCO categories.")


def _iter_split_images(coco: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for img in coco.get("images", []):
        if "id" in img and "file_name" in img and "width" in img and "height" in img:
            yield img


def _normalize_clamped(v: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, v / denom))


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


def _resolve_source_image_path(dataset_root: Path, file_name: str) -> Path:
    images_root = dataset_root / "images"
    if images_root.exists():
        return (images_root / file_name).resolve()
    return (dataset_root / file_name).resolve()


def _clip_bbox_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
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
    # Keep a square-ish context around the dial to preserve geometry.
    side = max(1.0, max(bw, bh))
    side = side * (1.0 + 2.0 * max(0.0, pad_ratio))
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    x1 = cx - side / 2.0
    y1 = cy - side / 2.0
    x2 = cx + side / 2.0
    y2 = cy + side / 2.0
    return _clip_bbox_xyxy(x1, y1, x2, y2, w=img_w, h=img_h)


def _write_split_labels(
    split_name: str,
    coco_path: Path,
    dataset_root: Path,
    yolo_root: Path,
    labels_root: Path,
    target_cat_id: int,
    keypoint_indices: list[int],
    crop_dial: bool,
    crop_pad_ratio: float,
) -> tuple[int, int]:
    coco = _read_json(coco_path)
    num_keypoints = len(keypoint_indices)
    max_keypoint_idx = max(keypoint_indices) if keypoint_indices else -1

    anns_by_image: Dict[int, list[Dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        bbox = ann.get("bbox")
        keypoints = ann.get("keypoints")
        if not isinstance(img_id, int):
            continue
        if cat_id != target_cat_id:
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        if not (
            isinstance(keypoints, list)
            and max_keypoint_idx >= 0
            and len(keypoints) >= 3 * (max_keypoint_idx + 1)
        ):
            continue
        anns_by_image[img_id].append(ann)

    labels_split_root = labels_root / split_name
    images_split_root = (yolo_root / "images" / split_name).resolve()

    if labels_split_root.exists():
        shutil.rmtree(labels_split_root)
    if images_split_root.exists():
        shutil.rmtree(images_split_root)
    labels_split_root.mkdir(parents=True, exist_ok=True)
    images_split_root.mkdir(parents=True, exist_ok=True)

    written_images = 0
    written_objects = 0
    for img in _iter_split_images(coco):
        img_id = int(img["id"])
        w0 = float(img["width"])
        h0 = float(img["height"])
        if w0 <= 0 or h0 <= 0:
            continue

        anns = anns_by_image.get(img_id, [])
        ann = _select_largest_ann(anns)
        if ann is None:
            continue

        file_name = str(img["file_name"])
        image_rel = Path(file_name)
        if image_rel.parts:
            if image_rel.parts[0] in {"train", "val", "test"}:
                image_rel = Path(*image_rel.parts[1:])
        image_rel = image_rel.with_suffix(image_rel.suffix or ".jpg")

        label_path = (labels_split_root / image_rel).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image_out_path = (images_split_root / image_rel).resolve()
        image_out_path.parent.mkdir(parents=True, exist_ok=True)

        x, y, bw, bh = [float(v) for v in ann["bbox"]]
        if bw <= 0 or bh <= 0:
            continue

        if crop_dial:
            src_image_path = _resolve_source_image_path(dataset_root, file_name)
            if not src_image_path.exists():
                continue
            with Image.open(src_image_path) as im:
                rgb = im.convert("RGB")
                src_w, src_h = rgb.size
                if src_w <= 1 or src_h <= 1:
                    continue
                crop_x1, crop_y1, crop_x2, crop_y2 = _crop_box_from_bbox(
                    x=x,
                    y=y,
                    bw=bw,
                    bh=bh,
                    img_w=src_w,
                    img_h=src_h,
                    pad_ratio=crop_pad_ratio,
                )
                crop = rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                crop.save(image_out_path)

            out_w = float(crop_x2 - crop_x1)
            out_h = float(crop_y2 - crop_y1)
            if out_w <= 1 or out_h <= 1:
                continue

            x = x - float(crop_x1)
            y = y - float(crop_y1)
            w = out_w
            h = out_h
        else:
            src_image_path = _resolve_source_image_path(dataset_root, file_name)
            if not src_image_path.exists():
                continue
            if not image_out_path.exists():
                shutil.copy2(src_image_path, image_out_path)
            w = w0
            h = h0

        cx = _normalize_clamped(x + bw / 2.0, w)
        cy = _normalize_clamped(y + bh / 2.0, h)
        nw = _normalize_clamped(bw, w)
        nh = _normalize_clamped(bh, h)

        kps = ann["keypoints"]
        kp_tokens: list[str] = []
        for src_idx in keypoint_indices:
            kx = float(kps[3 * src_idx + 0])
            ky = float(kps[3 * src_idx + 1])
            kv = int(float(kps[3 * src_idx + 2]))

            if crop_dial:
                kx -= float(crop_x1)
                ky -= float(crop_y1)

            if kv <= 0:
                kx_n = 0.0
                ky_n = 0.0
            else:
                kx_n = _normalize_clamped(kx, w)
                ky_n = _normalize_clamped(ky, h)

            kp_tokens.extend([f"{kx_n:.6f}", f"{ky_n:.6f}", str(kv)])

        line = f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(kp_tokens)
        label_path.write_text(line + "\n", encoding="utf-8")
        written_images += 1
        written_objects += 1

    return written_images, written_objects


def _write_data_yaml(
    out_yaml: Path,
    dataset_root: Path,
    available_splits: Iterable[str],
    category_name: str,
    num_keypoints: int,
    flip_idx: list[int],
) -> None:
    images_root = dataset_root / "images"
    use_images_root = images_root.exists()

    payload: Dict[str, Any] = {
        "path": str(images_root if use_images_root else dataset_root),
        "train": "train" if use_images_root else "images/train",
        "val": "val" if use_images_root else "images/val",
        "names": {0: category_name},
        "kpt_shape": [num_keypoints, 3],
        "flip_idx": flip_idx,
    }
    if "test" in set(available_splits):
        payload["test"] = "test" if use_images_root else "images/test"

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _resolve_keypoint_selection(
    cfg: Dict[str, Any],
    target_cat: Dict[str, Any],
) -> tuple[list[str], list[int]]:
    keypoints_cfg = cfg.get("keypoints", {})
    cat_keypoints = target_cat.get("keypoints", [])
    cat_names = [str(k) for k in cat_keypoints] if isinstance(cat_keypoints, list) else []

    cfg_keypoints = keypoints_cfg.get("names", [])
    if isinstance(cfg_keypoints, list) and cfg_keypoints:
        keypoint_names = [str(k) for k in cfg_keypoints]
    elif cat_names:
        keypoint_names = cat_names
    else:
        raise ValueError("Keypoint names are missing in config and COCO categories.")

    cfg_num = keypoints_cfg.get("num_keypoints")
    if cfg_num is not None and int(cfg_num) != len(keypoint_names):
        raise ValueError(
            f"num_keypoints={int(cfg_num)} does not match names count={len(keypoint_names)}"
        )

    if cat_names:
        idx_by_name = {name: idx for idx, name in enumerate(cat_names)}
        missing = [name for name in keypoint_names if name not in idx_by_name]
        if missing:
            raise ValueError(
                "Configured keypoints are not present in COCO category keypoints: "
                + ", ".join(missing)
            )
        keypoint_indices = [idx_by_name[name] for name in keypoint_names]
    else:
        keypoint_indices = list(range(len(keypoint_names)))

    return keypoint_names, keypoint_indices


def build_from_config(
    config_path: str | Path,
    raw_root: Optional[str | Path] = None,
    out_yaml: Optional[str | Path] = None,
) -> Path:
    cfg_path = _resolve_cfg_path(str(config_path))
    cfg = _load_yaml(cfg_path)

    dataset_root = _resolve_dataset_root(cfg, str(raw_root) if raw_root else None)
    split_coco = _resolve_split_coco_paths(cfg, dataset_root)

    train_coco = _read_json(split_coco["train"])
    category_name = _resolve_target_category_name(cfg, train_coco)
    target_cat = _find_category(train_coco.get("categories", []), category_name)
    target_cat_id = int(target_cat["id"])

    keypoints_cfg = cfg.get("keypoints", {})
    crop_dial = bool(keypoints_cfg.get("crop_dial", False))
    crop_pad_ratio = float(keypoints_cfg.get("crop_pad_ratio", 0.05))
    yolo_root = _resolve_yolo_root(
        cfg,
        dataset_root,
        prepare_images_view=not crop_dial,
    )
    if yolo_root.resolve() == dataset_root.resolve():
        raise ValueError(
            "paths.yolo_dataset_root must point to a processed directory, "
            "not to raw_ds_path."
        )
    keypoint_names, keypoint_indices = _resolve_keypoint_selection(cfg, target_cat)
    num_keypoints = len(keypoint_names)

    flip_idx = keypoints_cfg.get("flip_idx")
    if not isinstance(flip_idx, list) or len(flip_idx) != num_keypoints:
        flip_idx = list(range(num_keypoints))
    flip_idx = [int(v) for v in flip_idx]

    labels_root = (yolo_root / "labels").resolve()
    split_written: Dict[str, tuple[int, int]] = {}
    for split, coco_path in split_coco.items():
        split_written[split] = _write_split_labels(
            split_name=split,
            coco_path=coco_path,
            dataset_root=dataset_root,
            yolo_root=yolo_root,
            labels_root=labels_root,
            target_cat_id=target_cat_id,
            keypoint_indices=keypoint_indices,
            crop_dial=crop_dial,
            crop_pad_ratio=crop_pad_ratio,
        )

    cfg_out_yaml = cfg.get("paths", {}).get(
        "yolo_data_yaml", "configs/synthgauge_kp_yolo_crop_data.yaml"
    )
    out_yaml_path = (
        Path(out_yaml).resolve()
        if out_yaml
        else (PROJECT_ROOT / str(cfg_out_yaml)).resolve()
    )
    _write_data_yaml(
        out_yaml=out_yaml_path,
        dataset_root=yolo_root,
        available_splits=split_coco.keys(),
        category_name=category_name,
        num_keypoints=num_keypoints,
        flip_idx=flip_idx,
    )

    print(f"[OK] dataset_root: {dataset_root}")
    if yolo_root != dataset_root:
        print(f"[OK] yolo_root:    {yolo_root}")
    print(f"[OK] keypoints:    {', '.join(keypoint_names)}")
    print(f"[OK] crop_dial:    {crop_dial} (pad_ratio={crop_pad_ratio:.3f})")
    print(f"[OK] labels_root:  {labels_root}")
    for split in ["train", "val", "test"]:
        if split in split_written:
            n_images, n_objects = split_written[split]
            print(
                f"[OK] {split} labels written for {n_images} images, {n_objects} objects"
            )
    print(f"[OK] YOLO pose data yaml: {out_yaml_path}")
    return out_yaml_path


def main() -> None:
    args = _parse_args()
    build_from_config(args.config, raw_root=args.raw_root, out_yaml=args.out_yaml)


if __name__ == "__main__":
    main()
