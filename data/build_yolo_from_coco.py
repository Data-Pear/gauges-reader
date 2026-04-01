from __future__ import annotations

import argparse
import json
from collections import defaultdict
import os
from pathlib import Path
import stat
import shutil
from typing import Any, Dict, Iterable, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert COCO instance annotations to YOLO labels."
    )
    ap.add_argument("--config", type=str, default="configs/config_detection.yaml")
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


def _resolve_yolo_root(cfg: Dict[str, Any], dataset_root: Path) -> Path:
    yolo_root_cfg = cfg.get("paths", {}).get("yolo_dataset_root")
    if not yolo_root_cfg:
        return dataset_root
    yolo_root = Path(str(yolo_root_cfg)).resolve()
    _ensure_images_view(dataset_root=dataset_root, yolo_root=yolo_root)
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
        raise FileNotFoundError(
            "Both train and val COCO files are required in config paths."
        )
    return out


def _build_class_map(coco: Dict[str, Any]) -> tuple[Dict[int, int], Dict[int, str]]:
    categories = coco.get("categories", [])
    if not categories:
        raise ValueError("COCO categories are empty.")

    sorted_cats = sorted(categories, key=lambda c: int(c["id"]))
    cat_id_to_cls: Dict[int, int] = {}
    names: Dict[int, str] = {}
    for cls_idx, cat in enumerate(sorted_cats):
        cat_id = int(cat["id"])
        cat_name = str(cat.get("name", f"class_{cls_idx}"))
        cat_id_to_cls[cat_id] = cls_idx
        names[cls_idx] = cat_name
    return cat_id_to_cls, names


def _iter_split_images(coco: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for img in coco.get("images", []):
        if "id" in img and "file_name" in img and "width" in img and "height" in img:
            yield img


def _write_split_labels(
    coco_path: Path,
    labels_root: Path,
    cat_id_to_cls: Dict[int, int],
) -> int:
    coco = _read_json(coco_path)

    anns_by_image: Dict[int, list[Dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        bbox = ann.get("bbox")
        if not isinstance(img_id, int):
            continue
        if not isinstance(cat_id, int) or cat_id not in cat_id_to_cls:
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        anns_by_image[img_id].append(ann)

    written = 0
    for img in _iter_split_images(coco):
        img_id = int(img["id"])
        w = float(img["width"])
        h = float(img["height"])
        if w <= 0 or h <= 0:
            continue

        label_rel = Path(str(img["file_name"])).with_suffix(".txt")
        label_path = labels_root / label_rel
        label_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for ann in anns_by_image.get(img_id, []):
            cls = cat_id_to_cls[int(ann["category_id"])]
            x, y, bw, bh = [float(v) for v in ann["bbox"]]
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        text = "\n".join(lines)
        if text:
            text += "\n"
        label_path.write_text(text, encoding="utf-8")
        written += 1
    return written


def _write_data_yaml(
    out_yaml: Path,
    dataset_root: Path,
    available_splits: Iterable[str],
    names: Dict[int, str],
) -> None:
    images_root = dataset_root / "images"
    use_images_root = images_root.exists()

    payload: Dict[str, Any] = {
        "path": str(images_root if use_images_root else dataset_root),
        "train": "train" if use_images_root else "images/train",
        "val": "val" if use_images_root else "images/val",
        "names": names,
    }
    if "test" in set(available_splits):
        payload["test"] = "test" if use_images_root else "images/test"

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
    split_coco = _resolve_split_coco_paths(cfg, dataset_root)

    train_coco = _read_json(split_coco["train"])
    cat_id_to_cls, names = _build_class_map(train_coco)

    labels_root = (yolo_root / "labels").resolve()
    split_written: Dict[str, int] = {}
    for split, coco_path in split_coco.items():
        split_written[split] = _write_split_labels(
            coco_path=coco_path,
            labels_root=labels_root,
            cat_id_to_cls=cat_id_to_cls,
        )

    cfg_out_yaml = cfg.get("paths", {}).get(
        "yolo_data_yaml", "configs/synthgauge_det_yolo_data.yaml"
    )
    out_yaml_path = Path(out_yaml).resolve() if out_yaml else (PROJECT_ROOT / str(cfg_out_yaml)).resolve()
    _write_data_yaml(out_yaml_path, yolo_root, split_coco.keys(), names)

    print(f"[OK] dataset_root: {dataset_root}")
    if yolo_root != dataset_root:
        print(f"[OK] yolo_root:    {yolo_root}")
    print(f"[OK] labels_root:  {labels_root}")
    for split in ["train", "val", "test"]:
        if split in split_written:
            print(f"[OK] {split} labels written for {split_written[split]} images")
    print(f"[OK] YOLO data yaml: {out_yaml_path}")
    return out_yaml_path


def main() -> None:
    args = _parse_args()
    build_from_config(args.config, raw_root=args.raw_root, out_yaml=args.out_yaml)


if __name__ == "__main__":
    main()
