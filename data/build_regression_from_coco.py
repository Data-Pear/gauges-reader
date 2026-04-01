from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RegressionLayout:
    train_coco: Path
    val_coco: Path
    test_coco: Optional[Path]
    images_root: Path


def _required_files_from_cfg(paths_cfg: Dict[str, Any]) -> Tuple[str, str]:
    return str(paths_cfg["train_inst_coco"]), str(paths_cfg["val_inst_coco"])


def _optional_test_file_from_cfg(paths_cfg: Dict[str, Any]) -> Optional[str]:
    rel = paths_cfg.get("test_inst_coco")
    if rel is None:
        return None
    return str(rel)


def _infer_test_relpaths(train_rel: str, val_rel: str) -> List[str]:
    candidates: List[str] = []

    for src, old in [(train_rel, "train"), (val_rel, "val")]:
        if old in src:
            repl = src.replace(old, "test", 1)
            if repl not in candidates:
                candidates.append(repl)
    return candidates


def _resolve_regression_layout(
    dataset_dir: Path,
    required_files: Tuple[str, str],
    optional_test_file: Optional[str],
) -> Optional[RegressionLayout]:
    cfg_train_rel, cfg_val_rel = required_files
    cfg_train_rel_norm = cfg_train_rel.replace("\\", "/").lower()
    cfg_images_root = dataset_dir
    if cfg_train_rel_norm.startswith("annotations/"):
        candidate_images_root = dataset_dir / "images"
        if candidate_images_root.exists():
            cfg_images_root = candidate_images_root

    # Endava-like layout:
    #   ds/train__inst_coco.json, ds/val__inst_coco.json, file_name starts with "data/..."
    cfg_test_path: Optional[Path] = None
    if optional_test_file:
        candidate = dataset_dir / optional_test_file
        if candidate.exists():
            cfg_test_path = candidate
    else:
        for rel in _infer_test_relpaths(cfg_train_rel, cfg_val_rel):
            candidate = dataset_dir / rel
            if candidate.exists():
                cfg_test_path = candidate
                break

    cfg_layout = RegressionLayout(
        train_coco=dataset_dir / cfg_train_rel,
        val_coco=dataset_dir / cfg_val_rel,
        test_coco=cfg_test_path,
        images_root=cfg_images_root,
    )
    if cfg_layout.train_coco.exists() and cfg_layout.val_coco.exists():
        return cfg_layout

    # HF synthetic-analog-gauges layout:
    #   ds/annotations/instances_train.json, ds/annotations/instances_val.json, [instances_test.json]
    #   ds/images/{train,val,test}/...
    hf_test = dataset_dir / "annotations" / "instances_test.json"
    hf_layout = RegressionLayout(
        train_coco=dataset_dir / "annotations" / "instances_train.json",
        val_coco=dataset_dir / "annotations" / "instances_val.json",
        test_coco=hf_test if hf_test.exists() else None,
        images_root=dataset_dir / "images",
    )
    if hf_layout.train_coco.exists() and hf_layout.val_coco.exists():
        return hf_layout

    return None


def _is_dataset_dir(
    path: Path,
    required_files: Tuple[str, str],
    optional_test_file: Optional[str],
) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return (
        _resolve_regression_layout(path, required_files, optional_test_file) is not None
    )


def _dataset_sort_key(path: Path) -> Tuple[int, ...]:
    m = re.search(r"(\d+(?:\.\d+)*)", path.name)
    if not m:
        return (-1,)
    return tuple(int(p) for p in m.group(1).split("."))


def _resolve_raw_base(cfg: Dict[str, Any], raw_root_arg: str | None) -> Path:
    if raw_root_arg:
        return Path(raw_root_arg).resolve()

    cfg_raw = Path(str(cfg["paths"].get("raw_ds_path", ""))).resolve()
    if cfg_raw.exists():
        return cfg_raw

    fallback = Path("data/raw").resolve()
    if fallback.exists():
        return fallback

    return cfg_raw


def _discover_dataset_dirs(
    raw_base: Path,
    required_files: Tuple[str, str],
    optional_test_file: Optional[str],
) -> List[Path]:
    if _is_dataset_dir(raw_base, required_files, optional_test_file):
        return [raw_base]

    if not raw_base.exists():
        raise FileNotFoundError(
            f"Raw dataset path not found: {raw_base}\n"
            "Hint: set paths.raw_ds_path to an existing dataset directory "
            "or pass --raw-root data/raw"
        )

    candidates: List[Path] = []
    for child in raw_base.iterdir():
        if _is_dataset_dir(child, required_files, optional_test_file):
            candidates.append(child.resolve())

    candidates.sort(key=_dataset_sort_key)
    return candidates


def _select_dataset_dirs(candidates: List[Path], dataset_arg: str) -> List[Path]:
    if not candidates:
        raise FileNotFoundError(
            "No dataset directories with required COCO files were found."
        )

    mode = dataset_arg.strip()
    if mode == "all":
        return candidates
    if mode == "auto":
        return [candidates[-1]]

    selected = [d for d in candidates if d.name == mode]
    if not selected:
        available = ", ".join(d.name for d in candidates)
        raise ValueError(
            f"Dataset '{mode}' not found. Available: {available}. "
            "Use --dataset all|auto|<folder_name>."
        )
    return selected


def _extract_numeric_value(ann: Dict[str, Any], value_key: str) -> Optional[float]:
    direct = ann.get(value_key)
    if isinstance(direct, (int, float)):
        return float(direct)

    attrs = ann.get("attributes")
    if isinstance(attrs, dict):
        nested = attrs.get(value_key)
        if isinstance(nested, (int, float)):
            return float(nested)

    return None


def _find_category_id(coco: Dict[str, Any], category_name: str) -> Optional[int]:
    for cat in coco.get("categories", []):
        if cat.get("name") == category_name and isinstance(cat.get("id"), int):
            return int(cat["id"])
    return None


def _annotation_matches_category(
    ann: Dict[str, Any],
    category_name: str,
    category_id: Optional[int],
) -> bool:
    if ann.get("category_name") == category_name:
        return True
    if category_id is not None and ann.get("category_id") == category_id:
        return True
    return False


def _default_output_paths(
    paths_cfg: Dict[str, Any],
    selected_dirs: List[Path],
) -> Tuple[Path, Path, Path]:
    processed_root = Path(str(paths_cfg.get("processed_ds_path", "data/processed"))).resolve()
    dataset_label = "__".join(d.name for d in selected_dirs)
    out_dir = processed_root / f"{dataset_label}_reg"

    train_name = Path(str(paths_cfg["train_reg_output_json"])).name
    val_name = Path(str(paths_cfg["val_reg_output_json"])).name
    test_cfg = paths_cfg.get("test_reg_output_json")
    if test_cfg is not None:
        test_name = Path(str(test_cfg)).name
    elif "val" in val_name:
        test_name = val_name.replace("val", "test", 1)
    else:
        test_name = "test_regression.jsonl"
    return out_dir / train_name, out_dir / val_name, out_dir / test_name


def build_pairs_from_coco(
    coco: Dict[str, Any],
    images_root: Path,
    category_name: str,
    value_key: str,
) -> List[Tuple[str, float]]:
    """
    Делает пары (image_path, value) из Endava COCO:
    - image_path: images_root / images[*].file_name
    - value: берётся из annotations[*][value_key] для annotations[*].category_name == category_name
    """
    # image_id -> file_name
    id_to_file: Dict[int, str] = {}
    for img in coco.get("images", []):
        img_id = img.get("id")
        fn = img.get("file_name")
        if isinstance(img_id, int) and isinstance(fn, str) and fn:
            id_to_file[img_id] = fn

    target_category_id = _find_category_id(coco, category_name)

    # collect values per image_id from target category annotations
    values_by_img: Dict[int, List[float]] = {}
    for ann in coco.get("annotations", []):
        if not _annotation_matches_category(ann, category_name, target_category_id):
            continue

        img_id = ann.get("image_id")
        v = _extract_numeric_value(ann, value_key)

        if isinstance(img_id, int) and v is not None:
            values_by_img.setdefault(img_id, []).append(v)

    pairs: List[Tuple[str, float]] = []
    missing_target = 0
    missing_file = 0
    ambiguous = 0

    for img_id, fn in id_to_file.items():
        vals = values_by_img.get(img_id)
        if not vals:
            missing_target += 1
            continue

        v0 = vals[0]
        if any(abs(v - v0) > 1e-6 for v in vals[1:]):
            ambiguous += 1  # берём первый

        img_path = (images_root / fn).resolve()
        if not img_path.exists():
            missing_file += 1
            continue

        pairs.append((str(img_path), float(v0)))

    if missing_target:
        print(
            f"[WARN] Missing target for {missing_target} images (no '{category_name}' with '{value_key}')."
        )
    if missing_file:
        print(
            f"[WARN] {missing_file} images referenced in COCO were not found under images_root."
        )
        print(f"       images_root={images_root}")
    if ambiguous:
        print(f"[WARN] Ambiguous target values for {ambiguous} images (took first).")

    return pairs


def write_jsonl(pairs: List[Tuple[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for img_path, value in pairs:
            f.write(
                json.dumps({"image_path": img_path, "value": value}, ensure_ascii=False)
                + "\n"
            )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build regression JSONL index from one or more COCO dataset folders."
    )
    ap.add_argument("--config", type=str, default="configs/config_regression.yaml")
    ap.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Path to dataset folder OR parent folder containing multiple dataset folders.",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset selection mode: all | auto | <folder_name>.",
    )
    ap.add_argument(
        "--category-name",
        type=str,
        default=None,
        help="Override regression_target.category_name from config.",
    )
    ap.add_argument(
        "--value-key",
        type=str,
        default=None,
        help="Override regression_target.value_key from config.",
    )
    ap.add_argument("--out-train", type=str, default=None)
    ap.add_argument("--out-val", type=str, default=None)
    ap.add_argument("--out-test", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = cfg["paths"]

    required_files = _required_files_from_cfg(paths_cfg)
    optional_test_file = _optional_test_file_from_cfg(paths_cfg)
    raw_base = _resolve_raw_base(cfg, args.raw_root)
    candidates = _discover_dataset_dirs(raw_base, required_files, optional_test_file)
    selected_dirs = _select_dataset_dirs(candidates, args.dataset)

    if args.out_train:
        train_out = Path(args.out_train).resolve()
    else:
        train_out, _, _ = _default_output_paths(paths_cfg, selected_dirs)
    if args.out_val:
        val_out = Path(args.out_val).resolve()
    else:
        _, val_out, _ = _default_output_paths(paths_cfg, selected_dirs)
    if args.out_test:
        test_out = Path(args.out_test).resolve()
    else:
        _, _, test_out = _default_output_paths(paths_cfg, selected_dirs)

    category_name = args.category_name or cfg["regression_target"]["category_name"]
    value_key = args.value_key or cfg["regression_target"]["value_key"]

    print(f"[INFO] raw_base:     {raw_base}")
    print(f"[INFO] selected ds: {', '.join(d.name for d in selected_dirs)}")
    print(
        f"[INFO] target:       category_name='{category_name}', value_key='{value_key}'"
    )
    print(f"[INFO] out_train:    {train_out}")
    print(f"[INFO] out_val:      {val_out}")
    print(f"[INFO] out_test:     {test_out}")

    all_train_pairs: List[Tuple[str, float]] = []
    all_val_pairs: List[Tuple[str, float]] = []
    all_test_pairs: List[Tuple[str, float]] = []
    has_test_split = False

    for ds_dir in selected_dirs:
        layout = _resolve_regression_layout(ds_dir, required_files, optional_test_file)
        if layout is None:
            raise FileNotFoundError(
                f"Could not resolve COCO layout for dataset: {ds_dir}\n"
                f"Expected either: {required_files[0]} / {required_files[1]} "
                "or annotations/instances_train.json / annotations/instances_val.json"
            )
        images_root = layout.images_root
        train_coco_path = layout.train_coco
        val_coco_path = layout.val_coco

        print(f"[INFO] processing:  {ds_dir}")
        print(f"[INFO] train_coco:   {train_coco_path}")
        print(f"[INFO] val_coco:     {val_coco_path}")
        print(f"[INFO] images_root:  {images_root}")

        train_coco = _read_json(train_coco_path)
        train_pairs = build_pairs_from_coco(
            train_coco, images_root, category_name, value_key
        )
        all_train_pairs.extend(train_pairs)
        print(f"[OK] train pairs from {ds_dir.name}: {len(train_pairs)}")

        val_coco = _read_json(val_coco_path)
        val_pairs = build_pairs_from_coco(
            val_coco, images_root, category_name, value_key
        )
        all_val_pairs.extend(val_pairs)
        print(f"[OK] val pairs from {ds_dir.name}:   {len(val_pairs)}")

        if layout.test_coco is not None:
            has_test_split = True
            print(f"[INFO] test_coco:    {layout.test_coco}")
            test_coco = _read_json(layout.test_coco)
            test_pairs = build_pairs_from_coco(
                test_coco, images_root, category_name, value_key
            )
            all_test_pairs.extend(test_pairs)
            print(f"[OK] test pairs from {ds_dir.name}:  {len(test_pairs)}")
        else:
            print(f"[INFO] no test split for {ds_dir.name}, skipping.")

    write_jsonl(all_train_pairs, train_out)
    write_jsonl(all_val_pairs, val_out)
    print(f"[OK] wrote {len(all_train_pairs)} samples -> {train_out}")
    print(f"[OK] wrote {len(all_val_pairs)} samples -> {val_out}")
    if has_test_split:
        write_jsonl(all_test_pairs, test_out)
        print(f"[OK] wrote {len(all_test_pairs)} samples -> {test_out}")
    else:
        print("[INFO] test output not written (no test COCO split found).")


if __name__ == "__main__":
    main()
