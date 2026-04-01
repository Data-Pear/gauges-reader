# data/build_det_kp_dataset.py
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


@dataclass(frozen=True)
class TargetSpec:
    category_name: str
    keypoint_names: List[str]
    num_keypoints: int


@dataclass(frozen=True)
class DetKpLayout:
    train_inst: Path
    val_inst: Path
    train_kpts: Path
    val_kpts: Path
    test_inst: Optional[Path]
    test_kpts: Optional[Path]
    images_root: Path


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _required_files_from_cfg(paths_cfg: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(paths_cfg["train_inst_coco"]),
        str(paths_cfg["val_inst_coco"]),
        str(paths_cfg["train_kpts_coco"]),
        str(paths_cfg["val_kpts_coco"]),
    )


def _optional_test_files_from_cfg(paths_cfg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    inst = paths_cfg.get("test_inst_coco")
    kpts = paths_cfg.get("test_kpts_coco")
    inst_rel = str(inst) if inst is not None else None
    kpts_rel = str(kpts) if kpts is not None else None
    return inst_rel, kpts_rel


def _infer_test_relpaths(train_rel: str, val_rel: str) -> List[str]:
    candidates: List[str] = []
    for src, old in [(train_rel, "train"), (val_rel, "val")]:
        if old in src:
            repl = src.replace(old, "test", 1)
            if repl not in candidates:
                candidates.append(repl)
    return candidates


def _resolve_det_kp_layout(
    dataset_dir: Path,
    required_files: Tuple[str, str, str, str],
    optional_test_files: Tuple[Optional[str], Optional[str]],
) -> Optional[DetKpLayout]:
    (
        train_inst_rel,
        val_inst_rel,
        train_kpts_rel,
        val_kpts_rel,
    ) = required_files
    optional_test_inst, optional_test_kpts = optional_test_files
    train_inst_rel_norm = train_inst_rel.replace("\\", "/").lower()
    cfg_images_root = dataset_dir
    if train_inst_rel_norm.startswith("annotations/"):
        candidate_images_root = dataset_dir / "images"
        if candidate_images_root.exists():
            cfg_images_root = candidate_images_root

    # Endava-like layout with separate inst/kpts COCO files.
    cfg_test_inst: Optional[Path] = None
    cfg_test_kpts: Optional[Path] = None

    if optional_test_inst and optional_test_kpts:
        test_inst_candidate = dataset_dir / optional_test_inst
        test_kpts_candidate = dataset_dir / optional_test_kpts
        if test_inst_candidate.exists() and test_kpts_candidate.exists():
            cfg_test_inst = test_inst_candidate
            cfg_test_kpts = test_kpts_candidate
    else:
        inferred_inst_rels = _infer_test_relpaths(train_inst_rel, val_inst_rel)
        inferred_kpts_rels = _infer_test_relpaths(train_kpts_rel, val_kpts_rel)

        inferred_inst_path: Optional[Path] = None
        inferred_kpts_path: Optional[Path] = None
        for rel in inferred_inst_rels:
            p = dataset_dir / rel
            if p.exists():
                inferred_inst_path = p
                break
        for rel in inferred_kpts_rels:
            p = dataset_dir / rel
            if p.exists():
                inferred_kpts_path = p
                break

        if inferred_inst_path is not None and inferred_kpts_path is not None:
            cfg_test_inst = inferred_inst_path
            cfg_test_kpts = inferred_kpts_path

    cfg_layout = DetKpLayout(
        train_inst=dataset_dir / train_inst_rel,
        val_inst=dataset_dir / val_inst_rel,
        train_kpts=dataset_dir / train_kpts_rel,
        val_kpts=dataset_dir / val_kpts_rel,
        test_inst=cfg_test_inst,
        test_kpts=cfg_test_kpts,
        images_root=cfg_images_root,
    )
    if all(
        p.exists()
        for p in [
            cfg_layout.train_inst,
            cfg_layout.val_inst,
            cfg_layout.train_kpts,
            cfg_layout.val_kpts,
        ]
    ):
        return cfg_layout

    # HF synthetic-analog-gauges layout:
    # keypoints are stored in the same instances_*.json files.
    hf_train = dataset_dir / "annotations" / "instances_train.json"
    hf_val = dataset_dir / "annotations" / "instances_val.json"
    hf_test = dataset_dir / "annotations" / "instances_test.json"
    if hf_train.exists() and hf_val.exists():
        return DetKpLayout(
            train_inst=hf_train,
            val_inst=hf_val,
            train_kpts=hf_train,
            val_kpts=hf_val,
            test_inst=hf_test if hf_test.exists() else None,
            test_kpts=hf_test if hf_test.exists() else None,
            images_root=dataset_dir / "images",
        )

    return None


def _is_dataset_dir(
    path: Path,
    required_files: Tuple[str, str, str, str],
    optional_test_files: Tuple[Optional[str], Optional[str]],
) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return (
        _resolve_det_kp_layout(path, required_files, optional_test_files) is not None
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

    # keep old behavior as much as possible if nothing exists
    return cfg_raw


def _discover_dataset_dirs(
    raw_base: Path,
    required_files: Tuple[str, str, str, str],
    optional_test_files: Tuple[Optional[str], Optional[str]],
) -> List[Path]:
    if _is_dataset_dir(raw_base, required_files, optional_test_files):
        return [raw_base]

    if not raw_base.exists():
        raise FileNotFoundError(
            f"Raw dataset path not found: {raw_base}\n"
            "Hint: set paths.raw_ds_path to an existing dataset directory "
            "or pass --raw-root data/raw"
        )

    candidates: List[Path] = []
    for child in raw_base.iterdir():
        if _is_dataset_dir(child, required_files, optional_test_files):
            candidates.append(child.resolve())

    candidates.sort(key=_dataset_sort_key)
    return candidates


def _select_dataset_dirs(
    candidates: List[Path],
    dataset_arg: str,
    raw_base: Path,
    required_files: Tuple[str, str, str, str],
    optional_test_files: Tuple[Optional[str], Optional[str]],
) -> List[Path]:
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
        requested_dir = raw_base / mode
        if requested_dir.exists() and requested_dir.is_dir():
            layout = _resolve_det_kp_layout(
                requested_dir, required_files, optional_test_files
            )
            if layout is not None:
                return [requested_dir.resolve()]

            missing = [str(requested_dir / rel) for rel in required_files]
            raise ValueError(
                f"Dataset '{mode}' exists, but it is not det+kp-ready.\n"
                "Missing required files:\n"
                + "\n".join(f"  - {p}" for p in missing)
                + "\nSupported layouts:\n"
                "  - separate inst/kpts COCO files (Endava format)\n"
                "  - annotations/instances_{train,val}.json with keypoints field (HF synthetic format)"
            )
        available = ", ".join(d.name for d in candidates)
        raise ValueError(
            f"Dataset '{mode}' not found. Available: {available}. "
            "Use --dataset all|auto|<folder_name>."
        )
    return selected


def _default_output_paths(
    paths_cfg: Dict[str, Any],
    selected_dirs: List[Path],
) -> Tuple[Path, Path, Path]:
    processed_root = Path(str(paths_cfg.get("processed_ds_path", "data/processed"))).resolve()
    dataset_label = "__".join(d.name for d in selected_dirs)
    out_dir = processed_root / f"{dataset_label}_det_kp"

    train_name = Path(str(paths_cfg["train_det_kp_output_json"])).name
    val_name = Path(str(paths_cfg["val_det_kp_output_json"])).name
    test_cfg = paths_cfg.get("test_det_kp_output_json")
    if test_cfg is not None:
        test_name = Path(str(test_cfg)).name
    elif "val" in val_name:
        test_name = val_name.replace("val", "test", 1)
    else:
        test_name = "test_det_kp.jsonl"
    return out_dir / train_name, out_dir / val_name, out_dir / test_name


def _reindex_image_ids(records: List[Dict[str, Any]]) -> None:
    for idx, rec in enumerate(records):
        rec["image_id"] = idx


def _find_category_id(categories: List[Dict[str, Any]], name: str) -> int:
    for c in categories:
        if c.get("name") == name and isinstance(c.get("id"), int):
            return int(c["id"])
    # Fallback for datasets with a single category (e.g., gauge-only synthetic).
    if len(categories) == 1 and isinstance(categories[0].get("id"), int):
        only = categories[0]
        only_name = str(only.get("name", "<unknown>"))
        print(
            f"[WARN] Category '{name}' not found; using only available category '{only_name}'."
        )
        return int(only["id"])
    raise ValueError(f"Category '{name}' not found in COCO categories.")


def _bbox_xywh_to_xyxy(b: List[float]) -> List[float]:
    # COCO bbox: [x, y, w, h]
    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return [x, y, x + w, y + h]


def _select_one(
    anns: List[Dict[str, Any]],
    prefer_key: str = "area",
) -> Optional[Dict[str, Any]]:
    """
    If multiple annotations match (rare, but possible), pick the "best":
    - by largest 'area' if present
    - else by largest bbox area
    - else first
    """
    if not anns:
        return None
    if len(anns) == 1:
        return anns[0]

    # try 'area'
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for a in anns:
        if isinstance(a.get(prefer_key), (int, float)):
            scored.append((float(a[prefer_key]), a))
        else:
            bb = a.get("bbox")
            if isinstance(bb, list) and len(bb) == 4:
                scored.append((float(bb[2]) * float(bb[3]), a))
            else:
                scored.append((0.0, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _build_image_map(coco: Dict[str, Any]) -> Dict[int, str]:
    m: Dict[int, str] = {}
    for img in coco.get("images", []):
        img_id = img.get("id")
        fn = img.get("file_name")
        if isinstance(img_id, int) and isinstance(fn, str):
            m[int(img_id)] = fn
    return m


def _group_anns_by_image_and_instance(
    coco: Dict[str, Any],
    target_cat_id: int,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[Tuple[int, int], Dict[str, Any]]]:
    """
    Returns:
      - by_image[image_id] -> list of anns in target category
      - by_image_instance[(image_id, instance_id)] -> ann  (if instance_id present)
    """
    by_image: Dict[int, List[Dict[str, Any]]] = {}
    by_image_instance: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for ann in coco.get("annotations", []):
        if ann.get("category_id") != target_cat_id:
            continue
        img_id = ann.get("image_id")
        if not isinstance(img_id, int):
            continue

        by_image.setdefault(int(img_id), []).append(ann)

        inst_id = ann.get("instance_id")
        if isinstance(inst_id, int):
            by_image_instance[(int(img_id), int(inst_id))] = ann

    return by_image, by_image_instance


def _extract_keypoints_xyv(
    ann: Dict[str, Any],
    num_keypoints: int,
) -> Optional[List[List[float]]]:
    """
    COCO keypoints are typically stored as flat list: [x1,y1,v1, x2,y2,v2, ...]
    Return list of [x,y,v] length = num_keypoints.
    """
    kps = ann.get("keypoints")
    if not isinstance(kps, list):
        return None
    if len(kps) != 3 * num_keypoints:
        return None

    out: List[List[float]] = []
    for i in range(num_keypoints):
        x = float(kps[3 * i + 0])
        y = float(kps[3 * i + 1])
        v = float(kps[3 * i + 2])
        out.append([x, y, v])
    return out


def build_det_kp_index(
    inst_coco: Dict[str, Any],
    kpts_coco: Dict[str, Any],
    images_root: Path,
    target: TargetSpec,
) -> List[Dict[str, Any]]:
    # Validate categories & resolve category_id
    inst_cat_id = _find_category_id(
        inst_coco.get("categories", []), target.category_name
    )
    kpts_cat_id = _find_category_id(
        kpts_coco.get("categories", []), target.category_name
    )

    inst_img_map = _build_image_map(inst_coco)
    kpts_img_map = _build_image_map(kpts_coco)

    inst_by_image, inst_by_image_instance = _group_anns_by_image_and_instance(
        inst_coco, inst_cat_id
    )
    kpts_by_image, kpts_by_image_instance = _group_anns_by_image_and_instance(
        kpts_coco, kpts_cat_id
    )

    records: List[Dict[str, Any]] = []
    missing_inst = 0
    missing_kpts = 0
    missing_img = 0
    ambiguous = 0

    # Iterate over intersection of images present in both coco files
    image_ids = sorted(set(inst_img_map.keys()) & set(kpts_img_map.keys()))
    for img_id in image_ids:
        fn = inst_img_map.get(img_id) or kpts_img_map.get(img_id)
        if not fn:
            continue

        img_path = (images_root / fn).resolve()
        if not img_path.exists():
            missing_img += 1
            continue

        inst_candidates = inst_by_image.get(img_id, [])
        kpts_candidates = kpts_by_image.get(img_id, [])

        if not inst_candidates:
            missing_inst += 1
            continue
        if not kpts_candidates:
            missing_kpts += 1
            continue

        # Best-effort matching by instance_id if present.
        chosen_inst: Optional[Dict[str, Any]] = None
        chosen_kpts: Optional[Dict[str, Any]] = None

        # If kpts has instance_id, try to match to inst
        # (both files often contain instance_id for the same object)
        inst_ids_available = {
            a.get("instance_id")
            for a in inst_candidates
            if isinstance(a.get("instance_id"), int)
        }
        kpts_ids_available = {
            a.get("instance_id")
            for a in kpts_candidates
            if isinstance(a.get("instance_id"), int)
        }
        common_inst_ids = [i for i in inst_ids_available if i in kpts_ids_available]

        if common_inst_ids:
            # pick any common instance_id (should be one in this dataset)
            inst_id = int(sorted(common_inst_ids)[0])
            chosen_inst = inst_by_image_instance.get((img_id, inst_id))
            chosen_kpts = kpts_by_image_instance.get((img_id, inst_id))
        else:
            # fallback: choose "largest" in each and assume they correspond
            if len(inst_candidates) > 1 or len(kpts_candidates) > 1:
                ambiguous += 1
            chosen_inst = _select_one(inst_candidates)
            chosen_kpts = _select_one(kpts_candidates)

        if chosen_inst is None:
            missing_inst += 1
            continue
        if chosen_kpts is None:
            missing_kpts += 1
            continue

        bbox = chosen_inst.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            missing_inst += 1
            continue

        bbox_xyxy = _bbox_xywh_to_xyxy(bbox)

        kps_xyv = _extract_keypoints_xyv(chosen_kpts, target.num_keypoints)
        if kps_xyv is None:
            missing_kpts += 1
            continue

        records.append(
            {
                "image_id": int(img_id),
                "image_path": str(img_path),
                "bbox": bbox_xyxy,  # [x1,y1,x2,y2]
                "keypoints": kps_xyv,  # [[x,y,v], ...] in the order of keypoints_target.names
                "keypoint_names": target.keypoint_names,  # for clarity/debug; can be removed later
            }
        )

    if missing_img:
        print(
            f"[WARN] {missing_img} images referenced in COCO were not found under images_root={images_root}"
        )
    if missing_inst:
        print(
            f"[WARN] {missing_inst} images missing inst annotations for '{target.category_name}'"
        )
    if missing_kpts:
        print(
            f"[WARN] {missing_kpts} images missing keypoints annotations for '{target.category_name}' or wrong kp length"
        )
    if ambiguous:
        print(
            f"[WARN] {ambiguous} images had multiple candidates; used best-effort selection"
        )

    return records


def _write_jsonl(records: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build det+keypoints JSONL index from one or more COCO dataset folders."
    )
    ap.add_argument("--config", type=str, default="configs/config_det_kp.yaml")
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
        help="Override keypoints_target.category_name from config.",
    )
    ap.add_argument(
        "--keypoint-names",
        nargs="+",
        default=None,
        help="Override keypoint names (space-separated) in the same order as COCO keypoints.",
    )
    ap.add_argument(
        "--num-keypoints",
        type=int,
        default=None,
        help="Override keypoints_target.num_keypoints from config.",
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
    optional_test_files = _optional_test_files_from_cfg(paths_cfg)
    raw_base = _resolve_raw_base(cfg, args.raw_root)
    candidates = _discover_dataset_dirs(raw_base, required_files, optional_test_files)
    selected_dirs = _select_dataset_dirs(
        candidates,
        args.dataset,
        raw_base,
        required_files,
        optional_test_files,
    )

    if args.out_train:
        out_train = Path(args.out_train).resolve()
    else:
        out_train, _, _ = _default_output_paths(paths_cfg, selected_dirs)
    if args.out_val:
        out_val = Path(args.out_val).resolve()
    else:
        _, out_val, _ = _default_output_paths(paths_cfg, selected_dirs)
    if args.out_test:
        out_test = Path(args.out_test).resolve()
    else:
        _, _, out_test = _default_output_paths(paths_cfg, selected_dirs)

    layouts: Dict[Path, DetKpLayout] = {}
    for ds_dir in selected_dirs:
        layout = _resolve_det_kp_layout(ds_dir, required_files, optional_test_files)
        if layout is None:
            raise FileNotFoundError(f"Could not resolve det+kp layout for dataset: {ds_dir}")
        layouts[ds_dir] = layout

    # Synthetic-only convenience defaults.
    is_single_hf_synth_layout = False
    if len(selected_dirs) == 1:
        only_layout = layouts[selected_dirs[0]]
        is_single_hf_synth_layout = (
            only_layout.train_inst.name == "instances_train.json"
            and only_layout.train_kpts == only_layout.train_inst
            and only_layout.images_root.name == "images"
        )

    kp_cfg = cfg["keypoints_target"]
    default_category = str(kp_cfg["category_name"])
    default_names = list(kp_cfg["names"])
    default_num_keypoints = int(kp_cfg["num_keypoints"])

    category_name = (
        args.category_name
        or ("gauge" if is_single_hf_synth_layout else default_category)
    )
    keypoint_names = (
        list(args.keypoint_names)
        if args.keypoint_names
        else (
            ["center", "needle_tip", "scale_start", "scale_end"]
            if is_single_hf_synth_layout
            else default_names
        )
    )
    if args.num_keypoints is not None:
        num_keypoints = int(args.num_keypoints)
    elif args.keypoint_names:
        num_keypoints = len(keypoint_names)
    elif is_single_hf_synth_layout:
        num_keypoints = 4
    else:
        num_keypoints = default_num_keypoints

    if len(keypoint_names) != num_keypoints:
        raise ValueError(
            f"num_keypoints={num_keypoints} does not match number of keypoint names={len(keypoint_names)}."
        )

    target = TargetSpec(
        category_name=category_name,
        keypoint_names=keypoint_names,
        num_keypoints=num_keypoints,
    )

    print(f"[INFO] raw_base:     {raw_base}")
    print(f"[INFO] selected ds: {', '.join(d.name for d in selected_dirs)}")
    print(
        f"[INFO] target:       {target.category_name} | k={target.num_keypoints} | {target.keypoint_names}"
    )
    print(f"[INFO] out_train:    {out_train}")
    print(f"[INFO] out_val:      {out_val}")
    print(f"[INFO] out_test:     {out_test}")

    all_train_records: List[Dict[str, Any]] = []
    all_val_records: List[Dict[str, Any]] = []
    all_test_records: List[Dict[str, Any]] = []
    has_test_split = False

    for ds_dir in selected_dirs:
        layout = layouts[ds_dir]
        images_root = layout.images_root
        train_inst = layout.train_inst
        val_inst = layout.val_inst
        train_kpts = layout.train_kpts
        val_kpts = layout.val_kpts

        print(f"[INFO] processing:  {ds_dir}")
        print(f"[INFO] train_inst:   {train_inst}")
        print(f"[INFO] train_kpts:   {train_kpts}")
        print(f"[INFO] val_inst:     {val_inst}")
        print(f"[INFO] val_kpts:     {val_kpts}")
        if layout.test_inst is not None and layout.test_kpts is not None:
            print(f"[INFO] test_inst:    {layout.test_inst}")
            print(f"[INFO] test_kpts:    {layout.test_kpts}")
        else:
            print("[INFO] test split:   <none>")
        print(f"[INFO] images_root:  {images_root}")

        inst_coco = _read_json(train_inst)
        kpts_coco = _read_json(train_kpts)
        train_records = build_det_kp_index(inst_coco, kpts_coco, images_root, target)
        for rec in train_records:
            rec["source_dataset"] = ds_dir.name
        all_train_records.extend(train_records)
        print(f"[OK] train records from {ds_dir.name}: {len(train_records)}")

        inst_coco = _read_json(val_inst)
        kpts_coco = _read_json(val_kpts)
        val_records = build_det_kp_index(inst_coco, kpts_coco, images_root, target)
        for rec in val_records:
            rec["source_dataset"] = ds_dir.name
        all_val_records.extend(val_records)
        print(f"[OK] val records from {ds_dir.name}:   {len(val_records)}")

        if layout.test_inst is not None and layout.test_kpts is not None:
            has_test_split = True
            inst_coco = _read_json(layout.test_inst)
            kpts_coco = _read_json(layout.test_kpts)
            test_records = build_det_kp_index(inst_coco, kpts_coco, images_root, target)
            for rec in test_records:
                rec["source_dataset"] = ds_dir.name
            all_test_records.extend(test_records)
            print(f"[OK] test records from {ds_dir.name}:  {len(test_records)}")

    # If combining multiple dataset folders, make image ids unique within each split.
    if len(selected_dirs) > 1:
        _reindex_image_ids(all_train_records)
        _reindex_image_ids(all_val_records)
        _reindex_image_ids(all_test_records)

    _write_jsonl(all_train_records, out_train)
    _write_jsonl(all_val_records, out_val)
    print(f"[OK] wrote {len(all_train_records)} train records -> {out_train}")
    print(f"[OK] wrote {len(all_val_records)} val records   -> {out_val}")
    if has_test_split:
        _write_jsonl(all_test_records, out_test)
        print(f"[OK] wrote {len(all_test_records)} test records  -> {out_test}")
    else:
        print("[INFO] test output not written (no test COCO split found).")


if __name__ == "__main__":
    main()
