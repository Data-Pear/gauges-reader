# data/build_det_kp_dataset.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.config import load_config


@dataclass(frozen=True)
class TargetSpec:
    category_name: str
    keypoint_names: List[str]
    num_keypoints: int


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_category_id(categories: List[Dict[str, Any]], name: str) -> int:
    for c in categories:
        if c.get("name") == name and isinstance(c.get("id"), int):
            return int(c["id"])
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


def main() -> None:
    cfg = load_config("configs/config.yaml")

    raw_ds_path = Path(cfg["paths"]["raw_ds_path"]).resolve()
    images_root = raw_ds_path

    train_inst = raw_ds_path / cfg["paths"]["train_inst_coco"]
    val_inst = raw_ds_path / cfg["paths"]["val_inst_coco"]
    train_kpts = raw_ds_path / cfg["paths"]["train_kpts_coco"]
    val_kpts = raw_ds_path / cfg["paths"]["val_kpts_coco"]

    out_train = Path(cfg["paths"]["train_det_kp_output_json"]).resolve()
    out_val = Path(cfg["paths"]["val_det_kp_output_json"]).resolve()

    kp_cfg = cfg["keypoints_target"]
    target = TargetSpec(
        category_name=str(kp_cfg["category_name"]),
        keypoint_names=list(kp_cfg["names"]),
        num_keypoints=int(kp_cfg["num_keypoints"]),
    )

    # Sanity checks
    for p in [train_inst, val_inst, train_kpts, val_kpts]:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
    if not images_root.exists():
        raise FileNotFoundError(f"images_root not found: {images_root}")

    print(f"[INFO] raw_ds_path:  {raw_ds_path}")
    print(f"[INFO] images_root:  {images_root}")
    print(
        f"[INFO] target:       {target.category_name} | k={target.num_keypoints} | {target.keypoint_names}"
    )

    # Train
    inst_coco = _read_json(train_inst)
    kpts_coco = _read_json(train_kpts)
    train_records = build_det_kp_index(inst_coco, kpts_coco, images_root, target)
    _write_jsonl(train_records, out_train)
    print(f"[OK] wrote {len(train_records)} -> {out_train}")

    # Val
    inst_coco = _read_json(val_inst)
    kpts_coco = _read_json(val_kpts)
    val_records = build_det_kp_index(inst_coco, kpts_coco, images_root, target)
    _write_jsonl(val_records, out_val)
    print(f"[OK] wrote {len(val_records)} -> {out_val}")


if __name__ == "__main__":
    main()
