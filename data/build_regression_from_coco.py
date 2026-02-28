from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.config import load_config


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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

    # collect values per image_id from target category annotations
    values_by_img: Dict[int, List[float]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("category_name") != category_name:
            continue

        img_id = ann.get("image_id")
        v = ann.get(value_key)

        if isinstance(img_id, int) and isinstance(v, (int, float)):
            values_by_img.setdefault(img_id, []).append(float(v))

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


def main() -> None:
    cfg = load_config("configs/config.yaml")

    raw_ds_path = Path(cfg["paths"]["raw_ds_path"]).resolve()

    images_root = raw_ds_path

    train_coco_path = raw_ds_path / cfg["paths"]["train_coco"]
    val_coco_path = raw_ds_path / cfg["paths"]["val_coco"]

    train_out = Path(cfg["paths"]["train_output_json"]).resolve()
    val_out = Path(cfg["paths"]["val_output_json"]).resolve()

    category_name = cfg["regression_target"]["category_name"]
    value_key = cfg["regression_target"]["value_key"]

    # sanity checks
    if not train_coco_path.exists():
        raise FileNotFoundError(f"train_coco not found: {train_coco_path}")
    if not val_coco_path.exists():
        raise FileNotFoundError(f"val_coco not found: {val_coco_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"images_path not found: {images_root}")

    print(f"[INFO] raw_ds_path:  {raw_ds_path}")
    print(f"[INFO] images_root:  {images_root}")
    print(f"[INFO] train_coco:   {train_coco_path}")
    print(f"[INFO] val_coco:     {val_coco_path}")
    print(
        f"[INFO] target:       category_name='{category_name}', value_key='{value_key}'"
    )

    # build train
    train_coco = read_json(train_coco_path)
    train_pairs = build_pairs_from_coco(
        train_coco, images_root, category_name, value_key
    )
    write_jsonl(train_pairs, train_out)
    print(f"[OK] wrote {len(train_pairs)} samples -> {train_out}")

    # build val
    val_coco = read_json(val_coco_path)
    val_pairs = build_pairs_from_coco(val_coco, images_root, category_name, value_key)
    write_jsonl(val_pairs, val_out)
    print(f"[OK] wrote {len(val_pairs)} samples -> {val_out}")


if __name__ == "__main__":
    main()
