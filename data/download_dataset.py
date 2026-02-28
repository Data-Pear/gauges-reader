from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import kagglehub

DATASET_HANDLE = "endava/synthetic-data-for-precision-gauge-reading"


def safe_symlink_or_copy(src: Path, dst: Path) -> None:
    """
    Делает symlink, если возможно; иначе копирует.
    Удобно: кеш не дублируем, но проект получает стабильный путь.
    """
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def build_index(root_dir: Path, out_file: Path) -> None:
    """
    MVP-индекс: просто список изображений. Аннотации добавишь,
    когда разберёшь формат (csv/json) в датасете.
    """
    exts = {".jpg", ".jpeg", ".png"}
    images = sorted([p for p in root_dir.rglob("*") if p.suffix.lower() in exts])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for p in images:
            rec = {"image_path": str(p)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=str,
        default="data/raw/kaggle_endava_gauges",
        help="Куда положить датасет внутри проекта (папка/ссылка).",
    )
    ap.add_argument(
        "--version",
        type=int,
        default=None,
        help="Если хочешь закрепить версию: укажи номер версии Kaggle.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Перекачать заново (через kagglehub force_download).",
    )
    ap.add_argument(
        "--make-index",
        action="store_true",
        help="Сгенерировать data/processed/index.jsonl со списком изображений.",
    )
    args = ap.parse_args()

    # handle с версией
    handle = DATASET_HANDLE
    if args.version is not None:
        handle = f"{DATASET_HANDLE}/versions/{args.version}"

    # Скачивание (в кеш kagglehub)
    cache_path = Path(kagglehub.dataset_download(handle, force_download=args.force))

    # Стабильный путь в проекте
    out_dir = Path(args.out).resolve()
    safe_symlink_or_copy(cache_path, out_dir)

    print(f"[OK] Dataset cached at: {cache_path}")
    print(f"[OK] Project dataset path: {out_dir}")

    if args.make_index:
        idx_path = Path("data/processed/index.jsonl").resolve()
        build_index(out_dir, idx_path)
        print(f"[OK] Index saved to: {idx_path}")


if __name__ == "__main__":
    main()
