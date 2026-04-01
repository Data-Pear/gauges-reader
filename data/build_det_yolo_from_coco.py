from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.build_yolo_from_coco import build_from_config


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build YOLO detection labels/data yaml from COCO annotations."
    )
    ap.add_argument("--config", type=str, default="configs/config_detection.yaml")
    ap.add_argument("--raw-root", type=str, default=None)
    ap.add_argument("--out-yaml", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    build_from_config(args.config, raw_root=args.raw_root, out_yaml=args.out_yaml)


if __name__ == "__main__":
    main()
