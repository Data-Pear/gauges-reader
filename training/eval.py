from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(
        description="Unified evaluation entrypoint for detection, keypoints, and regression."
    )
    ap.add_argument(
        "--task",
        required=True,
        choices=["detection", "keypoints", "regression", "det_yolo", "det_kp"],
        help="Evaluation pipeline to run.",
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional explicit config path. If omitted, task default is used.",
    )
    return ap.parse_known_args()


def _task_defaults(task: str) -> tuple[Path, Path]:
    if task == "regression":
        return (
            PROJECT_ROOT / "training" / "eval_regression.py",
            PROJECT_ROOT / "configs" / "config_regression.yaml",
        )
    if task in ("keypoints", "det_kp"):
        return (
            PROJECT_ROOT / "training" / "eval_keypoints_yolo_pose.py",
            PROJECT_ROOT / "configs" / "config_keypoints.yaml",
        )
    return (
        PROJECT_ROOT / "training" / "eval_detection_yolo.py",
        PROJECT_ROOT / "configs" / "config_detection.yaml",
    )


def main() -> None:
    args, passthrough = _parse_args()
    script_path, default_config = _task_defaults(args.task)
    config_path = Path(args.config).resolve() if args.config else default_config.resolve()

    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        *passthrough,
    ]

    print(
        f"[INFO] task={args.task} script={script_path} "
        f"config={config_path}"
    )
    result = subprocess.run(cmd, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

