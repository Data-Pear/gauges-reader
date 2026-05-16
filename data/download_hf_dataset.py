from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'huggingface_hub'. Install with: uv sync"
    ) from exc

DEFAULT_REPO_ID = "Mileeena/synthetic-analog-gauges"
DEFAULT_OUT = "data/raw/synthetic-analog-gauges"


def _token_from_env_file(path: Path) -> str | None:
    if not path.is_file():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "HF_TOKEN":
            return value.strip().strip("\"'")
    return None


def _default_token() -> str | None:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or _token_from_env_file(Path(".env"))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset into data/raw."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT,
        help="Target directory inside the project.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Branch, tag or commit hash.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=_default_token(),
        help="HF token for private/gated datasets. "
        "Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN env var, then .env HF_TOKEN.",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Optional Hugging Face allow_patterns list, e.g. annotations/* images/*.",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=None,
        help="Optional Hugging Face ignore_patterns list, e.g. metadata/*.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum parallel Hub downloads. Lower this if HF returns 429.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if files already exist in cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=str(out_dir),
            token=args.token,
            force_download=args.force,
            allow_patterns=args.allow_patterns,
            ignore_patterns=args.ignore_patterns,
            max_workers=args.max_workers,
        )
    except GatedRepoError as exc:
        raise SystemExit(
            "Access denied: dataset is gated.\n"
            "1) Open https://huggingface.co/datasets/Mileeena/synthetic-analog-gauges and request/accept access.\n"
            "2) Create token: https://huggingface.co/settings/tokens\n"
            "3) Run either:\n"
            "   - huggingface-cli login\n"
            "   - or set env var HF_TOKEN and rerun script."
        ) from exc
    except HfHubHTTPError as exc:
        raise SystemExit(f"HF Hub request failed: {exc}") from exc

    print(f"[OK] Dataset downloaded: {args.repo_id}")
    print(f"[OK] Saved to: {out_dir}")


if __name__ == "__main__":
    main()
