from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from utils.config import load_config

N_SAMPLES = 10


def load_index(path: Path):
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def main():
    cfg = load_config("configs/config.yaml")

    index_path = Path(cfg["paths"]["train_output_json"]).resolve()
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    samples = load_index(index_path)
    if len(samples) == 0:
        raise RuntimeError("Index file is empty")

    n = min(N_SAMPLES, len(samples))
    chosen = random.sample(samples, n)

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for ax, rec in zip(axes, chosen):
        img_path = Path(rec["image_path"])
        value = rec["value"]

        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"value = {value:.3f}")
        ax.axis("off")

    # если ячеек больше, чем картинок
    for ax in axes[len(chosen) :]:
        ax.axis("off")

    fig.suptitle("Regression dataset sanity check", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
