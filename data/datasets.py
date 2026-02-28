from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GaugeValueDataset(Dataset):
    def __init__(self, index_jsonl: str | Path, transform: Optional[Callable] = None):
        self.index_path = Path(index_jsonl)
        self.transform = transform

        self.samples: list[tuple[str, float]] = []
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                self.samples.append((r["image_path"], float(r["value"])))

        if len(self.samples) == 0:
            raise ValueError(f"Index is empty: {self.index_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, value = self.samples[idx]

        # safer file handle usage
        with Image.open(img_path) as im:
            img = im.convert("RGB")

        if self.transform is not None:
            x = self.transform(img)  # expected torch.Tensor [3,H,W]
        else:
            # minimal default: HWC uint8 -> CHW float32 in [0,1]
            arr = np.asarray(img, dtype=np.uint8)
            x = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)

        # scalar target is удобнее для regression (loss будет проще)
        y = torch.tensor(value, dtype=torch.float32)
        return x, y


class DetKpDataset(Dataset):
    """
    Dataset для задачи:
      - bbox (циферблат / face_plate)
      - keypoints (dial_max, dial_min, dial_center, dial_tip)

    Ожидает jsonl, где каждая строка:
      {
        "image_id": int,
        "image_path": str,
        "bbox": [x1,y1,x2,y2],
        "keypoints": [[x,y,v], ...]  # len = K
      }

    Возвращает:
      image: torch.FloatTensor [3,H,W]
      target: dict:
        - boxes: FloatTensor [N,4] (xyxy)
        - labels: LongTensor [N] (в MVP один объект -> [1])
        - keypoints: FloatTensor [N,K,3] (x,y,v)
        - image_id: LongTensor [1]
    """

    def __init__(self, index_jsonl: str | Path, transform: Optional[Callable] = None):
        self.index_path = Path(index_jsonl)
        self.transform = transform

        self.samples: list[dict[str, Any]] = []
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

        if len(self.samples) == 0:
            raise ValueError(f"Index is empty: {self.index_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rec = self.samples[idx]

        img_path = Path(rec["image_path"])
        image_id = int(rec.get("image_id", idx))

        # Load image -> np HWC uint8
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        img_np = np.asarray(img, dtype=np.uint8)

        # One object per image (MVP)
        bbox = rec["bbox"]  # [x1,y1,x2,y2]
        if not (isinstance(bbox, list) and len(bbox) == 4):
            raise ValueError(f"Bad bbox at idx={idx}: {bbox}")

        kps = rec["keypoints"]  # [[x,y,v], ...]
        if not (isinstance(kps, list) and len(kps) > 0):
            raise ValueError(f"Bad keypoints at idx={idx}: {kps}")

        # Albumentations expects:
        # - bboxes as list of [x1,y1,x2,y2]
        # - keypoints as list of (x,y) OR (x,y,...) depending on keypoint_params(format=...)
        # We'll pass only (x,y) to transforms and keep v separately.
        kps_xy = [(float(p[0]), float(p[1])) for p in kps]
        kps_v = [float(p[2]) for p in kps]

        if self.transform is not None:
            out = self.transform(
                image=img_np,
                bboxes=[tuple(map(float, bbox))],
                keypoints=kps_xy,
                class_labels=[1],  # required if bbox_params has label_fields
            )
            img_np = out["image"]
            bboxes = out["bboxes"]
            keypoints = out["keypoints"]

            if len(bboxes) != 1:
                raise RuntimeError(
                    f"Expected 1 bbox after transform, got {len(bboxes)}"
                )
            bbox = list(map(float, bboxes[0]))

            if len(keypoints) != len(kps_v):
                raise RuntimeError(
                    f"Keypoints count changed after transform: {len(keypoints)} vs {len(kps_v)}"
                )
            kps_xy = [(float(x), float(y)) for (x, y) in keypoints]

        # Convert image -> torch [3,H,W] float in [0,1] if not already tensor
        if isinstance(img_np, torch.Tensor):
            # If transform used ToTensorV2, it returns CHW float
            image_t = img_np
            if image_t.dtype != torch.float32:
                image_t = image_t.float()
        else:
            image_t = torch.from_numpy(img_np).permute(2, 0, 1).float().div_(255.0)

        boxes_t = torch.tensor([bbox], dtype=torch.float32)  # [1,4]
        labels_t = torch.tensor([1], dtype=torch.int64)  # [1]
        keypoints_xyv = torch.tensor(
            [[[x, y, v] for (x, y), v in zip(kps_xy, kps_v)]],
            dtype=torch.float32,
        )  # [1,K,3]

        target: dict[str, torch.Tensor] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "keypoints": keypoints_xyv,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
        }

        return image_t, target
