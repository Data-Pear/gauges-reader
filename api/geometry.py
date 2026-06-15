from __future__ import annotations

import math
from typing import Any, Optional, Tuple

Point = Tuple[float, float]


def angle_cw_deg(center: Point, point: Point) -> float:
    cx, cy = center
    px, py = point
    return (math.degrees(math.atan2(px - cx, cy - py)) + 360.0) % 360.0


def normalized_reading_from_angles(
    *,
    start_angle: float,
    end_angle: float,
    needle_angle: float,
) -> Optional[float]:
    sweep = (end_angle - start_angle) % 360.0
    if sweep <= 1e-6:
        return None

    progress = (needle_angle - start_angle) % 360.0
    if progress <= sweep:
        return float(max(0.0, min(1.0, progress / sweep)))

    before_start = (start_angle - needle_angle) % 360.0
    after_end = (needle_angle - end_angle) % 360.0
    return 0.0 if before_start <= after_end else 1.0


def needle_tip_from_mask(mask: Any, center: Point) -> Optional[Point]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    cx, cy = center
    dist2 = (xs.astype(float) - cx) ** 2 + (ys.astype(float) - cy) ** 2
    top_n = max(1, int(round(0.01 * len(xs))))
    if top_n >= len(xs):
        idx = np.arange(len(xs))
    else:
        idx = np.argpartition(dist2, -top_n)[-top_n:]
    return float(xs[idx].mean()), float(ys[idx].mean())
