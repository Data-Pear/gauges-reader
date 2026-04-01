from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class RegressionMeter:
    """
    Агрегатор метрик для регрессии:
      - mae
      - rmse
      - r2
      - acc@tol (опционально)
    Работает по батчам: meter.update(y_pred, y_true)
    """

    tol: Optional[float] = None  # если задан, считаем acc@tol

    # internal accumulators
    _n: int = 0
    _sum_abs: float = 0.0
    _sum_sq: float = 0.0
    _sum_y: float = 0.0
    _sum_y2: float = 0.0
    _sum_correct: int = 0

    def reset(self) -> None:
        self._n = 0
        self._sum_abs = 0.0
        self._sum_sq = 0.0
        self._sum_y = 0.0
        self._sum_y2 = 0.0
        self._sum_correct = 0

    @torch.no_grad()
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        y_pred: shape [B] or [B,1]
        y_true: shape [B] or [B,1]
        """
        yp = y_pred.detach().float().view(-1)
        yt = y_true.detach().float().view(-1)

        if yp.numel() != yt.numel():
            raise ValueError(
                f"y_pred and y_true must have same numel, got {yp.numel()} vs {yt.numel()}"
            )

        diff = yp - yt
        b = int(yt.numel())

        self._n += b
        self._sum_abs += float(diff.abs().sum().item())
        self._sum_sq += float((diff * diff).sum().item())
        self._sum_y += float(yt.sum().item())
        self._sum_y2 += float((yt * yt).sum().item())

        if self.tol is not None:
            self._sum_correct += int((diff.abs() <= self.tol).sum().item())

    def compute(self) -> Dict[str, float]:
        if self._n == 0:
            drr_key = f"drr@{self.tol:.2f}" if self.tol is not None else None
            return {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                **({"acc@tol": float("nan")} if self.tol is not None else {}),
                **({drr_key: float("nan")} if drr_key is not None else {}),
            }

        mae = self._sum_abs / self._n
        rmse = (self._sum_sq / self._n) ** 0.5

        # R^2 = 1 - SSE/SST, где SST = sum((y - mean)^2)
        mean_y = self._sum_y / self._n
        sst = self._sum_y2 - self._n * (mean_y * mean_y)
        sse = self._sum_sq
        # если все y одинаковые, sst может быть 0
        r2 = 1.0 - (sse / sst) if sst > 1e-12 else float("nan")

        out = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

        if self.tol is not None:
            acc = float(self._sum_correct / self._n)
            out["acc@tol"] = acc
            out[f"drr@{self.tol:.2f}"] = acc

        return out


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Удобный формат для логов.
    """
    items = []
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if v != v:  # NaN check
            items.append(f"{prefix}{k}=nan")
        else:
            items.append(f"{prefix}{k}={v:.6f}")
    return " ".join(items)


@dataclass
class KeypointMeter:
    """
    Метрики для det + keypoints:
      - kpt_mae: средняя L2-ошибка (px) по видимым GT-keypoints
      - kpt_pck@thr: доля keypoints с ошибкой <= thr * max(bbox_w, bbox_h)
      - det_recall: доля изображений, где удалось взять предсказание >= score_thr
    """

    pck_thr: float = 0.05
    score_thr: float = 0.3

    _num_images: int = 0
    _num_images_with_pred: int = 0
    _num_points: int = 0
    _sum_dist: float = 0.0
    _num_pck_correct: int = 0

    def reset(self) -> None:
        self._num_images = 0
        self._num_images_with_pred = 0
        self._num_points = 0
        self._sum_dist = 0.0
        self._num_pck_correct = 0

    @staticmethod
    def _select_prediction(
        pred: Dict[str, torch.Tensor], score_thr: float
    ) -> Optional[torch.Tensor]:
        """
        Возвращает keypoints [K,2|3] у лучшей детекции, либо None.
        Приоритет: max score >= score_thr, иначе None.
        """
        kps = pred.get("keypoints")
        if kps is None or kps.numel() == 0:
            return None

        scores = pred.get("scores")
        if scores is None or scores.numel() == 0:
            return kps[0]

        keep = torch.nonzero(scores >= float(score_thr), as_tuple=False).view(-1)
        if keep.numel() == 0:
            return None

        best_local = keep[torch.argmax(scores[keep])]
        return kps[int(best_local)]

    @torch.no_grad()
    def update_batch(
        self, outputs: list[Dict[str, torch.Tensor]], targets: list[Dict[str, Any]]
    ) -> None:
        if len(outputs) != len(targets):
            raise ValueError(
                f"outputs/targets batch mismatch: {len(outputs)} vs {len(targets)}"
            )

        for pred, tgt in zip(outputs, targets):
            self._num_images += 1

            gt_kps = tgt["keypoints"]  # [1,K,3]
            gt_boxes = tgt["boxes"]  # [1,4]
            if gt_kps.ndim != 3 or gt_kps.shape[0] < 1:
                continue
            if gt_boxes.ndim != 2 or gt_boxes.shape[0] < 1:
                continue

            gt_kps = gt_kps[0].detach().float().cpu()  # [K,3]
            gt_box = gt_boxes[0].detach().float().cpu()  # [4]

            vis_mask = gt_kps[:, 2] > 0
            n_vis = int(vis_mask.sum().item())
            if n_vis == 0:
                continue

            bw = max(float((gt_box[2] - gt_box[0]).item()), 1.0)
            bh = max(float((gt_box[3] - gt_box[1]).item()), 1.0)
            scale = max(bw, bh, 1.0)
            pck_radius = float(self.pck_thr) * scale

            self._num_points += n_vis

            chosen = self._select_prediction(pred, score_thr=self.score_thr)
            if chosen is None:
                # если детекция не найдена, считаем это полной ошибкой
                self._sum_dist += scale * n_vis
                continue

            self._num_images_with_pred += 1

            pred_xy = chosen.detach().float().cpu()[:, :2]
            gt_xy = gt_kps[:, :2]
            d = torch.linalg.norm(pred_xy[vis_mask] - gt_xy[vis_mask], dim=1)

            self._sum_dist += float(d.sum().item())
            self._num_pck_correct += int((d <= pck_radius).sum().item())

    def compute(self) -> Dict[str, float]:
        pck_name = f"kpt_pck@{self.pck_thr:.2f}"

        if self._num_points <= 0:
            return {
                "kpt_mae": float("nan"),
                pck_name: float("nan"),
                "det_recall": float("nan"),
            }

        kpt_mae = self._sum_dist / self._num_points
        kpt_pck = self._num_pck_correct / self._num_points
        det_recall = (
            self._num_images_with_pred / self._num_images
            if self._num_images > 0
            else float("nan")
        )

        return {
            "kpt_mae": float(kpt_mae),
            pck_name: float(kpt_pck),
            "det_recall": float(det_recall),
        }
