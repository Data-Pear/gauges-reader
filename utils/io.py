from typing import Any, List, Tuple

import torch


def det_collate_fn(batch: List[Tuple[torch.Tensor, Any]]):
    images, targets = zip(*batch)
    return list(images), list(targets)
