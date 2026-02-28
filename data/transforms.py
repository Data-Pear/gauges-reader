from __future__ import annotations

from typing import Any, Callable, Dict

from albumentations.core.composition import BaseCompose
from albumentations.core.transforms_interface import BasicTransform
from torchvision import transforms as T


def _get_mean_std(cfg: Dict[str, Any]) -> tuple[list[float], list[float]]:
    mean = cfg.get("mean", [0.485, 0.456, 0.406])
    std = cfg.get("std", [0.229, 0.224, 0.225])
    return mean, std


def build_transforms(cfg: Dict[str, Any], split: str) -> Callable:
    """
    split: 'train' or 'val' (или 'test')
    Возвращает callable, который принимает PIL.Image и возвращает torch.Tensor [3,H,W]
    """
    tcfg = cfg.get("transforms", {})
    img_size = int(tcfg.get("img_size", 256))
    normalize = bool(tcfg.get("normalize", True))
    mean, std = _get_mean_std(tcfg)

    # Базовые преобразования (общие)
    base = [
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),  # -> float [0,1], CHW
    ]

    if split == "train":
        train_cfg = tcfg.get("train", {})

        # Важно: здесь только "безопасные" аугментации для MVP
        aug = []

        # ColorJitter
        if float(train_cfg.get("color_jitter_p", 0.0)) > 0:
            b, c, s, h = train_cfg.get("color_jitter", [0.2, 0.2, 0.2, 0.05])
            aug.append(
                T.RandomApply(
                    [T.ColorJitter(b, c, s, h)], p=float(train_cfg["color_jitter_p"])
                )
            )

        # RandomAffine (маленькие повороты/сдвиги/скейл)
        if float(train_cfg.get("random_affine_p", 0.0)) > 0:
            degrees = float(train_cfg.get("degrees", 7))
            translate = float(train_cfg.get("translate", 0.02))
            scale = train_cfg.get("scale", [0.95, 1.05])
            aug.append(
                T.RandomApply(
                    [
                        T.RandomAffine(
                            degrees=degrees,
                            translate=(translate, translate),
                            scale=(float(scale[0]), float(scale[1])),
                            interpolation=T.InterpolationMode.BILINEAR,
                            fill=0,
                        )
                    ],
                    p=float(train_cfg["random_affine_p"]),
                )
            )

        # Blur
        if float(train_cfg.get("blur_p", 0.0)) > 0:
            aug.append(
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                    p=float(train_cfg["blur_p"]),
                )
            )

        # HFlip (обычно выключено)
        hflip_p = float(train_cfg.get("hflip_p", 0.0))
        if hflip_p > 0:
            aug.append(T.RandomHorizontalFlip(p=hflip_p))

        pipeline = T.Compose(aug + base)
    else:
        pipeline = T.Compose(base)

    # Нормализация последним шагом
    if normalize:
        pipeline = T.Compose([pipeline, T.Normalize(mean=mean, std=std)])

    return pipeline


def build_transforms_det_kp(cfg: Dict[str, Any], split: str):
    """
    Albumentations transforms для задачи bbox + keypoints.
    Совместимо с DetKpDataset, который вызывает:
      transform(image=img_np, bboxes=[...], keypoints=[...], class_labels=[...])

    cfg expects:
      transforms_det_kp:
        backend: albumentations
        img_size: int
        bbox_format: pascal_voc
        clip: bool
        min_visibility: float
        train:
          photometric_p: float
          rotate_limit: float
          scale_limit: float
          translate_limit: float
          blur_p: float
          jpeg_p: float
    """
    tcfg = cfg.get("transforms_det_kp", {})
    backend = str(tcfg.get("backend", "albumentations")).lower()
    if backend != "albumentations":
        raise ValueError(
            "build_transforms_det_kp expects transforms_det_kp.backend=albumentations"
        )

    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except Exception as e:
        raise RuntimeError(
            "Albumentations is required for det+kps transforms. "
            "Install: pip install albumentations albumentations[pytorch]"
        ) from e

    img_size = int(tcfg.get("img_size", 256))
    bbox_format = str(tcfg.get("bbox_format", "pascal_voc"))
    clip = bool(tcfg.get("clip", True))
    min_visibility = float(tcfg.get("min_visibility", 0.0))

    # BBox + Keypoint params
    bbox_params = A.BboxParams(
        format=bbox_format,  # "pascal_voc" == xyxy
        label_fields=["class_labels"],  # DetKpDataset передаёт class_labels
        min_visibility=min_visibility,
        clip=clip,
    )
    keypoint_params = A.KeypointParams(
        format="xy",
        remove_invisible=False,  # visibility мы храним отдельно (v), сами не удаляем
    )

    # Базовые операции: resize всегда
    ops: list[BasicTransform | BaseCompose] = [
        A.Resize(img_size, img_size, interpolation=1)
    ]  # cv2.INTER_LINEAR

    if split == "train":
        tr = tcfg.get("train", {})

        photometric_p = float(tr.get("photometric_p", 0.7))
        rotate_limit = float(tr.get("rotate_limit", 0.0))
        scale_limit = float(tr.get("scale_limit", 0.0))
        translate_limit = float(tr.get("translate_limit", 0.0))
        blur_p = float(tr.get("blur_p", 0.0))
        jpeg_p = float(tr.get("jpeg_p", 0.0))

        # 1) Фотометрия (не ломает геометрию bbox/kps)
        if photometric_p > 0:
            ops.append(
                A.Compose(
                    [
                        A.RandomBrightnessContrast(p=0.5),
                        A.HueSaturationValue(p=0.3),
                        A.RandomGamma(p=0.3),
                    ],
                    p=photometric_p,
                )
            )

        # 2) Геометрия (аккуратно, малые значения — как в твоём конфиге)
        # Используем Affine: одновременно scale/translate/rotate
        if rotate_limit > 0 or scale_limit > 0 or translate_limit > 0:
            ops.append(
                A.Affine(
                    rotate=(-rotate_limit, rotate_limit)
                    if rotate_limit > 0
                    else (0.0, 0.0),
                    scale=(1.0 - scale_limit, 1.0 + scale_limit)
                    if scale_limit > 0
                    else (1.0, 1.0),
                    translate_percent=(-translate_limit, translate_limit)
                    if translate_limit > 0
                    else (0.0, 0.0),
                    interpolation=1,
                    border_mode=0,  # cv2.BORDER_CONSTANT
                    fit_output=False,
                    p=0.5,
                )
            )
        # 3) Деградации
        if blur_p > 0:
            ops.append(A.GaussianBlur(blur_limit=(3, 7), p=blur_p))

        if jpeg_p > 0:
            ops.append(A.ImageCompression(p=jpeg_p))

    # Для det/kp нормализация обычно не обязательна именно внутри A.Normalize,
    # но если хочешь пользоваться torchvision-pretrained, лучше нормализовать.
    # Возьмём mean/std из transforms_reg (если есть), иначе ImageNet дефолт.
    mean = cfg.get("transforms_reg", {}).get("mean", [0.485, 0.456, 0.406])
    std = cfg.get("transforms_reg", {}).get("std", [0.229, 0.224, 0.225])

    ops.append(A.Normalize(mean=mean, std=std, max_pixel_value=255.0))
    ops.append(ToTensorV2())

    return A.Compose(ops, bbox_params=bbox_params, keypoint_params=keypoint_params)
