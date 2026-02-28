# gauges-reader

Проект для чтения показаний манометров по изображению.

## Обучение det+keypoints

1. Подготовить det+kp индексы из COCO-аннотаций:

```bash
python data/build_det_kp_from_coco.py
```

2. Запустить обучение Keypoint R-CNN:

```bash
python training/train_det_kp.py --config configs/config.yaml
```

Чекпоинты сохраняются в `models/weights/det_kp/`:
- `last.pt`
- `best.pt`

Лог обучения сохраняется в `data/processed/train_det_kp.log`.

## Полезные параметры

В `configs/config.yaml`:
- `training_det_kp.best_metric`: `kpt_mae` или `kpt_pck@0.05`
- `training_det_kp.pck_thr`: порог для PCK-метрики
- `training_det_kp.score_thr`: порог confidence при валидации
- `training_det_kp.pretrained_coco`: использовать COCO-pretrained веса
