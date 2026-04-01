# gauges-reader

Gauge reading pipeline on `synthetic-analog-gauges`.

## Run From Project Root

Always run scripts from the repository root via `uv run`.

## Tasks

- Detection: YOLOv8n (`configs/config_detection.yaml`)
- Keypoint detection: YOLOv8n-pose (`configs/config_keypoints.yaml`)
- Regression: ResNet-18 (`configs/config_regression.yaml`)

## Metrics

- Detection: `precision`, `recall`, `mAP@0.5`, `mAP@0.5:0.95` (saved to `data/processed/detection_metrics.json`)
- Keypoint detection: `pose_precision`, `pose_recall`, `pose_mAP@0.5`, `pose_mAP@0.5:0.95`, `PCK@0.05`, `PCK@0.10`, `mean_angular_error_deg` (saved to `data/processed/keypoints_metrics.json`)
- Regression: `mae`, `drr@0.02` (Dial Recognition Rate), plus `rmse` and `r2` (logged during validation)

## 1) Prepare Datasets

Regression index (JSONL):

```powershell
uv run .\data\build_regression_from_coco.py --raw-root data/raw --dataset synthetic-analog-gauges --category-name gauge --value-key reading_normalized
```

Detection labels/data yaml (YOLO bbox):

```powershell
uv run .\data\build_det_yolo_from_coco.py --config configs/config_detection.yaml
```

Keypoint labels/data yaml (YOLO pose):

```powershell
uv run .\data\build_kp_yolo_pose_from_coco.py --config configs/config_keypoints.yaml
```

Detection and keypoint pipelines use separate YOLO dataset roots (`paths.yolo_dataset_root`), so labels do not overwrite each other.

## 2) Train Models

YOLOv8n detection:

```powershell
uv run .\training\train_detection_yolo.py --config configs/config_detection.yaml
```

YOLOv8n-pose keypoint detection:

```powershell
uv run .\training\train_keypoints_yolo_pose.py --config configs/config_keypoints.yaml
```

Regression (ResNet-18, `reading_normalized`):

```powershell
uv run .\training\train_regression.py --config configs/config_regression.yaml
```

Unified entrypoint:

```powershell
uv run .\training\train.py --task detection
uv run .\training\train.py --task keypoints
uv run .\training\train.py --task regression
```

Weights are saved to:

```powershell
models/weights/{dataset_name}/{det|kp|reg}_{model_name}
```

Example:

```powershell
models/weights/synthetic-analog-gauges/det_yolov8n
models/weights/synthetic-analog-gauges/kp_yolov8n-pose
models/weights/synthetic-analog-gauges/reg_resnet18
```

## 3) Inference Visualizations

Detection (bbox pred vs gt):

```powershell
uv run .\inference\visualize_detection_predictions.py --config configs/config_detection.yaml --split val --num-samples 10 --save data/processed/det_pred_samples.png
```

Keypoints (pred vs gt):

```powershell
uv run .\inference\visualize_keypoints_predictions.py --config configs/config_keypoints.yaml --split val --num-samples 10 --save data/processed/kp_pred_samples.png
```

Regression (pred value vs gt):

```powershell
uv run .\inference\visualize_regression_predictions.py --config configs/config_regression.yaml --split val --num-samples 12 --save data/processed/reg_pred_samples.png
```
