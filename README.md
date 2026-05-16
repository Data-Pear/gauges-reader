# Analog Gauge Reading Pipeline

Gauge reading pipeline for `Synthetic Analog Gauges Dataset v2.0`.

Raw v2.0 data is expected at:

```bash
data/raw/dataset_v2_0
```

Regression is intentionally left on its existing config and scripts.

## Tasks

- Dial detection: YOLO11n (`configs/config_detection.yaml`)
- Dial keypoints: YOLO11n-pose (`configs/config_keypoints.yaml`)
- Needle segmentation: YOLO11n-seg (`configs/config_segmentation.yaml`)
- Regression: existing regression pipeline (`configs/config_regression.yaml`)

## Metrics

- Detection: `precision`, `recall`, `mAP@0.5`, `mAP@0.5:0.95`
- Keypoints: `pose_precision`, `pose_recall`, `pose_mAP@0.5`, `pose_mAP@0.5:0.95`, `PCK@0.05`, `PCK@0.10`, `mean_keypoint_error_px`, `normalized_mean_keypoint_error`
- Needle segmentation: `mask_precision`, `mask_recall`, `mask_mAP@0.5`, `mask_mAP@0.5:0.95`, `needle_iou`, `needle_dice`, `needle_pixel_precision`, `needle_pixel_recall`
- Full pipeline: `reading_mae`, `reading_acc@5%`, `kpt_error`, `kpt_error_px`, `segmentation_iou`, `needle_angle_mae_deg`
- Regression: unchanged (`mae`, `rmse`, `r2`, `drr@0.02` where configured)

Segmentation uses both instance-style mask AP and pixel metrics. Mask AP checks whether the YOLO-seg model detects a usable needle instance over IoU thresholds; IoU and Dice measure direct overlap with the thin needle mask; pixel precision/recall separate over-segmentation from missed needle pixels. Dice is especially useful because the needle occupies a very small fraction of a 640x640 image.

## 1) Prepare Datasets

Dial detection labels/data yaml:

```bash
uv run --no-sync ./data/build_det_yolo_from_coco.py --config configs/config_detection.yaml
```

Keypoint labels/data yaml for `center`, `scale_start`, `scale_end`:

```bash
uv run --no-sync ./data/build_kp_yolo_pose_from_coco.py --config configs/config_keypoints.yaml
```

Needle segmentation crop labels/data yaml from indexed semantic masks and COCO dial boxes:

```bash
uv run --no-sync ./data/build_needle_seg_yolo.py --config configs/config_segmentation.yaml
```

## 2) Visualize Dataset Samples

```bash
uv run --no-sync ./data/visualize_v2_dataset_samples.py --split val --num-samples 6 --save data/processed/v2_dataset_samples.png
```

## 3) Train Models

```bash
uv run --no-sync ./training/train.py --task detection
uv run --no-sync ./training/train.py --task keypoints
uv run --no-sync ./training/train.py --task segmentation
```

Direct entrypoints are also available:

```bash
uv run --no-sync ./training/train_detection_yolo.py --config configs/config_detection.yaml
uv run --no-sync ./training/train_keypoints_yolo_pose.py --config configs/config_keypoints.yaml
uv run --no-sync ./training/train_needle_seg_yolo.py --config configs/config_segmentation.yaml
```

Weights are saved under:

```bash
models/weights/synthetic-analog-gauges-v2_0/det_yolo11n
models/weights/synthetic-analog-gauges-v2_0/kp_yolo11n-pose
models/weights/synthetic-analog-gauges-v2_0/seg_yolo11n-seg
```

## 4) Evaluate Models

```bash
uv run --no-sync ./training/eval.py --task detection --split test
uv run --no-sync ./training/eval.py --task keypoints --split test
uv run --no-sync ./training/eval.py --task segmentation --split test
uv run --no-sync ./training/eval.py --task full_pipeline --split test
```

Saved metric files:

```bash
data/processed/detection_metrics.json
data/processed/keypoints_metrics.json
data/processed/needle_segmentation_metrics.json
data/processed/full_pipeline_metrics.json
```

The full pipeline eval uses `configs/config_full_pipeline.yaml` and runs the deployed chain:
dial detection, detected dial crop, crop keypoints, crop needle mask, angle recovery, and
normalized reading on the `[0, 1]` scale. For a quick smoke run:

```bash
uv run --no-sync ./training/eval.py --task full_pipeline --split test --max-samples 20
```

Real-image Roboflow landmarks test set:

```bash
uv run --no-sync ./training/eval.py --task full_pipeline --config configs/config_full_pipeline_real_landmarks.yaml --split test
```

## 5) Visualize Pipeline Stages

Detection predictions:

```bash
uv run --no-sync ./inference/visualize_detection_predictions.py --split val --num-samples 6 --save data/processed/det_pred_samples.png
```

Keypoint predictions:

```bash
uv run --no-sync ./inference/visualize_keypoints_predictions.py --split val --num-samples 6 --save data/processed/kp_pred_samples.png
```

Needle segmentation predictions on dial crops, upscaled for display and without GT overlay by default:

```bash
uv run --no-sync ./inference/visualize_needle_seg_predictions.py --split val --num-samples 6 --save data/processed/needle_seg_pred_samples.png
```

Full three-stage pipeline visualization:

```bash
uv run --no-sync ./inference/visualize_full_pipeline_predictions.py --split val --num-samples 6 --save data/processed/full_pipeline_pred_samples.png
```

Real-image Roboflow landmarks full pipeline visualization:

```bash
uv run --no-sync ./inference/visualize_full_pipeline_real_landmarks.py --split test --num-samples 6 --save data/processed/full_pipeline_real_landmarks_samples.png
```

## 6) Unified Predict

```bash
uv run --no-sync ./inference/predict.py --task detection --image path/to/image.jpg
uv run --no-sync ./inference/predict.py --task keypoints --image path/to/image.jpg
uv run --no-sync ./inference/predict.py --task segmentation --image path/to/image.jpg
```
