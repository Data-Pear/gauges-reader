# SynthGauge: A Large-Scale Synthetic Dataset for Analog Gauge Reading with Domain Randomization

---

## Abstract

Automatic reading of analog gauges is essential for industrial monitoring, yet training robust computer vision models is hindered by the scarcity of annotated data. We present **SynthGauge**, a synthetic dataset of 9,000 photorealistic images covering 10+ instrument types with 42 scale configurations. Generated procedurally in Blender using Domain Randomization across 97 parameters, the dataset captures diverse lighting conditions, camera angles, materials, and weathering effects found in real industrial environments. Each image includes COCO-format bounding boxes, keypoints (`center`, `needle_tip`, `scale_start`, `scale_end`), and normalized needle readings, enabling detection, keypoint detection, and regression tasks. We release the dataset, generation pipeline, and trained models under CC BY 4.0 license.

---

## 1. Introduction

Analog gauges - pressure meters, voltmeters, ammeters, thermometers, and tachometers - remain prevalent in industrial facilities worldwide. Manual reading of these instruments is labor-intensive, error-prone, and potentially hazardous in hostile environments. Automated gauge reading through computer vision can enable continuous monitoring, reduce human exposure to dangerous conditions, and eliminate transcription errors.

Deep learning has achieved remarkable success in related tasks such as optical character recognition and object detection. However, applying these methods to analog gauge reading faces a fundamental obstacle: the lack of large-scale annotated datasets. Existing public datasets contain at most a few thousand images with limited diversity in instrument types and imaging conditions. Collecting real images requires physical access to operational equipment, while annotation demands expertise to correctly identify needle positions and scale parameters.

Synthetic data generation offers a solution to this scarcity. Procedural generation with Domain Randomization (DR) can produce unlimited labeled samples at negligible marginal cost. By randomizing visual parameters during rendering - lighting, textures, camera pose, materials - the resulting models learn features robust to any specific appearance, including real-world conditions.

In this work, we introduce **SynthGauge**, a procedurally generated dataset designed for gauge detection, keypoint localization, and reading estimation. Our contributions are:

1. **A large-scale synthetic dataset** of 9,000 photorealistic images covering 42 scale configurations across 10+ instrument types (manometers, voltmeters, ammeters, thermometers, tachometers, speedometers), with COCO-format annotations including normalized needle readings.

2. **Comprehensive Domain Randomization** spanning 97 parameters across seven categories: lighting (including industrial IES profiles), camera pose, PBR materials, gauge geometry, procedural weathering, post-processing effects, and position variation.

3. **A reproducible generation pipeline** based on Blender 4.2 LTS with complete configuration files, enabling researchers to generate custom variants or extend the dataset.

4. **Baseline experiments** for three repository pipelines: gauge detection, keypoint detection, and reading estimation on SynthGauge train/val/test splits.

---

**[FIGURE 1 - TEASER IMAGE HERE]**

**Figure 1.** Sample images from SynthGauge showing diversity of instrument types and imaging conditions. From left to right: manometer with color zones (0–150 bar), tachometer (0–8000 RPM), compound pressure gauge (−15 to +60 PSI), thermometer (0–250°C), tachometer with safety zones (0–6000 RPM). All images generated procedurally with Domain Randomization.

---

## 2. Related Work

### 2.1 Gauge Reading Datasets

Prior work on automatic gauge reading has been limited by data availability. UFPR-ADMR [1] provides 5,000 images of dial meters with reading annotations but focuses on utility meters rather than industrial gauges. SyntheticGauges [2] offers 10,000 synthetic images restricted to pressure gauges with limited visual diversity. The Roboflow Pointer Instrument dataset contains 1,500 real images with detection annotations only. Table 1 summarizes existing datasets.

**Table 1.** Comparison with existing gauge datasets.

| Dataset | Images | Synthetic | Types | Reading | COCO |
|---------|--------|-----------|-------|---------|------|
| UFPR-ADMR [1] | 5,000 | No | 1 | Yes | No |
| SyntheticGauges [2] | 10,000 | 100% | 1 | Yes | No |
| Roboflow Pointer | 1,500 | No | Mixed | No | Yes |
| **SynthGauge (Ours)** | **9,000** | **100%** | **10+** | **Yes** | **Yes** |

### 2.2 Synthetic Data and Domain Randomization

Domain Randomization [3] addresses the sim-to-real gap by introducing controlled variability during rendering, forcing models to learn domain-invariant features. Prior work has demonstrated its effectiveness for robotic manipulation [4], object detection in manufacturing [5], and pose estimation [6]. BlenderProc [7] provides infrastructure for photorealistic synthetic data generation with automatic annotation.

Our work extends these approaches with domain-specific randomization for industrial instrumentation, including IES lighting profiles, procedural weathering effects, and gauge-specific geometry variations.

### 2.3 Gauge Reading Methods

Classical approaches employ Hough transforms and template matching for needle detection [8]. Recent CNN-based methods [1,9] achieve higher accuracy through learned features. Multi-stage pipelines decomposing detection, segmentation, and reading estimation have shown practical success [10]. Our dataset supports both detection-based and end-to-end approaches.

---

## 3. Dataset

SynthGauge supports three computer vision tasks: (1) gauge face detection via bounding box localization, (2) keypoint detection (`center`, `needle_tip`, `scale_start`, `scale_end`), and (3) needle reading estimation via regression. We describe the generation pipeline, randomization parameters, and dataset statistics.

### 3.1 Generation Pipeline

Figure 2 illustrates the generation architecture comprising six components:

**Configuration.** YAML files define 97 DR parameter ranges and 42 scale configurations covering manometers (bar, kPa, PSI), voltmeters, ammeters, thermometers (°C), tachometers (RPM), speedometers (km/h), and specialized instruments (vacuum, compound, differential pressure gauges).

**Domain Randomization.** For each image, parameters are sampled from configured distributions across seven groups: lighting, camera, materials, geometry, weathering, post-processing, and position.

**3D Gauge Model.** Parametric geometry generated in Blender includes housing, bezel, glass cover, dial face with procedural scale markings, and needle. Dial textures support configurable tick marks, numeric labels, color zones, and manufacturer logos.

**Scene Setup.** Camera pose, lighting sources (point, sun, area, IES profiles), and HDRI environment maps selected from 484 industrial backgrounds (Poly Haven, CC0).

**Rendering.** Blender Cycles path tracer at 128 samples produces photorealistic output at 640×640 pixels.

**Post-processing.** Camera simulation effects including film grain, Gaussian blur, vignette, chromatic aberration, barrel distortion, bloom, and JPEG compression.

**Annotation.** COCO JSON with bounding boxes and custom fields: `reading_normalized` ∈ [0,1], `needle_angle_cw_deg`, scale parameters, and complete DR metadata per image.

---

**[FIGURE 2 - PIPELINE DIAGRAM HERE: pipeline.png]**

**Figure 2.** SynthGauge generation pipeline. Configuration files define 97 DR parameters and 42 scale configurations. Domain Randomization samples parameters for each image. The 3D gauge model and scene setup feed into Cycles rendering, followed by post-processing effects. COCO JSON annotations include bounding boxes, normalized readings, and full DR metadata.

---

### 3.2 Domain Randomization Parameters

We randomize 97 parameters across seven groups. Table 2 summarizes key parameters; complete specifications are available in the dataset documentation.

**Table 2.** Domain Randomization parameter groups.

| Group | Params | Key Ranges |
|-------|--------|------------|
| Lighting | 9 | 2–4 sources, 2700–6500K, 484 HDRIs, IES profiles (30%), dim mode (5%) |
| Camera | 6 | 35–85mm focal, 0.4–0.55m distance, ±15° tilt/azimuth |
| Materials | 18 | 11 housing materials, 4 needle shapes, 8 colors, glass types |
| Geometry | 6 | 42 scale configs, 0–270° needle angle, body shapes |
| Weathering | 4 | Edge wear, scratches, oil stains, glass smudges |
| Post-process | 12 | Noise, JPEG, blur, vignette, chromatic aberration, barrel distortion |
| Position | 2 | 30% off-center, up to 25% offset |

**Lighting** encompasses 2–4 sources with randomized type, intensity (1.1–1.25), and color temperature (2700–6500K). A dim lighting mode (5% of images) simulates poorly-lit industrial environments. Industrial IES luminaire profiles (30% of images) add realistic factory lighting patterns. HDRI environment maps provide contextual reflections and backgrounds.

**Materials** define visual appearance of gauge components. Housing materials include stainless steel variants, plastics, aluminum, brass, and rusty metal with procedural roughness. Glass covers are clean (70%) or scratched (30%). Needle shapes follow industrial standards: knife (70%), lollipop (15%), line (10%), spade (5%).

**Geometry** encompasses 42 scale configurations spanning pressure (bar, kPa, PSI), electrical (V, A), temperature (°C), and speed (RPM, km/h) measurements. Sweep angles range from 90° to 320°; most use 270° clockwise with counter-clockwise variants. Color zones (35%) and dual scales (15%) add visual diversity.

**Weathering** applies procedural aging via shader nodes: edge wear revealing bare metal, surface scratches, oil stains, and glass fingerprints simulate real industrial equipment in service.

**Post-processing** simulates camera characteristics: film grain (ISO 200–1600 equivalent), JPEG compression (quality 60–95), blur, vignette, chromatic aberration, and barrel distortion (40% of images).

---

**[FIGURE 3 - SAMPLE GRID HERE: figure2_sample_grid.jpg]**

**Figure 3.** Sample images from SynthGauge demonstrating diversity across instrument types (manometers, tachometers, thermometers, ammeters), scale configurations, materials (stainless steel, plastic, brass), dial colors, needle shapes, color zones, backgrounds (industrial, warehouse, outdoor), and imaging conditions (lighting, angles, weathering effects).

---

### 3.3 Annotation Format

Annotations follow COCO JSON format with custom extensions:

```json
{
  "annotations": [{
    "bbox": [x, y, width, height],
    "keypoints": [
      center_x, center_y, center_v,
      needle_tip_x, needle_tip_y, needle_tip_v,
      scale_start_x, scale_start_y, scale_start_v,
      scale_end_x, scale_end_y, scale_end_v
    ],
    "reading_normalized": 0.472,
    "attributes": {
      "needle_angle_cw_deg": 127.5,
      "scale_min": 0, "scale_max": 10,
      "scale_unit": "bar"
    }
  }]
}
```

Keypoints are ordered as `center -> needle_tip -> scale_start -> scale_end` and stored in standard COCO `[x, y, v]` triplets.

The `reading_normalized` field provides the primary regression target, with actual readings computed as: `v = scale_min + reading_normalized * (scale_max - scale_min)`.

Per-image metadata records all 97 DR parameter values, enabling fine-grained ablation studies and dataset filtering by specific conditions.

### 3.4 Dataset Statistics

SynthGauge contains 9,000 images split 7,000/1,000/1,000 for train/val/test. All images are 640×640 JPEG.

**Instrument distribution:** Manometers 34%, voltmeters 30%, ammeters 30%, thermometers 2.5%, tachometers 1.5%, speedometers 1%, other 1%.

**Bounding box statistics:** Mean width 445px, mean height 449px, gauge coverage 55–75% of frame.

**Reading distribution:** Mean 0.50, range [0.02, 0.98], approximately uniform.

---

**[FIGURE 4 - DR VARIATIONS HERE: figure3_dr_variations.jpg]**

**Figure 4.** Examples showing Domain Randomization effects. Different dial colors (cream, white, dark), lighting conditions, backgrounds, and units (bar, RPM, kPa) demonstrate the visual diversity achieved through procedural generation.

---

## 4. Experiments

This repository currently contains three independent, reproducible training pipelines on `synthetic-analog-gauges`: (1) gauge detection, (2) keypoint detection, and (3) reading regression.

### 4.1 Reproducible Protocol in This Repository

All scripts are run from project root with `uv run`. Dataset splits are defined by COCO files:
`annotations/instances_train.json`, `annotations/instances_val.json`, `annotations/instances_test.json`.

Data preparation scripts:

- Detection labels (YOLO bbox): `data/build_det_yolo_from_coco.py`
- Keypoint labels (YOLO pose): `data/build_kp_yolo_pose_from_coco.py`
- Regression index (JSONL): `data/build_regression_from_coco.py`

Training scripts:

- Detection: `training/train_detection_yolo.py`
- Keypoints: `training/train_keypoints_yolo_pose.py`
- Regression: `training/train_regression.py`

Checkpoints are stored as:
`models/weights/{dataset_name}/{det|kp|reg}_{model_name}`.

### 4.2 Gauge Detection (Bounding Box)

**Model.** YOLOv8n, COCO-pretrained.

**Training setup.** 100 epochs, batch size 16, AdamW, initial learning rate `1e-3`, cosine LR schedule, input size `640x640`.

**Metrics.** `precision`, `recall`, `mAP@0.5`, `mAP@0.5:0.95`.

**Artifacts.** Metrics are saved to `data/processed/detection_metrics.json`.

**Table 3.** Detection results on SynthGauge test split.

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| YOLOv8n | [TBD] | [TBD] | [TBD] | [TBD] |

### 4.3 Gauge Keypoint Detection (Pose)

**Model.** YOLOv8n-pose, pretrained initialization.

**Training setup.** 100 epochs, batch size 16, AdamW, initial learning rate `1e-3`, cosine LR schedule, input size `640x640`.

**Keypoint schema.** Ground-truth keypoints are ordered as:
`center -> needle_tip -> scale_start -> scale_end`.

**Metrics.**

- Pose detection quality: `pose_precision`, `pose_recall`, `pose_mAP@0.5`, `pose_mAP@0.5:0.95`
- Keypoint localization quality: `PCK@0.05`, `PCK@0.10`
- Needle geometry quality: `mean_angular_error_deg`

`PCK` and angular error are computed by a custom post-validation evaluator that matches the best predicted gauge instance to the ground-truth gauge per image.

**Artifacts.** Metrics are saved to `data/processed/keypoints_metrics.json`.

**Table 4.** Keypoint detection results on SynthGauge test split.

| Model | pose_P | pose_R | pose_mAP@0.5 | pose_mAP@0.5:0.95 | PCK@0.05 | PCK@0.10 | Mean Angular Error (deg) |
|-------|--------|--------|--------------|-------------------|----------|----------|--------------------------|
| YOLOv8n-pose | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### 4.4 Reading Estimation (Regression)

**Model.** ResNet-18 regression head predicting `reading_normalized` from cropped gauge images.

**Training setup.** 50 epochs, MSE loss, AdamW optimizer (default config: batch size 32, lr `3e-4`, weight decay `1e-4`).

**Metrics.** `MAE` (normalized), `DRR@0.02` (Dial Recognition Rate, i.e., percent within 2% absolute error). Auxiliary logged metrics: `RMSE`, `R2`.

**Artifacts.** Validation metrics are logged during training (`data/processed/train_regression.log`) and stored in checkpoint metadata.

**Table 5.** Reading estimation results on SynthGauge test split.

| Model | MAE | DRR@0.02 | RMSE | R2 |
|-------|-----|----------|------|----|
| ResNet-18 | [TBD] | [TBD] | [TBD] | [TBD] |

### 4.5 Scope Alignment

At the moment, the repository implements synthetic train/val/test experiments only. Cross-domain sim-to-real evaluation and DR ablations are planned extensions, but are not part of the current reproducible baseline scripts.

## 5. Limitations and Future Work

SynthGauge focuses on round analog gauges with single needles. Multi-needle instruments, digital displays, and non-circular gauges are out of scope. The 42 scale configurations, while diverse, do not exhaustively cover all industrial variants.

Despite extensive DR, systematic sim-to-real differences persist for factors not modeled: motion blur, extreme contamination, partial occlusion, and non-standard gauge modifications.

Future directions include expanding to additional instrument types, incorporating diffusion-based refinement for enhanced realism, and few-shot adaptation protocols for rapid deployment to new gauge types.

---

## 6. Conclusion

We introduced SynthGauge, a synthetic dataset of 9,000 analog gauge images with comprehensive Domain Randomization across 97 parameters. The dataset covers 42 scale configurations spanning 10+ instrument types, annotated in COCO format with bounding boxes, ordered keypoints, and normalized readings. By releasing the dataset, generation pipeline, and baseline models under CC BY 4.0, we provide a foundation for research on synthetic data for industrial computer vision where real annotated data is scarce.

---

## References

[1] G. Salomon, R. Laroca, D. Menotti. "Image-based Automatic Dial Meter Reading in Unconstrained Scenarios." Measurement, 2022.

[2] Cambridge SyntheticGauges Dataset, 2021.

[3] J. Tobin et al. "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World." IROS, 2017.

[4] OpenAI. "Learning Dexterous In-Hand Manipulation." IJRR, 2020.

[5] D. Horvath et al. "Object Detection Using Sim2Real Domain Randomization for Robotic Applications." IEEE T-RO, 2022.

[6] M. Elsisi et al. "Domain Randomization for Object Detection in Manufacturing Applications." IEEE Access, 2024.

[7] M. Denninger et al. "BlenderProc: Reducing the Reality Gap with Photorealistic Rendering." arXiv:1911.01911, 2019.

[8] R. Laroca et al. "Convolutional Neural Networks for Automatic Meter Reading." JEI, 2019.

[9] A. Barucci et al. "A Deep Learning Approach to Ancient Egyptian Hieroglyphs Classification." IEEE Access, 2021.

[10] Y. Lin et al. "A Pointer Type Instrument Intelligent Reading System Design Based on CNNs." Frontiers in Physics, 2020.

---

## Dataset and Code Availability

**Dataset:** https://huggingface.co/datasets/[anonymous]/synthetic-analog-gauges

**Code:** https://github.com/[anonymous]/synthgauge

**License:** CC BY 4.0

---

*Generated with Blender 4.2.1 LTS on NVIDIA RTX 3060. Total generation time: ~12.5 hours for 9,000 images.*
