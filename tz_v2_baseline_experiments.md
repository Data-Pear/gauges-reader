# ТЗ v2: Baseline эксперименты для статьи SynthGauge

**Обновлено:** учтён выбор Roboflow как real test set

**Дедлайн:** завтра (нужны цифры для статьи)

---

## Что нужно получить

Заполнить **3 таблицы** в статье конкретными числами:

### Таблица 3 — Detection

| Training Data | mAP@0.5 (Synth Test) | mAP@0.5 (Real Test) |
|---------------|----------------------|---------------------|
| SynthGauge | ? | ? |

### Таблица 4 — Reading Estimation

| Method | MAE (Synth) | DRR (Synth) |
|--------|-------------|-------------|
| ResNet-18 | ? | ? |

### Таблица 5 — Ablation (опционально, если успеешь)

| Configuration | mAP@0.5 (Real) |
|---------------|----------------|
| Full DR | ? |
| − Lighting | ? |
| No DR | ? |

---

## Данные

### Синтетический датасет (SynthGauge)

**Источник:** https://huggingface.co/datasets/Mileeena/synthetic-analog-gauges

| Split | Кол-во |
|-------|--------|
| Train | 7,000 |
| Val | 1,000 |
| Test | 1,000 |

**Формат:** COCO JSON

**Ключевые поля:**
- `bbox` — для detection
- `reading_normalized` — для regression (значение от 0 до 1)

### Real Test Set (Roboflow)

**Источник:** https://universe.roboflow.com/mnzn-li-j1jzu/pointer-instrument-7afwm

**Размер:** ~1,500 изображений реальных приборов

**Как скачать:**
```bash
pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="твой_ключ")  # бесплатная регистрация
project = rf.workspace("mnzn-li-j1jzu").project("pointer-instrument-7afwm")
dataset = project.version(2).download("coco")
```

**Важно:** В Roboflow есть только bbox (detection), нет reading annotations. Поэтому:
- Detection тестируем на Roboflow (real)
- Reading тестируем только на SynthGauge test (synth)

---

## Эксперимент 1: Detection (обязательно)

### Задача
Обучить YOLOv8n на SynthGauge, протестировать на:
1. SynthGauge test (synth→synth)
2. Roboflow (synth→real)

### Код

```python
from ultralytics import YOLO

# Конвертировать COCO в YOLO формат (или использовать встроенный)
model = YOLO("yolov8n.pt")

model.train(
    data="path/to/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    seed=42,
    optimizer="AdamW",
    lr0=1e-3,
    cos_lr=True,
    augment=False,  # DR уже в данных
)

# Eval на synth test
results_synth = model.val(data="path/to/synthgauge_test.yaml")
print(f"Synth mAP@0.5: {results_synth.box.map50}")

# Eval на real (Roboflow)
results_real = model.val(data="path/to/roboflow_test.yaml")
print(f"Real mAP@0.5: {results_real.box.map50}")
```

### Что записать в таблицу 3

- `mAP@0.5 (Synth Test)` — результат на SynthGauge test
- `mAP@0.5 (Real Test)` — результат на Roboflow

---

## Эксперимент 2: Reading Estimation (обязательно)

### Задача
Обучить ResNet-18 регрессию на SynthGauge, предсказывать `reading_normalized`.

### Код

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

class GaugeReadingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.backbone(x))

# Датасет: crop по bbox, resize 224x224, target = reading_normalized
# Loss: MSE
# Optimizer: AdamW, lr=1e-4
# Epochs: 50

# После обучения на train, eval на test:
# MAE = mean(|pred - gt|)
# DRR = процент где |pred - gt| < 0.02
```

### Что записать в таблицу 4

- `MAE (Synth)` — средняя абсолютная ошибка (число от 0 до 1, например 0.03)
- `DRR (Synth)` — процент с ошибкой < 2% (например 87%)

---

## Эксперимент 3: Ablation (опционально)

Если успеваешь — сравни Full DR vs No DR.

**No DR** = обучить на подмножестве SynthGauge где параметры максимально фиксированы (или просто взять первые 1000 картинок как proxy).

Это покажет, что Domain Randomization важен для sim-to-real.

---

## Deliverables (что прислать)

### Обязательно:

```
Таблица 3:
- SynthGauge → Synth Test: mAP@0.5 = X.XX
- SynthGauge → Real Test (Roboflow): mAP@0.5 = X.XX

Таблица 4:
- ResNet-18 MAE (Synth): X.XXX
- ResNet-18 DRR (Synth): XX%
```

### Опционально:
- Ablation результаты
- Чекпоинты моделей
- Логи обучения

---

## Частые вопросы

**Q: Как конвертировать COCO в YOLO?**

A: Ultralytics умеет читать COCO напрямую, или используй скрипт:
```python
# bbox COCO [x, y, w, h] → YOLO [x_center, y_center, w, h] normalized
x_center = (x + w/2) / img_width
y_center = (y + h/2) / img_height
w_norm = w / img_width
h_norm = h / img_height
```

**Q: Что делать если Roboflow требует API key?**

A: Бесплатная регистрация на roboflow.com, ключ в настройках аккаунта.

**Q: Reading на Roboflow?**

A: Не делаем — там нет reading annotations. Только detection.

---

## Контакт

Если что-то непонятно — пиши сразу, времени мало!
