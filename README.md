# LungScan3D

**LungScan3D** — Python-пакет для обучения, валидации, упаковки и инференса 3D CNN-модели, решающей задачу бинарной классификации кандидатов лёгочных узелков на компьютерной томографии.

---

# 1. Описание проекта

## 1.1. Постановка задачи

В проекте решается задача бинарной классификации 3D-фрагментов компьютерной томографии лёгких:

> по трёхмерному CT-фрагменту вокруг кандидата определить, является ли кандидат настоящим лёгочным узелком или ложным срабатыванием.

Модель принимает на вход нормализованный CT-фрагмент и предсказывает вероятность положительного класса:

```text
0 — ложный кандидат / non-nodule / false positive candidate
1 — настоящий лёгочный узелок / true nodule
```

Важно: текущая версия проекта **не решает задачу определения злокачественности**. LUNA16 содержит координаты узелков и список кандидатов, но не является датасетом с прямой клинической меткой «злокачественный / доброкачественный». Поэтому постановка проекта строго такая:

```text
candidate -> nodule / non-nodule
```

## 1.2. Медицинская и практическая мотивация

Анализ CT-исследований лёгких часто строится как многоступенчатый пайплайн:

1. поиск кандидатов подозрительных областей;
2. фильтрация ложных кандидатов;
3. ранжирование найденных областей для дальнейшей интерпретации врачом.

LungScan3D покрывает второй этап: **false-positive reduction**. Эта часть важна, потому что алгоритмы-кандидатогенераторы обычно оптимизируются на высокую чувствительность и возвращают много ложных срабатываний. 3D CNN-классификатор снижает количество ложных кандидатов, не теряя настоящие узелки.

## 1.3. Датасет LUNA16

Основной датасет проекта — **LUNA16**. Он предоставляет CT-исследования в медицинском 3D-формате:

```text
*.mhd
*.raw
```

Также используются CSV-файлы:

```text
annotations.csv
candidates.csv
```

Для текущей задачи главным является `candidates.csv`, потому что именно он задаёт бинарную метку кандидата:

```text
class = 0 — ложный кандидат
class = 1 — настоящий узелок
```

Ожидаемая структура данных:

```text
data/raw/luna16/
├── subset0/
│   ├── *.mhd
│   └── *.raw
├── subset1/
│   ├── *.mhd
│   └── *.raw
├── ...
├── subset9/
│   ├── *.mhd
│   └── *.raw
├── annotations.csv
└── candidates.csv
```

Пример строки из `candidates.csv`:

```csv
seriesuid,coordX,coordY,coordZ,class
1.3.6.1.4.1.14519.5.2.1.6279.6001.example,-128.70,-175.30,-298.40,1
```

| Поле | Описание |
|---|---|
| `seriesuid` | уникальный идентификатор CT-исследования |
| `coordX` | X-координата кандидата в мировых координатах, мм |
| `coordY` | Y-координата кандидата в мировых координатах, мм |
| `coordZ` | Z-координата кандидата в мировых координатах, мм |
| `class` | бинарная метка: `0` — ложный кандидат, `1` — настоящий узелок |

## 1.4. Формат входа модели

После препроцессинга каждый объект превращается в 3D-тензор:

```text
(C, D, H, W)
```
Пример processed-объекта:

```python
{
    "volume": np.ndarray,  # shape: (1, 32, 48, 48), dtype=float32
    "label": int,          # 0 или 1
    "seriesuid": str,
    "center_xyz_mm": tuple[float, float, float],
}
```

## 1.5. Выход модели

Baseline-модель возвращает два объекта:

```python
linear_output, probabilities = model(input_batch)
```

где:

```text
linear_output: Tensor[B, 2]
probabilities: Tensor[B, 2] = softmax(linear_output)
```

Для унификации training/inference pipeline двухклассовый выход приводится к одному positive-class logit:

```text
positive_logit = logit_class_1 - logit_class_0
```

После sigmoid:

```text
probability = sigmoid(positive_logit)
```

Это эквивалентно вероятности класса `1` из softmax для двухклассовой модели.

Постобработка:

```text
probability >= threshold -> class 1
probability < threshold  -> class 0
```

Порог хранится в конфиге и может быть подобран автоматически на validation split:

```yaml
postprocess:
  threshold: 0.5
  threshold_selection_metric: recall_at_min_precision
  min_precision: 0.5
```

## 1.6. Метрики качества

Для оценки модели используются пять метрик:

| Метрика | Назначение | Ожидаемый ориентир |
|---|---|---|
| `ROC-AUC` | основная ranking-метрика для бинарной классификации | `>= 0.85` |
| `PR-AUC` | качество при дисбалансе классов | `>= 0.45` |
| `Recall` | доля найденных настоящих узелков | `>= 0.80` |
| `Precision` | доля настоящих узелков среди положительных предсказаний | `>= 0.40` |
| `F1-score` | баланс Precision и Recall | `>= 0.55` |

Основной акцент делается на `Recall`, потому что в задаче фильтрации кандидатов важно не отбросить настоящий узелок. Но `Precision` и `PR-AUC` также важны из-за сильного дисбаланса классов и большого количества ложных кандидатов.

Ожидаемые значения являются ориентирами для учебного проекта и должны уточняться после запуска экспериментов на полном LUNA16 split.

## 1.7. Валидация и тестирование

Данные разделяются на три части:

```text
train: 70%
validation: 15%
test: 15%
```

Разделение должно выполняться на уровне CT-исследований / пациентов, а не отдельных 3D-фрагментов, чтобы избежать утечки данных между train и validation/test.

Для воспроизводимости:

- фиксируется `random seed`;
- split-файлы сохраняются в `data/splits`;
- гиперпараметры хранятся в Hydra YAML-конфигах;
- метрики, параметры и git commit логируются в MLflow;
- дополнительно ведётся TensorBoard-логирование.

## 1.8. Бейзлайн из Deep Learning with PyTorch

Пайплайн бейзлайна:

```text
CT volume
-> candidate coordinates
-> 3D crop вокруг кандидата
-> HU clipping
-> normalization
-> LunaModel
-> binary prediction
```

Архитектура:

```text
Input: (B, 1, 32, 48, 48)

BatchNorm3d(1)

LunaBlock(1, 8):
  Conv3d(1, 8, kernel_size=3, padding=1)
  ReLU
  Conv3d(8, 8, kernel_size=3, padding=1)
  ReLU
  MaxPool3d(2, 2)

LunaBlock(8, 16)
LunaBlock(16, 32)
LunaBlock(32, 64)

Flatten
Linear(1152, 2)
Softmax(dim=1)
```
Функция потерь в training pipeline:

```text
BCEWithLogitsLoss
```

При этом двухклассовый выход baseline приводится к одному positive-class logit через:

```text
logit_1 - logit_0
```

Так сохраняется совместимость с threshold optimization, метриками и production-инференсом.

## 1.9. Основная архитектура `resnet3d_se`

Мотивация архитектуры:

- 3D CNN использует объёмную структуру CT-фрагмента, а не отдельные 2D-срезы;
- residual/shortcut connections помогают обучать более глубокие сети и уменьшают риск затухания градиента;
- multi-scale подход учитывает одновременно локальную область вокруг центра кандидата и более широкий контекст;
- Squeeze-and-Excitation блоки усиливают информативные каналы признаков;
- архитектура ориентирована на задачу false-positive reduction: отличить настоящий узелок от ложного кандидата.

Реализованная модель называется:

```text
MultiScaleResNet3DSE
```

Фактическая структура:

```text
Input context patch: (B, 1, D, H, W)

Scale 1 — context branch:
  full preprocessed CT patch
  -> ResNet3DEncoder
  -> context feature vector

Scale 2 — local branch:
  center crop from the same CT patch
  -> resize to original patch shape
  -> ResNet3DEncoder
  -> local feature vector

Feature fusion:
  concat(local_features, context_features)

Classifier head:
  Dropout
  Linear
  ReLU
  Dropout
  Linear(..., 1)
```

Каждый `ResNet3DEncoder` состоит из:

```text
Stem:
  Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)
  BatchNorm3d
  ReLU

Stages:
  ResidualSEBlock3D x blocks_per_stage[0]
  ResidualSEBlock3D x blocks_per_stage[1]
  ResidualSEBlock3D x blocks_per_stage[2]

Each ResidualSEBlock3D:
  Conv3d
  BatchNorm3d
  ReLU
  Conv3d
  BatchNorm3d
  SqueezeExcitation3D
  Residual shortcut
  ReLU

Encoder output:
  AdaptiveAvgPool3d(1)
  Flatten
```

Конфиг основной модели:

```yaml
model:
  name: resnet3d_se
  in_channels: 1
  base_channels: 16
  blocks_per_stage: [2, 2, 2]
  se_reduction: 8
  dropout: 0.25
  local_crop_fraction: 0.5
```

`local_crop_fraction: 0.5` означает, что локальная ветка получает центральную область размером примерно 50% по каждой пространственной оси. Полный patch остаётся входом контекстной ветки.

## 1.9.1. Loss для основной модели

Для борьбы с дисбалансом классов используется модифицированный loss:

```text
loss = focal_weight * FocalLossWithLogits + bce_weight * BCEWithLogitsLoss
```

Конфиг:

```yaml
loss:
  name: focal
  alpha: 0.75
  gamma: 2.0
  focal_weight: 1.0
  bce_weight: 0.25
  pos_weight: null
```

`FocalLoss` фокусирует обучение на сложных примерах, а BCE-компонента стабилизирует оптимизацию. Для baseline можно использовать обычный `BCEWithLogitsLoss`.

## 1.9.2. Методы борьбы с дисбалансом

В проекте реализованы три механизма:

1. **Weighted sampling** в train dataloader: положительные и отрицательные кандидаты получают веса, обратные частотам классов.
2. **Focal Loss** для усиления вклада сложных примеров.
3. **Hard negative mining** через сохранённый список сложных отрицательных примеров: выбранные ложные кандидаты могут получать повышенный sampling weight в следующем запуске обучения.

Команда для выбора hard negatives из сохранённых массивов меток и вероятностей (доступно после первого обучения):

```bash
lungscan3d select-hard-negatives \
  --labels=artifacts/predictions/train_labels.npy \
  --probabilities=artifacts/predictions/train_probabilities.npy \
  --output=artifacts/hard_negatives/train_hard_negatives.npy \
  --top-fraction=0.25 \
  --min-probability=0.5
```

После этого можно включить hard negative sampling:

```bash
lungscan3d train \
  data=luna16 \
  model=resnet3d_se \
  loss=focal \
  data.hard_negative_mining.enabled=true
```

## 1.9.3. Аугментации и обучение

Для повышения обобщающей способности в train dataset применяются 3D-аугментации:

- случайные отражения по пространственным осям;
- случайные повороты на 90 градусов в 3D-плоскостях;
- случайные смещения области интереса;
- добавление гауссовского шума.

Конфиг:

```yaml
preprocessing:
  augment:
    enabled: true
    random_flip: true
    random_rotate90: true
    gaussian_noise_std: 0.02
    random_shift_voxels: 2
```

Training pipeline использует:

- PyTorch Lightning;
- mixed precision через `trainer.precision`, например `16-mixed`;
- `ReduceLROnPlateau` learning-rate scheduler;
- gradient clipping через `trainer.gradient_clip_val`;
- MLflow и TensorBoard loggers;
- сохранение лучшего checkpoint по `val/loss`.

## 1.10. Препроцессинг

Препроцессинг включает:

1. чтение `.mhd/.raw` через `SimpleITK`;
2. извлечение volume, spacing, origin;
3. перевод мировых координат кандидата в voxel coordinates;
4. clipping HU-значений;
5. нормализацию;
6. crop 3D-фрагмента вокруг кандидата;
7. padding, если crop выходит за границы volume;
8. сохранение processed samples;
9. создание train/val/test split.

Параметры задаются в Hydra:

```yaml
data:
  patch_size: [32, 48, 48]

preprocessing:
  clip_hu_min: -1000.0
  clip_hu_max: 400.0
```

## 1.11. Постобработка и подбор порога

Постобработка модели:

1. получить positive-class logit;
2. применить sigmoid;
3. загрузить оптимизированный threshold, если он есть;
4. сравнить вероятность с threshold;
5. вернуть JSON-ответ.

Подбор порога выполняется отдельной командой:

```bash
lungscan3d optimize-threshold data=luna16 checkpoint=artifacts/checkpoints/best.ckpt split=val
```

Стратегия по умолчанию:

```text
recall_at_min_precision
```

То есть выбирается порог, который максимизирует `Recall`, но сохраняет `Precision` не ниже заданного минимума.

Пример результата:

```json
{
  "threshold": 0.271,
  "strategy": "recall_at_min_precision",
  "precision": 0.52,
  "recall": 0.86,
  "f1": 0.65,
  "roc_auc": 0.88,
  "pr_auc": 0.61
}
```

Артефакт сохраняется в:

```text
artifacts/postprocessing/threshold.json
```

## 1.12. Внедрение

Модель оформляется как Python-пакет и поддерживает:

- обучение;
- валидацию;
- инференс;
- экспорт в ONNX;
- подготовку TensorRT engine;
- запуск через Triton Inference Server.

Форматы модели:

| Формат | Назначение |
|---|---|
| `.ckpt` | PyTorch Lightning checkpoint для дообучения и воспроизводимости |
| `.onnx` | переносимый production inference format |
| `.engine` / `.plan` | TensorRT engine для ускоренного GPU-инференса |
| Triton model repository | структура для model serving |

---

# 2. Техническая часть

## 2.1. Стек технологий

### Язык и пакетирование

- Python `>=3.11,<3.13`;
- `uv` для управления зависимостями и lock-файлом;
- `hatchling` как build backend;
- `pyproject.toml` как единый конфигурационный файл проекта.

### Машинное обучение

- PyTorch;
- PyTorch Lightning;
- TorchMetrics;
- NumPy;
- Pandas;
- Scikit-learn;
- SimpleITK.

### Конфигурация и воспроизводимость

- Hydra;
- OmegaConf;
- DVC;

### Логирование

- стандартный Python `logging` для пользовательских CLI-логов;
- MLflow для experiment tracking;
- TensorBoard для локального просмотра обучения;
- matplotlib-графики в `plots/`.

### Качество кода

- pre-commit;
- ruff;
- pytest;
- pytest-cov;
- type annotations;
- Google-style docstrings.

### Production inference

- ONNX;
- ONNX Runtime;
- TensorRT;
- Triton Inference Server.

---

# 3. Структура репозитория

```text
lung-scan-3d/
├── .dvc/
│   └── config
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── config.yaml
│   ├── data/
│   │   ├── luna16.yaml
│   │   └── synthetic.yaml
│   ├── preprocessing/
│   │   └── default.yaml
│   ├── model/
│   │   ├── dlwpt_baseline.yaml
│   │   └── resnet3d_se.yaml
│   ├── trainer/
│   │   └── default.yaml
│   ├── loss/
│   │   ├── bce.yaml
│   │   └── focal.yaml
│   ├── logging/
│   │   └── mlflow_tensorboard.yaml
│   ├── postprocess/
│   │   └── threshold.yaml
│   └── infer/
│       └── onnx.yaml
├── data/
│   └── .gitkeep
├── lungscan3d/
│   ├── commands.py
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── serving/
│   └── utils/
├── plots/
│   └── .gitkeep
├── scripts/
├── tests/
├── triton_model_repository/
├── .gitignore
├── .pre-commit-config.yaml
├── dvc.yaml
├── infer.py
├── pyproject.toml
├── README.md
└── uv.lock
```

---

# 4. Установка, сборка и установка пакета

## 4.1. Требования

Необходимо установить:

- Python 3.11;
- `uv`;
- `git`;
- DVC;
- опционально CUDA/TensorRT для production GPU inference.

Установка `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Проверка:

```bash
uv --version
```

## 4.2. Клонирование репозитория

```bash
git clone https://github.com/DimaAtamanov/lungscan3d.git
cd lungscan3d
```

## 4.3. Установка окружения разработки

```bash
uv sync --extra dev
source .venv/bin/activate
```

После активации окружения все команды запускаются как команды установленного Python-пакета, без `uv run`:

```bash
lungscan3d show-config
lungscan3d train data=synthetic trainer.max_epochs=3
```

## 4.4. Сборка wheel/sdist

Так как проект использует `uv`, предпочтительная команда сборки:

```bash
uv build
```

`uv build` читает стандартный блок `[build-system]` из `pyproject.toml` и использует указанный backend `hatchling`. Альтернативная PEP 517-команда тоже корректна, если установлен пакет `build`:

```bash
python -m build
```

После сборки появятся артефакты:

```text
dist/lungscan3d-0.1.0-py3-none-any.whl
dist/lungscan3d-0.1.0.tar.gz
```

## 4.5. Установка собранного пакета

В чистом окружении:

```bash
python -m venv .venv-prod
source .venv-prod/bin/activate
uv pip install dist/lungscan3d-0.1.0-py3-none-any.whl
```

Можно использовать и обычный `pip install`, но в проекте предпочтительно использовать `uv pip install`, чтобы оставаться в одном tooling-стеке.

Проверка установленной CLI-команды:

```bash
lungscan3d show-config
```

## 4.6. Установка pre-commit

```bash
pre-commit install
pre-commit run -a
```

---

# 5. Конфигурация проекта

Проект использует Hydra. Главная точка входа:

```text
configs/config.yaml
```

Переопределение параметров выполняется через CLI:

```bash
lungscan3d train trainer.max_epochs=5 data=synthetic
```

Выбор LUNA16-режима:

```bash
lungscan3d train data=luna16
```

Быстрый smoke-test:

```bash
lungscan3d train data=synthetic trainer.max_epochs=3
```

---

# 6. Управление данными через DVC

## 6.1. DVC-команды через пакет

Для удобства основные DVC-операции обёрнуты в CLI пакета:

```bash
lungscan3d dvc-pull
lungscan3d dvc-pull --target=data/processed/luna16 --remote=data_storage
lungscan3d dvc-push --target=artifacts/onnx/lungscan3d.onnx --remote=model_storage
```

Если нужно выполнить расширенную DVC-операцию, можно использовать обычный `dvc` напрямую, но базовый пользовательский сценарий покрыт командами пакета.

## 6.2. Remote storage

```text
data_storage  — для данных
model_storage — для моделей и production-артефактов
```
---

# 7. Получение LUNA16

## 7.1. Рекомендуемый вариант

Полный LUNA16 занимает десятки гигабайт. Поэтому безопасный и воспроизводимый сценарий такой:

1. для CI и проверки использовать `data=synthetic`;
2. для реального обучения скачать LUNA16 вручную или через CLI-утилиту;
3. после подготовки данных добавить processed-артефакты в DVC.

## 7.2. Автоматическая загрузка через пакет

В пакет добавлена команда:

```bash
lungscan3d download-luna16
```

По умолчанию она скачивает все 10 subset-архивов и metadata. Это большой объём данных, поэтому для разработки лучше ограничить число частей.

Скачать только metadata и `subset0`:

```bash
lungscan3d download-luna16 --subsets=0
```

Скачать первые две части:

```bash
lungscan3d download-luna16 --max-subsets=2
```

Скачать конкретные части:

```bash
lungscan3d download-luna16 --subsets=0,3,7
```

Скачать архивы, распаковать и удалить zip-файлы после распаковки:

```bash
lungscan3d download-luna16 --subsets=0 --extract=True --keep-archives=False
```

Назначение параметров:

| Параметр | Значение |
|---|---|
| `--raw-dir` | куда положить данные, по умолчанию `data/raw/luna16` |
| `--subsets` | список subset ids через запятую, например `0,1,2` |
| `--max-subsets` | скачать первые N subset-архивов |
| `--include-metadata` | скачать `annotations.csv` и `candidates.csv` |
| `--extract` | распаковать zip-архивы |
| `--keep-archives` | оставить zip после распаковки |
| `--overwrite` | перекачать существующие файлы |

Команда скачивает архивы с Zenodo и распаковывает их в структуру:

```text
data/raw/luna16/
├── subset0/
├── subset1/
├── ...
├── annotations.csv
└── candidates.csv
```

## 7.3. Ручная загрузка

Если автоматическая загрузка нежелательна, можно скачать данные вручную:

1. открыть страницу LUNA16 на Grand Challenge или Zenodo;
2. скачать `annotations.csv`, `candidates.csv` и нужные `subset*.zip`;
3. распаковать архивы в `data/raw/luna16/`;
4. проверить структуру:

```bash
find data/raw/luna16 -maxdepth 2 -type f | head
```

## 7.4. Интеграция загрузки в пайплайн

По умолчанию `data=luna16` сначала пытается сделать DVC pull. Интернет-загрузка выключена, чтобы случайно не скачать 60+ ГБ.

Чтобы разрешить интернет-загрузку из train/preprocess pipeline:

```bash
lungscan3d preprocess \
  data=luna16 \
  data.allow_internet_download=true \
  data.download_subsets=0
```

---

# 8. Обучение модели

## 8.1. Быстрая проверка на synthetic dataset

```bash
lungscan3d train data=synthetic trainer.max_epochs=3
```

Этот режим нужен для автоматической проверки:

- пакет импортируется;
- DataModule работает;
- модель делает forward pass;
- loss считается;
- training loop запускается;
- loss должен снижаться на простом synthetic signal.

## 8.2. Препроцессинг LUNA16

```bash
lungscan3d preprocess data=luna16
```

Если нужно скачать только `subset0` перед препроцессингом:

```bash
lungscan3d download-luna16 --subsets=0
lungscan3d preprocess data=luna16
```

## 8.3. Обучение baseline на LUNA16

```bash
lungscan3d train data=luna16 model=dlwpt_baseline loss=bce trainer.max_epochs=20
```

## 8.4. Обучение улучшенной модели

```bash
lungscan3d train data=luna16 model=resnet3d_se loss=focal trainer.max_epochs=50
```

## 8.5. Что пользователь видит в логах

Через стандартный `logging` выводятся этапы:

- загрузка/проверка данных;
- DVC pull/push;
- скачивание LUNA16;
- распаковка архивов;
- начало препроцессинга;
- количество найденных CT-файлов;
- количество обработанных кандидатов;
- создание train/val/test split;
- запуск обучения;
- путь к лучшему checkpoint;
- сохранение графиков;
- экспорт ONNX/TensorRT;
- результат инференса.

Пример:

```text
2026-05-22 12:00:01 | INFO | lungscan3d.training.train | Starting training: project=lungscan3d, data=synthetic, model=dlwpt_baseline
2026-05-22 12:00:02 | INFO | lungscan3d.data.download | Synthetic dataset saved: volumes=data/processed/synthetic/volumes.npy
2026-05-22 12:00:03 | INFO | lungscan3d.data.datamodule | Dataset split sizes: train=134, val=28, test=30
```

---

# 9. Логирование экспериментов

## 9.1. MLflow

Локальный запуск MLflow:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Обучение:

```bash
lungscan3d train data=synthetic trainer.max_epochs=3
```

В MLflow логируются:

- train loss;
- validation loss;
- ROC-AUC;
- PR-AUC;
- Recall;
- Precision;
- F1-score;
- hyperparameters;
- git commit id;
- checkpoints;
- plots.

## 9.2. TensorBoard

```bash
tensorboard --logdir lightning_logs
```

---

# 10. Подбор порога

После обучения:

```bash
lungscan3d optimize-threshold \
  data=luna16 \
  model=dlwpt_baseline \
  checkpoint=artifacts/checkpoints/best.ckpt \
  split=val
```

Порог сохраняется в:

```text
artifacts/postprocessing/threshold.json
```

Инференс автоматически использует этот файл, если включено:

```yaml
postprocess:
  use_threshold_artifact: true
```

---

# 11. Инференс

## 11.1. Инференс по processed patch

Формат входа:

```text
.npy файл
shape: (1, 32, 48, 48)
dtype: float32
```

Команда:

```bash
lungscan3d infer data/examples/sample_patch.npy
```

Пример ответа:

```json
{
  "probability": 0.873,
  "threshold": 0.5,
  "label": 1,
  "class_name": "nodule"
}
```

---

# 12. Экспорт модели в ONNX

```bash
lungscan3d export-onnx \
  checkpoint=artifacts/checkpoints/best.ckpt \
  output=artifacts/onnx/lungscan3d.onnx
```

После экспорта выполняются проверки:

- `onnx.checker.check_model`;
- smoke-test через `onnxruntime`.

Добавление ONNX-артефакта в DVC:

```bash
lungscan3d dvc-add artifacts/onnx/lungscan3d.onnx
lungscan3d dvc-push --target=artifacts/onnx/lungscan3d.onnx --remote=model_storage
```

---

# 13. Экспорт в TensorRT

TensorRT export требует:

- NVIDIA GPU;
- CUDA;
- установленный TensorRT;
- доступный `trtexec`.

```bash
lungscan3d export-tensorrt output=artifacts/tensorrt/lungscan3d.engine
```

Или shell-обёртка:

```bash
bash scripts/export_tensorrt.sh output=artifacts/tensorrt/lungscan3d.engine
```

---

# 14. Triton Inference Server

Структура model repository:

```text
triton_model_repository/
└── lungscan3d/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

Перед запуском:

```bash
cp artifacts/onnx/lungscan3d.onnx triton_model_repository/lungscan3d/1/model.onnx
```

Запуск:

```bash
bash scripts/run_triton.sh
```

Проверка клиента:

```bash
lungscan3d triton-client data/examples/sample_patch.npy --url=localhost:8000
```

---

# 15. Тестирование

```bash
pytest
```

С coverage:

```bash
pytest --cov=lungscan3d
```

Тестами покрываются:

- dataset loading;
- synthetic data generation;
- preprocessing utilities;
- coordinate conversion;
- model forward pass;
- loss functions;
- postprocessing;
- threshold optimization;
- plot generation;
- inference utilities.

---

# 16. Качество кода

Проверка:

```bash
ruff check .
```

Форматирование:

```bash
ruff format .
```

Полная проверка:

```bash
pre-commit run -a
```
---

# 17. Быстрый сценарий проверки проекта

```bash
git clone https://github.com/DimaAtamanov/lungscan3d.git
cd lungscan3d

uv sync --extra dev
source .venv/bin/activate

pre-commit install
pre-commit run -a
pytest

lungscan3d train data=synthetic trainer.max_epochs=3
```

Ожидаемый результат:

- зависимости устанавливаются;
- пакет доступен как CLI-команда `lungscan3d`;
- pre-commit проходит;
- тесты проходят;
- обучение запускается;
- loss снижается;
- графики сохраняются в `plots/`;
- метрики логируются;
- ONNX экспортируется;
- инференс возвращает JSON.

---

# 18. Основные команды

```bash
# установка окружения разработки
uv sync --extra dev
source .venv/bin/activate

# сборка пакета
python -m build

# хуки и тесты
pre-commit install
pre-commit run -a
pytest

# показать конфиг
lungscan3d show-config

# скачать LUNA16 subset0 и metadata
lungscan3d download-luna16 --subsets=0

# DVC
lungscan3d dvc-pull
lungscan3d dvc-add artifacts/onnx/lungscan3d.onnx
lungscan3d dvc-push --remote=model_storage

# smoke train
lungscan3d train data=synthetic trainer.max_epochs=3

# preprocess LUNA16
lungscan3d preprocess data=luna16

# train LUNA16
lungscan3d train data=luna16 model=dlwpt_baseline loss=bce

# threshold optimization
lungscan3d optimize-threshold data=luna16 checkpoint=artifacts/checkpoints/best.ckpt split=val

# infer
lungscan3d infer data/examples/sample_patch.npy

# export ONNX
lungscan3d export-onnx

# export TensorRT
lungscan3d export-tensorrt

# run Triton
bash scripts/run_triton.sh

# Triton client
lungscan3d triton-client data/examples/sample_patch.npy
```
