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

| Поле        | Описание                                                      |
| ----------- | ------------------------------------------------------------- |
| `seriesuid` | уникальный идентификатор CT-исследования                      |
| `coordX`    | X-координата кандидата в мировых координатах, мм              |
| `coordY`    | Y-координата кандидата в мировых координатах, мм              |
| `coordZ`    | Z-координата кандидата в мировых координатах, мм              |
| `class`     | бинарная метка: `0` — ложный кандидат, `1` — настоящий узелок |

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

| Метрика     | Назначение                                              | Ожидаемый ориентир |
| ----------- | ------------------------------------------------------- | ------------------ |
| `ROC-AUC`   | основная ranking-метрика для бинарной классификации     | `>= 0.85`          |
| `PR-AUC`    | качество при дисбалансе классов                         | `>= 0.45`          |
| `Recall`    | доля найденных настоящих узелков                        | `>= 0.80`          |
| `Precision` | доля настоящих узелков среди положительных предсказаний | `>= 0.40`          |
| `F1-score`  | баланс Precision и Recall                               | `>= 0.55`          |

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

## 1.9.3. Аугментации и обучение

Для повышения обобщающей способности в train dataset применяются 3D-аугментации:

- случайные отражения по пространственным осям;
- случайные повороты на 90 градусов в квадратной плоскости вокселя;
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
- опциональные loggers: `logging.mode=none`, `logging.mode=mlflow`, `logging.mode=tensorboard`, `logging.mode=all`;
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
lungscan3d optimize-threshold data=luna16 infer.checkpoint_path=artifacts/checkpoints/best.ckpt postprocess.split=val
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

| Формат                  | Назначение                                                      |
| ----------------------- | --------------------------------------------------------------- |
| `.ckpt`                 | PyTorch Lightning checkpoint для дообучения и воспроизводимости |
| `.onnx`                 | переносимый production inference format                         |
| `.engine` / `.plan`     | TensorRT engine для ускоренного GPU-инференса                   |
| Triton model repository | структура для model serving                                     |

---

# 2. Техническая часть

Эта часть README — практический manual по пакету. Все команды запускаются через установленный entrypoint `lungscan3d`. После установки пакета команда доступна как обычная CLI-команда окружения.

## 2.1. Где что лежит

```text
lungscan3d/
├── configs/                         # Hydra-конфиги всех режимов запуска
│   ├── config.yaml                   # корневой конфиг и defaults
│   ├── data/                         # synthetic / LUNA16, batch, split, download
│   ├── infer/                        # ONNX/input/output/checkpoint параметры
│   ├── logging/                      # default.yaml: mode/ports/URI для логгеров
│   ├── loss/                         # BCE и focal loss
│   ├── model/                        # 3D baseline и ResNet3D-SE
│   ├── postprocess/                  # threshold и подбор порога
│   ├── preprocessing/                # HU clipping, patch size, chunking, progress
│   ├── tensorrt/                     # trtexec, dynamic shapes, precision, engine path
│   ├── trainer/                      # PyTorch Lightning trainer
│   └── triton/                       # Triton repository/client/docker параметры
├── lungscan3d/
│   ├── commands.py                   # единый Hydra-first CLI entrypoint
│   ├── data/                         # download, preprocessing, dataset, patient split
│   ├── inference/                    # infer, ONNX export, TensorRT export, thresholds
│   ├── models/                       # архитектуры моделей
│   ├── serving/                      # Triton HTTP client
│   ├── training/                     # Lightning module, train loop, callbacks, plots
│   └── utils/                        # DVC, git, logging, paths
├── scripts/                          # thin wrappers around lungscan3d / сервисы
├── tests/                            # unit/smoke проверки
├── triton_model_repository/          # Triton model repository
├── dvc.yaml                          # DVC stage для препроцессинга LUNA16
├── pyproject.toml                    # зависимости, entrypoint, dev extras
└── README.md                         # этот manual
```

## 2.2. Сборка, настройка окружения, установка пакета

CLI entrypoint у пакета всегда называется `lungscan3d`, но разворачивать окружение, фиксировать зависимости и собирать пакет рекомендуется через `uv`. Это даёт воспроизводимое окружение по `uv.lock` и одинаковый workflow на локальной машине, CI и сервере.

Подготовка к работе:

Установка `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Клонирование репозитория

```bash
git clone https://github.com/DimaAtamanov/lungscan3d.git
cd lungscan3d
```

Базовая установка для разработки:

```bash
uv sync --extra dev --extra triton --extra tensorrt
uv run lungscan3d show-config
```

После `uv sync` можно либо запускать команды через `uv run lungscan3d ...`, либо активировать виртуальное окружение и пользоваться чистым entrypoint `lungscan3d`:

```bash
source .venv/bin/activate
lungscan3d show-config
```

Сборка wheel/sdist также выполняется через `uv`:

```bash
uv build
```

Для TensorRT-конвертации есть две части:

1. Python-зависимости проекта:

```bash
uv sync --extra dev --extra triton --extra tensorrt
```

2. Системный NVIDIA TensorRT CLI `trtexec`. Python-пакет не всегда кладёт `trtexec` в `PATH`, поэтому проверка обязательна:

```bash
uv run python -c "import tensorrt; print(tensorrt.__version__)"
nvidia-smi
trtexec --version
```

Если `trtexec` отсутствует, установите TensorRT на хост или выполняйте экспорт в NVIDIA TensorRT container. Для Triton нужен Docker с NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Triton client-зависимости ставятся тем же способом:

```bash
uv sync --extra dev --extra triton
```

## 2.3. Общая модель CLI

```bash
lungscan3d <command> key=value group=value group.key=value
```

Примеры:

```bash
lungscan3d show-config data=luna16 model=resnet3d_se logging.mode=none
lungscan3d train data=luna16 trainer.max_epochs=30 data.batch_size=64 logging.mode=tensorboard
lungscan3d export-onnx data=luna16 infer.checkpoint_path=artifacts/checkpoints/best.ckpt infer.onnx_path=artifacts/onnx/lungscan3d.onnx
```

То же самое можно выполнить без активации `.venv`, если явно обернуть entrypoint через `uv run`:

```bash
uv run lungscan3d train data=luna16 trainer.max_epochs=30 logging.mode=none
```

Список команд:

```text
show-config         вывести итоговый Hydra config
download-data       подготовить данные согласно data=...
download-luna16     скачать metadata/subset архивы LUNA16
preprocess          сделать preprocessing
train               обучить модель
infer               локальный inference по .npy patch
optimize-threshold  подобрать threshold на val/test
export-onnx         экспортировать checkpoint в ONNX
export-tensorrt     собрать TensorRT plan через trtexec
triton-client       вызвать Triton HTTP endpoint
dvc-add             dvc add для target из dvc.target
dvc-pull            dvc pull для target/remote из dvc.*
dvc-push            dvc push для target/remote из dvc.*
self-test           smoke-проверка synthetic → train → ONNX → TensorRT dry-run → pytest
```

## 2.4. Быстрый smoke test без LUNA16

Этот сценарий ничего не скачивает из LUNA16. Он проверяет, что пакет установлен, CLI работает, synthetic data создаётся, модель обучается, ONNX экспортируется, TensorRT-команда собирается, Triton repository имеет ожидаемую структуру, а тесты проходят.

```bash
lungscan3d self-test
```

Более быстрый режим без запуска pytest внутри self-test:

```bash
lungscan3d self-test self_test.run_pytest=false
```

Отдельно можно прогнать обычный набор тестов:

```bash
pytest
```

## 2.5. Скачивание LUNA16

Полный LUNA16 большой, поэтому для первого запуска обычно берут один subset:

```bash
lungscan3d download-luna16 data=luna16 data.download_subsets=[0]
```

Несколько subset:

```bash
lungscan3d download-luna16 data=luna16 data.download_subsets=[0,1,2]
```

Первые N subset:

```bash
lungscan3d download-luna16 data=luna16 data.download_max_subsets=3
```

Только metadata без архивов subset не является полноценным датасетом для обучения, но удобно для проверки доступа:

```bash
lungscan3d download-luna16 data=luna16 data.download_metadata=true data.download_max_subsets=0
```

Ожидаемая структура после скачивания:

```text
data/raw/luna16/
├── annotations.csv
├── candidates.csv
├── subset0.zip
├── subset0/
│   ├── *.mhd
│   └── *.raw
└── ...
```

## 2.6. DVC workflow

Если данные или processed artifacts уже лежат в DVC remote:

```bash
lungscan3d dvc-pull dvc.target=data/processed/luna16
lungscan3d dvc-pull dvc.target=data/processed/luna16 dvc.remote=data_storage
```

Добавить processed dataset в DVC:

```bash
lungscan3d dvc-add dvc.target=data/processed/luna16
lungscan3d dvc-push dvc.target=data/processed/luna16 dvc.remote=data_storage
```

DVC stage из `dvc.yaml` запускает preprocessing так же через entrypoint пакета:

```bash
dvc repro preprocess_luna16
```

## 2.7. Препроцессинг LUNA16

Препроцессинг читает `candidates.csv`, ищет соответствующие `*.mhd`, нормализует HU, вырезает patch вокруг кандидата и сохраняет chunked dataset:

```bash
lungscan3d preprocess data=luna16
```

Результат:

```text
data/processed/luna16/
├── labels.npy
├── manifest.csv
└── chunks/
    ├── volumes_000000.npy
    ├── labels_000000.npy
    ├── volumes_000001.npy
    └── labels_000001.npy
```

`manifest.csv` содержит `seriesuid`, `chunk_index`, `local_index`, пути к chunk-файлам и label. Это важно для lazy loading и patient-level split.

Во время препроцессинга есть progress bar по строкам `candidates.csv`. Chunk size по умолчанию уменьшен до `256`, чтобы держать RAM в районе 12 GB даже на обычной рабочей станции.

Примеры настройки:

```bash
lungscan3d preprocess data=luna16 preprocessing.chunk_size=128
lungscan3d preprocess data=luna16 data.patch_size=[32,48,48] preprocessing.max_cached_ct_volumes=1
lungscan3d preprocess data=luna16 preprocessing.progress=false
```

## 2.8. Patient-level split и lazy loading

Для LUNA16 по умолчанию включено:

```yaml
data.split_by_patient: true
data.group_column: seriesuid
```

Это означает, что split делается не по отдельным patch, а по `seriesuid`. Один CT/patient не может оказаться одновременно в train и validation/test. Это сохраняется и при lazy loading, потому что lazy dataset получает уже готовые global indices, построенные на основе `manifest.csv`.

При `data.save_splits=true` split-файлы сохраняются в:

```text
data/splits/luna16/
├── train_idx.npy
├── val_idx.npy
├── test_idx.npy
└── groups.json
```

## 2.9. Рекомендованные ресурсы и дефолты LUNA16

Дефолты подобраны под ориентир: около 12 GB RAM и 16 GB GPU VRAM.

| Сценарий            |   patch size | chunk size | batch size |      RAM |  GPU VRAM | Комментарий                                        |
| ------------------- | -----------: | ---------: | ---------: | -------: | --------: | -------------------------------------------------- |
| smoke/debug         | `[16,16,16]` |         64 |       8–32 |   2–4 GB |    2–4 GB | Быстрые проверки, не для качества                  |
| workstation default | `[32,48,48]` |        256 |         64 | ~8–12 GB | ~10–16 GB | Дефолт проекта для LUNA16                          |
| осторожный режим    | `[32,48,48]` |        128 |         32 | ~6–10 GB |  ~8–12 GB | Если есть OOM на CPU/GPU                           |
| крупнее patch       | `[48,64,64]` |     64–128 |       8–16 | 12–24 GB | 16–24+ GB | Только если нужна большая область вокруг кандидата |
| ResNet3D-SE         | `[32,48,48]` |    128–256 |      16–32 | ~8–12 GB | 12–16+ GB | Модель тяжелее baseline                            |

Оценка памяти на один float32 patch:

```text
1 * D * H * W * 4 bytes
[32,48,48] ≈ 0.28 MB на sample до активаций модели
```

На GPU основную память занимают не входы, а активации и градиенты. Если ловите CUDA OOM, уменьшайте параметры в таком порядке:

1. `data.batch_size`
2. `model.base_channels` или `model.conv_channels`
3. `data.patch_size`
4. `trainer.precision=16-mixed` при совместимой GPU

## 2.10. Обучение

По умолчанию логирование экспериментов отключено:

```bash
lungscan3d train data=luna16 logging.mode=none
```

MLflow:

```bash
scripts/run_mlflow.sh
lungscan3d train data=luna16 logging.mode=mlflow
```

MLflow UI по умолчанию:

```text
http://127.0.0.1:8080
```

TensorBoard:

```bash
scripts/run_tensorboard.sh
lungscan3d train data=luna16 logging.mode=tensorboard
```

TensorBoard UI по умолчанию:

```text
http://127.0.0.1:6006
```

Оба логгера сразу:

```bash
scripts/run_mlflow.sh
scripts/run_tensorboard.sh
lungscan3d train data=luna16 logging.mode=all
```

Конфигурация логгеров хранится в одном файле `configs/logging/default.yaml`.

```bash
lungscan3d train data=luna16 \
  logging.mode=mlflow \
  logging.mlflow_tracking_uri=http://127.0.0.1:8080 \
  logging.experiment_name=lungscan3d

lungscan3d train data=luna16 \
  logging.mode=tensorboard \
  logging.tensorboard_save_dir=lightning_logs
```

Baseline:

```bash
lungscan3d train data=luna16 model=dlwpt_baseline trainer.max_epochs=30
```

ResNet3D-SE:

```bash
lungscan3d train data=luna16 model=resnet3d_se trainer.max_epochs=50 data.batch_size=32
```

Checkpoint сохраняется только один — лучший по `val/loss`:

```text
artifacts/checkpoints/best.ckpt
```

## 2.11. Подбор threshold

```bash
lungscan3d optimize-threshold data=luna16 infer.checkpoint_path=artifacts/checkpoints/best.ckpt postprocess.split=val
```

Результат сохраняется в:

```text
artifacts/postprocessing/threshold.json
```

Управление стратегией:

```bash
lungscan3d optimize-threshold data=luna16 postprocess.threshold_selection_metric=recall_at_min_precision postprocess.min_precision=0.5
```

## 2.12. Локальный inference

На вход нужен `.npy` patch формы `(1,D,H,W)` или batch `(B,1,D,H,W)`:

```bash
lungscan3d infer data=luna16 infer.input_path=data/examples/sample_patch.npy infer.checkpoint_path=artifacts/checkpoints/best.ckpt
```

## 2.13. Экспорт в ONNX

```bash
lungscan3d export-onnx data=luna16 infer.checkpoint_path=artifacts/checkpoints/best.ckpt infer.onnx_path=artifacts/onnx/lungscan3d.onnx
```

Экспорт валидируется через `onnx.checker` и `onnxruntime`.

## 2.14. Экспорт в TensorRT

Сначала убедитесь, что ONNX есть:

```bash
lungscan3d export-onnx data=luna16
```

Затем TensorRT:

```bash
lungscan3d export-tensorrt data=luna16 tensorrt.engine_path=artifacts/tensorrt/lungscan3d.plan
```

Dynamic shapes задаются через Hydra:

```bash
lungscan3d export-tensorrt data=luna16 tensorrt.min_batch_size=1 tensorrt.opt_batch_size=16 tensorrt.max_batch_size=64 tensorrt.precision=fp16
```

Dry-run без запуска `trtexec`, удобно для CI:

```bash
lungscan3d export-tensorrt data=luna16 tensorrt.dry_run=true
```

После успешной сборки положите plan в Triton repository:

```bash
mkdir -p triton_model_repository/lungscan3d/1
cp artifacts/tensorrt/lungscan3d.plan triton_model_repository/lungscan3d/1/model.plan
```

## 2.15. Triton Server

В репозитории уже есть конфиг TensorRT backend:

```text
triton_model_repository/lungscan3d/config.pbtxt
```

Запуск:

```bash
scripts/run_triton.sh
```

Адреса по умолчанию:

```text
HTTP:    localhost:8000
gRPC:    localhost:8001
metrics: localhost:8002
```

Проверка клиента:

```bash
lungscan3d triton-client triton.input_path=data/examples/sample_patch.npy triton.client_url=localhost:8000
```

## 2.16. Полный workflow от данных до Triton

Минимальный LUNA16 workflow:

```bash
# 1. Установка
uv sync --extra dev --extra triton
source .venv/bin/activate

# 2. Скачивание одного subset для первого запуска
lungscan3d download-luna16 data=luna16 data.download_subsets=[0]

# 3. Препроцессинг
lungscan3d preprocess data=luna16

# 4. При необходимости сохранить processed data в DVC
lungscan3d dvc-add dvc.target=data/processed/luna16
lungscan3d dvc-push dvc.target=data/processed/luna16

# 5. Обучение без внешнего логирования
lungscan3d train data=luna16 logging.mode=none trainer.max_epochs=30

# 6. Подбор threshold
lungscan3d optimize-threshold data=luna16 postprocess.split=val

# 7. ONNX
lungscan3d export-onnx data=luna16

# 8. TensorRT
lungscan3d export-tensorrt data=luna16

# 9. Triton model repository
mkdir -p triton_model_repository/lungscan3d/1
cp artifacts/tensorrt/lungscan3d.plan triton_model_repository/lungscan3d/1/model.plan

# 10. Triton Server
scripts/run_triton.sh

# 11. Triton client
lungscan3d triton-client triton.input_path=data/examples/sample_patch.npy
```

## 2.17. Таблица управляющих параметров

Ниже перечислены параметры, которыми пользователь управляет через Hydra overrides. Формат:

```bash
lungscan3d <command> parameter=value
```

| Параметр                                    |                                    Дефолт | Где                   | Комментарий / допустимые значения                      |
| ------------------------------------------- | ----------------------------------------: | --------------------- | ------------------------------------------------------ |
| `seed`                                      |                                      `42` | `configs/config.yaml` | Глобальная воспроизводимость                           |
| `project_name`                              |                              `lungscan3d` | `configs/config.yaml` | Имя проекта для логгеров                               |
| `paths.data_dir`                            |                                    `data` | `configs/config.yaml` | Корень данных                                          |
| `paths.raw_dir`                             |                                `data/raw` | `configs/config.yaml` | Корень raw данных                                      |
| `paths.processed_dir`                       |                          `data/processed` | `configs/config.yaml` | Корень processed данных                                |
| `paths.splits_dir`                          |                             `data/splits` | `configs/config.yaml` | Куда сохраняются split indices                         |
| `paths.examples_dir`                        |                           `data/examples` | `configs/config.yaml` | Примеры `.npy` для inference                           |
| `paths.artifacts_dir`                       |                               `artifacts` | `configs/config.yaml` | Корень артефактов                                      |
| `paths.plots_dir`                           |                                   `plots` | `configs/config.yaml` | PNG-графики обучения                                   |
| `paths.checkpoints_dir`                     |                   `artifacts/checkpoints` | `configs/config.yaml` | Директория checkpoint                                  |
| `dvc.target`                                |                                    `null` | `configs/config.yaml` | DVC target для `dvc-add/pull/push`                     |
| `dvc.remote`                                |                                    `null` | `configs/config.yaml` | DVC remote, например `data_storage`                    |
| `self_test.run_pytest`                      |                                    `true` | `configs/config.yaml` | Запускать ли pytest внутри `self-test`                 |
| `self_test.pytest_args`                     |                                  `['-q']` | `configs/config.yaml` | Аргументы pytest                                       |
| `data`                                      |                               `synthetic` | Hydra defaults        | Выбор data config: `synthetic`, `luna16`               |
| `data.name`                                 |                    `synthetic` / `luna16` | `configs/data/*.yaml` | Имя датасета                                           |
| `data.num_samples`                          |                                     `192` | `synthetic`           | Количество synthetic samples                           |
| `data.positive_fraction`                    |                                    `0.35` | `synthetic`           | Доля positive synthetic samples                        |
| `data.raw_dir`                              |                         `data/raw/luna16` | `luna16`              | Где лежит LUNA16 raw                                   |
| `data.candidates_csv`                       |          `data/raw/luna16/candidates.csv` | `luna16`              | CSV кандидатов                                         |
| `data.processed_dir`                        |                   `data/processed/luna16` | `luna16`              | Processed LUNA16 output                                |
| `data.patch_size`                           |                              `[32,48,48]` | `data`                | `(D,H,W)` patch size                                   |
| `data.batch_size`                           |                `8` synthetic, `64` LUNA16 | `data`                | Batch size dataloader/training                         |
| `data.num_workers`                          |                 `0` synthetic, `2` LUNA16 | `data`                | Dataloader workers                                     |
| `data.train_fraction`                       |                                     `0.7` | `data`                | Train split fraction                                   |
| `data.val_fraction`                         |                                    `0.15` | `data`                | Validation split fraction                              |
| `data.test_fraction`                        |                                    `0.15` | `data`                | Test split fraction                                    |
| `data.split_by_patient`                     |          `false` synthetic, `true` LUNA16 | `data`                | Делить по patient/seriesuid                            |
| `data.group_column`                         |                               `seriesuid` | `data`                | Колонка группировки в manifest                         |
| `data.save_splits`                          |                                    `true` | `data`                | Сохранять split indices                                |
| `data.ensure_data`                          |                                    `true` | `data`                | Автоматически готовить данные перед train              |
| `data.task`                                 |                    `nodule_vs_non_nodule` | `luna16`              | Текущая постановка задачи                              |
| `data.dvc_target`                           |                   `data/processed/luna16` | `luna16`              | DVC target для авто-восстановления                     |
| `data.allow_internet_download`              |                                    `true` | `luna16`              | Разрешить download из Zenodo                           |
| `data.download_subsets`                     |                                    `null` | `luna16`              | Явный список subset, например `[0,1,2]`                |
| `data.download_max_subsets`                 |                                       `1` | `luna16`              | Скачать первые N subset                                |
| `data.download_metadata`                    |                                    `true` | `luna16`              | Скачать `annotations.csv`, `candidates.csv`            |
| `data.extract_archives`                     |                                    `true` | `luna16`              | Распаковывать zip                                      |
| `data.keep_archives`                        |                                    `true` | `luna16`              | Оставлять zip после распаковки                         |
| `data.overwrite_downloads`                  |                                   `false` | `luna16`              | Перекачивать/перераспаковывать                         |
| `data.weighted_sampling.enabled`            |                                    `true` | `data`                | WeightedRandomSampler для дисбаланса                   |
| `preprocessing.spacing_mm`                  |                           `[1.0,1.0,1.0]` | preprocessing         | Целевой spacing; сейчас справочный параметр            |
| `preprocessing.clip_hu_min`                 |                                 `-1000.0` | preprocessing         | Нижний HU clipping                                     |
| `preprocessing.clip_hu_max`                 |                                   `400.0` | preprocessing         | Верхний HU clipping                                    |
| `preprocessing.normalize_to`                |                              `[-1.0,1.0]` | preprocessing         | Диапазон нормализации                                  |
| `preprocessing.chunk_size`                  |                                     `256` | preprocessing         | Samples per chunk при preprocessing                    |
| `preprocessing.cache_processed_patches`     |                                    `true` | preprocessing         | Флаг совместимости; chunked format всегда disk-backed  |
| `preprocessing.max_cached_ct_volumes`       |                                       `1` | preprocessing         | Сколько CT volume держать в RAM                        |
| `preprocessing.progress`                    |                                    `true` | preprocessing         | Показывать progress bars                               |
| `preprocessing.augment.enabled`             |                                    `true` | preprocessing         | Включить train augmentations                           |
| `preprocessing.augment.random_flip`         |                                    `true` | preprocessing         | Случайные flip по осям                                 |
| `preprocessing.augment.random_rotate90`     |                                    `true` | preprocessing         | Случайные повороты на 90°                              |
| `preprocessing.augment.gaussian_noise_std`  |                                    `0.02` | preprocessing         | Std гауссова шума                                      |
| `preprocessing.augment.random_shift_voxels` |                                       `2` | preprocessing         | Случайный shift в voxel                                |
| `model`                                     |                          `dlwpt_baseline` | Hydra defaults        | Выбор модели: `dlwpt_baseline`, `resnet3d_se`          |
| `model.name`                                |                          `dlwpt_baseline` | model                 | Имя архитектуры                                        |
| `model.in_channels`                         |                                       `1` | model                 | Каналов входного CT patch                              |
| `model.conv_channels`                       |                                       `8` | dlwpt_baseline        | Ширина baseline CNN                                    |
| `model.patch_size`                          |                      `${data.patch_size}` | dlwpt_baseline        | Patch size модели                                      |
| `model.expected_patch_size`                 |                              `[32,48,48]` | dlwpt_baseline        | Проверка ожидаемой формы                               |
| `model.spacing_mm`                          |                           `[1.0,1.0,1.0]` | dlwpt_baseline        | Metadata/совместимость                                 |
| `model.clip_hu_min`                         |                                 `-1000.0` | dlwpt_baseline        | Metadata/совместимость                                 |
| `model.clip_hu_max`                         |                                   `400.0` | dlwpt_baseline        | Metadata/совместимость                                 |
| `model.normalize_to`                        |                              `[-1.0,1.0]` | dlwpt_baseline        | Metadata/совместимость                                 |
| `model.cache_processed_patches`             |                                    `true` | dlwpt_baseline        | Metadata/совместимость                                 |
| `model.base_channels`                       |                                      `16` | resnet3d_se           | Ширина ResNet3D-SE                                     |
| `model.blocks_per_stage`                    |                                 `[2,2,2]` | resnet3d_se           | Количество блоков по стадиям                           |
| `model.se_reduction`                        |                                       `8` | resnet3d_se           | Reduction в SE блоках                                  |
| `model.dropout`                             |                                    `0.25` | resnet3d_se           | Dropout                                                |
| `model.local_crop_fraction`                 |                                     `0.5` | resnet3d_se           | Доля local crop, если используется моделью             |
| `model.augment.random_flip`                 |                                    `true` | resnet3d_se           | Модельный augmentation-флаг совместимости              |
| `model.augment.random_rotate90`             |                                    `true` | resnet3d_se           | Модельный augmentation-флаг совместимости              |
| `model.augment.gaussian_noise_std`          |                                    `0.02` | resnet3d_se           | Модельный augmentation-флаг совместимости              |
| `model.augment.random_shift_voxels`         |                                       `2` | resnet3d_se           | Модельный augmentation-флаг совместимости              |
| `loss`                                      |                                     `bce` | Hydra defaults        | Выбор loss: `bce`, `focal`                             |
| `loss.name`                                 |                           `bce` / `focal` | loss                  | Тип loss                                               |
| `loss.pos_weight`                           |                                    `null` | loss                  | Positive class weight или auto/число, если реализовано |
| `loss.alpha`                                |                                    `0.75` | focal                 | Alpha focal loss                                       |
| `loss.gamma`                                |                                     `2.0` | focal                 | Gamma focal loss                                       |
| `loss.focal_weight`                         |                                     `1.0` | focal                 | Вес focal компоненты                                   |
| `loss.bce_weight`                           |                                    `0.25` | focal                 | Вес BCE компоненты                                     |
| `trainer.max_epochs`                        |                                       `5` | trainer               | Epoch count                                            |
| `trainer.learning_rate`                     |                                  `0.0003` | trainer               | Learning rate                                          |
| `trainer.weight_decay`                      |                                  `0.0001` | trainer               | Weight decay                                           |
| `trainer.accelerator`                       |                                    `auto` | trainer               | `auto`, `cpu`, `gpu`                                   |
| `trainer.devices`                           |                                    `auto` | trainer               | `auto`, `1`, `[0]`, etc.                               |
| `trainer.precision`                         |                                      `32` | trainer               | `32`, `16-mixed`, `bf16-mixed`                         |
| `trainer.gradient_clip_val`                 |                                     `1.0` | trainer               | Gradient clipping                                      |
| `trainer.log_every_n_steps`                 |                                       `1` | trainer               | Частота train logging                                  |
| `trainer.num_sanity_val_steps`              |                                       `0` | trainer               | Sanity validation steps                                |
| `trainer.fast_dev_run`                      |                                   `false` | trainer               | Быстрый dev-run Lightning                              |
| `trainer.deterministic`                     |                                   `false` | trainer               | Deterministic kernels                                  |
| `trainer.benchmark`                         |                                   `false` | trainer               | CuDNN benchmark                                        |
| `logging`                                   |                                 `default` | Hydra defaults        | Подключает единый файл `configs/logging/default.yaml`  |
| `logging.mode`                              |                                    `none` | logging               | Режим логирования                                      |
| `logging.mlflow_tracking_uri`               |                   `http://127.0.0.1:8080` | logging               | MLflow tracking URI                                    |
| `logging.mlflow_port`                       |                                    `8080` | logging               | Документированный порт MLflow UI                       |
| `logging.experiment_name`                   |                              `lungscan3d` | logging               | MLflow experiment                                      |
| `logging.log_git_commit`                    |                                    `true` | logging               | Логировать git commit                                  |
| `logging.log_hyperparameters`               |                                    `true` | logging               | Логировать config                                      |
| `logging.tensorboard_save_dir`              |                          `lightning_logs` | logging               | TensorBoard logdir                                     |
| `logging.tensorboard_port`                  |                                    `6006` | logging               | Документированный порт TensorBoard UI                  |
| `postprocess.threshold`                     |                                    `0.35` | postprocess           | Classification threshold                               |
| `postprocess.threshold_artifact_path`       | `artifacts/postprocessing/threshold.json` | postprocess           | Файл подобранного threshold                            |
| `postprocess.use_threshold_artifact`        |                                    `true` | postprocess           | Читать threshold artifact при inference                |
| `postprocess.threshold_selection_metric`    |                 `recall_at_min_precision` | postprocess           | Стратегия подбора                                      |
| `postprocess.min_precision`                 |                                     `0.5` | postprocess           | Минимальная precision для стратегии                    |
| `postprocess.min_recall`                    |                                     `0.8` | postprocess           | Минимальная recall, если стратегия использует          |
| `postprocess.split`                         |                                     `val` | postprocess           | `val` или `test` для threshold search                  |
| `infer.checkpoint_path`                     |         `artifacts/checkpoints/best.ckpt` | infer                 | Checkpoint для infer/export                            |
| `infer.onnx_path`                           |          `artifacts/onnx/lungscan3d.onnx` | infer                 | ONNX path                                              |
| `infer.input_path`                          |                                    `null` | infer                 | `.npy` input для local infer                           |
| `infer.output_name`                         |                                   `logit` | infer                 | Имя ONNX/Triton output                                 |
| `infer.input_name`                          |                                   `input` | infer                 | Имя ONNX/Triton input                                  |
| `infer.opset_version`                       |                                      `17` | infer                 | ONNX opset                                             |
| `tensorrt.engine_path`                      |      `artifacts/tensorrt/lungscan3d.plan` | tensorrt              | TensorRT plan output                                   |
| `tensorrt.workspace_mb`                     |                                    `4096` | tensorrt              | Workspace memory pool MB                               |
| `tensorrt.precision`                        |                                    `fp16` | tensorrt              | `fp32`, `fp16`, `int8`                                 |
| `tensorrt.min_batch_size`                   |                                       `1` | tensorrt              | Dynamic shape min batch                                |
| `tensorrt.opt_batch_size`                   |                                      `16` | tensorrt              | Dynamic shape opt batch                                |
| `tensorrt.max_batch_size`                   |                                      `64` | tensorrt              | Dynamic shape max batch                                |
| `tensorrt.dry_run`                          |                                   `false` | tensorrt              | Не запускать `trtexec`, только собрать команду         |
| `tensorrt.trtexec_path`                     |                                 `trtexec` | tensorrt              | Путь к `trtexec`                                       |
| `tensorrt.extra_args`                       |                                      `[]` | tensorrt              | Дополнительные аргументы `trtexec`                     |
| `triton.model_repository`                   |                 `triton_model_repository` | triton                | Triton model repository                                |
| `triton.model_name`                         |                              `lungscan3d` | triton                | Triton model name                                      |
| `triton.input_path`                         |                                    `null` | triton                | `.npy` input для Triton client                         |
| `triton.client_url`                         |                          `localhost:8000` | triton                | HTTP endpoint Triton                                   |
| `triton.http_port`                          |                                    `8000` | triton                | HTTP port                                              |
| `triton.grpc_port`                          |                                    `8001` | triton                | gRPC port                                              |
| `triton.metrics_port`                       |                                    `8002` | triton                | Metrics port                                           |
| `triton.docker_image`                       |   `nvcr.io/nvidia/tritonserver:24.05-py3` | triton                | Docker image для сервера                               |

## 2.18. Типовые проблемы

### `trtexec` not found

Установите TensorRT CLI на хост или запускайте экспорт в контейнере NVIDIA TensorRT. Python dependency `tensorrt` полезна для Python-интеграций, но production export в проекте выполняется через `trtexec`.

### CUDA OOM

Уменьшите:

```bash
data.batch_size=32
```

Если не помогло:

```bash
model=resnet3d_se model.base_channels=8
```

или уменьшите patch:

```bash
data.patch_size=[24,40,40]
```

### Слишком много RAM на preprocessing

Уменьшите:

```bash
preprocessing.chunk_size=128 preprocessing.max_cached_ct_volumes=1
```

### Нет MLflow/TensorBoard UI

Логирование в train и UI-сервер — разные процессы. Сначала поднимите UI:

```bash
scripts/run_mlflow.sh
scripts/run_tensorboard.sh
```

Затем запускайте обучение с нужным режимом:

```bash
lungscan3d train logging.mode=all
```
