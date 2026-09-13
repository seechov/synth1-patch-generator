# Synth1GAN — Project Overview

> **This Synth1 Bank Does Not Exist** — генерация новых пресетов для VST-синтезатора [Synth1](https://daichilab.sakura.ne.jp/softsynth/) с помощью нейросети WGAN-GP.

---

## 1. О проекте

Synth1GAN генерирует новые, ранее не существовавшие пресеты (`.sy1`) для бесплатного VST-синтезатора **Synth1**. В основе лежит генеративно-состязательная сеть с градиентным штрафом (Wasserstein GAN with Gradient Penalty, WGAN-GP), обученная на реальных банках звуков.

Проект состоит из двух независимых частей:

| Часть | Язык | Назначение |
|-------|------|------------|
| **Обучение** (`training/`) | Python 3.12 / PyTorch | Парсинг пресетов, подготовка данных, обучение модели, экспорт в ONNX |
| **Генерация** (`gui/`) | Rust / egui / tract-onnx | Кроссплатформенное приложение для генерации пресетов из обученной модели |

Итоговый рабочий цикл:

```mermaid
flowchart LR
    A[.sy1 банки звуков] --> B[Парсинг пресетов]
    B --> C[Feature engineering]
    C --> D[Обучение WGAN-GP]
    D --> E[generator.onnx + normalization.json]
    E --> F[GUI приложение]
    F --> G[Новые .sy1 пресеты]
```

---

## 2. Структура репозитория

```
Synth1GAN/
├── training/                  # Python: парсинг пресетов и обучение модели
│   ├── train.py               # Единый скрипт обучения (весь пайплайн)
│   ├── requirements.txt       # Python-зависимости (PyTorch ставится отдельно)
│   ├── TRAIN_EXPLAINED.md     # Подробное объяснение обучения (англ.)
│   ├── TRAIN_EXPLAINED_ru.md  # Подробное объяснение обучения (рус.)
│   └── model/                 # Результаты обучения (в .gitignore)
├── gui/                       # Rust: кроссплатформенное GUI
│   ├── Cargo.toml             # Манифест и зависимости
│   ├── Cargo.lock             # Зафиксированные версии зависимостей
│   ├── src/main.rs            # Вся логика GUI (один файл)
│   └── assets/
│       └── synth1gan.desktop  # .desktop-файл для Linux
├── installer/                 # Конфигурации упаковки
│   ├── windows/setup.iss      # Сценарий Inno Setup (Windows-инсталлятор)
│   ├── macos/Info.plist       # Метаданные macOS .app-бандла
│   └── trained-model/         # Модель по умолчанию для инсталлятора
│       ├── generator.onnx     # Экспортированный генератор
│       └── normalization.json # Метаданные нормализации
├── presets/                   # Входные/выходные пресеты (в .gitignore)
├── docs/
│   ├── overview.md                 # Этот документ (англ.)
│   ├── overview_ru.md              # Русская версия этого документа
│   ├── training-improvements.md    # План улучшений методики обучения (англ.)
│   └── training-improvements_ru.md # Русская версия плана улучшений
├── .devcontainer/             # DevContainer для Zed / VS Code
│   ├── Dockerfile             # Python 3.11 + системные зависимости
│   └── devcontainer.json      # Конфигурация окружения
├── .github/workflows/
│   └── release.yml            # CI-сборка релизов (Windows/Linux/macOS)
├── .zed/settings.json         # Настройки редактора Zed
├── .gitignore
├── LICENSE                    # MIT (Copyright © 2026 Aleksei Sychev)
├── README.md                  # Быстрый старт и инструкции (англ.)
└── README_ru.md               # Быстрый старт и инструкции (рус.)
```

---

## 3. Как это работает

### 3.1 Этап обучения (`training/train.py`)

Скрипт выполняет весь пайплайн за 5 шагов:

1. **Загрузка** — рекурсивный поиск `.sy1`-файлов и их парсинг в словари.
2. **Построение датасета** — преобразование пресетов в `DataFrame`, переименование числовых идентификаторов параметров в человекочитаемые имена.
3. **Feature engineering** — удаление шумных/разреженных параметров и one-hot кодирование категориальных признаков (с логированием числа отброшенных колонок/строк).
4. **Нормализация** — масштабирование *только непрерывных* параметров в диапазон `[-1, 1]` (под выходной слой `Tanh`); one-hot признаки остаются в `[0, 1]`.
5. **Обучение и экспорт** — обучение WGAN-GP, экспорт EMA-версии генератора в ONNX и сохранение метаданных нормализации.

#### Параметры Synth1

Пресет Synth1 описывается набором параметров, сопоставленных через `COL_RENAME` (например, `filter freq`, `amp attack`, `lfo1 speed`). В ходе подготовки:

- **Обучение** использует две группы признаков: непрерывные параметры + one-hot закодированные категории (их количество зависит от набора данных).
- Часть параметров **исключается из обучения** (арпеджиатор, эквалайзер, pitch bend и другие) как слишком шумные или разреженные.
- Категориальные переменные (формы осцилляторов, типы фильтров, назначения LFO и т.д.) обрабатываются через `scikit-learn` `OneHotEncoder`.
- Непрерывные признаки масштабируются в `[-1, 1]`; one-hot признаки остаются `0/1`.

### 3.2 Архитектура модели

| Компонент | Архитектура |
|-----------|-------------|
| **Generator** | noise(N) → общий ствол (128→256→512→1024, BatchNorm + LeakyReLU) → «голова» `Tanh` для непрерывных + «голова» `softmax` на каждую категориальную переменную |
| **Discriminator** | preset(F) → Linear(256) → Linear(128) → Linear(64) → Linear(1), LeakyReLU(0.2), опциональная спектральная нормализация |
| **Обучение** | WGAN-GP, градиентный штраф λ=10, 5 шагов критика на 1 шаг генератора, Adam lr_g/lr_d (β₁=0.5, β₂=0.999), EMA весов генератора |

### 3.3 Результаты обучения

После успешного обучения создаётся директория модели:

```
model/
├── generator.onnx        ← загружается GUI-приложением
├── normalization.json    ← метаданные нормализации (min/max, one-hot кодировки)
└── checkpoints/          ← периодические снапшоты весов (каждые 500 эпох)
```

`normalization.json` описывает, как сопоставить выход генератора обратно в реальные значения параметров: непрерывные параметры денормализуются из `[-1, 1]` в `[min, max]`, категориальные восстанавливаются через `argmax` по one-hot-срезу (выход softmax-головы).

### 3.4 Этап генерации (`gui/src/main.rs`)

GUI написано на Rust с использованием:

- **[eframe/egui](https://github.com/emilk/egui)** — нативный кроссплатформенный интерфейс.
- **[tract-onnx](https://github.com/sonos/tract)** — чисто-Rust рантайм ONNX без внешних DLL.
- **rfd** — нативные диалоги выбора папок.
- **rand / rand_distr** — генерация нормального шума для входа генератора.

Процесс генерации:

1. При запуске приложение ищет модель по умолчанию рядом с исполняемым файлом (каталог `model/` рядом с exe/бинарником, либо `/usr/share/synth1gan/model` на Linux). Если найдены `generator.onnx` и `normalization.json`, модель загружается автоматически.
2. Выходной каталог по умолчанию — `Documents/Synth1GAN/presets` (создаётся при необходимости).
3. При нажатии **⚡ Generate** для каждого из `N` пресетов:
   - генерируется латентный вектор шума (нормальное распределение);
   - модель выдаёт вектор, где непрерывные параметры идут первыми, а затем one-hot группы (каждая — результат softmax-головы);
   - выход декодируется обратно в карту `param_id → значение`;
   - записывается `.sy1`-файл (заголовок + отсортированные параметры).
4. Готовые пресеты загружаются в Synth1 через **File → Load Bank**.

> Если у приложения нет модели по умолчанию, её нужно указать вручную: `generator.onnx` + `normalization.json` должны лежать в одной папке, которую нужно выбрать и загрузить кнопкой **Load model**.

---

## 4. Использование

### 4.1 Обучение модели

Требуется Python 3.12. PyTorch ставится отдельно (GPU через CUDA 12.1 или CPU). Обучение поддерживается на Python 3.12; более новые версии могут не работать.

```powershell
# GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Затем зависимости и запуск:

```powershell
pip install -r training/requirements.txt
python training/train.py --presets-dir C:\path\to\presets --output-dir .\model
```

Основные опции:

| Опция | По умолчанию | Описание |
|-------|--------------|----------|
| `--epochs` | 20000 | Число эпох (убывающая отдача после ~10k) |
| `--batch-size` | 64 | При превышении размера датасета подстраивается автоматически |
| `--latent-dim` | 32 | Размер латентного вектора шума |
| `--lr-g` | 0.0002 | Скорость обучения генератора |
| `--lr-d` | 0.0002 | Скорость обучения дискриминатора |
| `--n-critic` | 5 | Шагов дискриминатора на шаг генератора |
| `--lambda-gp` | 10.0 | Вес градиентного штрафа |
| `--ema-decay` | 0.999 | Коэффициент EMA весов генератора (`0` отключает) |
| `--spectral-norm` | выкл | Спектральная нормализация дискриминатора |
| `--d-noise-std` | 0.0 | СКО гауссова шума на входах дискриминатора |
| `--seed` | — | Сид для воспроизводимости |

> GPU настоятельно рекомендуется (~8 ч на GTX 1070, ~30 мин на RTX 3080).

### 4.2 Сборка GUI

```powershell
cd gui
cargo build --release
```

Бинарники: `gui\target\release\synth1gan.exe` (Windows) или `gui/target/release/synth1gan` (Linux).

### 4.3 Генерация пресетов

1. Запустить приложение — если модель поставляется вместе с инсталлятором, она загрузится автоматически и появится индикатор «● model ready».
2. При необходимости выбрать/загрузить другую папку модели (с `generator.onnx` + `normalization.json`) кнопкой **Load model**.
3. Выбрать выходную папку (по умолчанию `Documents/Synth1GAN/presets`).
4. Задать имя банка и количество пресетов (1–128).
5. Нажать **⚡ Generate**.

---

## 5. Сборка и CI/CD

### 5.1 Локальная разработка (DevContainer)

В репозитории есть конфигурация DevContainer (`.devcontainer/`), которую автоматически распознают Zed и VS Code. Контейнер включает Python 3.11, PyTorch (CPU-сборку) и полный инструментарий Rust (rust-analyzer, rustfmt, clippy).

> Для обучения на GPU `train.py` запускается нативно на Windows — прокидывание GPU в Docker требует WSL2 + NVIDIA Container Toolkit.

### 5.2 Автоматическая сборка релизов

`.github/workflows/release.yml` собирает артефакты при публикации версионного тега (например `v1.0.0`):

| Задача | Платформа | Результат |
|--------|-----------|-----------|
| `build-windows` | `windows-latest` | Установщик `.exe` (Inno Setup) |
| `build-linux` | `ubuntu-22.04` | Пакеты `.deb` (cargo-deb) и `.rpm` (cargo-generate-rpm) |
| `build-macos` | `macos-latest` | `.app`-бандл и `.dmg` (hdiutil) |

Все артефакты выгружаются в GitHub Release через `softprops/action-gh-release`.

### 5.3 Упаковка

- **Windows** — `installer/windows/setup.iss` (Inno Setup, локализация EN/RU). Модель по умолчанию (`installer/trained-model/`) упаковывается в каталог `model/` рядом с исполняемым файлом.
- **macOS** — `installer/macos/Info.plist` (идентификатор `com.seechov.synth1gan`). Модель помещается в `Contents/MacOS/model/` внутри `.app`-бандла.
- **Linux** — `gui/assets/synth1gan.desktop` + метаданные `cargo-deb`/`cargo-generate-rpm` в `Cargo.toml`. Модель упаковывается в `/usr/share/synth1gan/model`.

---

## 6. Лицензия и авторство

- **GUI и код проекта**: MIT (см. `LICENSE`, Copyright © 2026 Aleksei Sychev).
- **Код обучения** (`training/`): **GPL-3.0**, производная работа от оригинального проекта [jskripchuk/Synth1GAN](https://github.com/jskripchuk/Synth1GAN) (см. `training/LICENSE`).
- **Автор GUI**: Aleksei Sychev `seechov@protonmail.com` (см. `gui/Cargo.toml`).
- **Synth1** — VST-синтезатор от Daichi Laboratory (ICHIRO TODA), см. <https://daichilab.sakura.ne.jp/softsynth/>.

---

## 7. Полезные ссылки

- [Synth1 (официальный сайт)](https://daichilab.sakura.ne.jp/softsynth/)
- [tract-onnx — Rust-рантайм ONNX](https://github.com/sonos/tract)
- [egui — библиотека GUI на Rust](https://github.com/emilk/egui)
- [PyTorch — установка под свою платформу](https://pytorch.org/get-started/locally/)
- Подробное описание обучения: `training/TRAIN_EXPLAINED.md` и `training/TRAIN_EXPLAINED_ru.md`
