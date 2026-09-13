# Synth1GAN — This Synth1 Bank Does Not Exist

Генерирует новые пресеты VST-синтезатора [Synth1](https://www.taktech.org/takumi/synth1/) с помощью нейросети WGAN-GP, обученной на реальных банках звуков.

---

## Структура проекта

```
Synth1GAN/
├── training/          # Python: парсинг пресетов и обучение модели
│   ├── train.py       # Единый скрипт обучения
│   └── requirements.txt
├── gui/               # Rust: кроссплатформенное GUI для генерации пресетов
│   ├── Cargo.toml
│   └── src/main.rs
├── installer/         # Конфигурации упаковки (Windows/macOS/Linux)
│   └── trained-model/ # Модель по умолчанию для инсталляторов
├── .devcontainer/     # Конфигурация DevContainer для Zed / VS Code
└── .zed/              # Настройки редактора Zed
```

---

## Шаг 1 — Обучение модели (Windows, нативно)

Настоятельно рекомендуется GPU (~8 ч на GTX 1070, ~30 мин на RTX 3080). Обучение на CPU работает, но очень медленно.

### 1a. Установка Python 3.12

Скачайте с https://www.python.org/downloads/windows/

> Обучение поддерживается на Python 3.12. Более новые версии могут не работать.

### 1b. Установка PyTorch

Откройте терминал в папке `training/` и выполните **один** из вариантов:

```powershell
# GPU (CUDA 12.1) — рекомендуется при наличии NVIDIA-видеокарты
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Только CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Затем установите остальные зависимости:

```powershell
pip install -r requirements.txt
```

### 1c. Запуск обучения

```powershell
python train.py --presets-dir C:\path\to\your\sy1-presets --output-dir .\model
```

`--presets-dir` может указывать на любую папку с `.sy1` файлами — скрипт сканирует её рекурсивно.  
Используйте фабричный банк Synth1, бесплатные банки звуков или любую их комбинацию.

**Полезные опции:**

| Опция | По умолчанию | Описание |
|--------|-------------|-------------|
| `--epochs` | 20000 | Больше = лучше качество (убывающая отдача после ~10k) |
| `--batch-size` | 64 | Автоматически ограничивается, если превышает размер датасета |
| `--latent-dim` | 32 | Размер латентного вектора шума |
| `--lr-g` | 0.0002 | Скорость обучения генератора |
| `--lr-d` | 0.0002 | Скорость обучения дискриминатора |
| `--batch-norm` | вкл | BatchNorm в стволе генератора (`--no-batch-norm` отключает) |
| `--ema-decay` | 0.999 | Коэффициент EMA весов генератора (`0` отключает) |
| `--spectral-norm` | выкл | Спектральная нормализация дискриминатора |
| `--d-noise-std` | 0.0 | СКО гауссова шума на входах дискриминатора |
| `--seed` | — | Сид для воспроизводимости |

По завершении получите:

```
model/
├── generator.onnx        ← загружается GUI-приложением
├── normalization.json    ← загружается GUI-приложением
└── checkpoints/          ← периодические снапшоты весов
```

---

## Шаг 2 — Сборка GUI-приложения

GUI написано на Rust. Использует [tract-onnx](https://github.com/sonos/tract) (чисто-Rust рантайм ONNX, без внешних DLL) и [egui](https://github.com/emilk/egui).

### 2a. Установка Rust

Скачайте с https://rustup.rs/ и запустите установщик.

### 2b. Сборка

```powershell
cd gui
cargo build --release
```

Готовые бинарники:
- Windows: `gui\target\release\synth1gan.exe`
- Linux:   `gui/target/release/synth1gan`

Либо просто запустите напрямую:

```powershell
cargo run --release
```

---

## Шаг 3 — Генерация пресетов

Инсталляторы включают модель по умолчанию (`installer/trained-model/`), поэтому при первом запуске GUI автоматически находит её рядом с исполняемым файлом и сразу загружает. Папка вывода по умолчанию — `Documents/Synth1GAN/presets`.

1. Запустите приложение — вы увидите зелёный индикатор «● model ready» (поставляемая модель загружена автоматически)
2. Выберите **Output folder** (по умолчанию `Documents/Synth1GAN/presets`)
3. Задайте имя банка и количество пресетов (1–128)
4. Нажмите **⚡ Generate**

Если вы обучили свою модель, укажите в **Model folder** каталог, содержащий `generator.onnx` + `normalization.json`, и нажмите **Load model**.

Загрузите выходную папку в Synth1 через **File → Load Bank**.

---

## Разработка в Zed с DevContainer

Откройте репозиторий в Zed. Он обнаружит `.devcontainer/devcontainer.json` и предложит переоткрыть в контейнере. Контейнер включает Python 3.11, PyTorch (CPU-сборку) и полный инструментарий Rust с rust-analyzer.

> **Примечание:** для обучения на GPU запускайте `train.py` нативно на Windows — прокидывание GPU в Docker требует WSL2 + NVIDIA Container Toolkit и опционально.

---

## Архитектура модели

| Компонент | Архитектура |
|-----------|-------------|
| **Generator** | noise(N) → общий ствол (128→256→512→1024, BatchNorm + LeakyReLU) → «голова» `Tanh` для непрерывных + «голова» `softmax` на каждую категориальную переменную |
| **Discriminator** | preset(F) → Linear(256) → Linear(128) → Linear(64) → Linear(1), LeakyReLU(0.2), опциональная спектральная нормализация |
| **Обучение** | WGAN-GP, λ=10, 5 шагов критика на шаг генератора, Adam lr_g/lr_d=0.0002 (β₁=0.5, β₂=0.999), экспоненциальное скользящее среднее весов генератора |
| **Признаки** | непрерывные параметры масштабируются в [-1, 1]; категориальные — one-hot (остаются 0/1) |
