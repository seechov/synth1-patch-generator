# Seechov Forge — This Synth1 Bank Does Not Exist

Generates novel [Synth1](https://daichilab.sakura.ne.jp/softsynth/) VST presets using a WGAN-GP neural network trained on real soundbanks.

The training code is based on the original [jskripchuk/Synth1GAN](https://github.com/jskripchuk/Synth1GAN) project, licensed under **GPL-3.0**.

---

## Project structure

```
Seechov Forge/
├── training/          # Python: parse presets and train the model
│   ├── train.py       # All-in-one training script
│   └── requirements.txt
├── gui/               # Rust: cross-platform GUI for generating presets
│   ├── Cargo.toml
│   └── src/main.rs
├── installer/         # Packaging configs (Windows/macOS/Linux)
│   └── trained-model/ # Bundled default model for the installers
├── .devcontainer/     # DevContainer config for Zed / VS Code
└── .zed/              # Zed editor settings
```

---

## Step 1 — Train the model (Windows, native)

GPU is strongly recommended (~8 h on GTX 1070, ~30 min on RTX 3080). CPU training works but is very slow.

### 1a. Install Python 3.12

Download from https://www.python.org/downloads/windows/

> Training is supported on Python 3.12. Newer versions may not work.

### 1b. Install PyTorch

Open a terminal inside the `training/` folder and run **one** of:

```powershell
# GPU (CUDA 12.1) — recommended if you have an NVIDIA card
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

### 1c. Run training

```powershell
python train.py --presets-dir C:\path\to\your\sy1-presets --output-dir .\model
```

`--presets-dir` can point at any folder containing `.sy1` files — the script scans recursively.  
Use the Synth1 factory bank, free soundbanks, or any combination.

**Useful options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 20000 | More = better quality (diminishing returns after ~10k) |
| `--batch-size` | 64 | Clamped automatically if it exceeds the dataset size |
| `--latent-dim` | 32 | Latent noise vector size |
| `--lr-g` | 0.0002 | Generator learning rate |
| `--lr-d` | 0.0002 | Discriminator learning rate |
| `--batch-norm` | on | Use BatchNorm in the generator trunk (`--no-batch-norm` disables) |
| `--ema-decay` | 0.999 | EMA decay for generator weights (`0` disables) |
| `--spectral-norm` | off | Use spectral normalization in the discriminator |
| `--d-noise-std` | 0.0 | Std of Gaussian noise added to discriminator inputs |
| `--seed` | — | Random seed for reproducibility |

When done you will have:

```
model/
├── generator.onnx        ← loaded by the GUI app
├── normalization.json    ← loaded by the GUI app
└── checkpoints/          ← periodic weight snapshots
```

---

## Step 2 — Build the GUI app

The GUI is written in Rust. It uses [tract-onnx](https://github.com/sonos/tract) (pure Rust ONNX runtime, no external DLLs) and [egui](https://github.com/emilk/egui).

### 2a. Install Rust

Download from https://rustup.rs/ and run the installer.

### 2b. Build

```powershell
cd gui
cargo build --release
```

Binary output:
- Windows: `gui\target\release\seechov-forge.exe`
- Linux:   `gui/target/release/seechov-forge`

Or just run directly:

```powershell
cargo run --release
```

---

## Step 3 — Generate presets

The installers bundle a default model (`installer/trained-model/`), so on first launch the GUI auto-discovers it next to the executable and loads it right away. The default output folder is `Documents/Seechov Forge/presets`.

1. Launch the app — you should see a green "● model ready" indicator (bundled model auto-loaded)
2. Pick a **Output folder** (defaults to `Documents/Seechov Forge/presets`)
3. Set a bank name and preset count (1–128)
4. Click **⚡ Generate**

If you trained your own model, point **Model folder** to the directory containing `generator.onnx` + `normalization.json` and click **Load model** instead.

Load the output folder into Synth1 via **File → Load Bank**.

> **Windows SmartScreen:** the installer is not digitally signed, so Windows or
> Edge may show a "Windows protected your PC" warning. This is expected for
> unsigned open-source binaries — click **More info**, then **Run anyway**.

---

## Development in Zed with DevContainer

Open the repository in Zed. It will detect `.devcontainer/devcontainer.json` and offer to reopen in a container. The container includes Python 3.11, PyTorch (CPU build), and the full Rust toolchain with rust-analyzer.

> **Note:** For GPU training, run `train.py` natively on Windows — GPU passthrough in Docker requires WSL2 + NVIDIA Container Toolkit and is optional.

---

## Model architecture

| Component | Architecture |
|-----------|-------------|
| **Generator** | noise(N) → shared trunk (128→256→512→1024, BatchNorm + LeakyReLU) → a `Tanh` head for continuous features + a `softmax` head per categorical variable |
| **Discriminator** | preset(F) → Linear(256) → Linear(128) → Linear(64) → Linear(1), LeakyReLU(0.2), optional spectral norm |
| **Training** | WGAN-GP, λ=10, 5 critic steps per generator step, Adam lr_g/lr_d=0.0002 (β₁=0.5, β₂=0.999), exponential moving average of generator weights |
| **Features** | continuous parameters scaled to [-1, 1]; categoricals one-hot encoded (kept 0/1) |

---

## License

This repository contains code under two licenses:

- **GUI and general project code** — MIT (see `LICENSE`, © 2026 Aleksei Sychev).
- **Training code** (`training/`) — **GPL-3.0**, derived from the original [jskripchuk/Synth1GAN](https://github.com/jskripchuk/Synth1GAN) project. Full text in `training/LICENSE`.
