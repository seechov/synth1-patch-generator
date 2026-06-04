# Synth1GAN — This Synth1 Bank Does Not Exist

Generates novel [Synth1](https://www.taktech.org/takumi/synth1/) VST presets using a WGAN-GP neural network trained on real soundbanks.

---

## Project structure

```
Synth1GAN/
├── training/          # Python: parse presets and train the model
│   ├── train.py       # All-in-one training script
│   └── requirements.txt
├── gui/               # Rust: cross-platform GUI for generating presets
│   ├── Cargo.toml
│   └── src/main.rs
├── .devcontainer/     # DevContainer config for Zed / VS Code
└── .zed/              # Zed editor settings
```

---

## Step 1 — Train the model (Windows, native)

GPU is strongly recommended (~8 h on GTX 1070, ~30 min on RTX 3080). CPU training works but is very slow.

### 1a. Install Python 3.11+

Download from https://www.python.org/downloads/windows/

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
| `--batch-size` | 64 | Reduce to 32 if you get out-of-memory errors |
| `--latent-dim` | 10 | Latent noise vector size |

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
- Windows: `gui\target\release\synth1gan.exe`
- Linux:   `gui/target/release/synth1gan`

Or just run directly:

```powershell
cargo run --release
```

---

## Step 3 — Generate presets

1. Launch the app
2. Click **…** next to **Model folder** → select the `model/` directory from Step 1
3. Click **Load model** — you should see a green "● model ready" indicator
4. Click **…** next to **Output folder** → choose where to save the `.sy1` files
5. Set a bank name and preset count (1–128)
6. Click **⚡ Generate**

Load the output folder into Synth1 via **File → Load Bank**.

---

## Development in Zed with DevContainer

Open the repository in Zed. It will detect `.devcontainer/devcontainer.json` and offer to reopen in a container. The container includes Python 3.11, PyTorch (CPU build), and the full Rust toolchain with rust-analyzer.

> **Note:** For GPU training, run `train.py` natively on Windows — GPU passthrough in Docker requires WSL2 + NVIDIA Container Toolkit and is optional.

---

## Model architecture

| Component | Architecture |
|-----------|-------------|
| **Generator** | noise(10) → Linear(128) → Linear(256) → Linear(512) → Linear(1024) → Linear(93), BatchNorm + LeakyReLU(0.2), Tanh output |
| **Discriminator** | preset(93) → Linear(64) → Linear(32) → Linear(1), LeakyReLU(0.2) |
| **Training** | WGAN-GP, λ=10, 5 critic steps per generator step, Adam lr=0.0002 (β₁=0.5, β₂=0.999) |
| **Features** | 93 dims: 52 continuous params + 41 one-hot encoded categorical params |
