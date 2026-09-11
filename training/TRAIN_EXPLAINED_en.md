# `train.py` Walkthrough — Synth1GAN

> This document explains the training script line by line.  
> It is written for readers who are comfortable with Python programming but have never worked with neural network training.  
> Each section includes a brief explanation of the underlying concept and links to articles for deeper study.

---

## Table of Contents

1. [What This Script Does — the Big Picture](#1-what-this-script-does--the-big-picture)
2. [Shebang and Docstring — Lines 1–15](#2-shebang-and-docstring--lines-1-15)
3. [Imports — Lines 17–32](#3-imports--lines-17-32)
4. [Synth1 Parameter Dictionary — Lines 36–112](#4-synth1-parameter-dictionary--lines-36-112)
5. [Dropped Columns TO_DROP — Lines 115–131](#5-dropped-columns-to_drop--lines-115-131)
6. [Categorical Variables CATEGORICAL_VARS — Lines 133–145](#6-categorical-variables-categorical_vars--lines-133-145)
7. [Reading Preset Files — Lines 151–186](#7-reading-preset-files--lines-151-186)
8. [Building the DataFrame — Lines 192–199](#8-building-the-dataframe--lines-192-199)
9. [Feature Engineering — Lines 202–251](#9-feature-engineering--lines-202-251)
10. [Data Normalization — Lines 254–260](#10-data-normalization--lines-254-260)
11. [Neural Network Architecture — Lines 266–315](#11-neural-network-architecture--lines-266-315)
12. [Gradient Penalty — Lines 321–340](#12-gradient-penalty--lines-321-340)
13. [Training Loop — Lines 343–432](#13-training-loop--lines-343-432)
14. [ONNX Export — Lines 438–454](#14-onnx-export--lines-438-454)
15. [Saving Normalization Metadata — Lines 460–515](#15-saving-normalization-metadata--lines-460-515)
16. [Entry Point main() — Lines 521–602](#16-entry-point-main--lines-521-602)
17. [Full Data Flow Diagram](#17-full-data-flow-diagram)

---

## 1. What This Script Does — the Big Picture

This script trains a **GAN** (Generative Adversarial Network) on a collection of Synth1 synthesizer presets in `.sy1` format. Once trained, the network can **generate new, plausible presets** — synthesizer settings that never existed before but sound stylistically consistent with the training data.

**What is a GAN?**  
A GAN is two neural networks locked in competition:
- The **Generator** — an artist that produces fake paintings (presets).
- The **Discriminator** — a detective that decides whether what it sees is a genuine preset or a forgery.

Both networks are trained simultaneously and each drives the other to improve. Eventually the generator learns to produce presets so convincing that the discriminator can no longer tell them apart from real ones.

**Further reading:**
- [Original GAN paper (Goodfellow, 2014)](https://arxiv.org/abs/1406.2661)
- [Understanding GANs — Towards Data Science](https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29)

---

## 2. Shebang and Docstring — Lines 1–15

```python
#!/usr/bin/env python3
"""
Synth1GAN — WGAN-GP trainer for Synth1 VST presets.

Usage:
    python train.py --presets-dir C:/path/to/presets --output-dir ./model
    python train.py --presets-dir ./synth_patches/all --output-dir ./model --epochs 5000
...
"""
```

**Line 1 — shebang** (`#!/usr/bin/env python3`):  
A Unix directive that tells the shell which interpreter to use when the file is executed directly (`./train.py`) rather than via `python train.py`. On Windows this line is ignored, but it is kept for cross-platform compatibility.

**Lines 2–15 — module docstring:**  
The triple-quoted string immediately following the shebang is the module-level documentation string. Python stores it in the module's `__doc__` attribute and displays it when `help(train)` is called. It lists what the script does, how to invoke it, and its five-step pipeline.

---

## 3. Imports — Lines 17–32

```python
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
```

The imports fall into two groups:

### Python Standard Library (first block)

| Library | Purpose |
|---|---|
| `argparse` | Parses command-line arguments (`--presets-dir`, `--epochs`, etc.) |
| `csv` | Imported but not used directly (legacy from an earlier version) |
| `glob` | Finds files matching a pattern (`*.sy1`) recursively |
| `json` | Reads and writes JSON files (saving normalization parameters) |
| `os` | File system operations: creating directories, building paths |
| `sys` | Provides `sys.exit()` to halt on errors and `sys.stdout.flush()` to flush the output buffer |
| `pathlib.Path` | Object-oriented path handling (imported, used indirectly) |

### Third-Party Libraries (second block)

**`numpy` (np)**  
The foundational library for numerical computing in Python. It works with arrays of numbers far faster than Python lists because data is stored contiguously in memory and all heavy lifting is delegated to optimised C code.  
→ [NumPy: Absolute Beginners Guide](https://numpy.org/doc/stable/user/absolute_beginners.html)

**`pandas` (pd)**  
A library for working with tabular data — think of it as Excel inside Python. The core object is a `DataFrame` (a table with named rows and columns). Used here to load, clean, and transform preset data.  
→ [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)

**`torch` (PyTorch)**  
The main machine-learning framework in this script. PyTorch is a library for building and training neural networks. Its defining feature is **automatic differentiation** (autograd): PyTorch records every operation performed on tensors and can automatically compute the gradients needed for training.  
→ [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

**`torch.autograd`**  
The automatic differentiation engine. Used here to compute gradients manually inside the gradient penalty function. Without autograd, gradients would have to be derived analytically — complex and error-prone.

**`torch.nn`**  
High-level module for building neural networks. Provides ready-made layers: `Linear` (fully-connected layer), `BatchNorm1d` (batch normalisation), `LeakyReLU` (activation function), `Sequential` (container for a sequence of layers).

**`sklearn.preprocessing.OneHotEncoder`**  
From the scikit-learn library. Converts categorical variables (e.g. oscillator waveform type: sine, sawtooth, square) into numerical vectors. Covered in detail in section 9.

**`torch.utils.data.DataLoader, Dataset`**  
Infrastructure for feeding data to the network in batches. `Dataset` describes how to retrieve a single sample. `DataLoader` iterates over the dataset, shuffles the data, and assembles batches.

---

## 4. Synth1 Parameter Dictionary — Lines 36–112

```python
COL_RENAME = {
    0: "osc1 shape",
    1: "osc2 shape",
    2: "osc2 pitch",
    ...
    74: "portament auto mode",
}
```

A `.sy1` file stores synthesizer parameters as numeric identifiers: `0`, `42`, `12`, and so on. Working with raw numbers is inconvenient. `COL_RENAME` is a **translation table**: the key is the numeric parameter ID found in the preset file; the value is a human-readable name.

When the DataFrame is first built, columns are named `"0"`, `"1"`, `"2"`. After `df.rename(columns=col_rename_str)` they become `"osc1 shape"`, `"osc2 shape"`, etc.

Synth1 has 75 parameters in total (IDs 0–74), and this dictionary covers all of them.

---

## 5. Dropped Columns TO_DROP — Lines 115–131

```python
TO_DROP = [
    "name",
    "osc1 fm modulation",
    "osc2 pitch",
    "osc2 kbd track",
    "osc key shift",
    "arp. on/off",
    "arp type",
    "arp range",
    "arp beat",
    "arp gate",
    "equalizer tone",
    "equalizer freq.",
    "equalizer level",
    "equalizer Q",
    "pitch bend range",
]
```

Not all synthesizer parameters are equally useful for training. These columns are excluded for the following reasons:

- **`name`** — a text string with the preset name; not a synthesis parameter.
- **Arpeggiator (`arp.*`)** — often absent (most presets do not use the arpeggiator). Sparse data with many zeros degrades training.
- **Equalizer (`equalizer.*`)** — similarly rarely used.
- **`pitch bend range`** — almost always the same value (2 semitones), meaning it carries no information about the sonic character of a preset.
- **`osc2 pitch`, `osc2 kbd track`, `osc key shift`** — low variance or complex interdependencies.

**Why remove such columns?**  
In machine learning this is called **feature selection**. Irrelevant features are not just useless — they add noise and expand the search space, slowing training and reducing model quality.  
→ [Feature Selection Techniques — Towards Data Science](https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e)

---

## 6. Categorical Variables CATEGORICAL_VARS — Lines 133–145

```python
CATEGORICAL_VARS = [
    "osc1 shape",
    "osc2 shape",
    "osc mod env dest.",
    "filter type",
    "chorus type",
    "play mode type",
    "lfo1 destination",
    "lfo1 type",
    "lfo2 destination",
    "lfo2 type",
]
```

Some Synth1 parameters are not numbers in the usual sense but **enumerations**. For example, the oscillator waveform type: 0 = sine, 1 = triangle, 2 = sawtooth, 3 = square. The numbers 0, 1, 2, 3 are **labels**, not quantities. It would be incorrect to say that "square" (3) is "three times sine" (0).

The network must handle these parameters differently from ordinary continuous values (volume, attack time, etc.). The solution is **one-hot encoding**, explained in section 9.

---

## 7. Reading Preset Files — Lines 151–186

### Function `read_sy1_file` (Lines 151–173)

```python
def read_sy1_file(filepath: str) -> dict | None:
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            lines = f.read().split("\n")
        if len(lines) < 4:
            return None
        header = lines[:3]
        preset: dict = {}
        preset["name"] = header[0].strip()
        preset["color"] = (
            header[1].split("=", 1)[1].strip() if "=" in header[1] else "red"
        )
        preset["ver"] = (
            header[2].split("=", 1)[1].strip() if "=" in header[2] else "106"
        )
        for line in lines[3:]:
            line = line.strip()
            if "," in line:
                param_id, value = line.split(",", 1)
                preset[param_id.strip()] = value.strip()
        return preset
    except Exception:
        return None
```

The `.sy1` format is a plain-text file. The first three lines form a header:
```
Bass Wobble
color=blue
ver=106
```
Everything after that consists of lines like `0,127` (parameter ID, value).

**Line-by-line breakdown:**

- `open(filepath, encoding="utf-8", errors="ignore")` — opens the file. The `errors="ignore"` flag means: if a byte cannot be decoded as UTF-8, skip it silently. This protects against crashes caused by unusual encodings.
- `f.read().split("\n")` — reads the entire file as a single string, then splits it on newline characters.
- `if len(lines) < 4: return None` — a validity check: fewer than four lines (3 header + at least one parameter) means the file is malformed.
- `header[0].strip()` — `.strip()` removes leading and trailing whitespace and newline characters.
- `header[1].split("=", 1)[1]` — splits `color=blue` on the first `=` and takes the second part (`blue`). The `1` argument means "at most one split", preventing breakage on values like `key=a=b`.
- The `for line in lines[3:]` loop iterates over all lines after the header and parses `param_id,value` pairs.
- `except Exception: return None` — if anything goes wrong (corrupted file, unexpected format), the function returns `None` instead of crashing the entire run.

### Function `load_presets` (Lines 176–186)

```python
def load_presets(presets_dir: str) -> list[dict]:
    files = glob.glob(os.path.join(presets_dir, "**", "*.sy1"), recursive=True)
    files += glob.glob(os.path.join(presets_dir, "*.sy1"))
    files = list(set(files))
    presets = [r for f in files if (r := read_sy1_file(f)) is not None]
    return presets
```

- `glob.glob(..., recursive=True)` with the `**/*.sy1` pattern — a recursive file search. `**` means "any depth of subdirectories".
- The second `glob.glob` targets the root directory itself, in case presets are stored directly there rather than in subfolders.
- `list(set(files))` — a `set` automatically removes duplicates (the same file could appear in both glob results).
- `[r for f in files if (r := read_sy1_file(f)) is not None]` — a list comprehension using the **walrus operator** (`:=`, Python 3.8+). It assigns the result of `read_sy1_file(f)` to `r` right inside the condition, avoiding a double call.

---

## 8. Building the DataFrame — Lines 192–199

```python
def build_dataframe(presets: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(presets)
    df = df.dropna(axis=1, thresh=int(len(df) * 0.5))
    col_rename_str = {str(k): v for k, v in COL_RENAME.items()}
    df = df.rename(columns={k: v for k, v in col_rename_str.items() if k in df.columns})
    return df
```

- `pd.DataFrame(presets)` — converts the list of dictionaries into a table. Each dictionary (preset) becomes a row; dictionary keys become column names. If some presets have a parameter that others lack, pandas fills the gap with `NaN` (Not a Number) — a special marker for a missing value.

- `df.dropna(axis=1, thresh=int(len(df) * 0.5))` — drops **columns** (`axis=1`) where fewer than 50% of rows (`thresh=0.5 × row count`) contain a real value. If fewer than half of the presets include a given parameter, it is considered irrelevant or optional and is removed.

  - `axis=0` would drop rows instead
  - `thresh=N` means "keep the column only if it has at least N non-null values"

- `col_rename_str` — creates a copy of `COL_RENAME` with keys cast from `int` to `str`, because pandas stores column names from the `.sy1` file as strings (`"0"`, `"1"`).

- `df.rename(columns=...)` — renames columns according to the dictionary, leaving untouched any column not present in the mapping.

**Further reading:**
- [pandas DataFrame — official docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
- [Working with missing data in pandas](https://pandas.pydata.org/docs/user_guide/missing_data.html)

---

## 9. Feature Engineering — Lines 202–251

```python
def engineer_features(df):
    ...
```

This is the most involved part of preprocessing. Let's walk through it step by step.

### Step 1: Dropping unwanted columns (Lines 206–210)

```python
drop_existing = [c for c in TO_DROP if c in df.columns]
df = df.drop(columns=drop_existing)
for meta in ["color", "ver", "pack"]:
    if meta in df.columns:
        df = df.drop(columns=[meta])
```

Columns listed in `TO_DROP` (explained in section 5) are removed, along with preset metadata (`color`, `ver`, `pack`) that does not describe the sound.

### Step 2: Converting everything to numbers (Lines 213–216)

```python
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(axis=1, thresh=int(len(df) * 0.8))
df = df.fillna(df.median(numeric_only=True))
```

- `pd.to_numeric(df[col], errors="coerce")` — attempts to parse each value as a number. Where conversion fails (e.g. the column contains text), it inserts `NaN`. The `errors="coerce"` flag means "on failure, substitute NaN" rather than raising an exception.
- After conversion, columns with fewer than 80% real values are dropped again (stricter threshold for the modelling stage).
- `df.fillna(df.median())` — fills remaining `NaN` values with the **column median**. The median is preferred over the mean because it is robust to outliers. For example, a single preset with an extreme volume setting will not skew the imputed values.

### Step 3: One-hot encoding of categorical variables (Lines 219–247)

```python
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
values = df[col].values.reshape(-1, 1)
enc.fit(values)
encoders[col] = enc

encoded = enc.transform(values)
encoded_df = pd.DataFrame(
    encoded,
    columns=[f"{col}-{i}" for i in range(encoded.shape[1])],
    index=df.index,
)
```

**Why can't we simply leave the numbers for categories?**

Suppose oscillator 1 waveform is encoded as: 0 = sine, 1 = triangle, 2 = sawtooth, 3 = square.  
If these numbers are fed directly to the network, it will infer that "square" (3) is "three times sine" (0), or that "triangle" lies midway between "sine" and "sawtooth". That is mathematically meaningless.

**One-hot encoding** replaces a single number with a vector of zeros containing a single `1` at the relevant position:

```
sine     → [1, 0, 0, 0]
triangle → [0, 1, 0, 0]
sawtooth → [0, 0, 1, 0]
square   → [0, 0, 0, 1]
```

All categories are now equidistant — the network makes no false arithmetic assumptions.

- `enc.fit(values)` — the encoder inspects all unique values in the column and memorises them.
- `enc.transform(values)` — applies the encoding.
- `reshape(-1, 1)` — scikit-learn expects data in shape `(N, 1)` (N rows, 1 column). `-1` tells numpy: "infer this dimension automatically."
- `f"{col}-{i}"` — new columns are named, for example, `osc1 shape-0`, `osc1 shape-1`, …
- `encoders[col] = enc` — the encoder is saved so that the same mapping can be applied in reverse during preset generation (decoding the network output back to a real parameter value).

**Further reading:**
- [One-Hot Encoding explained — Machine Learning Mastery](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
- [scikit-learn OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

### Step 4: Removing duplicates (Line 247)

```python
df = df.drop_duplicates()
```

Duplicate presets introduce bias — the network learns "this combination of parameters appears more often" and over-generates it. Removing duplicates makes the training distribution more uniform.

---

## 10. Data Normalization — Lines 254–260

```python
def normalize(df):
    min_vals = df.min()
    max_vals = df.max()
    range_vals = (max_vals - min_vals).replace(0, 1)
    df_norm = 2.0 * ((df - min_vals) / range_vals) - 1.0
    return df_norm, min_vals, max_vals
```

Synthesizer parameters live in very different numerical ranges:
- Amplifier volume: 0 to 127
- Attack time: 0 to 1
- Delay time: 0 to 100

Feeding these raw values into the network allows large-magnitude parameters to dominate. Neural networks train poorly on data at wildly different scales.

**The normalisation formula** maps all values to the range `[-1, 1]`:

```
x_norm = 2 × (x − min) / (max − min) − 1
```

- When `x = min`: `2 × 0 / range − 1 = −1`
- When `x = max`: `2 × 1 − 1 = 1`
- When `x = midpoint`: ≈ 0

The `[-1, 1]` range is chosen deliberately — it matches the output range of the `Tanh` activation function at the end of the generator (see section 11).

`.replace(0, 1)` guards against division by zero: if a column has `min == max` (all values are identical), the range is 0. Replacing it with 1 avoids `inf`.

The function also returns `min_vals` and `max_vals` for **denormalisation** during preset generation (translating the network's `[-1, 1]` output back to real synthesizer values).

**Further reading:**
- [Why normalize data for neural networks](https://www.jeremyjordan.me/batch-normalization/)
- [Feature Scaling — Wikipedia](https://en.wikipedia.org/wiki/Feature_scaling)

---

## 11. Neural Network Architecture — Lines 266–315

### Class `Generator` (Lines 266–287)

```python
class Generator(nn.Module):
    def __init__(self, latent_dim: int, data_size: int):
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, data_size),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)
```

**What does the generator do?**

The generator receives a **noise vector** `z` (random numbers) of size `latent_dim` (default 10) and expands it into a vector of synthesizer parameters of size `data_size` (~60 numbers after encoding).

Think of it as unfolding a seed into a fully grown plant — structured, meaningful data emerging from randomness.

**Architecture** — four blocks, each widening the vector:
```
10 → 128 → 256 → 512 → 1024 → data_size
```

**What is `nn.Linear`?**  
A fully-connected (dense) layer. Every neuron in the input is connected to every neuron in the output. Mathematically: `output = W × input + b`, where `W` is a weight matrix and `b` is a bias vector. These numbers — `W` and `b` — are exactly what gets updated during training.

**What is `nn.BatchNorm1d`?**  
Batch Normalization normalises activations within a batch. This stabilises training: without it, values can "explode" (grow very large) or "vanish" (collapse towards zero) as data passes through many layers.  
→ [Batch Normalization: Accelerating Deep Network Training (original paper)](https://arxiv.org/abs/1502.03167)

**What is `nn.LeakyReLU`?**  
An activation function. Without non-linear activations, a stack of linear layers is mathematically equivalent to a single layer — the network could never learn anything complex.

`LeakyReLU(0.2)` works as follows:
```
if x > 0: output = x
if x ≤ 0: output = 0.2 × x
```

This is an improvement over plain ReLU (`max(0, x)`), which completely "kills" neurons with negative activations (the "dying ReLU" problem).  
→ [Activation Functions explained](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)

**`nn.Tanh` at the end:**  
The hyperbolic tangent squashes any number into `(-1, 1)`. This perfectly matches the data normalisation range of `[-1, 1]` — the generator's output is directly interpreted as normalised preset parameters without any extra conversion step.

**`nn.Sequential`:**  
A container that applies layers one after another. The `*block(...)` operator "unpacks" a list of layers into Sequential's arguments.

**`def forward(self, z)`:**  
The mandatory PyTorch method defining the forward pass: how data flows through the network. When you call `generator(z)`, PyTorch automatically invokes `forward`.

---

### Class `Discriminator` (Lines 290–304)

```python
class Discriminator(nn.Module):
    def __init__(self, data_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(data_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )
```

The discriminator works in the reverse direction: it receives a preset parameter vector and "compresses" it down to a single number:
```
data_size → 256 → 128 → 64 → 1
```

In a standard GAN this output would be a probability in `[0, 1]`. In WGAN-GP (the variant used here) the discriminator is called a **critic** and outputs an arbitrary real number — the higher the number, the more "real" the preset is judged to be.

Notice there is **no BatchNorm** here. This is intentional — WGAN-GP requires the critic to have no BatchNorm, because batch normalisation would violate the Lipschitz condition that the gradient penalty is designed to enforce.

---

### Class `PresetDataset` (Lines 307–315)

```python
class PresetDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df.values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.FloatTensor(self.data[idx])
```

This is an adapter between a pandas DataFrame and PyTorch. `DataLoader` (which handles batching) requires a `Dataset` object with two methods:
- `__len__` — how many samples exist in total.
- `__getitem__` — return the sample at index `idx`.

- `df.values` — extracts data from the DataFrame as a numpy array.
- `.astype(np.float32)` — converts to 32-bit floating-point numbers. This is the standard dtype for neural networks: `float32` uses half the memory of `float64` and is processed faster by GPUs.
- `torch.FloatTensor(...)` — wraps the numpy array in a PyTorch tensor.

**Further reading:**
- [Creating custom datasets in PyTorch](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

---

## 12. Gradient Penalty — Lines 321–340

```python
def compute_gradient_penalty(D, real, fake, device):
    alpha = torch.rand(real.size(0), 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interpolates)
    ones = torch.ones(real.size(0), 1, device=device, requires_grad=False)
    grads = autograd.grad(
        outputs=d_interp,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()
```

This is the heart of **WGAN-GP** (Wasserstein GAN with Gradient Penalty). It is the most mathematically involved part, so we break it down carefully.

**The problem with vanilla GANs:**  
In a classic GAN the discriminator can become "too good" and start returning values like `0.0000001` for fakes and `0.9999999` for real data — the gradient becomes nearly zero and the generator stops learning. This is known as the **vanishing gradient** or **mode collapse** problem.

**The WGAN solution:**  
Instead of probabilities, the critic outputs arbitrary real numbers, and the loss function measures the **Wasserstein distance** — the "transportation distance" between the distributions of real and generated data. This is far more stable.

**The WGAN problem:**  
The Wasserstein distance is theoretically valid only if the critic satisfies the **Lipschitz condition**: its gradient must not exceed 1 anywhere. The original WGAN enforced this by brutally clipping weights — a poor approach.

**The WGAN-GP solution:**  
Instead of weight clipping, a **penalty** is added to the loss: if the gradient norm of the critic evaluated on points interpolated between real and fake data deviates from 1, the deviation is penalised.

**Code breakdown:**

1. `alpha = torch.rand(batch_size, 1)` — a random number between 0 and 1 for each sample in the batch.

2. `interpolates = alpha * real + (1 - alpha) * fake` — an **interpolation** between a real and a fake preset. At `alpha=1` you get the real preset; at `alpha=0` the fake; at `alpha=0.5` a blend of both.

3. `.requires_grad_(True)` — tells PyTorch to track gradients for this tensor (not tracked by default).

4. `d_interp = D(interpolates)` — runs the interpolated samples through the critic.

5. `autograd.grad(...)` — computes the gradient of the critic's output with respect to its input (the interpolated points). This is `∂D/∂interpolates`.
   - `create_graph=True` — necessary so that we can later differentiate through the penalty itself (gradient of a gradient).
   - `retain_graph=True` — do not discard the computation graph after taking the gradient.

6. `grads.view(grads.size(0), -1)` — flattens the gradient tensor into a vector for each sample in the batch.

7. `grads.norm(2, dim=1)` — the L2 norm (Euclidean length) of the gradient vector for each sample.

8. `((norm - 1) ** 2).mean()` — the penalty: how far the norm deviates from 1. Ideally this should be 0.

**Further reading:**
- [Original WGAN-GP paper](https://arxiv.org/abs/1704.00028)
- [Wasserstein GAN explained simply](https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490)

---

## 13. Training Loop — Lines 343–432

```python
def train(df_norm, output_dir, n_epochs, batch_size, latent_dim, lr, n_critic, lambda_gp, sample_interval):
```

This is the main training function. Let's break it down section by section.

### Device Setup and Directory Creation (Lines 354–365)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.85)
```

**GPU vs CPU:**  
Neural networks train 10–100× faster on a GPU (graphics card) than on a CPU. The reason is that a GPU consists of thousands of small cores optimised for the parallel matrix operations that neural networks require.

`torch.cuda.is_available()` — checks whether a CUDA-capable GPU (NVIDIA) is present. If so, it uses the GPU; otherwise it falls back to the CPU.

`set_per_process_memory_fraction(0.85)` — caps GPU memory usage at 85%. This prevents `CUDA out of memory` errors on cards with limited VRAM.

### DataLoader (Lines 367–375)

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
)
```

- `batch_size=64` — the network trains not on one sample at a time, nor on all at once, but on **batches** (mini-subsets). Size 64 is a typical balance between speed and training stability.
- `shuffle=True` — reshuffles the data before each epoch, preventing the network from memorising sample order.
- `drop_last=True` — if the last batch is smaller than `batch_size` (the dataset does not divide evenly), it is discarded. BatchNorm works poorly on very small batches.
- `num_workers=0` — number of parallel worker processes for data loading. 0 means "run in the main thread." On Windows, higher values can trigger multiprocessing issues.
- `pin_memory=True` — stores data in "pinned" (page-locked) CPU memory, which accelerates transfer to the GPU.

**What is an epoch?**  
One epoch = one full pass through the entire dataset. With 1,000 presets and `batch_size=64`, one epoch contains `1000 // 64 = 15` batches (iterations).

### Optimisers (Lines 380–381)

```python
opt_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
```

An **optimiser** is the algorithm that updates the network's weights based on computed gradients. **Adam** (Adaptive Moment Estimation) is one of the most widely used algorithms; it adapts the learning rate per parameter.

- `lr=0.0002` — the **learning rate**, controlling how much the weights change after each batch. Too large → unstable training. Too small → painfully slow convergence.
- `betas=(0.5, 0.999)` — exponential decay coefficients for the first and second moment estimates. `beta1=0.5` (instead of the standard 0.9) is standard practice for GANs, recommended in the DCGAN paper.

**Further reading:**
- [Adam optimizer — original paper](https://arxiv.org/abs/1412.6980)
- [An overview of gradient descent optimisation algorithms](https://www.ruder.io/optimizing-gradient-descent/)

### The Main Loop (Lines 389–428)

```python
for epoch in range(n_epochs):
    for i, real_imgs in enumerate(dataloader):
```

The outer loop iterates over epochs; the inner loop over batches within each epoch.

#### Discriminator / Critic Step

```python
opt_D.zero_grad()
z = torch.randn(real_imgs.size(0), latent_dim, device=device)
fake_imgs = generator(z).detach()

real_val = discriminator(real_imgs)
fake_val = discriminator(fake_imgs)
gp = compute_gradient_penalty(discriminator, real_imgs, fake_imgs, device)
d_loss = -real_val.mean() + fake_val.mean() + lambda_gp * gp
d_loss.backward()
opt_D.step()
```

1. `opt_D.zero_grad()` — zeroes out accumulated gradients. PyTorch **adds** gradients by default rather than overwriting them — if you skip this step, gradients from previous batches will contaminate the current update.

2. `z = torch.randn(...)` — samples a **noise vector** from the standard normal distribution N(0, 1). Each number is drawn from a Gaussian (bell-curve) distribution.

3. `fake_imgs = generator(z).detach()` — generates fake presets. `.detach()` disconnects the result from the generator's computation graph, because we do not want to update the generator in this step.

4. `real_val` and `fake_val` — the critic's scores for real and fake presets respectively.

5. **WGAN-GP loss:**
   ```
   L_D = −E[D(real)] + E[D(fake)] + λ × gradient_penalty
   ```
   The critic wants to maximise `D(real) − D(fake)`. The loss is the negation of that difference (because we always **minimise** the loss).

6. `d_loss.backward()` — **backpropagation**. PyTorch traverses the computation graph in reverse and computes how each weight's change affects the loss.

7. `opt_D.step()` — applies the weight update: `w := w − lr × gradient`.

**What is backpropagation?**  
An algorithm for computing gradients in a neural network using the chain rule of calculus: `∂L/∂w = ∂L/∂y × ∂y/∂w`, applied recursively from the output layer back to the input.  
→ [Backpropagation — 3Blue1Brown (video)](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

#### Generator Step (Lines 406–421)

```python
if i % n_critic == 0:
    opt_G.zero_grad()
    z = torch.randn(real_imgs.size(0), latent_dim, device=device)
    fake_imgs = generator(z)
    g_loss = -discriminator(fake_imgs).mean()
    g_loss.backward()
    opt_G.step()
```

- `if i % n_critic == 0` — the generator is updated **less frequently** than the critic: once every `n_critic=5` critic iterations. This is a key feature of WGAN: the critic needs more steps to develop an accurate assessment before the generator acts on its feedback. Equal update rates leave the critic perpetually underdeveloped.

- `g_loss = -discriminator(fake_imgs).mean()` — the generator's loss function: it wants the critic to **score its fakes highly**. The generator is updated to maximise `D(fake)`, which means minimising `−D(fake)`.

- This time there is **no `.detach()`** — gradients must flow all the way back through the discriminator into the generator.

### Checkpoints (Lines 425–428)

```python
if epoch > 0 and epoch % 500 == 0:
    ckpt = os.path.join(checkpoints_dir, f"generator_{epoch}.pt")
    torch.save(generator.state_dict(), ckpt)
```

Every 500 epochs a **checkpoint** is saved — a snapshot of the generator's weights. This allows:
- Resuming training after an interruption.
- Comparing quality at different training stages.
- Rolling back to an earlier version if quality degrades later (overfitting).

`generator.state_dict()` returns a dictionary `{parameter_name: weight_tensor}`. This is the standard PyTorch format for saving models.

---

## 14. ONNX Export — Lines 438–454

```python
def export_onnx(generator, latent_dim, output_dir):
    generator.eval()
    dummy = torch.randn(1, latent_dim).to(next(generator.parameters()).device)
    path = os.path.join(output_dir, "generator.onnx")
    torch.onnx.export(
        generator,
        dummy,
        path,
        input_names=["noise"],
        output_names=["preset"],
        dynamic_axes={"noise": {0: "batch"}, "preset": {0: "batch"}},
        opset_version=13,
        do_constant_folding=True,
    )
```

**What is ONNX?**  
ONNX (Open Neural Network Exchange) is a **universal format** for neural networks, independent of the training framework. A PyTorch model exported to `.onnx` can be loaded in C++, Rust, Java — anywhere — without any dependency on Python or PyTorch.

The GUI application written in Rust uses the ONNX Runtime to run the trained generator.

**Code breakdown:**

- `generator.eval()` — switches the model to **inference mode**. BatchNorm (and Dropout, if present) behave differently during inference than during training. Forgetting this call is a common bug.

- `dummy = torch.randn(1, latent_dim)` — a "dummy" input is needed for tracing: PyTorch runs real data through the model, records every operation, and that record becomes the ONNX graph.

- `.to(next(generator.parameters()).device)` — ensures the tensor is on the same device (GPU/CPU) as the model weights.

- `input_names=["noise"]` — labels the input tensor in the ONNX graph (for convenience in the GUI code).
- `output_names=["preset"]` — labels the output tensor.
- `dynamic_axes={"noise": {0: "batch"}}` — tells ONNX that the first dimension (batch size) can be arbitrary. Without this, the model would be compiled for `batch_size=1` only.
- `opset_version=13` — the version of the ONNX operator set. Version 13 is supported by most modern ONNX runtimes.
- `do_constant_folding=True` — optimisation: pre-computes constant sub-expressions in the graph.

**Further reading:**
- [ONNX official site](https://onnx.ai/)
- [Exporting a PyTorch model to ONNX](https://pytorch.org/docs/stable/onnx.html)

---

## 15. Saving Normalization Metadata — Lines 460–515

```python
def save_normalization(df_reduced, min_vals, max_vals, cat_vars, encoders, col_names, latent_dim, output_dir):
```

After training, the generator outputs numbers in `[-1, 1]`. To convert them back into real synthesizer parameters, the GUI must know:
- For each continuous parameter: what does `−1` correspond to, and what does `1` correspond to (the original `min` and `max`).
- For categorical parameters: which category label maps to which numeric ID in the synthesizer.

This function writes all of that information to `normalization.json`.

```python
continuous_stats[col] = {
    "col_idx": idx,
    "min": float(min_vals[col]),
    "max": float(max_vals[col]),
    "param_id": name_to_id.get(col, col),
}
```

For each continuous column:
- `col_idx` — the position in the output vector (so the GUI knows which generated number corresponds to which parameter).
- `min`, `max` — for denormalisation: `x_real = (x_norm + 1) / 2 × (max − min) + min`.
- `param_id` — the numeric Synth1 parameter ID to set.

```python
categorical_encodings[col] = {
    "start_idx": start_idx,
    "num_classes": len(categories),
    "categories": [int(c) for c in categories],
    "param_id": name_to_id.get(col, col),
}
```

For categorical parameters:
- `start_idx` — where the one-hot block begins in the output tensor.
- `num_classes` — the length of the one-hot vector.
- `categories` — the list of real values: `[0, 1, 2, 3]` for 4 waveform types.

The GUI takes the `argmax` (index of the maximum value) within the one-hot block and looks up which parameter ID that corresponds to.

---

## 16. Entry Point main() — Lines 521–602

```python
def main() -> None:
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--presets-dir", required=True, ...)
    parser.add_argument("--output-dir", default="./model", ...)
    parser.add_argument("--epochs", type=int, default=20000, ...)
    ...
    args = parser.parse_args()
```

### Command-Line Arguments

`argparse` is Python's standard module for parsing command-line arguments. Here are the parameters and their meaning:

| Argument | Default | Purpose |
|---|---|---|
| `--presets-dir` | required | Folder containing `.sy1` files |
| `--output-dir` | `./model` | Where to save the trained model |
| `--epochs` | 20000 | Number of full passes over the dataset |
| `--batch-size` | 64 | Mini-batch size |
| `--latent-dim` | 10 | Size of the noise vector fed to the generator |
| `--lr` | 0.0002 | Learning rate |
| `--n-critic` | 5 | Critic update steps per generator step |
| `--lambda-gp` | 10.0 | Weight of the gradient penalty |
| `--sample-interval` | 400 | How often to print a log line (in batches) |

### Execution Order (Lines 549–598)

```python
# 1. Load
presets = load_presets(args.presets_dir)
if len(presets) < args.batch_size:
    sys.exit(1)

# 2. Build DataFrame
df = build_dataframe(presets)

# 3. Feature engineering
df_reduced, cat_vars, encoders = engineer_features(df)

# 4. Normalize
df_norm, min_vals, max_vals = normalize(df_reduced)
col_names = list(df_norm.columns)

# 5. Train
generator = train(df_norm, args.output_dir, ...)

# 6. Export
export_onnx(generator, args.latent_dim, args.output_dir)
save_normalization(...)
```

The guard `if len(presets) < args.batch_size` is an important safety check. If the dataset is smaller than a single batch, the network cannot train: `DataLoader` with `drop_last=True` will simply discard the only incomplete batch, leaving the training loop with nothing to iterate over.

### Final Lines (Lines 601–602)

```python
if __name__ == "__main__":
    main()
```

This is the standard Python idiom: the code in this block executes **only when the file is run directly** (`python train.py`), not when it is imported as a module (`import train`). This allows other scripts to import and call individual functions from `train.py` without triggering a full training run.

---

## 17. Full Data Flow Diagram

```
.sy1 files on disk
    │
    ▼  read_sy1_file()
list of dicts {param_id: value, ...}
    │
    ▼  build_dataframe()
DataFrame with numeric IDs renamed to human-readable column names
    │
    ▼  engineer_features()
    ├── drop noisy columns (TO_DROP)
    ├── convert all values to float
    ├── one-hot encode categorical parameters
    └── remove duplicate rows
DataFrame with engineered features
    │
    ▼  normalize()
DataFrame in range [-1, 1]  +  min/max saved for later denormalisation
    │
    ▼  train()
    │
    ├── each batch:
    │   ├── 5× critic step:
    │   │   ├── sample noise z → G(z) = fake_preset
    │   │   ├── score real:  D(real_preset)
    │   │   ├── score fake:  D(fake_preset)
    │   │   ├── compute gradient penalty
    │   │   └── update critic weights
    │   └── 1× generator step:
    │       ├── sample noise z → G(z)
    │       ├── critic scores: D(G(z))
    │       └── update generator weights so D(G(z)) ↑
    │
    └── every 500 epochs: save checkpoint
    │
    ▼  export_onnx()
generator.onnx  (framework-independent format for the Rust GUI)
    │
    ▼  save_normalization()
normalization.json  (min/max + category mappings for decoding)
    │
    ▼
GUI feeds a random noise vector → ONNX Runtime → denormalise → Synth1 preset
```

---

## Key Concepts for Further Study

| Concept | What it is | Resource |
|---|---|---|
| Neural network | Universal function approximator | [3Blue1Brown: Neural Networks (video series)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) |
| Backpropagation | Algorithm for computing gradients | [Yes, you should understand backprop — Andrej Karpathy](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) |
| GAN | Generative Adversarial Network | [GAN — Ian Goodfellow (original paper)](https://arxiv.org/abs/1406.2661) |
| WGAN-GP | GAN with Wasserstein distance + gradient penalty | [WGAN-GP paper](https://arxiv.org/abs/1704.00028) |
| Batch Normalization | Normalising activations within a batch | [BN paper](https://arxiv.org/abs/1502.03167) |
| Adam optimizer | Adaptive learning-rate optimiser | [Adam paper](https://arxiv.org/abs/1412.6980) |
| One-hot encoding | Encoding categories as binary vectors | [ML Mastery](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/) |
| ONNX | Portable neural network format | [onnx.ai](https://onnx.ai/) |
| PyTorch | Neural network framework | [PyTorch Tutorials](https://pytorch.org/tutorials/) |
