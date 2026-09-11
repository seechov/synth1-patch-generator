#!/usr/bin/env python3
"""
Synth1GAN — WGAN-GP trainer for Synth1 VST presets.

Usage:
    python train.py --presets-dir C:/path/to/presets --output-dir ./model
    python train.py --presets-dir ./synth_patches/all --output-dir ./model --epochs 5000

The script will:
  1. Recursively scan --presets-dir for *.sy1 files
  2. Parse and preprocess the preset data
  3. Train a WGAN-GP neural network
  4. Export the generator as generator.onnx
  5. Save normalization.json (needed by the GUI app)
"""

import argparse
import csv
import glob
import json
import os
import random
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

# ─── Synth1 parameter definitions ────────────────────────────────────────────

COL_RENAME = {
    0: "osc1 shape",
    1: "osc2 shape",
    2: "osc2 pitch",
    3: "osc2 tune",
    4: "osc2 kbd track",
    5: "osc mix",
    6: "osc sync",
    7: "osc ring modulation",
    8: "osc pulse width",
    9: "osc key shift",
    10: "osc mod env on/off",
    11: "osc mod env amount",
    12: "osc mod env attack",
    13: "osc p.env decay",
    14: "filter type",
    15: "filter attack",
    16: "filter decay",
    17: "filter sustain",
    18: "filter release",
    19: "filter freq",
    20: "filter resonance",
    21: "filter amount",
    22: "filter kbd track",
    23: "filter distortion",
    24: "filter velocity switch",
    25: "amp attack",
    26: "amp decay",
    27: "amp sustain",
    28: "amp release",
    29: "amp gain",
    30: "amp velocity sens.",
    31: "arp type",
    32: "arp range",
    33: "arp beat",
    34: "arp gate",
    35: "delay time",
    36: "delay feedback",
    37: "delay level",
    38: "play mode type",
    39: "play mode portament",
    40: "pitch bend range",
    41: "lfo1 destination",
    42: "lfo1 type",
    43: "lfo1 speed",
    44: "lfo1 depth",
    45: "osc1 fm modulation",
    46: "lfo2 destination",
    47: "lfo2 type",
    48: "lfo2 speed",
    49: "lfo2 depth",
    50: "wheel lfo1 depth sens.",
    51: "wheel lfo1 speed sens.",
    52: "chorus delay time",
    53: "chorus depth",
    54: "chorus rate",
    55: "chorus feedback",
    56: "chorus level",
    57: "lfo1 on/off",
    58: "lfo2 on/off",
    59: "arp. on/off",
    60: "equalizer tone",
    61: "equalizer freq.",
    62: "equalizer level",
    63: "equalizer Q",
    64: "chorus type",
    65: "delay on/off",
    66: "chorus on/off",
    67: "lfo1 tempo sync",
    68: "lfo1 key sync",
    69: "lfo2 tempo sync",
    70: "lfo2 key sync",
    71: "osc mod env dest.",
    72: "osc1,2 tune",
    73: "unison mode",
    74: "portament auto mode",
}

# Columns dropped from training (arpeggiator, equalizer, pitch bend — too noisy/sparse)
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

# These are treated as categorical (one-hot encoded)
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


# ─── Data loading ─────────────────────────────────────────────────────────────


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


def load_presets(presets_dir: str) -> list[dict]:
    print(f"Scanning {presets_dir} for .sy1 files...")
    # Support both flat and nested folder structures
    files = glob.glob(os.path.join(presets_dir, "**", "*.sy1"), recursive=True)
    files += glob.glob(os.path.join(presets_dir, "*.sy1"))
    files = list(set(files))
    print(f"  Found {len(files)} .sy1 files")

    presets = [r for f in files if (r := read_sy1_file(f)) is not None]
    print(f"  Successfully parsed {len(presets)} presets")
    return presets


# ─── Feature engineering ──────────────────────────────────────────────────────


def build_dataframe(presets: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(presets)
    # Drop columns where the majority of presets have no value
    df = df.dropna(axis=1, thresh=int(len(df) * 0.5))
    # Rename numeric param IDs to human-readable names
    col_rename_str = {str(k): v for k, v in COL_RENAME.items()}
    df = df.rename(columns={k: v for k, v in col_rename_str.items() if k in df.columns})
    return df


def engineer_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, OneHotEncoder], set[str]]:
    """Build the training DataFrame and return stats needed downstream.

    Returns (df, cat_vars, encoders, one_hot_cols) where one_hot_cols is the set
    of generated one-hot column names (e.g. "filter type-0").
    """
    # Drop metadata and non-useful synthesis columns
    drop_existing = [c for c in TO_DROP if c in df.columns]
    df = df.drop(columns=drop_existing)
    for meta in ["color", "ver", "pack"]:
        if meta in df.columns:
            df = df.drop(columns=[meta])

    # Convert everything to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(axis=1, thresh=int(len(df) * 0.8))
    df = df.fillna(df.median(numeric_only=True))

    # One-hot encode known categorical variables
    cat_vars: dict[str, int] = {}
    encoders: dict[str, OneHotEncoder] = {}
    one_hot_cols: set[str] = set()

    new_cols = []
    for col in CATEGORICAL_VARS:
        if col not in df.columns:
            continue
        n_unique = int(df[col].nunique())
        if n_unique < 2:
            continue
        cat_vars[col] = n_unique

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
        one_hot_cols.update(encoded_df.columns.tolist())
        new_cols.append(encoded_df)
        df = df.drop(columns=[col])

    if new_cols:
        df = pd.concat([df] + new_cols, axis=1)
    df = df.drop_duplicates()

    print(f"  Categorical columns: {list(cat_vars.keys())}")
    print(f"  Total features after encoding: {df.shape[1]}")
    return df, cat_vars, encoders, one_hot_cols


def normalize(
    df: pd.DataFrame,
    one_hot_cols: set[str],
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    """Scale continuous columns to [-1, 1]; leave one-hot columns as 0/1.

    Returns (df_out, min_vals, max_vals). min_vals/max_vals are None when there
    are no continuous columns (all features are one-hot encoded).
    """
    cont_cols = [c for c in df.columns if c not in one_hot_cols]
    if not cont_cols:
        return df.copy(), None, None

    min_vals = df[cont_cols].min()
    max_vals = df[cont_cols].max()
    range_vals = (max_vals - min_vals).replace(0, 1)  # avoid div/0
    df_cont = 2.0 * ((df[cont_cols] - min_vals) / range_vals) - 1.0

    out = df.copy()
    out[cont_cols] = df_cont
    return out, min_vals, max_vals


# ─── Model architecture ───────────────────────────────────────────────────────


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_cont: int,
        group_sizes: list[int],
        batch_norm: bool = True,
    ):
        super().__init__()

        def block(in_feat: int, out_feat: int, normalize: bool = True) -> list:
            layers: list = [nn.Linear(in_feat, out_feat)]
            if normalize and batch_norm:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Shared trunk: maps latent vector → high-level feature representation.
        self.trunk = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
        )

        # Continuous head outputs values in [-1, 1] for tanh-style denormalization.
        self.cont_head = nn.Linear(1024, n_cont) if n_cont > 0 else None

        # Categorical heads: one head per categorical variable, each emitting a
        # softmax distribution over that variable's classes.
        self.cat_heads = (
            nn.ModuleList([nn.Linear(1024, n_cls) for n_cls in group_sizes])
            if group_sizes
            else None
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.trunk(z)
        parts: list[torch.Tensor] = []

        if self.cont_head is not None:
            parts.append(torch.tanh(self.cont_head(h)))

        if self.cat_heads is not None:
            for head in self.cat_heads:
                parts.append(torch.softmax(head(h), dim=1))

        return torch.cat(parts, dim=1)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PresetDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df.values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.FloatTensor(self.data[idx])


# ─── Training ─────────────────────────────────────────────────────────────────


def compute_gradient_penalty(
    D: Discriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
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


def train(
    df_norm: pd.DataFrame,
    output_dir: str,
    n_epochs: int,
    batch_size: int,
    latent_dim: int,
    lr: float,
    n_critic: int,
    lambda_gp: float,
    sample_interval: int,
    one_hot_cols: set[str],
    cat_vars: dict[str, int],
    batch_norm: bool = True,
) -> Generator:
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85)  # ~3.5 ГБ из 4

    # Order output features: continuous columns first, then one-hot groups. The
    # generator builds exactly this layout so reconstruction is unambiguous.
    cont_cols = [c for c in df_norm.columns if c not in one_hot_cols]
    one_hot_cols_ordered = [c for c in df_norm.columns if c in one_hot_cols]
    ordered_cols = cont_cols + one_hot_cols_ordered
    df_train = df_norm[ordered_cols]

    data_size = df_train.shape[1]
    n_cont = len(cont_cols)
    n_one_hot = len(one_hot_cols_ordered)

    # Group sizes (number of classes per categorical head) in the same order as
    # one_hot_cols_ordered. cat_vars preserves CATEGORICAL_VARS order (skipping
    # variables with < 2 unique values), matching the column concat order.
    group_sizes: list[int] = []
    for col in CATEGORICAL_VARS:
        if col in cat_vars:
            group_sizes.append(cat_vars[col])

    print(
        f"  Data size: {data_size} features "
        f"({n_cont} continuous, {n_one_hot} one-hot in {len(group_sizes)} groups), "
        f"{len(df_train)} presets"
    )
    if batch_size > len(df_train):
        print(
            f"  Warning: batch_size ({batch_size}) > dataset size ({len(df_train)})"
            "; training will skip every batch (drop_last)."
        )

    dataset = PresetDataset(df_train)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    generator = Generator(latent_dim, n_cont, group_sizes, batch_norm).to(device)
    discriminator = Discriminator(data_size).to(device)

    opt_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    print(
        f"\nStarting training: {n_epochs} epochs, batch={batch_size}, latent_dim={latent_dim}"
    )
    print("─" * 70)

    batches_done = 0
    for epoch in range(n_epochs):
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(device)

            # ── Discriminator step ──
            opt_D.zero_grad()
            z = torch.randn(real_imgs.size(0), latent_dim, device=device)
            fake_imgs = generator(z).detach()

            real_val = discriminator(real_imgs)
            fake_val = discriminator(fake_imgs)
            gp = compute_gradient_penalty(discriminator, real_imgs, fake_imgs, device)
            d_loss = -real_val.mean() + fake_val.mean() + lambda_gp * gp
            d_loss.backward()
            opt_D.step()

            # ── Generator step (every n_critic batches) ──
            if i % n_critic == 0:
                opt_G.zero_grad()
                z = torch.randn(real_imgs.size(0), latent_dim, device=device)
                fake_imgs = generator(z)
                g_loss = -discriminator(fake_imgs).mean()
                g_loss.backward()
                opt_G.step()

                if batches_done % sample_interval == 0:
                    print(
                        f"[Epoch {epoch:>5}/{n_epochs}] "
                        f"[Batch {i:>4}/{len(dataloader)}] "
                        f"[D: {d_loss.item():+.4f}] "
                        f"[G: {g_loss.item():+.4f}]"
                    )
                    sys.stdout.flush()

                batches_done += 1

        if epoch > 0 and epoch % 500 == 0:
            ckpt = os.path.join(checkpoints_dir, f"generator_{epoch}.pt")
            torch.save(
                {
                    "state_dict": generator.state_dict(),
                    "batch_norm": batch_norm,
                    "n_cont": n_cont,
                    "group_sizes": group_sizes,
                },
                ckpt,
            )
            print(f"  Checkpoint saved: {ckpt}")

    print("─" * 70)
    print("Training complete.")
    return generator


# ─── ONNX export ──────────────────────────────────────────────────────────────


def export_onnx(generator: Generator, latent_dim: int, output_dir: str) -> str:
    generator.eval()
    # dummy = torch.randn(1, latent_dim)
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
    print(f"  ONNX model → {path}")
    return path


# ─── Normalization metadata ───────────────────────────────────────────────────


def save_normalization(
    df_reduced: pd.DataFrame,
    min_vals: pd.Series | None,
    max_vals: pd.Series | None,
    cat_vars: dict[str, int],
    encoders: dict[str, OneHotEncoder],
    col_names: list[str],
    latent_dim: int,
    output_dir: str,
) -> str:
    # Map human-readable name → numeric param ID
    name_to_id = {v: str(k) for k, v in COL_RENAME.items()}

    # Which columns are part of one-hot encodings
    one_hot_col_names: set[str] = set()
    for col, n in cat_vars.items():
        for i in range(n):
            one_hot_col_names.add(f"{col}-{i}")

    # Continuous columns with their index in the output vector. Only continuous
    # columns have min/max; one-hot columns are emitted directly by softmax.
    continuous_stats: dict = {}
    for idx, col in enumerate(col_names):
        if col in one_hot_col_names:
            continue
        if min_vals is None or col not in min_vals.index:
            continue
        continuous_stats[col] = {
            "col_idx": idx,
            "min": float(min_vals[col]),
            "max": float(max_vals[col]),
            "param_id": name_to_id.get(col, col),
        }

    # Categorical encodings. Since continuous columns are ordered first and
    # one-hot columns are appended in CATEGORICAL_VARS order, each group's
    # start_idx is its position within the full output vector.
    categorical_encodings: dict = {}
    for col, enc in encoders.items():
        start_idx = next(i for i, n in enumerate(col_names) if n == f"{col}-0")
        categories = enc.categories_[0].tolist()
        categorical_encodings[col] = {
            "start_idx": start_idx,
            "num_classes": len(categories),
            "categories": [int(c) for c in categories],
            "param_id": name_to_id.get(col, col),
        }

    data = {
        "latent_dim": latent_dim,
        "output_dim": len(col_names),
        "column_names": col_names,
        "continuous_stats": continuous_stats,
        "categorical_encodings": categorical_encodings,
    }

    path = os.path.join(output_dir, "normalization.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Normalization params → {path}")
    return path


# ─── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Synth1GAN and export ONNX model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--presets-dir",
        required=True,
        help="Path to folder containing .sy1 preset files (searched recursively)",
    )
    parser.add_argument("--output-dir", default="./model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20000, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=10, help="Noise vector size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument(
        "--n-critic", type=int, default=5, help="Discriminator steps per generator step"
    )
    parser.add_argument(
        "--lambda-gp", type=float, default=10.0, help="Gradient penalty weight"
    )
    parser.add_argument(
        "--sample-interval", type=int, default=400, help="Log every N batches"
    )
    parser.add_argument(
        "--batch-norm",
        action="store_true",
        default=True,
        help="Use BatchNorm in the generator trunk (default: on)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print("\n=== Synth1GAN Training ===\n")

    # 1. Load
    print("Step 1/5 — Loading presets")
    presets = load_presets(args.presets_dir)
    if len(presets) < args.batch_size:
        print(f"Error: need at least {args.batch_size} presets, found {len(presets)}")
        sys.exit(1)

    # 2. Build DataFrame
    print("\nStep 2/5 — Building dataset")
    df = build_dataframe(presets)

    # 3. Feature engineering
    print("\nStep 3/5 — Feature engineering")
    df_reduced, cat_vars, encoders, one_hot_cols = engineer_features(df)
    print(f"  Dataset shape: {df_reduced.shape}")

    # 4. Normalize (continuous only; one-hot stays 0/1)
    df_norm, min_vals, max_vals = normalize(df_reduced, one_hot_cols)
    col_names = list(df_norm.columns)

    # 5. Train
    print("\nStep 4/5 — Training WGAN-GP")
    generator = train(
        df_norm,
        args.output_dir,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        lr=args.lr,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
        sample_interval=args.sample_interval,
        one_hot_cols=one_hot_cols,
        cat_vars=cat_vars,
        batch_norm=args.batch_norm,
    )

    # 6. Export
    print("\nStep 5/5 — Exporting")
    export_onnx(generator, args.latent_dim, args.output_dir)
    save_normalization(
        df_reduced,
        min_vals,
        max_vals,
        cat_vars,
        encoders,
        col_names,
        args.latent_dim,
        args.output_dir,
    )

    print(f"\n✓ Done! Model files are in: {os.path.abspath(args.output_dir)}")
    print("  Open the GUI app and point it to this folder to generate presets.")


if __name__ == "__main__":
    main()
