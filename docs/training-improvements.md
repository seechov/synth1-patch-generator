# Generator Training Methodology — Improvement Plan

This document records the improvement plan for `training/train.py`, so work can
proceed sequentially and nothing gets lost.

> Status: ✅ — done, 🔜 — queued, ⏸ — deferred/under discussion.

---

## Priorities

| Priority | Improvement | Effect | Status |
|----------|-------------|--------|--------|
| 🔴 High | Split generator heads for continuous/categorical | Clean categories, meaningful presets | ✅ (1a) |
| 🔴 High | Generator EMA + EMA export | Stable quality, less "broken" output | ✅ |
| 🔴 High | Fix `batches_done`, seed, real dataset check | Reproducibility, correct logs | ✅ |
| 🟡 Medium | Separate `lr` for G/D, spectral norm, feature noise | WGAN-GP stability | ✅ |
| 🟡 Medium | Progressive batch size, increase `latent_dim` | Better GP estimate, diversity | ✅ |
| 🟢 Low | Train/val monitoring, log dropped features | Collapse diagnostics | ✅ |

---

## Details

### 1a. Split handling of continuous and categorical outputs ✅

**Problem.** A single `normalize()` scaled *all* features, including one-hot
encoded categories, into `[-1, 1]`, and the generator ended with a single
`Tanh()`. Categorical branches were "blurred", and the model did not produce
clean one-hot outputs.

**Solution (implemented).**
- `Generator` is split into a shared trunk + `cont_head` (`Tanh`) and
  `cat_heads` (`softmax` per categorical variable).
- `normalize()` scales only continuous columns; one-hot stay `0/1`.
- `engineer_features()` returns the set of one-hot columns.
- `save_normalization()` writes `min`/`max` only for continuous features;
  `continuous_stats` and `categorical_encodings` are explicitly separated.
- The `normalization.json` format and ONNX interface remain GUI-compatible.

---

### Generator EMA + EMA export ✅

Accumulate an exponential moving average (EMA) of the generator weights and
export the **EMA version** to ONNX rather than the last weights. Smooths quality
spikes during final export.

**Tasks:**
- Create an EMA copy `gen_ema` in `train()`.
- Update after each generator step: `ema_avg = decay * ema_avg + (1 - decay) * weights`.
- Export the EMA in `export_onnx()`.
- Store the EMA weights in checkpoints.

---

### Separate `lr`, spectral norm, and feature noise ✅

**Tasks:**
- Add `--lr-g` and `--lr-d` (previously a shared `--lr`).
- Optional spectral normalization of the discriminator
  (`torch.nn.utils.spectral_norm`), flag `--spectral-norm`.
- Optional input noise for the discriminator (feature noise),
  flag `--d-noise-std`. Reduces mode collapse.

---

### Progressive batch size and `latent_dim` ✅

**Tasks:**
- Batch = `min(batch_size, len(df) // 4)`, log a warning.
- Raise the default `latent_dim` from `10` to `32` for diversity
  (already parameterized; only the default changed).

---

### Train/val monitoring and diagnostics ✅

**Tasks:**
- Periodically compute a diversity metric (e.g. mean L2 distance between
  generated and nearest real presets) for early mode-collapse detection.
- Log how many features/rows were dropped in `engineer_features()`.

---

## Context

- Input data — `.sy1` files (the `presets/` folder), parameter names in
  `COL_RENAME`, categorical variables in `CATEGORICAL_VARS`, dropped columns in
  `TO_DROP`.
- The GUI (`gui/src/main.rs`) reads `generator.onnx` + `normalization.json`;
  continuous parameters are denormalized from `[-1, 1]`, categoricals are
  recovered via `argmax` over a slice (`start_idx`, `num_classes`).
