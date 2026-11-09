---
name: neural-network
description: Rules for writing, training and evaluating ReverbNet models (1D CNN in TensorFlow/Keras on per-band TAE) - architecture, leak-free group splits, target normalization, losses, callbacks, seeds, checkpoints, batched prediction, metrics and shortcut detection. Use when touching code/model.py, the training loop in run.py, or any model, training-dataset or inference code.
---

# Neural networks in ReverbNet

## Current state (baseline v0)

- `code/model.py::model()`: per-band 1D CNN. Input `(200, 1)` (TAE), 4 conv layers
  (`filters=[32,18,8,4]`, `kernel_size=[10,5,5,5]`, ReLU), MaxPool + BatchNorm, Dropout 0.4,
  `Flatten` → linear `Dense(4)`. ~4.6k parameters. Adam lr=1e-3, MSE.
- 7 independent models (one per band), targets `[T30, C50, C80, D50]` divided by their p95.
- `EarlyStopping(patience=100, restore_best_weights=True)`, 500 epochs, `validation_split=0.1`.
- TF 2.16.1 on native Windows → **CPU only** (TRAIN-06).
- The Spanish layer names (`'Audio de entrada'`, `'capa1_conv'`…) are kept so the checkpoints
  in `results/exp1..4` can still be loaded. New architectures may use English names, but the
  baseline architecture is not touched.

## Data rules for training

1. **Group splits, never per-record splits** (BUG-04, implemented in `code/splits.py`):
   - Real RIRs: by **room** (great_hall / octagon / classroom). Development split holds out
     `classroom`; milestone baselines run leave-one-room-out folds.
   - Augmented RIRs: **training only** (DATA-11). Test RIRs are never augmented; the test
     set holds original real and synthetic RIRs only.
   - Synthetic RIRs: per file (20 % to test).
   - Speech: by **speaker** (`F4`, `M4`, `M9` in test).
   - Validation: `validation_mask` by source RIR (same decision in every band), passed as
     `validation_data`. Never use Keras' `validation_split`.
   - Any new split logic goes into `code/splits.py` with a disjointness test.
2. **Target normalization** with statistics from **train only** (`normalize_descriptors(y_train,
   y_val, y_test)`), saved with the results and reused at inference. Z-score is TRAIN-03.
3. Model input produced by **the same function** used at inference (`dsp` skill).
4. Physically linked descriptors: D50 is derived from C50 (DATA-01); check C80 ≥ C50.

## Architecture

- Builders are pure functions parameterized by the config (`filters`, `kernel_size`…); no
  magic constants (`200` comes from `duration × TAE rate`).
- Prefer `GlobalAveragePooling1D` over `Flatten` in new models (variable-length input, fewer
  parameters).
- For multi-band (TRAIN-01): input `(T, n_bands)` and output `(n_bands, n_targets)`.
- Every new architecture is compared against the baseline in an ablation with the **same data
  and splits** (one change at a time).

## Training

- Seeds: `seed_training(training_seed(seed, band, replica))` before building each band's model
  (TRAIN-09). Reproducible on CPU as measured; determinism on other devices is not guaranteed.
- Loss: MSE (baseline). Alternatives: Huber; Gaussian NLL with a (μ, log σ²) output for
  uncertainty (TRAIN-03). With several targets, mind their relative scale (weights).
- Optimizer and LR: AdamW + `ReduceLROnPlateau` or cosine decay. In v0 the best epoch is almost
  always close to 500: check convergence before concluding anything (TRAIN-04).
- Minimal callbacks: EarlyStopping on the group-split `val_loss`, `ModelCheckpoint` of the best
  model, `CSVLogger` with the full history.
- Large batches (1024) are fine on CPU; record the time per epoch.

## Saving artifacts

For every trained model, in `results/exp<N>/`:
- weights (`.weights.h5` or `.keras`),
- the full config used and the **commit** (`git rev-parse HEAD`),
- normalization statistics,
- training history,
- test predictions **unrounded** and in physical units, with per-sample metadata (RIR name,
  real/synthetic/augmented type, room, speaker, SNR).

Never overwrite another experiment's results.

## Inference and evaluation

- `model.predict(X, batch_size=1024)`; never a one-sample-at-a-time loop (TRAIN-07).
- Do not round predictions or errors.
- Metrics in physical units: MAE, RMSE, bias, Spearman and Pearson, % within JND; per band,
  per descriptor, per RIR type and per room (`data-analysis` skill).
- Besides the synthetic test set, always evaluate on the **field recordings**
  (`data/Measurements/room*`) against the REW measurements.

## Detecting shortcuts (shortcut learning)

Before trusting an improvement, ask whether the model could have learned something that does
not exist in a real recording. Checks:
- Does the input carry information about the RIR's length or position? (BUG-01 was exactly this.)
- Performance per RIR type: if it only improves on synthetic RIRs, be suspicious.
- Perturb the input (shift the window by a few hundred ms, change the speech crop): the
  prediction should not change much.
- Compare against a trivial baseline (predict the train mean): ρ and MAE must be clearly better.

## Framework decision

Migrating to PyTorch or Keras 3 (torch backend) is under discussion (TRAIN-06). Until it is
decided and recorded in `docs/improvements/log.md`, new code stays in TF/Keras and keeps the
lazy `tensorflow` import inside functions.
