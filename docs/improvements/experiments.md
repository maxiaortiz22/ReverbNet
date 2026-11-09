# Experiment registry

Each new experiment appends an entry **at the end** following the template.
Tables are generated with:

```bash
python scripts/summarize_results.py --exp <N> [<N> ...]
```

Metrics: ρ = Spearman correlation; MAE = mean absolute error in physical units
(de-normalized predictions and targets). JND reference (thesis, table 14):
T30 5 %, C50 1 dB, C80 1 dB, D50 5 %.

## Template

```markdown
### EXP-<id> — <short title>
- **Date**: YYYY-MM-DD
- **Branch / commit**: `<branch>` @ `<hash>`
- **Config**: `configs/expN.py` (or the relevant file)
- **Database**: `cache/<db_name>`
- **Changes vs.**: EXP-<previous id> — <what changed and why> (items: BUG-xx, ...)
- **Results**: table from `summarize_results.py`
- **Conclusion**: <what we learned; accepted, dropped or iterate>
```

---

## Baseline v0 — thesis results (exp1–exp4)

- **Computed on**: 2026-11-07, from `results/exp1..4/results_<band>.pickle`.
- **Branch / commit**: `main` @ `f1a75ae` (thesis code, before any fix).
- **Configs**: `configs/exp1.py` … `configs/exp4.py`.

| Exp | Noise threshold (`max_ruido_dB`) | Pink noise |
| --- | --- | --- |
| exp1 | −60 dB | no |
| exp2 | −45 dB | no |
| exp3 | −60 dB | yes (SNR −5 to 20 dB) |
| exp4 | −45 dB | yes (SNR −5 to 20 dB) |

> **Validity warning**: these metrics are affected by BUG-01 (convolution shift that leaks
> the RT), BUG-02 (C50/C80/D50 from sample 0), BUG-03 (noise added incorrectly in
> exp3/exp4) and BUG-04 (test set shares rooms and speakers with train). They are a starting
> point for comparison, **not** a measure of generalization. Stored predictions are rounded
> to 2 decimals in the normalized space (TRAIN-07), so MAE carries a small quantization error.

**Observations**
- In almost every band the best epoch is close to 500 → training did not converge (TRAIN-04).
- D50 MAE (4–15 %) is almost always above the 5 % JND, even when ρ is high (DATA-01).
- The noisy models (exp3/exp4) degrade mostly in T30 (ρ 0.62–0.84 vs. 0.78–0.98),
  consistent with BUG-03.

#### exp1

| Band [Hz] | N test | Best epoch / total | T30 ρ / MAE [s] | C50 ρ / MAE [dB] | C80 ρ / MAE [dB] | D50 ρ / MAE [%] |
| --- | --- | --- | --- | --- | --- | --- |
| 125 | 6492 | 474 / 500 | 0.834 / 0.41 | 0.678 / 2.95 | 0.794 / 2.37 | 0.694 / 11.97 |
| 250 | 6658 | 493 / 500 | 0.960 / 0.18 | 0.732 / 2.37 | 0.907 / 1.84 | 0.738 / 10.32 |
| 500 | 7162 | 483 / 500 | 0.965 / 0.18 | 0.831 / 2.00 | 0.898 / 1.69 | 0.833 / 8.63 |
| 1000 | 7265 | 499 / 500 | 0.964 / 0.19 | 0.912 / 1.57 | 0.953 / 1.37 | 0.909 / 6.08 |
| 2000 | 7428 | 331 / 431 | 0.979 / 0.13 | 0.946 / 1.33 | 0.972 / 1.20 | 0.945 / 5.60 |
| 4000 | 7634 | 496 / 500 | 0.964 / 0.16 | 0.966 / 1.17 | 0.979 / 1.15 | 0.963 / 4.79 |
| 8000 | 7592 | 460 / 500 | 0.947 / 0.16 | 0.978 / 1.17 | 0.987 / 1.18 | 0.974 / 4.25 |

#### exp2

| Band [Hz] | N test | Best epoch / total | T30 ρ / MAE [s] | C50 ρ / MAE [dB] | C80 ρ / MAE [dB] | D50 ρ / MAE [%] |
| --- | --- | --- | --- | --- | --- | --- |
| 125 | 7487 | 500 / 500 | 0.775 / 0.45 | 0.661 / 3.23 | 0.803 / 2.39 | 0.676 / 12.37 |
| 250 | 7796 | 483 / 500 | 0.946 / 0.20 | 0.714 / 2.63 | 0.887 / 1.84 | 0.715 / 10.99 |
| 500 | 8260 | 443 / 500 | 0.947 / 0.21 | 0.834 / 2.27 | 0.900 / 1.76 | 0.833 / 9.56 |
| 1000 | 8260 | 371 / 471 | 0.969 / 0.20 | 0.913 / 1.51 | 0.952 / 1.19 | 0.907 / 6.30 |
| 2000 | 8260 | 486 / 500 | 0.974 / 0.15 | 0.940 / 1.36 | 0.967 / 1.10 | 0.937 / 6.04 |
| 4000 | 8260 | 331 / 431 | 0.958 / 0.16 | 0.969 / 1.46 | 0.980 / 1.36 | 0.965 / 5.23 |
| 8000 | 8260 | 223 / 323 | 0.944 / 0.18 | 0.978 / 1.21 | 0.987 / 1.23 | 0.975 / 3.96 |

#### exp3

| Band [Hz] | N test | Best epoch / total | T30 ρ / MAE [s] | C50 ρ / MAE [dB] | C80 ρ / MAE [dB] | D50 ρ / MAE [%] |
| --- | --- | --- | --- | --- | --- | --- |
| 125 | 6392 | 464 / 500 | 0.623 / 0.54 | 0.532 / 3.49 | 0.636 / 3.26 | 0.551 / 14.86 |
| 250 | 6609 | 468 / 500 | 0.702 / 0.46 | 0.671 / 2.88 | 0.812 / 2.46 | 0.676 / 12.97 |
| 500 | 7197 | 450 / 500 | 0.687 / 0.49 | 0.779 / 2.52 | 0.820 / 2.42 | 0.780 / 11.02 |
| 1000 | 7318 | 448 / 500 | 0.793 / 0.39 | 0.888 / 2.14 | 0.922 / 1.99 | 0.890 / 7.87 |
| 2000 | 7487 | 493 / 500 | 0.694 / 0.45 | 0.906 / 1.72 | 0.925 / 1.75 | 0.910 / 7.03 |
| 4000 | 7724 | 455 / 500 | 0.766 / 0.37 | 0.933 / 1.73 | 0.941 / 1.72 | 0.933 / 6.44 |
| 8000 | 7664 | 306 / 406 | 0.814 / 0.32 | 0.949 / 2.11 | 0.959 / 2.23 | 0.947 / 7.54 |

#### exp4

| Band [Hz] | N test | Best epoch / total | T30 ρ / MAE [s] | C50 ρ / MAE [dB] | C80 ρ / MAE [dB] | D50 ρ / MAE [%] |
| --- | --- | --- | --- | --- | --- | --- |
| 125 | 6688 | 483 / 500 | 0.618 / 0.57 | 0.499 / 3.96 | 0.575 / 3.71 | 0.534 / 14.87 |
| 250 | 6997 | 395 / 495 | 0.725 / 0.44 | 0.625 / 3.15 | 0.785 / 2.56 | 0.633 / 13.51 |
| 500 | 7461 | 373 / 473 | 0.726 / 0.46 | 0.722 / 2.95 | 0.790 / 2.63 | 0.734 / 11.82 |
| 1000 | 7461 | 494 / 500 | 0.814 / 0.41 | 0.857 / 2.21 | 0.895 / 2.04 | 0.862 / 8.50 |
| 2000 | 7461 | 466 / 500 | 0.764 / 0.42 | 0.892 / 1.70 | 0.916 / 1.76 | 0.893 / 7.20 |
| 4000 | 7461 | 379 / 479 | 0.806 / 0.36 | 0.928 / 1.76 | 0.945 / 1.81 | 0.931 / 6.35 |
| 8000 | 7461 | 169 / 269 | 0.837 / 0.34 | 0.935 / 2.39 | 0.951 / 2.56 | 0.938 / 8.08 |

### Field measurements (thesis, §5.5)

Comparisons against in-situ measurements (3 rooms, 2 positions, REW reference) are in
tables 15–23 of the thesis and in `notebooks/exp*.ipynb`. Once the inference script exists
(CODE-06), record the per-band differences against the measured RIR here.

---

### EXP-5 — Baseline v1: clean data, classroom held out (development fold)
- **Date**: 2026-11-07 19:18 → 2026-11-08 07:46 (UTC−3).
- **Branch / commit**: `exp/exp5-baseline-v1` @ `dd30f73` (= `main` after PR #7; clean tree).
- **Command**: `PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 python run.py --config configs/exp5.py`
  (env `ReverbNet`, from the repo root; log in `results/exp5/train.log`, run metadata in
  `results/exp5/run_info.txt`).
- **Config**: `configs/exp5.py` → `configs/v1_base.py::make_config(exp_num=5, add_noise=False)`:
  seed 2222, Lundeby criterion −45 dB, test room `classroom` + 20 % of the synthetic RIRs,
  test speakers F4/M4/M9, validation by source RIR (10 %), thesis architecture, batch 1024,
  500 epochs, early stopping (patience 100).
- **Database**: `cache/data_base_-45_noise_False_traug_0.2_3.1_0.1_drraug_-6_19_1_snr_0_40_beta_1.0_2.0_clean_0.2_seed_2222_files_89442dc172_dv5`
  (data pipeline v5): 3368 RIRs → 1,331,112 band records in 27 partitions, 5.6 GB.
- **Discards** (`generation_summary.json`):

  | Reason | Unit | Count |
  | --- | --- | --- |
  | `band:lundeby_noise` | band record (speech × variant × band) | 18,026 |
  | `tr_variant:tr_augmentation_FloatingPointError` | whole TR variant | 303 |
  | `tr_variant:tr_augmentation_ValueError` | whole TR variant | 58 |
  | `drr_variant:unrealizable` | whole DRR variant | 158 |

  Lundeby discards per band (192,734 possible records per band minus the records used):
  125 Hz 11,376 (5.9 %) · 250 Hz 6,464 (3.4 %) · 500 Hz 48 · 1 kHz 68 · 2 kHz 16 · 4 kHz 8 ·
  8 kHz 46. Per the DATA-10 survey, no original real RIR passes −45 dB at 125 Hz, so the
  125 Hz test set is essentially synthetic-only; at 250 Hz only part of the real RIRs remain.
- **Records per band** (train / validation / test): 125 Hz 147,570 / 17,340 / 16,448 ·
  250 Hz 151,290 / 17,940 / 17,040 · 500 Hz 156,810 / 18,420 / 17,456 ·
  1 kHz 156,750 / 18,420 / 17,496 · 2 kHz 156,810 / 18,420 / 17,488 ·
  4 kHz 156,810 / 18,420 / 17,496 · 8 kHz 156,780 / 18,420 / 17,488.
- **Wall-clock time** (8 cores, CPU only):
  - Database generation: **2 h 03 min** (19:18 → 21:21). The 468 real RIRs (with TR/DRR
    augmentation) took ~1 h 15 min (~6 RIRs/min); the 2900 synthetic ones ~50 min
    (~55 RIRs/min). The earlier 20–30 min estimate (CODE-01) was 4–6× too low.
  - Training + evaluation of the 7 bands: **10 h 25 min** (21:21 → 07:46). Per band
    (load + fit + predict): 125 Hz 64 min · 250 Hz 102 · 500 Hz 63 · 1 kHz 81 · 2 kHz 112 ·
    4 kHz 91 · 8 kHz 112. Fitting runs at ~10 s/epoch; the sample-by-sample test prediction
    (TRAIN-07) takes ~25 min per band (measured on 125 Hz), i.e. ~3 h of the total.
  - Total: **12 h 28 min**.
- **Changes vs.**: baseline v0 (exp1, the closest v0 run: clean, Lundeby −60 dB) — every
  fix of phase 1–3: causal convolution (BUG-01), C50/C80/D50 from the direct-sound onset
  (BUG-02), group-aware splits by room/speaker/source RIR and train-only normalization
  (BUG-04), explicit discards and zero-padded short speech (BUG-05), reproducible
  augmentation built once per RIR (BUG-06, CODE-01), fixed TR augmentation (BUG-09), Lundeby
  −45 dB (DATA-10). Same architecture and training hyper-parameters as the thesis.
- **Results**:

| Band [Hz] | N test | Best epoch / total | T30 ρ / MAE [s] | C50 ρ / MAE [dB] | C80 ρ / MAE [dB] | D50 ρ / MAE [%] |
| --- | --- | --- | --- | --- | --- | --- |
| 125 | 16448 | 109 / 209 | 0.792 / 0.45 | 0.760 / 3.96 | 0.798 / 3.93 | 0.775 / 14.98 |
| 250 | 17040 | 388 / 488 | 0.892 / 0.29 | 0.871 / 2.22 | 0.914 / 1.99 | 0.879 / 9.87 |
| 500 | 17456 | 178 / 278 | 0.904 / 0.40 | 0.933 / 1.93 | 0.953 / 1.90 | 0.932 / 8.17 |
| 1000 | 17496 | 255 / 355 | 0.943 / 0.23 | 0.924 / 2.41 | 0.944 / 2.29 | 0.918 / 8.35 |
| 2000 | 17488 | 485 / 500 | 0.933 / 0.23 | 0.953 / 2.01 | 0.961 / 2.03 | 0.944 / 7.06 |
| 4000 | 17496 | 268 / 368 | 0.909 / 0.26 | 0.973 / 1.88 | 0.981 / 1.74 | 0.939 / 5.92 |
| 8000 | 17488 | 364 / 464 | 0.882 / 0.23 | 0.977 / 3.28 | 0.979 / 3.32 | 0.966 / 9.67 |

  Signed bias (mean of prediction − target, physical units; computed from the same pickles):

  | Band [Hz] | T30 [s] | C50 [dB] | C80 [dB] | D50 [%] |
  | --- | --- | --- | --- | --- |
  | 125 | −0.15 | −1.80 | −1.89 | −5.99 |
  | 250 | +0.06 | −0.46 | −0.61 | −1.70 |
  | 500 | +0.35 | −0.83 | −1.20 | −3.54 |
  | 1000 | +0.12 | −0.58 | −0.77 | −2.29 |
  | 2000 | +0.05 | −0.91 | −1.03 | −2.72 |
  | 4000 | +0.06 | −0.54 | −0.58 | −2.03 |
  | 8000 | +0.04 | −3.18 | −3.18 | −9.16 |

- **Comparison with v0 (exp1)** — **not directly comparable**: the test set changed from a
  random per-record split (rooms and speakers shared with train) to a held-out room
  (`classroom`) plus held-out synthetic RIRs and speakers (~2.4× more test records), the
  Lundeby criterion changed (−60 → −45 dB) and C50/C80/D50 are now defined from the
  direct-sound onset, which shifts their distribution (mean test C50 per band: v0 −2.6…+2.3 dB,
  v1 +1.3…+8.4 dB). Differences below describe the new reference, they do not measure a
  regression or an improvement of the model.
  - **T30**: ρ is lower in every band (0.79–0.94 vs. 0.83–0.98) and MAE higher (0.23–0.45 s vs.
    0.13–0.41 s). Expected: v0 benefited from the RT leak of BUG-01 and from seeing the test
    rooms in training.
  - **C50 / C80 / D50**: ρ is higher at 125–500 Hz (C50 0.76 / 0.87 / 0.93 vs. 0.68 / 0.73 /
    0.83) and similar or higher from 1 kHz up, but MAE is higher in every band (C50
    1.9–4.0 dB vs. 1.2–3.0 dB). No band reaches the JND (1 dB for C50/C80, 5 % for D50).
  - **Systematic under-estimation of the clarity descriptors**: C50/C80/D50 are biased low in
    every band. At 8 kHz the bias is almost the whole error (C50 −3.18 of 3.28 dB MAE, D50
    −9.2 of 9.7 %) while ρ stays at 0.98, so the ranking is right and the offset is wrong.
    T30 at 500 Hz is biased high (+0.35 of 0.40 s MAE). This is not explained by test targets
    beyond the training p95 (≤ 9 % of the test records exceed it). Hypothesis: the held-out
    room differs from the training distribution in those bands; checking it needs a per-subset
    breakdown (classroom real vs. synthetic test RIRs), which the results pickles do not allow
    because they store no RIR names (TRAIN-08).
  - **Convergence**: early stopping now triggers in 6 of 7 bands (best epoch 109–388), whereas
    in v0 it almost never did. 2 kHz still peaked at epoch 485/500 (TRAIN-04).
- **Conclusion**: accepted as **baseline v1** (development fold, clean). From now on new
  experiments are compared against EXP-5 (clean) or its noisy counterpart exp6; milestone
  claims need the leave-one-room-out folds (`exp5_great_hall`, `exp5_octagon`) as well.
  Before reading more into the bias, add the per-subset/per-room metrics of TRAIN-08; making
  prediction batched (TRAIN-07) would also save ~3 h per run.

---

### EXP-5 addendum — per-subset breakdown of baseline v1 (TRAIN-08)
- **Date**: 2026-11-08.
- **Branch / commit**: `feat/TRAIN-07-08-batched-prediction-metrics` (metadata recovery and metrics code; no retraining).
- **Method**: `python scripts/recover_test_metadata.py --config configs/exp5.py` rebuilt the
  test records of every band with the same reading path as `run.py` and checked them against
  the stored `X_test`/`y_test` (exact match, 7/7 bands). Tables with
  `python scripts/summarize_results.py --exp 5 --by rir_type`. The headline EXP-5 table above
  is unchanged.
- **Test composition** (≥ 500 Hz, per band): ~9,780 DRR-augmented + ~2,240 TR-augmented records
  from **13** classroom RIRs (69 %), 1,040 original classroom records from 130 RIRs (6 %),
  4,424 synthetic records from 553 RIRs (25 %). 125 Hz has no original real records; 250 Hz
  has 576.
- **Results** (MAE / bias / % of records within the JND):

| Band [Hz] | rir_type | N | T30 MAE / bias [s] / ≤JND | C50 MAE / bias [dB] / ≤JND | C80 MAE / bias [dB] / ≤JND | D50 MAE / bias [%] / ≤JND |
| --- | --- | --- | --- | --- | --- | --- |
| 125 | augmented | 12024 | 0.48 / -0.22 / 8 % | 4.46 / -2.31 / 14 % | 4.48 / -2.51 / 15 % | 16.49 / -8.24 / 18 % |
| 125 | synthetic | 4424 | 0.36 / +0.04 / 14 % | 2.60 / -0.41 / 26 % | 2.43 / -0.21 / 28 % | 10.88 / +0.14 / 29 % |
| 250 | augmented | 12040 | 0.28 / +0.05 / 17 % | 2.37 / -0.84 / 26 % | 2.11 / -1.03 / 32 % | 10.43 / -3.65 / 30 % |
| 250 | real | 576 | 0.22 / +0.06 / 31 % | 1.64 / +0.46 / 44 % | 1.45 / +0.61 / 45 % | 7.61 / +3.72 / 44 % |
| 250 | synthetic | 4424 | 0.31 / +0.11 / 17 % | 1.88 / +0.45 / 35 % | 1.74 / +0.38 / 38 % | 8.64 / +2.89 / 38 % |
| 500 | augmented | 11992 | 0.40 / +0.34 / 16 % | 2.12 / -1.47 / 27 % | 2.15 / -1.74 / 29 % | 8.62 / -6.26 / 31 % |
| 500 | real | 1040 | 0.44 / +0.42 / 12 % | 1.40 / -0.02 / 38 % | 1.12 / -0.16 / 51 % | 6.73 / +0.70 / 41 % |
| 500 | synthetic | 4424 | 0.39 / +0.35 / 15 % | 1.56 / +0.71 / 40 % | 1.40 / +0.01 / 45 % | 7.27 / +2.83 / 42 % |
| 1000 | augmented | 12032 | 0.22 / +0.10 / 22 % | 2.76 / -1.19 / 21 % | 2.64 / -1.28 / 22 % | 8.72 / -4.72 / 35 % |
| 1000 | real | 1040 | 0.29 / +0.21 / 19 % | 1.52 / +0.11 / 44 % | 1.35 / +0.08 / 46 % | 7.19 / +0.64 / 44 % |
| 1000 | synthetic | 4424 | 0.22 / +0.15 / 26 % | 1.65 / +0.90 / 44 % | 1.54 / +0.43 / 45 % | 7.63 / +3.63 / 42 % |
| 2000 | augmented | 12024 | 0.25 / +0.00 / 22 % | 2.29 / -1.57 / 32 % | 2.31 / -1.61 / 34 % | 7.26 / -5.16 / 46 % |
| 2000 | real | 1040 | 0.21 / +0.12 / 40 % | 1.12 / -0.22 / 52 % | 1.11 / -0.10 / 53 % | 5.74 / -0.47 / 52 % |
| 2000 | synthetic | 4424 | 0.21 / +0.16 / 28 % | 1.47 / +0.74 / 39 % | 1.48 / +0.36 / 44 % | 6.83 / +3.39 / 41 % |
| 4000 | augmented | 12032 | 0.29 / +0.05 / 17 % | 2.22 / -0.99 / 33 % | 2.05 / -0.98 / 39 % | 6.12 / -3.86 / 57 % |
| 4000 | real | 1040 | 0.23 / +0.13 / 15 % | 0.99 / -0.35 / 59 % | 0.79 / -0.22 / 70 % | 5.21 / -1.43 / 59 % |
| 4000 | synthetic | 4424 | 0.19 / +0.09 / 27 % | 1.16 / +0.66 / 56 % | 1.12 / +0.43 / 59 % | 5.53 / +2.81 / 54 % |
| 8000 | augmented | 12024 | 0.25 / -0.01 / 14 % | 4.24 / -4.22 / 5 % | 4.28 / -4.24 / 9 % | 11.47 / -11.45 / 14 % |
| 8000 | real | 1040 | 0.20 / +0.20 / 12 % | 2.32 / -2.26 / 15 % | 1.63 / -1.55 / 34 % | 11.90 / -11.47 / 16 % |
| 8000 | synthetic | 4424 | 0.19 / +0.12 / 26 % | 0.89 / -0.55 / 67 % | 1.11 / -0.69 / 61 % | 4.23 / -2.36 / 69 % |

  C50 split of the augmented records (mean true vs. predicted C50 [dB], bias [dB]):

  | Band [Hz] | DRR-augmented | TR-augmented | Original real | Synthetic |
  | --- | --- | --- | --- | --- |
  | 500 | 6.54 vs 4.75, −1.79 | −0.59 vs −0.67, −0.07 | −3.15 vs −3.16, −0.02 | −1.65 vs −0.94, +0.71 |
  | 1000 | 8.46 vs 6.98, −1.47 | 0.84 vs 0.91, +0.07 | −2.25 vs −2.13, +0.11 | −1.40 vs −0.50, +0.90 |
  | 8000 | 14.55 vs 10.10, −4.44 | 4.35 vs 1.09, −3.26 | 0.83 vs −1.43, −2.26 | −1.23 vs −1.77, −0.55 |

- **Conclusion**:
  - The negative clarity bias of the headline table comes from the **DRR-augmented variants**
    of 13 held-out RIRs, which are 56 % of the test records and have far higher C50 than
    anything else; the model compresses them towards the mean. This is a test-composition
    problem → new item **DATA-11**.
  - On the **original real classroom RIRs** the model is unbiased from 250 Hz to 4 kHz
    (C50 −0.35…+0.46 dB) with C50 MAE 1.0–1.6 dB (59 % within the JND at 4 kHz). These
    are the numbers closest to the use case.
  - **8 kHz** is biased low on every real subset (original real C50 −2.26 dB, D50 −11.5 %),
    not on synthetic RIRs → evidence added to DATA-08.
  - Synthetic RIRs are biased slightly high in C50/D50 (+0.5…+0.9 dB) at 250 Hz–4 kHz.
  - T30 at 500 Hz is biased high (+0.34…+0.42 s) in **every** subset, synthetic included, so it
    is a property of the 500 Hz model, not of the held-out room (open; TRAIN-04 territory).

---

### EXP-5 re-evaluation — baseline v1 under the DATA-11 test definition
- **Date**: 2026-11-08.
- **Branch / commit**: `fix/DATA-11-train-only-augmentation` (no retraining; same `results/exp5` models and predictions).
- **Why it is valid**: DATA-11 (data pipeline v6) only stops augmenting test RIRs. For exp5 the
  training RIRs, their augmentation (same 15 RIRs per training room), the source-RIR validation
  split and the train p95 factors are identical, so a v6 run of exp5 would train the same model
  (up to TF run-to-run nondeterminism) and its test set is exactly the original-RIR subset of the
  EXP-5 test set: 1,040 original classroom records (130 RIRs; 576 at 250 Hz, none at 125 Hz) +
  4,424 synthetic records (553 RIRs) per band.
- **Command**: `python scripts/summarize_results.py --exp 5 --exclude augmented [--by rir_type]`.
- **Results** (headline):

| Band [Hz] | N test | Best epoch / total | T30 ρ / MAE [s] | C50 ρ / MAE [dB] | C80 ρ / MAE [dB] | D50 ρ / MAE [%] |
| --- | --- | --- | --- | --- | --- | --- |
| 125 | 4424 | 109 / 209 | 0.836 / 0.36 | 0.610 / 2.60 | 0.725 / 2.43 | 0.656 / 10.88 |
| 250 | 5000 | 388 / 488 | 0.874 / 0.30 | 0.703 / 1.86 | 0.804 / 1.71 | 0.737 / 8.52 |
| 500 | 5464 | 178 / 278 | 0.884 / 0.40 | 0.755 / 1.53 | 0.843 / 1.35 | 0.769 / 7.17 |
| 1000 | 5464 | 255 / 355 | 0.919 / 0.24 | 0.809 / 1.62 | 0.854 / 1.50 | 0.812 / 7.54 |
| 2000 | 5464 | 485 / 500 | 0.943 / 0.21 | 0.866 / 1.40 | 0.880 / 1.41 | 0.851 / 6.63 |
| 4000 | 5464 | 268 / 368 | 0.938 / 0.20 | 0.924 / 1.13 | 0.945 / 1.06 | 0.915 / 5.47 |
| 8000 | 5464 | 364 / 464 | 0.962 / 0.19 | 0.940 / 1.16 | 0.948 / 1.21 | 0.941 / 5.69 |

  Per RIR type (MAE / bias / % within JND):

| Band [Hz] | rir_type | N | T30 MAE / bias [s] / ≤JND | C50 MAE / bias [dB] / ≤JND | C80 MAE / bias [dB] / ≤JND | D50 MAE / bias [%] / ≤JND |
| --- | --- | --- | --- | --- | --- | --- |
| 125 | synthetic | 4424 | 0.36 / +0.04 / 14 % | 2.60 / -0.41 / 26 % | 2.43 / -0.21 / 28 % | 10.88 / +0.14 / 29 % |
| 250 | real | 576 | 0.22 / +0.06 / 31 % | 1.64 / +0.46 / 44 % | 1.45 / +0.61 / 45 % | 7.61 / +3.72 / 44 % |
| 250 | synthetic | 4424 | 0.31 / +0.11 / 17 % | 1.88 / +0.45 / 35 % | 1.74 / +0.38 / 38 % | 8.64 / +2.89 / 38 % |
| 500 | real | 1040 | 0.44 / +0.42 / 12 % | 1.40 / -0.02 / 38 % | 1.12 / -0.16 / 51 % | 6.73 / +0.70 / 41 % |
| 500 | synthetic | 4424 | 0.39 / +0.35 / 15 % | 1.56 / +0.71 / 40 % | 1.40 / +0.01 / 45 % | 7.27 / +2.83 / 42 % |
| 1000 | real | 1040 | 0.29 / +0.21 / 19 % | 1.52 / +0.11 / 44 % | 1.35 / +0.08 / 46 % | 7.19 / +0.64 / 44 % |
| 1000 | synthetic | 4424 | 0.22 / +0.15 / 26 % | 1.65 / +0.90 / 44 % | 1.54 / +0.43 / 45 % | 7.63 / +3.63 / 42 % |
| 2000 | real | 1040 | 0.21 / +0.12 / 40 % | 1.12 / -0.22 / 52 % | 1.11 / -0.10 / 53 % | 5.74 / -0.47 / 52 % |
| 2000 | synthetic | 4424 | 0.21 / +0.16 / 28 % | 1.47 / +0.74 / 39 % | 1.48 / +0.36 / 44 % | 6.83 / +3.39 / 41 % |
| 4000 | real | 1040 | 0.23 / +0.13 / 15 % | 0.99 / -0.35 / 59 % | 0.79 / -0.22 / 70 % | 5.21 / -1.43 / 59 % |
| 4000 | synthetic | 4424 | 0.19 / +0.09 / 27 % | 1.16 / +0.66 / 56 % | 1.12 / +0.43 / 59 % | 5.53 / +2.81 / 54 % |
| 8000 | real | 1040 | 0.20 / +0.20 / 12 % | 2.32 / -2.26 / 15 % | 1.63 / -1.55 / 34 % | 11.90 / -11.47 / 16 % |
| 8000 | synthetic | 4424 | 0.19 / +0.12 / 26 % | 0.89 / -0.55 / 67 % | 1.11 / -0.69 / 61 % | 4.23 / -2.36 / 69 % |

- **Comparison with the EXP-5 headline table** (same models; only the test definition changed):
  - C50 MAE drops in every band (e.g. 1 kHz 2.41 → 1.62 dB, 8 kHz 3.28 → 1.16 dB) and the bias
    stays within −0.9…+0.8 dB, instead of being negative everywhere.
  - Clarity ρ drops (C50 at 1 kHz 0.924 → 0.809, at 125 Hz 0.760 → 0.610): the DRR sweep of the
    augmented records spread the C50 range and inflated the correlations.
  - T30 is similar (MAE 0.19–0.40 s); the +0.36 s bias at 500 Hz remains.
  - 125 Hz is synthetic-only: no original real RIR passes the Lundeby criterion (DATA-10).
- **Conclusion**: this table replaces the EXP-5 headline table as **baseline v1 (clean,
  development fold)**. New runs (pipeline v6) produce this test definition directly; compare
  them against this entry, always together with the `--by rir_type` table.

---

### EXP-6 — Noisy baseline v1: classroom held out, pipeline v6
- **Date**: 2026-11-08 19:34 → 2026-11-09 09:06 (UTC−3). A first attempt started at 10:31 died
  with the Claude Code session at 11:11 (partial cache deleted, log in
  `results/exp6/train_interrupted.log`); the run below was launched detached from the session.
- **Branch / commit**: `exp/exp6-noisy-baseline-v1` @ `e187c6e` (clean tree).
- **Command**: `run.py --config configs/exp6.py` through `results/exp6/launch_exp6.cmd`
  (env `ReverbNet`; log in `results/exp6/train.log`, metadata in `results/exp6/run_info.txt`).
- **Exit status**: not captured. The launcher wrote `echo %ERRORLEVEL%> exit_code.txt`, and in
  `cmd` a digit right before `>` is a stream handle, so the file came out empty. Evidence of a
  normal end: results and weights of 7/7 bands, no `Traceback` in the log, and the process ended
  6 s after saving the last band, which is the last thing `run.py` does.
- **Config**: `configs/exp6.py` → `make_config(exp_num=6, add_noise=True)`: as EXP-5 plus noise
  (broadband SNR 0–40 dB, β ∈ [1, 2], 20 % of the recordings clean; DATA-09). Augmentation on
  the 30 training RIRs only (DATA-11).
- **Database**: `cache/data_base_-45_noise_True_traug_0.2_3.1_0.1_drraug_-6_19_1_snr_0_40_beta_1.0_2.0_clean_0.2_seed_2222_files_2d9ce0ae14_dv6` (data pipeline v6): 3368 RIRs → 1,246,944 band records in 25
  partitions, 5.3 GB. Exactly 84,168 records fewer than EXP-5, which is the sum over the 7 bands
  of the augmented test records of EXP-5.
- **Discards** (`generation_summary.json`, EXP-5 → EXP-6): `band:lundeby_noise` 18,026 → 17,914,
  `tr_variant:tr_augmentation_FloatingPointError` 303 → 208,
  `tr_variant:tr_augmentation_ValueError` 58 → 0, `drr_variant:unrealizable` 158 → 156. Lundeby
  discards per band (180,694 possible records per band): 125 Hz 11,360 · 250 Hz 6,464 · 1 kHz 60 ·
  8 kHz 30 · none at 500 Hz, 2 kHz and 4 kHz.
- **Records per band**: train and validation counts identical to EXP-5 in every band; test 4,424
  (125 Hz), 5,000 (250 Hz), 5,464 (500 Hz and up).
- **Wall-clock time** (8 cores, CPU only):
  - Database generation: **2 h 01 min** (19:34 → 21:35).
  - Training + evaluation of the 7 bands: **11 h 31 min** (21:35 → 09:06), 3269 epochs in total
    (EXP-5: 2662). Per band (load + fit + predict): 125 Hz 74 min (500 epochs) · 250 Hz 66 (427) ·
    500 Hz 74 (451) · 1 kHz 78 (391) · 2 kHz 122 (500) · 4 kHz 137 (500) · 8 kHz 141 (500).
  - Total: **13 h 32 min**, above the ~10 h estimated before the run: batched prediction saved
    its ~3 h, but this run trained 23 % more epochs and the last three bands ran at 14–17 s/epoch
    against ~9–12 s/epoch for the first four (cause not investigated).
- **Changes vs.**: EXP-5 re-evaluation (baseline v1, clean) — only the noise (DATA-09). Training
  RIRs, augmentation, splits and architecture are the same.
- **Results**:

| Band [Hz] | N test | Best epoch / total | T30 ρ / MAE [s] | C50 ρ / MAE [dB] | C80 ρ / MAE [dB] | D50 ρ / MAE [%] |
| --- | --- | --- | --- | --- | --- | --- |
| 125 | 4424 | 404 / 500 | 0.847 / 0.36 | 0.653 / 2.69 | 0.755 / 2.45 | 0.682 / 12.48 |
| 250 | 5000 | 327 / 427 | 0.873 / 0.28 | 0.649 / 2.27 | 0.756 / 2.18 | 0.697 / 10.11 |
| 500 | 5464 | 351 / 451 | 0.893 / 0.28 | 0.781 / 1.38 | 0.826 / 1.38 | 0.775 / 6.62 |
| 1000 | 5464 | 291 / 391 | 0.895 / 0.25 | 0.849 / 1.20 | 0.904 / 1.16 | 0.866 / 5.70 |
| 2000 | 5464 | 460 / 500 | 0.925 / 0.21 | 0.815 / 1.41 | 0.873 / 1.44 | 0.836 / 7.01 |
| 4000 | 5464 | 455 / 500 | 0.929 / 0.20 | 0.903 / 1.23 | 0.916 / 1.28 | 0.895 / 5.95 |
| 8000 | 5464 | 468 / 500 | 0.895 / 0.25 | 0.798 / 1.59 | 0.851 / 1.59 | 0.824 / 7.44 |

  Per RIR type (MAE / bias / % within JND):

| Band [Hz] | rir_type | N | T30 MAE / bias [s] / ≤JND | C50 MAE / bias [dB] / ≤JND | C80 MAE / bias [dB] / ≤JND | D50 MAE / bias [%] / ≤JND |
| --- | --- | --- | --- | --- | --- | --- |
| 125 | synthetic | 4424 | 0.36 / +0.11 / 14 % | 2.69 / +0.79 / 25 % | 2.45 / +0.50 / 27 % | 12.48 / +6.14 / 25 % |
| 250 | real | 576 | 0.27 / -0.15 / 23 % | 1.92 / +0.99 / 36 % | 1.85 / +1.31 / 36 % | 8.85 / +5.95 / 38 % |
| 250 | synthetic | 4424 | 0.28 / -0.02 / 17 % | 2.31 / +1.10 / 30 % | 2.22 / +1.03 / 31 % | 10.28 / +5.35 / 32 % |
| 500 | real | 1040 | 0.31 / +0.26 / 20 % | 1.31 / -0.32 / 44 % | 1.17 / -0.09 / 50 % | 6.63 / -0.39 / 44 % |
| 500 | synthetic | 4424 | 0.27 / +0.17 / 22 % | 1.40 / +0.40 / 45 % | 1.43 / +0.11 / 46 % | 6.62 / +2.10 / 47 % |
| 1000 | real | 1040 | 0.26 / +0.03 / 23 % | 1.25 / -0.61 / 45 % | 1.00 / -0.09 / 60 % | 5.72 / -1.28 / 53 % |
| 1000 | synthetic | 4424 | 0.25 / +0.04 / 21 % | 1.19 / +0.39 / 56 % | 1.20 / +0.09 / 59 % | 5.70 / +2.18 / 55 % |
| 2000 | real | 1040 | 0.19 / +0.04 / 30 % | 1.27 / +0.06 / 50 % | 1.16 / +0.22 / 51 % | 6.41 / +1.56 / 47 % |
| 2000 | synthetic | 4424 | 0.21 / +0.09 / 22 % | 1.45 / +0.75 / 45 % | 1.50 / +0.25 / 47 % | 7.15 / +4.25 / 43 % |
| 4000 | real | 1040 | 0.18 / +0.03 / 27 % | 1.21 / -0.42 / 45 % | 1.04 / +0.14 / 56 % | 5.83 / -0.87 / 49 % |
| 4000 | synthetic | 4424 | 0.21 / +0.01 / 23 % | 1.24 / +0.51 / 60 % | 1.34 / +0.41 / 56 % | 5.97 / +2.89 / 54 % |
| 8000 | real | 1040 | 0.23 / +0.15 / 19 % | 2.19 / -1.71 / 26 % | 1.74 / -0.94 / 33 % | 11.22 / -8.36 / 25 % |
| 8000 | synthetic | 4424 | 0.26 / +0.04 / 20 % | 1.45 / +0.28 / 46 % | 1.55 / +0.03 / 49 % | 6.55 / +1.09 / 50 % |

  Per broadband SNR of the test recording, all RIR types (T30 MAE [s] / C50 MAE [dB] / C50 bias
  [dB]; about 1,090 records per bucket at 500 Hz and up):

| Band [Hz] | clean | 0-10 | 10-20 | 20-30 | 30-40 |
| --- | --- | --- | --- | --- | --- |
| 125 | 0.32 / 2.56 / +0.81 | 0.43 / 3.18 / +1.48 | 0.37 / 2.63 / +0.61 | 0.33 / 2.51 / +0.60 | 0.32 / 2.56 / +0.48 |
| 250 | 0.26 / 2.18 / +1.04 | 0.33 / 2.35 / +1.01 | 0.26 / 2.32 / +1.11 | 0.28 / 2.24 / +1.16 | 0.27 / 2.25 / +1.10 |
| 500 | 0.25 / 1.33 / +0.33 | 0.35 / 1.52 / +0.06 | 0.27 / 1.34 / +0.16 | 0.26 / 1.31 / +0.32 | 0.25 / 1.41 / +0.46 |
| 1000 | 0.24 / 1.18 / +0.14 | 0.28 / 1.27 / +0.26 | 0.25 / 1.24 / +0.29 | 0.24 / 1.11 / +0.14 | 0.24 / 1.22 / +0.16 |
| 2000 | 0.17 / 1.30 / +0.59 | 0.29 / 1.67 / +0.63 | 0.22 / 1.42 / +0.57 | 0.19 / 1.35 / +0.69 | 0.18 / 1.33 / +0.61 |
| 4000 | 0.18 / 1.15 / +0.16 | 0.25 / 1.47 / +0.69 | 0.22 / 1.23 / +0.29 | 0.18 / 1.13 / +0.24 | 0.18 / 1.19 / +0.29 |
| 8000 | 0.21 / 1.48 / -0.34 | 0.33 / 1.94 / +0.30 | 0.26 / 1.59 / -0.14 | 0.24 / 1.50 / -0.18 | 0.22 / 1.45 / -0.13 |

- **Comparison with baseline v1 (EXP-5 re-evaluation)**:
  - **Validation loss is higher in every band** (ratio noisy/clean from 1.10 at 500 Hz to 2.25 at
    4 kHz), as expected from a harder task.
  - **Test metrics are mixed, not uniformly worse**. C50 MAE [dB], EXP-5 → EXP-6: 125 Hz 2.60 → 2.69 ·
    250 Hz 1.86 → 2.27 · 500 Hz 1.53 → 1.38 · 1 kHz 1.62 → 1.20 · 2 kHz 1.40 → 1.41 · 4 kHz 1.13 →
    1.23 · 8 kHz 1.16 → 1.59. T30 MAE [s] is equal or within 0.02 s except 500 Hz 0.40 → 0.28
    (better) and 8 kHz 0.19 → 0.25 (worse). The two runs are single, unseeded trainings that stopped
    at very different epochs (500 Hz: best epoch 178 vs. 351), so the gains at 500 Hz–1 kHz **must
    not be read as noise helping** without repeating the runs (TRAIN-09).
  - **Error depends weakly on SNR above 10 dB**. Across the clean, 10–20, 20–30 and 30–40 dB
    buckets the C50 MAE of each band varies by at most ~0.15 dB. The 0–10 dB bucket is clearly
    worse than the clean recordings (T30 MAE +17 to +71 %, C50 MAE +8 to +31 %).
  - **Real classroom RIRs**: C50 MAE 1.2–1.3 dB from 500 Hz to 4 kHz (bias −0.6…+0.1 dB), 1.9 dB at
    250 Hz. **8 kHz stays biased low on real RIRs** (C50 −1.71 dB, D50 −8.4 %) and not on synthetic
    ones, as in the clean run, so that bias is not a noise effect (DATA-08).
  - **Convergence**: 4 of 7 bands hit the 500-epoch limit with best epochs 404–468 (TRAIN-04).
- **Comparison with v0** (exp3/exp4): not comparable; besides the changes listed in EXP-5, v0 mixed
  noise after the filter bank (BUG-03) with a different SNR range and color.
- **Conclusion**: accepted as **noisy baseline v1** (development fold). Differences against the
  clean baseline should not be interpreted band by band until TRAIN-09 gives a measure of
  run-to-run variation.
