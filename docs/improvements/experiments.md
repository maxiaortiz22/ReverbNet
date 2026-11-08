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
