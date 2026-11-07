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
