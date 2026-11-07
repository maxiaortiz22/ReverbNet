---
name: data-analysis
description: How to analyze data and results in ReverbNet - exploring the RIR and descriptor database, error metrics, plots, notebooks, comparison against field measurements, and leakage or bias checks. Use when writing or editing notebooks (notebooks/*.ipynb), analysis scripts, results tables or figures, or when interpreting an experiment's metrics.
---

# Data analysis in ReverbNet

## Where things live

- `notebooks/expN.ipynb`: exploration and figures for each experiment (run from `notebooks/`,
  hence the `../` paths).
- Reusable logic (database reading, metrics, inference) lives **in the package** and is
  imported from the notebooks. Do not copy functions into notebooks (today `read_dataset` and
  `get_tae_per_band` are duplicated, and `get_parameters` calls `filtered_signals`, which does
  not exist: CODE-09).
- Experiment parameters are loaded from the config (`import_configs_objs`), not copied by hand
  into a cell.
- Metrics tables: `scripts/summarize_results.py` → `docs/improvements/experiments.md`.

## Notebook hygiene

- Run top to bottom before committing (no out-of-order cells).
- Avoid committing heavy outputs (large images, huge DataFrames). If `nbstripout` is adopted,
  record it in the log.
- One question per section, with a written Markdown conclusion under the plot.

## Standard metrics

Always in **physical units** (de-normalize with the saved statistics) and with **N**:

| Metric | What for |
| --- | --- |
| MAE, RMSE | Error magnitude (s, dB, %). |
| Bias (mean of pred − true) | Systematic under/overestimation. |
| Spearman ρ and Pearson r | Ranking and linearity (ρ alone is not enough: it misses bias and scale). |
| % within JND | Perceptual relevance (T30 5 % rel., C50/C80 1 dB, D50 5 %). |

Mandatory breakdowns: by **band**, by **descriptor**, by **RIR type** (real / synthetic /
augmented, from the `speech|rir|tag` name) and, once group splits exist, by **room**. Bands
always ordered 125 → 8000 Hz.

## Plots

- matplotlib (seaborn optional). Axes with units, title with experiment and band.
- Predicted vs. true with the diagonal and ±JND bands.
- Bland-Altman (difference vs. mean) to spot value-dependent bias.
- Per-band error boxplots; per-band target distributions (spot outliers at 125/250 Hz).
- Save figures to `results/expN/figures/` with descriptive names (`t30_scatter_1000Hz.png`).

## Checks before concluding

- **Leakage**: are rooms, source RIRs or speakers shared between train and test? (BUG-04)
- **Distribution**: do the test targets cover the same range as train? Report min/p5/p95/max.
- **Composition**: 86 % of the RIRs are synthetic; a global metric can hide poor performance
  on real ones. Always look at the per-type breakdown.
- **Discards**: how many records were lost per band and why (NoiseError, Lundeby, failed
  augmentation). Less data at low bands is expected.
- **Physical consistency** of predictions: C80 ≥ C50; D50 consistent with C50; plausible ranges
  (`room-acoustics` skill).
- **Field measurements**: compare against the REW Excel files
  (`data/Measurements/room*/room*_rew.xlsx`) per band and position, with the difference
  relative to the JND.

## Reading the database without running out of memory

The current cache is ~4.3 GB of pickles per database (`cache/<db_name>/*.pkl`) holding Python
lists. Read once and filter by band and split in the same pass; do not concatenate DataFrames
inside a loop (CODE-02). For exploratory analysis, subsample (`sample_frac`) with a fixed seed.
