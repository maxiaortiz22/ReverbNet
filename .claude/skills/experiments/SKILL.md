---
name: experiments
description: Workflow for running, recording and comparing ReverbNet experiments - configs in configs/, database generation in cache/, training with run.py, artifacts in results/expN, metrics tables with scripts/summarize_results.py and recording in docs/improvements/experiments.md against the baseline. Use when planning or launching a training run or a data regeneration, when creating a new config, or when comparing results.
---

# Experiments in ReverbNet

## Before running

1. **Hypothesis**: what is expected to change and why, with the item ID (`DATA-02`, ...).
2. **One change at a time** relative to the reference experiment (the current baseline in
   `docs/improvements/experiments.md`). If there are several changes, run separate ablations.
3. **New config** in `configs/`, with an `exp_num` that does not exist in `results/`. Never
   reuse or overwrite another experiment's `results/expN`.
4. **Cache**: the name (`database.db_name`) only encodes `max_ruido_dB`, noise, augmentation
   and SNR. If the change touches data generation (convolution, filters, TAE, onset, noise,
   file lists), the old cache is **not valid** even if the name matches: rename or delete it,
   after confirming with the user (CODE-05).
5. **Disk and time**: each database takes ~4.3 GB; training runs on CPU (TRAIN-06) and takes
   hours. Warn the user before launching anything long.

## Config keys for noise and randomness

- `seed`: the **only** seed of the project. It drives the RIR train/test split in the config
  and every random process of data generation (noise today; augmentation once BUG-06 is done).
  Changing it changes both the split and the noise.
- `add_noise`: master switch for noise.
- `snr`: broadband SNR range in dB, drawn uniformly per recording.
- `noise_beta`: range of the noise spectral exponent (1/f^β), drawn per recording.
- `clean_fraction`: probability that a recording stays noise-free.
- Recommended noisy setup (DATA-09): `snr = [0, 40]`, `noise_beta = [1.0, 2.0]`,
  `clean_fraction = 0.2`. If `noise_beta`/`clean_fraction` are missing, `run.py` falls back to
  the legacy setup (always pink, no clean fraction), which is what the v0 configs exp1–4 use.

## v1 configs

- New experiments use `configs/v1_base.py::make_config(exp_num, add_noise, test_rooms=...)`;
  each config file is three lines. Never copy the split logic into a config again.
- `exp5` (clean) and `exp6` (noisy) hold out `classroom`. Their leave-one-room-out folds are
  `exp5_great_hall`, `exp5_octagon`, `exp6_great_hall`, `exp6_octagon`, with string ids
  (results in `results/exp5_octagon`, …; `summarize_results.py --exp 5 5_octagon` accepts them).
- Report LORO metrics per fold and their mean; the development fold alone is for iterating.
- Each fold builds its own database (the split is baked in): ~4.3 GB and hours per fold.
- The cache name includes a digest of the file lists and split (`files_<sha1>`), so folds and
  file changes never share a cache.

## Running

```bash
python run.py --config configs/expN.py
```

- Run it in the background if it will take more than a couple of minutes, logging to
  `results/expN/train.log`.
- If data generation is interrupted, the cache folder is left half-written and the next run
  treats it as complete (`any(db_dir.iterdir())`): delete it before retrying.
- For quick pipeline tests: a config with a small `sample_frac` and few epochs, with a test
  `exp_num` (e.g. 900+) that is deleted afterwards.

## Recording

1. Metrics table:
   ```bash
   python scripts/summarize_results.py --exp N
   ```
2. A new entry **at the end** of `docs/improvements/experiments.md` using the template: date,
   branch and commit, config, database, changes vs. the reference, table, conclusion.
3. If the experiment closes an item, update the board, backlog and log
   (`improvement-tracking` skill).

## Comparing

- Against the **current baseline**: v0 = thesis (exp1–4, with known bugs). After the BUG-*
  items are fixed, the first retraining becomes **baseline v1**, and comparisons are made
  against v1 from then on.
- Same split, same metrics, same bands. If the split changed (e.g. because of BUG-04), the
  numbers are not directly comparable: say so explicitly.
- Report improvements with their magnitude per band and descriptor (not just "it improved"),
  and whether the change is within seed-to-seed variation (run ≥ 2 seeds when the difference
  is small).
- An improvement on the synthetic test set that does not show up on the field measurements is
  suspicious (`neural-network` skill, shortcuts section).

## Artifacts per experiment (`results/expN/`)

| File | Content |
| --- | --- |
| `weights_<band>.weights.h5` | Best model weights per band. |
| `results_<band>.pickle` | History, predictions, errors, normalization factors, test X/y. |
| `train.log` | Run output (to be added). |
| `config.py` / `config.json` | Copy of the config used + commit (to be added). |
| `figures/` | Analysis plots (`data-analysis` skill). |

`.pickle` and `.wav` files are in `.gitignore`: results do not travel with the repo. Whatever
is needed to reproduce a run (config, commit, metrics) must be recorded in `experiments.md`.
