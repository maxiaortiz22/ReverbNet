# Log

Entries in chronological order, most recent **at the end**. Absolute dates
(YYYY-MM-DD). Each entry states what was done, what was decided and which items it touches.

---

## 2026-11-07 — Initial review and workflow setup

- Full review of the code (pipeline, augmentation, Lundeby, model, C++, notebooks) and of
  the thesis. Findings recorded in [backlog.md](backlog.md) with `BUG/DATA/TRAIN/CODE/RES` IDs.
- Checks performed during the review:
  - Onset delay of measured RIRs: great_hall ~33 ms, octagon ~31 ms, classroom ~17 ms
    (median) → evidence for BUG-02.
  - `fftconvolve(mode='same')` starts at `(len(rir)-1)//2` → evidence for BUG-01.
  - `import tensorflow` from the repo root fails because of the `code` package → BUG-08.
  - The `ReverbNet` env (TF 2.16.1, Python 3.12, numpy 1.26.4) sees no GPU → TRAIN-06.
- Computed **baseline v0** from `results/exp1..4` with the new
  `scripts/summarize_results.py` (see [experiments.md](experiments.md)).
- Created branch `chore/project-workflow-setup` with:
  - `CLAUDE.md` (project guide).
  - Skills in `.claude/skills/`: `python-code`, `cpp-code`, `dsp`, `room-acoustics`,
    `neural-network`, `data-analysis`, `experiments`, `improvement-tracking`.
  - This `docs/improvements/` folder.
- Also committed previously uncommitted work from `main`: `analyze_model.py`,
  `calculate_macs_manual.py`, `scripts/model_analysis_simple.py` (model size / MACs
  analysis, related to RES-04) and a change to `notebooks/exp2.ipynb`.
- **Decision**: all repository content (code, docstrings, comments, docs, skills, tracking,
  commit messages) is written in **English** for consistency with the rest of the repo.

## 2026-11-07 — BUG-01: causal convolution

- Replaced `fftconvolve(speech, rir, mode='same')` with `causal_convolve` (full convolution
  truncated to the speech length) in `code/convolution.py`, used by the database generator.
  Closes BUG-01 at code level.
- Added `DATA_PIPELINE_VERSION = 1` to the cache name (`_dv1`) so the thesis caches (v0) are
  kept for comparison and never reused by the fixed pipeline (partial CODE-05).
- Started the test suite: pytest installed in the `ReverbNet` env, `tests/` with 11 passing
  tests (partial CODE-07).
- **Decision**: do not regenerate the database or retrain after each bug. The database will
  be regenerated once all P0 bugs (BUG-01 to BUG-04) are fixed, and that run becomes
  baseline v1. Running `configs/exp1..4.py` now would build a `_dv1` cache but write into
  `results/exp1..4` and overwrite the v0 artifacts, so v1 needs new configs with new `exp_num`.

## 2026-11-07 — BUG-02: onset-referenced energy ratios

- Added `find_onset` / `trim_to_onset` (`code/parameters_calculation/onset.py`) and used the
  broadband onset for C50/C80/D50 in the database generator. `DATA_PIPELINE_VERSION` → 2.
- **Decision**: T30 and DRR keep their v0 definitions (referenced to the band peak) to isolate
  this change; revisiting them belongs to DATA-08 / BUG-07.
- Measured on all 468 real RIRs: C50 was biased low by +1.9 to +4.8 dB on average per band
  (up to +18 dB at p95 in low bands), C80 by +1.7 to +2.9 dB, D50 by +7.6 to +10.7 %.
  Synthetic RIRs are unaffected (onset at 0). Details in the BUG-02 entry of the backlog.
- Test suite: 26 passing tests. In sandboxed sessions the default pytest temp dir may not be
  writable; use `--basetemp=<writable dir>`.

## 2026-11-07 — BUG-03: broadband noise before the filter bank

- Added `code/mixing.py::mix_at_snr` and `DataBaseGenerator._build_input_bands`: noise is mixed
  broadband with the reverberant speech and the mixture goes through the filter bank, exactly
  like a real recording at inference time. `DATA_PIPELINE_VERSION` → 3.
- **Decisions**:
  - The SNR is now **broadband** (mean power over the 5 s window) and drawn **once per
    recording**; records store it in `snr` and the resulting per-band SNR in `snr_band`.
  - The SNR range was left at `[-5, 20]` dB for now. Measured on real speech and RIRs, a
    broadband SNR leaves the 4 and 8 kHz bands ~15–19 dB lower on average, so the range must be
    revisited before generating the noisy variant of baseline v1 → new item **DATA-09**.
  - The RNG is still NumPy's global state (`rng=None`); seeding is BUG-06.
- The C++ `AudioProcessor` is no longer used (noted in CODE-03).
- Corrected the speech counts in the tracking docs: 30 training files (not 33) and 8 test files.
- Test suite: 42 passing tests.

## 2026-11-07 — DATA-09 analysis (SNR range and definition)

- Measured on 28 speech files × 8 real RIRs (details and tables in the DATA-09 backlog entry):
  - Active-speech vs. whole-window SNR differ by only +0.3 dB (reverberation fills the
    pauses), so the SNR definition does not need a VAD.
  - The noise color, not the range, drives the per-band SNR: pink noise is a worst case for
    2–8 kHz, while room noise (NC/NR-like) lies between pink and brown.
  - In the dB domain, the TAE needs ~15 dB of band SNR to resemble the clean one (r ≥ 0.9).
- Found while running it: `F5s3.wav` and `M8s3.wav` are shorter than 5 s and all their records
  are silently dropped (evidence added to BUG-05).
- Decision on the range/color pending with the maintainer (options a/b/c in the backlog).

## 2026-11-07 — DATA-09: reproducible colored noise with a clean fraction

- **Decision** (maintainer): option (c) of DATA-09, and one **project-wide seed**: the
  config's `seed` drives every random process; no separate noise seed.
- Implemented `colored_noise`, the generator parameters `noise_beta`, `clean_fraction` and
  `seed` (passed by `run.py`, legacy defaults for the v0 configs), per-recording generators
  seeded with `(seed, RNG_STREAMS['noise'], crc32(recording_name))`, and the `noise_beta`
  record field. `DATA_PIPELINE_VERSION` → 4.
- Recommended values for the noisy variant of baseline v1: `snr = [0, 40]`,
  `noise_beta = [1.0, 2.0]`, `clean_fraction = 0.2` (to be set in the v1 configs).
- `RNG_STREAMS` is the pattern for the rest of BUG-06 (augmentation draws).
- Test suite: 57 passing tests.

## 2026-11-07 — BUG-04: group-aware splits and v1 configs

- **Decisions** (maintainer): development split holds out the `classroom` room; milestone
  baselines run leave-one-room-out folds. Test speakers `F4`, `M4`, `M9`.
- **Decision**: the v1 configs keep the thesis model and `max_ruido_dB = -60` (as exp1), so
  baseline v1 isolates the effect of the data fixes.
- Implemented `code/splits.py` and `code/random_streams.py`, explicit validation by source RIR
  and train-only target normalization in `run.py`, pooled speech folders in the generator,
  and `configs/v1_base.py` with `exp5` (clean) / `exp6` (noisy) plus their LORO folds.
- Found and fixed while validating the configs: LORO folds would have shared one cache
  because the cache name did not encode the split; it now includes a digest of the file lists.
- Checked that Keras 3 handles the `(n, 4, 1)` targets correctly (no MSE broadcasting bug).
- With BUG-01 to BUG-04 done, the P0 items are closed. Next: decide whether to run baseline v1
  now or first fix BUG-05/BUG-06 (both change the generated data).
- Test suite: 87 passing tests.

## 2026-11-07 — BUG-05 / BUG-06 (+ CODE-01): explicit failures, reproducible augmentation

- Surveyed which exceptions the generator was swallowing on the real data: only the Lundeby
  noise criterion for descriptors, the `DRRR_buscado` typo for unrealizable DRR targets, and
  numeric failures inside `tr_augmentation` (BUG-05 backlog entry).
- Only those expected failures are now caught and counted by reason; everything else
  propagates. Short speech files are zero-padded to 5 s (F5s3 and M8s3 are back).
  `run.py` writes `generation_summary.json` last and refuses interrupted database folders.
- Augmentation is reproducible from the project seed (`drr_subset`, `tr_tail` streams) and,
  since it no longer varies per speech file, it is built once per RIR together with its
  descriptors (CODE-01 done). Serial and multiprocess runs give identical records.
  `DATA_PIPELINE_VERSION` → 5.
- **Decision**: zero-pad short speech instead of rejecting it (realistic: the speaker stops
  and the recording continues).
- New findings, recorded as items:
  - **BUG-09**: TR augmentation fails for 26 of the 45 selected real RIRs (`IndexError`).
  - **DATA-10**: with `max_ruido_dB = -60` no real RIR survives at 125/250 Hz and ~40 % are lost
    at 500 Hz–1 kHz; −45 dB (the ISO minimum for T30) keeps all of them from 500 Hz up.
- Rough cost of a full v1 database after CODE-01: ~20–30 min on 8 cores (to be confirmed).
- Test suite: 100 passing tests.

## 2026-11-07 — DATA-10 (−45 dB) and BUG-09 (TR augmentation)

- **Decision** (maintainer): v1 uses the Lundeby criterion `max_ruido_dB = -45` (ISO minimum
  for T30) instead of −60 dB; set as the default of `configs/v1_base.py::make_config`.
- BUG-09 root cause: the Lundeby iteration inside `tr_augmentation` fitted the late decay below
  the noise floor. Fixed to fit above it, with robust interval search and explicit errors.
  TR-analysis failures on the 45 selected RIRs: 58 % → 4.4 %.
- Running the full generator path (augmentation + descriptors) on all 45 selected RIRs
  surfaced two more silent failures, now handled: NaN/inf augmented RIRs (overflow when
  lengthening short decays a lot; counted as `FloatingPointError`) and a `ValueError` in
  `lundeby` for missing dynamic range (now a `NoiseError`).
- Known limitation recorded in BUG-09: augmented RIRs are biased towards short RTs because
  long targets (≥ 1.7 s) often overflow.
- Full-path validation after the fixes: 45 RIRs, 4706 variants, 0 unexpected exceptions; 72 % of
  the TR variants are usable (details in BUG-09). Test suite: 113 passing tests.

## 2026-11-08 — EXP-5: baseline v1 (clean, classroom held out)

- Ran `configs/exp5.py` on branch `exp/exp5-baseline-v1` @ `dd30f73` (2026-11-07 19:18 →
  2026-11-08 07:46), exit code 0, no unexpected exceptions. Full entry in
  [experiments.md](experiments.md#exp-5--baseline-v1-clean-data-classroom-held-out-development-fold).
- Database (pipeline v5): 1,331,112 records, 5.6 GB, **2 h 03 min** on 8 cores, 4–6× the
  20–30 min estimate (CODE-01 evidence updated; the `experiments` skill now quotes the measured
  time). Discards: 18,026 band records by the Lundeby criterion (almost all at 125/250 Hz,
  DATA-10), 361 TR variants (303 overflow, 58 failed TR analysis; BUG-09) and 158
  unrealizable DRR variants.
- Training + evaluation of the 7 bands: **10 h 25 min**; ~3 h of it is the per-sample test
  prediction (TRAIN-07 evidence).
- **Decision**: EXP-5 is **baseline v1** (development fold, clean). New experiments are
  compared against it (clean) or against exp6 (noisy); v0 numbers are kept only as historical
  reference because the split, the Lundeby criterion and the clarity definitions changed.
- Findings recorded as evidence on existing items:
  - Systematic low bias of C50/C80/D50 in every band (−3.2 dB C50 at 8 kHz, 97 % of the MAE)
    and high T30 bias at 500 Hz (+0.35 s); the results pickles store no RIR names, so the
    held-out-room vs. synthetic breakdown that would explain it is not possible yet (TRAIN-08).
  - Early stopping now triggers in 6 of 7 bands; 2 kHz still peaks at epoch 485/500 (TRAIN-04).
  - The clean database name still encodes the unused noise parameters (CODE-05).
- Open question for the maintainer: next run exp6 (noisy) or the clean LORO folds
  (`exp5_great_hall`, `exp5_octagon`); each costs ~12.5 h with the current pipeline.
