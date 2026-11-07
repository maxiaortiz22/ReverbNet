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
