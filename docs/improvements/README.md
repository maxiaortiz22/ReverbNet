# ReverbNet improvement tracking

This folder is the **source of truth** for what we want to improve, what is in progress,
and what is done. It is maintained with the `improvement-tracking` skill
(`.claude/skills/improvement-tracking/SKILL.md`).

| File | Purpose |
| --- | --- |
| [`README.md`](README.md) | This board: status of every item (the only place where status lives). |
| [`backlog.md`](backlog.md) | Item details: problem, evidence, proposal, acceptance criteria. |
| [`experiments.md`](experiments.md) | Experiment registry and metrics (includes the thesis baseline). |
| [`log.md`](log.md) | Chronological log of what was done and decided. |

## Conventions

- **IDs**: `BUG-xx` (correctness errors), `DATA-xx` (data and processing),
  `TRAIN-xx` (model and training), `CODE-xx` (code and engineering),
  `RES-xx` (research). IDs are never reused.
- **Status**: `todo` · `in progress` · `blocked` · `done` · `dropped`.
- **Priority**: `P0` (affects the validity of results) · `P1` (high impact) ·
  `P2` (improvement) · `P3` (exploratory).
- **Branches**: `fix/BUG-01-...`, `feat/DATA-02-...`, `refactor/CODE-04-...`,
  `exp/...`. Commits mention the ID.
- Status lives **only on this board**; the backlog holds the details.

## Suggested phases

0. **Workflow setup**: CLAUDE.md, skills and this tracking folder.
1. **Correctness**: all `BUG-*` items (P0). Until these are fixed, metrics are not reliable.
2. **Pipeline**: refactor data generation and storage (`CODE-01`, `CODE-02`, `CODE-04`, `CODE-05`).
3. **New baseline (v1)**: retrain with the fixed pipeline and compare against v0 in `experiments.md`.
4. **Model and data improvements**: `DATA-*`, `TRAIN-*`.
5. **Research**: `RES-*`.

## Board

| ID | Title | Priority | Phase | Status | Branch / PR |
| --- | --- | --- | --- | --- | --- |
| BUG-01 | `mode='same'` convolution shifts the signal and leaks the RT | P0 | 1 | done | `fix/BUG-01-causal-convolution` |
| BUG-02 | C50/C80/D50 integrate from sample 0 instead of the direct sound | P0 | 1 | done | `fix/BUG-01-causal-convolution` |
| BUG-03 | Pink noise added after the filter bank and unfiltered | P0 | 1 | done | `fix/BUG-01-causal-convolution` |
| BUG-04 | Train/val/test leakage (normalization, per-record validation, shared rooms and speakers) | P0 | 1 | done | `fix/BUG-01-causal-convolution` |
| BUG-05 | Silenced exceptions (`except Exception: continue`) and `DRRR_buscado` typo | P1 | 1 | todo | |
| BUG-06 | Reproducibility: per-iteration `random.sample`, clock-based seed, `os.listdir` order | P1 | 1 | in progress | `fix/BUG-01-causal-convolution` (noise RNG, splits, sorted listings) |
| BUG-07 | `argmax(rir)` without `abs` to detect the direct sound | P2 | 1 | todo | |
| BUG-08 | The `code` package shadows the stdlib `code` module (breaks `import tensorflow` from the root) | P1 | 1 | todo | |
| DATA-01 | Derive D50 from C50 (deterministic relation) | P1 | 4 | todo | |
| DATA-02 | Higher-resolution and/or dB TAE | P1 | 4 | todo | |
| DATA-03 | More realistic synthetic RIRs (per-band T60, direct sound, early reflections) | P1 | 4 | todo | |
| DATA-04 | Add real RIR datasets (ACE, BUT ReverbDB, MIT, OpenAIR, Arni…) | P1 | 4 | todo | |
| DATA-05 | C50/C80 augmentation (early energy) | P2 | 4 | todo | |
| DATA-06 | Larger speech corpus, random crops and VAD | P1 | 4 | todo | |
| DATA-07 | Real noise types and a clean/noisy mix | P2 | 4 | in progress | `fix/BUG-01-causal-convolution` (noise color + clean mix) |
| DATA-08 | Filter influence on descriptors (zero-phase / time-reversed filtering, 8 kHz band) | P2 | 4 | todo | |
| DATA-09 | Revisit the SNR range/definition: broadband −5…20 dB leaves 4–8 kHz bands at −24…+6 dB | P1 | 3 | done | `fix/BUG-01-causal-convolution` |
| TRAIN-01 | One multi-band model instead of 7 networks | P1 | 4 | todo | |
| TRAIN-02 | Higher-capacity architecture with global pooling | P2 | 4 | todo | |
| TRAIN-03 | Z-score normalization with train stats, Huber/NLL loss with uncertainty | P1 | 4 | todo | |
| TRAIN-04 | LR schedule: most runs reached 500 epochs without early stopping | P1 | 4 | todo | |
| TRAIN-05 | On-the-fly data generation during training | P2 | 2 | todo | |
| TRAIN-06 | GPU training (TF on native Windows has no GPU): WSL2 / PyTorch / Keras 3 | P1 | 2 | todo | |
| TRAIN-07 | Batched prediction without rounding | P1 | 1 | todo | |
| TRAIN-08 | Metrics in physical units, % within JND, per RIR type and per room | P1 | 3 | todo | |
| CODE-01 | Augment and compute descriptors once per RIR (currently repeated ~30×) | P1 | 2 | todo | |
| CODE-02 | Efficient per-band storage format (npz/HDF5/parquet) | P1 | 2 | todo | |
| CODE-03 | Decide the future of the C++ extension | P2 | 2 | todo | |
| CODE-04 | Installable `reverbnet` package with `pyproject.toml` | P1 | 2 | todo | |
| CODE-05 | De-duplicated configs and config-hashed cache | P1 | 2 | in progress | `fix/BUG-01-causal-convolution` (cache version + file-list digest, v1 shared config) |
| CODE-06 | Inference CLI `predict.py` (windows + VAD + aggregation) | P1 | 4 | todo | |
| CODE-07 | pytest tests (analytic cases) and CI | P1 | 2 | in progress | `fix/BUG-01-causal-convolution` (pytest + first tests) |
| CODE-08 | Pinned and cleaned-up dependencies; updated README | P2 | 2 | todo | |
| CODE-09 | Remove duplicated code (serial generator, functions copied into notebooks) | P2 | 2 | todo | |
| RES-01 | Predict the per-band EDC and derive all descriptors | P3 | 5 | todo | |
| RES-02 | Other input representations (gammatone, log-mel, WavLM) | P3 | 5 | todo | |
| RES-03 | Benchmark with the ACE Challenge protocol | P2 | 5 | todo | |
| RES-04 | Embedded / real-time inference | P3 | 5 | todo | |
| RES-05 | New field measurements with controlled noise | P2 | 5 | todo | |
| RES-06 | Non-shoebox room geometries | P3 | 5 | todo | |
