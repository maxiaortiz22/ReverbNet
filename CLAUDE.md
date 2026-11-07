# CLAUDE.md — ReverbNet

Guide for working in this repo. Read it before touching code.

## What it is

ReverbNet blindly estimates room-acoustic parameters (**T30, C50, C80, D50** per octave
band, 125 Hz–8 kHz) from a reverberant speech recording. It is Maximiliano Ortiz's thesis
project (`thesis/*.pdf`).

Current pipeline:
1. Anechoic speech (`data/Speech/{train,test}`) ⊛ RIRs (`data/RIRs`: 468 real ones from
   Great Hall, Octagon and Classroom + 2900 synthetic `sintetica_*`), with RT and DRR
   augmentation on a subset of the real RIRs.
2. Octave filter bank (`code/filters.py`, scipy SOS Butterworth).
3. Descriptors of each band-filtered RIR (Lundeby + Schroeder for T30; C50/C80/D50 in C++,
   integrated from the broadband direct-sound onset since data pipeline v2).
4. Per-band TAE of the reverberant speech (Hilbert → 20 Hz low-pass → 40 Hz → 200 samples).
5. One 1D CNN per band (TF/Keras) predicting the 4 descriptors normalized by their p95.

## Status and priorities

Improvement work is tracked in **`docs/improvements/`**:
- `README.md`: board with the status of each item (`BUG-xx`, `DATA-xx`, `TRAIN-xx`, `CODE-xx`, `RES-xx`).
- `backlog.md`: details of each item.
- `experiments.md`: metrics of each experiment; **baseline v0** is the thesis.
- `log.md`: decision log.

There are known correctness bugs (BUG-01 to BUG-08) that partially invalidate the current
metrics. **Before proposing or implementing anything, check the backlog** so you neither
duplicate nor contradict what was already decided. When a change is finished, update the
tracking (`improvement-tracking` skill).

## Project skills (`.claude/skills/`)

| Skill | When to use it |
| --- | --- |
| `improvement-tracking` | When starting, finishing or discovering an improvement; maintains `docs/improvements/`. |
| `python-code` | When writing or modifying any Python code. |
| `cpp-code` | When touching `code/cpp/` (C++17 + pybind11 + CMake). |
| `dsp` | Filters, convolution, envelopes, resampling, noise, normalization. |
| `room-acoustics` | Computing or validating T30/EDT/C50/C80/D50/DRR, Lundeby, onset, JND. |
| `neural-network` | Architecture, training, splits, target normalization, checkpoints. |
| `data-analysis` | Notebooks, metrics, plots, analysis of the database or of results. |
| `experiments` | Running, recording and comparing experiments; configs, cache and results. |

## Environment

- Windows 11 + conda. Main env: **`ReverbNet`** (Python 3.12.0, numpy 1.26.4, scipy 1.15.3,
  librosa 0.11.0, pandas 2.3.0, TensorFlow 2.16.1). There is also a `ReverbNetTest` env.
- Env interpreter: `C:/Users/maxia/anaconda3/envs/ReverbNet/python.exe`.
- **TF sees no GPU** (native Windows, TF ≥ 2.11): training runs on CPU (TRAIN-06).
- `pytest` is installed in the env but not yet listed in the dependency files (CODE-07).
- The compiled C++ extension lives at `code/cpp/build/Release/audio_processing.cp312-win_amd64.pyd`.

## Commands

```bash
git submodule update --init --recursive      # pybind11
python build.py                              # build the C++ extension (Windows, CMake)
python scripts/test_compilation.py           # smoke test for the extension
python -m pytest -q tests                    # unit tests (add --basetemp=<dir> if the temp dir is not writable)
python run.py --config configs/exp1.py       # build the database (if not cached) and train all 7 bands
python scripts/summarize_results.py --exp 1  # metrics table for experiments.md
```

## Known pitfalls

- **The package is named `code` and shadows the stdlib `code` module** (BUG-08). From the
  repo root, `python -c "import tensorflow"` fails. `run.py` works because it runs
  `sys.path.append('code')` before importing. For quick checks that import TF, run from
  another directory or through `run.py`.
- `code/generate_database.py` uses `from filters import BandpassFilter` (an absolute import
  that depends on `sys.path`).
- Paths are **relative to the repo root** (`data/`, `cache/`, `results/`); configs run
  `os.listdir('data/RIRs')` at import time. Notebooks use `../`.
- The Spanish layer names in `code/model.py` (`'Audio de entrada'`, `'capa1_conv'`…) are kept
  for compatibility with the checkpoints in `results/`.
- Data generation uses `ProcessPoolExecutor` (spawn on Windows): everything passed to workers
  must be picklable and the entry point needs `if __name__ == "__main__":`.
- `cache/` takes ~17 GB (4 databases of ~4.3 GB, the thesis/v0 caches without suffix). The cache
  name ends with `_dv<DATA_PIPELINE_VERSION>` (`code/generate_database.py`): **bump that constant**
  whenever a change alters the generated features or labels (CODE-05).
- `configs/exp1..4.py` write into `results/exp1..4` (baseline v0). New runs need new configs
  with a new `exp_num`.
- `data/RIRs/*.wav`, `cache/**/*.pkl` and `results/**/*.pickle` are in `.gitignore`.

## Conventions

- **Language**: everything in the repository (code, docstrings, comments, docs, skills,
  tracking, commit messages) is written in **English**. Conversation with the maintainer
  may be in Spanish.
- Audio conventions: `fs = 16000`, mono, 5 s windows, bands
  `[125, 250, 500, 1000, 2000, 4000, 8000]`.
- **Branches**: one per item or coherent group: `fix/BUG-01-...`, `feat/...`, `refactor/...`,
  `exp/...`, `chore/...`. Never work directly on `main`. Commits mention the item ID.
- Every change that affects data or metrics is validated with an experiment recorded in
  `docs/improvements/experiments.md`, compared against the current baseline.
- Whatever is processed at training time and at inference time must go through **the same
  function** (filters, TAE, normalization).
- Never modify the `results/` of previous experiments: each new run uses a new `exp_num`.
