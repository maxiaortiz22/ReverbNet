# ReverbNet

Estimating room‑acoustic parameters directly from *in‑situ* speech recordings.

ReverbNet is a research codebase that:

* Builds a large, labeled database by **convolving dry speech recordings with measured (and synthetic) room impulse responses (RIRs)**.
* Optionally performs **data augmentation** in several acoustic dimensions:

  * Reverberation‑time (TR) scaling.
  * Direct‑to‑Reverberant Ratio (DRR) scaling.
  * Pink‑noise addition across a specified SNR range.
* Extracts **Temporal Amplitude Envelope (TAE)** features per octave band using a compiled C++ **OctaveFilterBank**.
* Trains a compact **1‑D CNN** (TensorFlow/Keras) to estimate the per‑band acoustic descriptors **T30, C50, C80, and D50** from the TAE features of reverberant speech.

The project mixes Python (data pipeline, training) and C++ (high‑performance audio primitives exposed via `pybind11`).

---

## Table of Contents

* [Project Status](#project-status)
* [Key Features](#key-features)
* [Repository Layout](#repository-layout)
* [Getting Started](#getting-started)

  * [Clone + Submodules](#clone--submodules)
  * [Conda Environment](#conda-environment)
  * [Python Dependencies](#python-dependencies)
  * [Build the C++ Extension](#build-the-c-extension)
  * [Smoke Tests](#smoke-tests)
* [Data Preparation](#data-preparation)

  * [Directory Structure](#directory-structure)
  * [Audio Format Assumptions](#audio-format-assumptions)
  * [RIR Naming Conventions](#rir-naming-conventions)
* [Experiment Configuration Files](#experiment-configuration-files)

  * [Shared Config Keys](#shared-config-keys)
  * [Provided Experiments (exp1--exp4)](#provided-experiments-exp1--exp4)
* [Database Generation & Caching](#database-generation--caching)
* [Training the Model](#training-the-model)
* [Results & Artifacts](#results--artifacts)
* [Augmentation Utilities](#augmentation-utilities)
* [C++ Audio Primitives](#c-audio-primitives)
* [Reproducibility Notes](#reproducibility-notes)
* [Troubleshooting](#troubleshooting)
* [Citation](#citation)
* [License](#license)

---

## Project Status

This is **research / experimental code**. Interfaces may change; error handling is minimal; no formal packaging yet. Use in controlled experiments and read the source before production use.

---

## Key Features

* End‑to‑end pipeline from raw data to trained CNN acoustic estimator.
* Multiprocessing **database builder** that writes chunked pickles to `cache/` to avoid exhausting RAM.
* Flexible **data augmentation** (TR, DRR, pink noise) controlled via experiment config files.
* **Octave‑band feature extraction** implemented in C++ (pybind11) for speed.
* **Band‑wise training loop:** a separate model is trained for each octave band.
* Experiment reproducibility via explicit config scripts (`configs/exp*.py`).

---

## Repository Layout

A simplified view:

```
ReverbNet/
├─ code/
│  ├─ __init__.py                # top-level Python API exports
│  ├─ model.py                   # CNN model + training utilities
│  ├─ generate_database.py       # DataBaseGenerator class (multiprocessing worker logic)
│  ├─ data_reader.py             # Partitioned DB reader (per band, per split)
│  ├─ filters.py                 # Filter bank
│  ├─ utils.py                   # Dynamic config loader
│  ├─ parameters_calculation/    # Acoustic descriptors + augmentation fns
│  │   ├─ tae.py, tr_lundeby.py, drr_augmentation.py, ...
│  └─ cpp/                       # C++ sources + build artifacts
│      ├─ audio_processor.[ch]pp
│      ├─ bindings.cpp           # pybind11 module (audio_processing)
│      ├─ build/                 # CMake out-of-source build tree
│      │   ├─clarity_calculator.[ch]pp
│      └   └─definition_calculator.[ch]pp
├─ configs/
│  ├─ exp1.py ... exp4.py        # experiment parameter sets
├─ data/
│  ├─ RIRs/                      # measured + synthetic RIR WAVs
│  └─ Speech/
│      ├─ train/                 # dry speech for training convolutions
│      └─ test/                  # dry speech for test convolutions
├─ cache/                        # auto-created; chunked DB pickles live here
├─ results/                      # experiment outputs (weights, pickles)
├─ run.py                        # main entry point for DB build + training
├─ build.py                      # helper to build C++ extension on Windows
├─ gen_synthetic_RIRs.py         # synthetic RIR generator (Schroeder‑style)
└─ test_*.py                     # quick functional tests of C++ bindings
```

---

## Getting Started

### Clone + Submodules

This repo vendors **pybind11** as a git submodule. After cloning:

```bash
git submodule update --init --recursive
```

### Conda Environment

Export (from an existing configured machine):

```bash
conda env export --name ReverbNet > ReverbNet.yml
```

Create on a new machine:

```bash
conda env create -f ReverbNet.yml
```

Activate / deactivate:

```bash
conda activate ReverbNet
# ... work ...
conda deactivate
```

### Python Dependencies

If you prefer `pip` inside the environment (or need to update):

```bash
pip install -r requirements.txt
```

> **Tip:** Some dependencies (TensorFlow, sounddevice) can be platform‑specific;
> ensure your environment matches the Python ABI of the compiled extension (see below).

### Build the C++ Extension

The high‑performance audio routines live in `code/cpp` and are bound to Python
as the module `audio_processing` (a `.pyd` on Windows). A helper script is
provided:

```bash
python build.py
```

`build.py` will:

1. Remove any existing `code/cpp/build/` directory.
2. Recreate it and invoke CMake.
3. Build in **Release** configuration.
4. Copy any produced `.lib` artifacts back into `code/cpp/` (Windows).

The pybind11 wrapper `audio_processor.py` automatically searches common build
output paths (e.g., `build/Release/`) for the compiled extension and imports it.
If import fails, you'll see an informative error with searched paths.

### Smoke Tests

Quick sanity checks for the compiled module & calculators:

```bash
python test_compilation.py        # import the extension
python test_filter.py             # run OctaveFilterBank + listen + plot
python test_definition.py         # batch Definition (D50) calc over sample RIRs
python test_clarity.py            # batch C50/C80 calc over sample RIRs
```

---

## Data Preparation

### Directory Structure

Your `data/` directory should contain:

```
data/
├─ RIRs/            # *.wav impulse responses (measured + synthetic)
└─ Speech/
   ├─ train/        # dry speech WAVs used for training convolutions
   └─ test/         # dry speech WAVs used for evaluation convolutions
```

### Audio Format Assumptions

* **Sample rate:** 16 kHz (hard‑coded in many configs and the C++ filter bank design).
* **Channels:** Mono.
* **Duration (speech):** \~5 s segments are assumed; downstream code asserts that a TAE resampled at 40 Hz equals 200 samples (5 s × 40 Hz).

### RIR Naming Conventions

Several selection / augmentation rules in the configs and database builder rely
on filename substrings:

* `sintetica` → synthetic RIRs generated by `gen_synthetic_RIRs.py`.
* `great_hall`, `octagon`, `classroom` → example room categories used when
  sampling RIRs for augmentation.

Ensure your RIR filenames follow a consistent pattern if you extend the data.

---

## Experiment Configuration Files

Experiment configs are **Python scripts** (not JSON/YAML). They define a set of
global variables that the pipeline imports dynamically at runtime. See
`configs/exp1.py` .. `exp4.py` for working examples.

### Shared Config Keys

Below are the keys consumed by `run.py` and downstream utilities. (All are
required.)

| Key                                        | Description                                                       |
| ------------------------------------------ | ----------------------------------------------------------------- |
| `seed`                                     | Master random seed used to select files and splits.               |
| `exp_num`                                  | Experiment identifier used to name result dirs.                   |
| `train` / `test`                           | Fractions used when partitioning RIRs per category.               |
| `files_rirs`                               | List of *all* RIR filenames in `data/RIRs`.                       |
| `to_augmentate`                            | Subset of RIR filenames selected for TR/DRR augmentation.         |
| `rirs_for_training` / `rirs_for_testing`   | Partitioned RIR filename lists.                                   |
| `files_speech_train` / `files_speech_test` | Dry speech filenames for convolution.                             |
| `bands`                                    | Octave‑band center freqs (Hz) to model (e.g., `[125, ... 8000]`). |
| `filter_type`                              | String label (`'octave band'` currently).                         |
| `fs`                                       | Sample rate (Hz), must match audio data & filter design (16k).    |
| `order`                                    | OctaveFilterBank order (2 or 4).                                  |
| `max_ruido_dB`                             | Noise threshold (dB) for Lundeby TR estimation.                   |
| `add_noise`                                | Bool; if True, inject pink noise when computing TAE.              |
| `snr`                                      | `[low, high]` SNR range (dB) for noise injection.                 |
| `tr_aug`                                   | `[start, stop, step]` TR augmentation sweep (seconds).            |
| `drr_aug`                                  | `[start_dB, stop_dB, step]` DRR augmentation sweep (dB).          |
| `sample_frac`                              | Fraction of DB to sample when reading for training.               |
| `random_state`                             | Seed for DataFrame sampling in `read_dataset`.                    |
| `filters`                                  | CNN conv filter counts per layer.                                 |
| `kernel_size`                              | CNN kernel lengths per layer.                                     |
| `activation`                               | Activation fn names per conv layer.                               |
| `pool_size`                                | MaxPool factors per pooling layer.                                |
| `learning_rate`                            | Optimizer LR for Adam.                                            |
| `validation_split`                         | Fraction of training data for validation.                         |
| `batch_size`                               | Training batch size.                                              |
| `epochs`                                   | Max training epochs (early stopping used).                        |

### Provided Experiments (exp1–exp4)

The included configs explore two noise thresholds (−60 dB vs −45 dB) × pink noise off/on:

| Config    | Noise Threshold | Pink Noise | Notes                               |
| --------- | --------------- | ---------- | ----------------------------------- |
| `exp1.py` | −60 dB          | Off        | Baseline, stricter S/N requirement. |
| `exp2.py` | −45 dB          | Off        | Looser S/N requirement.             |
| `exp3.py` | −60 dB          | On         | Adds pink noise (SNR −5 to +20 dB). |
| `exp4.py` | −45 dB          | On         | As above w/ looser threshold.       |

All four share the same CNN hyperparameters: `filters=[32,18,8,4]`,
`kernel_size=[10,5,5,5]`, ReLU activations, etc. See the files for exact values.

---

## Database Generation & Caching

The heavy lifting happens in `run.py` via the `DataBaseGenerator` class.

### What Gets Generated?

For each (speech file × RIR variant × band) combination, a record is created:

* Reverbed speech name (`ReverbedAudio` → `speech|rir|tag`).
* `type_data` (`train` or `test`, inherited from RIR split).
* `band` (Hz).
* `descriptors` `[T30, C50, C80, D50]` computed from the **band‑filtered RIR**.
* `drr` (per‑record DRR).
* `tae` (200‑sample TAE vector from reverbed speech; noise added if enabled).
* `snr` (actual SNR used when noise injected; NaN otherwise).

### Chunked Pickle Output

Records are accumulated and written in **50k‑row chunks** to:

```
cache/<db_name>/0.pkl
cache/<db_name>/1.pkl
...
```

The `db_name` string encodes noise / augmentation settings so you can build and
reuse multiple databases side‑by‑side. If the cache directory already contains
pickles, generation is skipped.

---

## Training the Model

Invoke training by pointing `run.py` at a config script:

```bash
python run.py --config configs/exp1.py
```

The script:

1. Loads the config via dynamic import.
2. Builds the database (multiprocessing) if needed.
3. Iterates over each octave band in `config['bands']`:

   * Reads the band‑specific subset of the cached DB (`read_dataset`).
   * Extracts TAE inputs (`X`) and descriptor targets (`y`).
   * Normalizes descriptor targets by their 95th percentile (computed jointly over train+test slices so scales match across splits).
   * Builds the CNN (`model()`).
   * Trains with early stopping (`create_early_stopping_callback`).
   * Evaluates on holdout (`prediction`, `descriptors_err`).
   * Saves weights + full experiment artifact pickle (`save_exp_data`).

Console output reports best epoch & validation loss for each band.

---

## Results & Artifacts

Training artifacts are stored under `results/exp<N>/`:

* `weights_<band>.weights.h5` – Keras weights per band.
* `results_<band>.pickle` – Python pickle containing:

  * Loss / val\_loss history.
  * Best epoch / val\_loss.
  * Predictions & per‑descriptor errors.
  * Normalization percentiles (T30, C50, C80, D50).
  * Raw `X_test`, `y_test` arrays for downstream analysis.

Use Jupyter notebooks or your own analysis scripts to inspect model behavior.

---

## Augmentation Utilities

Utilities live under `code/parameters_calculation/` and are imported by the DB generator:

* **`tr_augmentation.py`** – Adjust reverberation time by modeling late‑field decay per band, estimating cross‑points (Lundeby‑style), denoising tails, and rescaling to a target TR.
* **`drr_augmentation.py`** – Adjust DRR by scaling the early portion of an RIR (Hamming crossfade) to meet a target direct/late energy ratio.
* **`pink_noise.py`** – Generate peak‑normalized pink (1/f) noise for SNR augmentation.
* **`tr_lundeby.py`** – Lundeby‑corrected T30 estimation and related helpers.
* **`tae.py`** – Temporal Amplitude Envelope extraction (Hilbert magnitude → low‑pass → resample to 40 Hz → normalize; yields 200 samples for 5 s input).

You can also generate **synthetic exponential‑decay RIRs** across a TR sweep via `gen_synthetic_RIRs.py` (Schroeder reconstruction) to populate `data/RIRs/`.

---

## C++ Audio Primitives

Bound to Python as the module `audio_processing` (see `code/cpp/`):

| Class                  | Purpose                                             |
| ---------------------- | --------------------------------------------------- |
| `AudioProcessor`       | Scalar utilities: RMS, SNR, SNR compensation gain.  |
| `ClarityCalculator`    | Clarity Cx (C50, C80) metrics from IR energy.       |
| `DefinitionCalculator` | Definition (D50) metric from IR energy.             |
| `OctaveFilterBank`     | Cascaded biquad octave‑band analyzer (125‑8000 Hz). |

The filter bank center frequencies and coefficients are currently designed for
**16 kHz** sampling. Use `calculate_cpp_filter_coefficients.py` to regenerate
coefficients if you target a different sample rate or design.

---

## Reproducibility Notes

* Global selection randomness is driven by `seed` in the config; speech/RIR
  sampling uses Python's `random` module.
* The database worker reseeds per RIR using wall‑clock time to diversify nested
  augmentation draws; exact dataset reproduction across machines may therefore
  vary unless you fully control the RIR augmentation loops.
* TensorFlow determinism is not enforced; results may differ across hardware /
  driver stacks.

---

## Troubleshooting

**ImportError: cannot find audio\_processing**
Ensure you ran `python build.py` (or a manual CMake build) *inside* the active
Conda environment and that the Python version matches your interpreter (e.g.,
`cp312` for Python 3.12). The wrapper searches common paths but you can also
append the built module's directory to `PYTHONPATH`.

**AssertionError in TAE()**
Input speech segments must be \~5 s at 16 kHz so that resampling to 40 Hz yields
200 samples. Verify your speech data length.

**Out of memory during DB build**
The generator writes 50k‑row pickles incrementally; if you still see issues,
reduce the number of TR/DRR variations or limit the speech/RIR set.

---

## Citation

If you use ReverbNet in academic work, please cite the repository. A formal
citation entry (BibTeX) will be added here when a paper/preprint is available.

---

## License

TODO – add a license file clarifying terms of use.

---

### One‑Line Training Example

```bash
python run.py --config configs/exp1.py
```

Happy experimenting!

* Pink‑noise addition over a uniform SNR range you specify.
* Per‑band acoustic descriptor calculation using a **C++/pybind11 octave filter bank** for speed and numerical stability.
* Chunked, multiprocessing database generation that scales to very large corpora without exhausting RAM.
* Training of a compact 1‑D CNN (TensorFlow / Keras) to regress T30, C50, C80, and D50 from the speech‑band temporal amplitude envelope (TAE).

The codebase mixes **Python** (data orchestration, ML) and **C++** (fast signal processing exposed through pybind11). It targets 16 kHz mono audio and 5 s speech clips by default (other durations are possible if you adapt the asserts / model input shapes).