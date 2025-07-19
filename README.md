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
│  ├─ utils.py                   # Dynamic config loader
│  ├─ parameters_calculation/    # Acoustic descriptors + augmentation fns
│  │   ├─ tae.py, tr_lundeby.py, drr_augmentation.py, ...
│  └─ cpp/                       # C++ sources + build artifacts
│      ├─ audio_processor.[ch]pp
│      ├─ clarity_calculator.[ch]pp
│      ├─ definition_calculator.[ch]pp
│      ├─ octave_filter_bank.[ch]pp
│      ├─ bindings.cpp           # pybind11 module (audio_processing)
│      └─ build/                 # CMake out-of-source build tree
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
├─ calculate_cpp_filter_coefficients.py  # design/emit C++ coeff init code
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

---

## Table of Contents

* [Project Status](#project-status)
* [Quick Start](#quick-start)

  * [Clone & Submodules](#clone--submodules)
  * [Conda Environment](#conda-environment)
  * [Python Dependencies](#python-dependencies)
  * [Build the C++ Extension](#build-the-c-extension)
  * [Smoke Tests](#smoke-tests)
* [Project Layout](#project-layout)
* [Data Preparation](#data-preparation)

  * [Directory Structure](#directory-structure)
  * [Audio Assumptions](#audio-assumptions)
  * [RIR Naming Conventions](#rir-naming-conventions)
* [Experiment Configuration Files](#experiment-configuration-files)

  * [What Each Config Must Define](#what-each-config-must-define)
  * [Provided Experiments (exp1--exp4)](#provided-experiments-exp1--exp4)
* [Database Generation Pipeline](#database-generation-pipeline)

  * [Caching & Chunking](#caching--chunking)
  * [Record Schema](#record-schema)
* [Model & Training](#model--training)

  * [Input Features (TAE)](#input-features-tae)
  * [CNN Architecture](#cnn-architecture)
  * [Training Loop Per Band](#training-loop-per-band)
  * [Early Stopping](#early-stopping)
* [Results & Artifacts](#results--artifacts)
* [Augmentation Utilities](#augmentation-utilities)

  * [TR Augmentation](#tr-augmentation)
  * [DRR Augmentation](#drr-augmentation)
  * [Pink Noise](#pink-noise)
* [C++ Signal‑Processing Backend](#c-signalprocessing-backend)
* [Test Scripts](#test-scripts)
* [Reproducibility Notes](#reproducibility-notes)
* [Troubleshooting](#troubleshooting)
* [Contributing](#contributing)
* [Citation](#citation)
* [License](#license)

---

## Project Status

**Research / prototype quality.** APIs and data formats may change. Expect to read the code.

---

## Quick Start

### Clone & Submodules

This repository vendors **pybind11** as a git submodule (needed to build the C++ extension). After cloning:

```bash
git submodule update --init --recursive
```

### Conda Environment

Export (from an existing environment):

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

If you prefer pip after activation:

```bash
pip install -r requirements.txt
```

> **Tip:** Ensure the Python version you use matches the one the C++ extension was built against (see the `.pyd` filename, e.g., `cp312` for CPython 3.12).

### Build the C++ Extension

The pybind11 module lives under `code/cpp/` and is built with CMake. A helper script is provided:

```bash
python build.py
```

What `build.py` does:

1. Removes any existing `code/cpp/build/` directory.
2. Runs `cmake ..` from that directory.
3. Builds in `Release` configuration.
4. Copies resulting `.lib` (Windows) artifacts up to `code/cpp/`.

The compiled Python extension (`audio_processing*.pyd`) is searched for dynamically at import time (see `code/cpp/audio_processor.py`).

### Smoke Tests

After building, try a few sanity checks from the repo root:

```bash
# Verify the extension loads
python code/cpp/test_compilation.py

# Listen to filtered noise & view magnitude responses
python code/test_filter.py

# Compute C50/C80 on a handful of RIRs
python code/test_clarity.py

# Compute D50 on a handful of RIRs
python code/test_definition.py
```

---

## Project Layout

```
ReverbNet/
├── run.py                    # End‑to‑end DB generation + training driver
├── build.py                  # Helper: build C++ extension via CMake
├── gen_synthetic_RIRs.py     # Bulk synthesize exponential‑decay RIRs
├── calculate_cpp_filter_coefficients.py  # Design octave filters & emit C++ inits
├── configs/                  # Experiment config files (exp1.py .. exp4.py, etc.)
├── data/
│   ├── RIRs/                 # Measured & synthetic room impulse responses
│   └── Speech/
│       ├── train/            # Dry speech clips for training convolutions
│       └── test/             # Dry speech clips for validation/testing convolutions
├── cache/                    # Chunked pickled DB partitions (auto‑generated)
├── results/                  # Model weights & experiment results (auto‑generated)
└── code/
    ├── __init__.py
    ├── model.py              # CNN, training utilities, metrics, save helpers
    ├── generate_database.py  # DataBaseGenerator class (multiprocess friendly)
    ├── data_reader.py        # read_dataset() to assemble band‑specific subsets
    ├── utils.py              # import_configs_objs() dynamic loader
    ├── parameters_calculation/
    │   ├── __init__.py
    │   ├── tae.py
    │   ├── tr_lundeby.py
    │   ├── pink_noise.py
    │   ├── drr_augmentation.py
    │   └── tr_augmentation.py
    └── cpp/
        ├── audio_processor.py        # Dynamic loader for compiled extension
        ├── bindings.cpp              # pybind11 bindings
        ├── audio_processor.[ch]pp    # RMS / SNR utilities
        ├── clarity_calculator.[ch]pp # Cx metrics (C50/C80)
        ├── definition_calculator.[ch]pp # D50 metric
        ├── octave_filter_bank.[ch]pp # Multi‑band IIR filter bank
        └── ... (CMakeLists, build/, etc.)
```

---

## Data Preparation

### Directory Structure

Place your raw assets under `data/` with the following hierarchy:

```
data/
├── RIRs/            # .wav RIR files (measured + synthetic)
└── Speech/
    ├── train/       # Dry speech for training set convolutions
    └── test/        # Dry speech for held‑out testing convolutions
```

### Audio Assumptions

* **Sample rate:** 16 kHz (hard‑coded in several places, including C++ filters).
* **Channels:** Mono.
* **Duration:** \~5 s speech clips are assumed by default. The TAE extractor
  downsamples envelopes to 40 Hz and asserts length == 200 samples.
* Files are peak‑normalized within the pipeline before use.

### RIR Naming Conventions

Experiment configs use filename substring matching to group RIRs by room type
and to select augmentation candidates. The following substrings are expected:

* `sintetica`  → synthetic RIRs generated via `gen_synthetic_RIRs.py`.
* `great_hall`, `octagon`, `classroom` → example real rooms.

Feel free to adopt your own naming scheme; just update the config logic accordingly.

---

## Experiment Configuration Files

Configs are plain Python scripts (see `configs/exp1.py` .. `exp4.py`) that define
a set of module‑level variables. They are dynamically imported at runtime by
`import_configs_objs()` and converted to a dictionary. **Every key becomes a
config entry**, so keep the file clean—only define what you mean to use.

### What Each Config Must Define

Below are the keys read by `run.py` / `DataBaseGenerator` and training code.

| Key                                                                  | Description                                                            |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `seed`                                                               | Global Python RNG seed used for RIR selection and reproducibility.     |
| `exp_num`                                                            | Experiment ID (used to name result folders).                           |
| `files_rirs`                                                         | `os.listdir('data/RIRs')` style list of available RIR filenames.       |
| `to_augmentate`                                                      | Subset of RIR filenames selected for TR/DRR augmentation.              |
| `rirs_for_training`, `rirs_for_testing`                              | Train/test split (filenames).                                          |
| `files_speech_train`, `files_speech_test`                            | Dry speech filename lists.                                             |
| `bands`                                                              | Octave band centers (e.g., `[125, 250, 500, 1000, 2000, 4000, 8000]`). |
| `filter_type`                                                        | String label (currently `'octave band'`).                              |
| `fs`                                                                 | Sample rate (Hz). Must match your audio & filter design (16k).         |
| `order`                                                              | Octave filter order (2 or 4) passed to C++ filter bank.                |
| `max_ruido_dB`                                                       | Max allowed noise floor (dB) for Lundeby T30 calc; acts as S/N gate.   |
| `add_noise`                                                          | Bool: add pink noise to speech before TAE extraction?                  |
| `snr`                                                                | `[min_dB, max_dB]` SNR range for noise injection (uniform random).     |
| `tr_aug`                                                             | `[start, stop, step]` TR60 range (s) for augmentation.                 |
| `drr_aug`                                                            | `[start_dB, stop_dB, step_dB]` DRR range for augmentation.             |
| `sample_frac`                                                        | Fraction of rows to sample when reading DB partitions.                 |
| `random_state`                                                       | Seed for pandas sampling.                                              |
| `filters`, `kernel_size`, `activation`, `pool_size`, `learning_rate` | CNN hyperparams.                                                       |
| `validation_split`, `batch_size`, `epochs`                           | Training hyperparams.                                                  |

> **Note:** Additional module constants (e.g., `train`, `test`, etc.) may appear
> in the config files; only the keys used by the pipeline matter.

### Provided Experiments (exp1–exp4)

| Experiment | Noise Threshold (dB) | Pink Noise? | SNR Range (dB) | Description                            |
| ---------- | -------------------- | ----------- | -------------- | -------------------------------------- |
| exp1       | -60                  | No          | \[-5, 20]      | Baseline full DB, stricter noise gate. |
| exp2       | -45                  | No          | \[-5, 20]      | Same as exp1 but relaxed noise gate.   |
| exp3       | -60                  | Yes         | \[-5,          |                                        |





















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
│  ├─ utils.py                   # Dynamic config loader
│  ├─ parameters_calculation/    # Acoustic descriptors + augmentation fns
│  │   ├─ tae.py, tr_lundeby.py, drr_augmentation.py, ...
│  └─ cpp/                       # C++ sources + build artifacts
│      ├─ audio_processor.[ch]pp
│      ├─ clarity_calculator.[ch]pp
│      ├─ definition_calculator.[ch]pp
│      ├─ octave_filter_bank.[ch]pp
│      ├─ bindings.cpp           # pybind11 module (audio_processing)
│      └─ build/                 # CMake out-of-source build tree
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
├─ calculate_cpp_filter_coefficients.py  # design/emit C++ coeff init code
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

The repo ships with four ready‑to‑run experiment scripts under `configs/` that
sweep two axes: **noise‑floor acceptance** for Lundeby T30 estimation,
and **pink‑noise injection** into the speech prior to TAE extraction.
All other model hyperparameters are identical across these examples.

| Experiment | `max_ruido_dB` | `add_noise` | SNR Range (dB)     | Comment                                              |
| ---------- | -------------- | ----------- | ------------------ | ---------------------------------------------------- |
| `exp1.py`  | -60            | False       | \[-5, 20] (unused) | Baseline; strict S/N gate; clean speech only.        |
| `exp2.py`  | -45            | False       | \[-5, 20] (unused) | Looser gate allows noisier RIRs; still clean speech. |
| `exp3.py`  | -60            | True        | \[-5, 20]          | Strict gate; inject pink noise over uniform SNR.     |
| `exp4.py`  | -45            | True        | \[-5, 20]          | Looser gate + pink‑noise SNR augmentation.           |

> **Note**: The `snr` range is read in all configs; it is only *applied* when
> `add_noise=True`.

---

## Database Generation & Caching

The heavy lifting happens in **`run.py`**, which instantiates
`DataBaseGenerator` with the parameters from your chosen config.

### High‑Level Flow

1. **Config load** via `import_configs_objs()`.
2. **DataBaseGenerator** constructed (paths, augmentation ranges, noise settings, etc.).
3. **Multiprocessing RIR loop**:

   * For each RIR → load, normalize.
   * Determine train/test split membership.
   * For each relevant speech file → convolve (FFT), normalize.
   * Always emit an *original* record.
   * If the RIR is in `to_augmentate` *and* not synthetic (`'sintetica'`), sweep TR augmentation.
   * Optionally nest DRR sweeps within a random subset of TR variants.
   * For each (speech, RIR variant) pair → per‑band processing (OctaveFilterBank → descriptors & TAE).
4. Records buffered and periodically flushed to pickled DataFrames in `cache/<db_name>/<part>.pkl` chunks (50k rows per chunk).

If `cache/<db_name>/` already contains pickles, generation is skipped so you can re‑train quickly.

### Record Schema

Each row in the cached database contains:

| Column          | Type       | Description                                                               |     |                                   |
| --------------- | ---------- | ------------------------------------------------------------------------- | --- | --------------------------------- |
| `ReverbedAudio` | str        | Identifier \`speech                                                       | rir | tag\` (tag encodes augmentation). |
| `type_data`     | str        | `'train'` or `'test'` (inherited from RIR split).                         |     |                                   |
| `band`          | int        | Octave‑band center frequency (Hz).                                        |     |                                   |
| `descriptors`   | list\[4]   | `[T30, C50, C80, D50]` from band‑filtered RIR.                            |     |                                   |
| `drr`           | float      | DRR (dB) of the processed band.                                           |     |                                   |
| `tae`           | list\[200] | 200‑sample TAE vector from reverbed speech (pink noise added if enabled). |     |                                   |
| `snr`           | float/NaN  | Actual SNR drawn for noise injection; NaN when `add_noise=False`.         |     |                                   |

---

## Model & Training

Training is launched from the command line by pointing `run.py` at a config:

```bash
python run.py --config configs/exp1.py
```

`run.py` loads (or builds) the cached database, then iterates over each octave
band in `config['bands']` training a *separate* CNN per band.

### Input Features (TAE)

* TAE computed from the **reverbed speech** in that band.
* Envelope: Hilbert magnitude → low‑pass (SOS) → resample to 40 Hz.
* 5 s @ 16 kHz ⇒ 200 samples; enforced by `assert` in `tae.py`.
* Peak normalized.

### Target Variables

* 4 regression targets per sample: `[T30, C50, C80, D50]` estimated on the **band‑filtered RIR**.
* Targets normalized by their joint (train+test slice) 95th percentile per descriptor.

### CNN Architecture (see `model.py`)

* Input shape: `(200, 1)`.
* 4 × Conv1D → MaxPool → BatchNorm blocks (dropout after 2nd conv).
* Flatten → Dense(4) output (linear) for the descriptors.
* Loss: MSE; Optimizer: Adam (configurable learning rate).

### Training Loop

* `validation_split` fraction carved from training data (Keras built‑in).
* Batch size & epoch count from config.
* Early stopping callback (`create_early_stopping_callback`) monitors `val_loss`,
  patience=100 epochs, and restores best weights.
* Best epoch and min `val_loss` reported after training.

### Evaluation & Saving

After each band model trains:

1. Predict on the band’s *test* set (`prediction()`).
2. Compute per‑descriptor errors (`descriptors_err()`).
3. Save Keras weights (`weights_<band>.weights.h5`).
4. Serialize a results pickle with training curves, predictions, errors, normalization constants, and test data (`save_exp_data()`).

Artifacts are stored in `results/exp<EXP_NUM>/`.

---

## Results & Artifacts

Inside `results/expN/` you will find, per band:

* `weights_<band>.weights.h5` – Trained Keras weights.
* `results_<band>.pickle` – Dict with:

  * `loss`, `val_loss` arrays.
  * `best_epoch`, `best_val_loss`, `total_epochs`.
  * `prediction` arrays and per‑descriptor error lists.
  * Normalization percentiles (T30, C50, C80, D50).
  * `X_test`, `y_test` copies for downstream analysis.

---

## Augmentation Utilities

### TR Augmentation

Adjust reverberation time by:

1. Band‑splitting the RIR.
2. Estimating decay + noise floor (Lundeby‑style) per band.
3. Denoising / modeling late decay.
4. Rescaling to a target TR60.
5. Recombining bands.

See `parameters_calculation/tr_augmentation.py`.

### DRR Augmentation

Scale the direct/early energy relative to the late tail using a Hamming‑window
crossfade and quadratic solve for the required gain (`bhaskara`).

See `parameters_calculation/drr_augmentation.py`.

### Pink Noise

Generate peak‑normalized pink (1/f) noise of arbitrary length for SNR
augmentation in speech bands.

See `parameters_calculation/pink_noise.py`.

---

## C++ Signal‑Processing Backend

Bound into Python as the `audio_processing` module (pybind11).

| Class                  | What it does                                             |
| ---------------------- | -------------------------------------------------------- |
| `AudioProcessor`       | RMS, SNR, and gain factor utilities.                     |
| `ClarityCalculator`    | Cx (C50/C80) clarity metrics in dB.                      |
| `DefinitionCalculator` | D50 speech definition metric (%).                        |
| `OctaveFilterBank`     | Multi‑band octave analyzer (125–8000 Hz; 2nd/4th order). |

The compiled extension is located automatically by `code/cpp/audio_processor.py`.
Rebuild with `python build.py` after editing C++ code.

---

## Test Scripts

Quick functional checks (run from repo root unless noted):

```bash
python test_compilation.py   # imports extension
python test_filter.py        # filters noise, plays audio, plots spectra
python test_definition.py    # computes D50 over sample RIRs
python test_clarity.py       # computes C50/C80 over sample RIRs
```

---

## Reproducibility Notes

* Python RNG seeded via `seed` in config; affects RIR selection & augmentation sampling.
* Multiprocessing workers reseed from wall‑clock time; exact dataset replication across machines may vary unless you freeze augmentation lists.
* TensorFlow training nondeterminism (GPU kernels, etc.) may yield slight metric deltas run‑to‑run.
* Cached DB pickles capture the *generated* dataset; archive `cache/<db_name>/` to reproduce training exactly.

---

## Troubleshooting

**ImportError: audio\_processing not found**  → Rebuild (`python build.py`), confirm Python version matches the `.pyd` tag, ensure `pybind11` submodule initialized.

**AssertionError in `TAE()`**  → Input speech duration must produce 200 samples at 40 Hz after downsampling (≈5 s @ 16 kHz).

**Insufficient S/N (Lundeby NoiseError)**  → Your RIR’s noise floor is above `max_ruido_dB`; relax the threshold in the config (e.g., use exp2/exp4) or clean the RIR.

**Out of memory while building DB**  → Reduce augmentation ranges, fewer speech files, or lower chunk size in `run.py` (default 50k rows).

---

## Contributing

PRs welcome! Please open an issue describing:

* Platform (OS, Python version, TF version).
* Reproduction steps.
* Relevant logs / error traces.

Small doc fixes or portability patches (Linux/macOS build helpers) are especially helpful.

---

## Citation

If you publish work using ReverbNet, please cite the repository. A BibTeX entry
will be provided once a paper/preprint is available.

---

## License

A license file will be added; until then, treat this as "all rights reserved"
for personal / research use only. Contact the author for other uses.

---

### One‑Line Training Example

```bash
python run.py --config configs/exp1.py
```

Happy experimenting!
