# Improvement backlog

Details for each item. **Status** is tracked only on the board ([README.md](README.md)).
Item format: problem → evidence → proposal → acceptance criteria.
File paths refer to the code as it was during the initial review (2026-11-07).

---

## BUG — Correctness (phase 1)

### BUG-01 — `mode='same'` convolution shifts the signal and leaks the RT
- **Files**: `code/generate_database.py` (`_process_entry`).
- **Problem**: `fftconvolve(speech, rir, mode='same')` returns the **central** part of the
  full convolution, starting at index `(len(rir)-1)//2`. The thesis (§4.7.1) describes
  taking the **first** 5 s.
- **Evidence**: measured RIRs are 2.0 s long, so ~1 s of the onset is lost. Synthetic RIRs
  are `RT + 0.5` s long, so the shift is `(RT+0.5)/2` and **encodes the RT**. Since speech is
  cut at 5 s, a clean free decay appears at the end, which does not exist in a real recording.
- **Proposal**: `fftconvolve(speech, rir)[:len(speech)]` (or `oaconvolve`). Use the same
  function for training and inference.
- **Acceptance**: test checking causal alignment (impulse at t=0 → output = RIR);
  regenerated database; v0 vs v1 comparison in `experiments.md`.

### BUG-02 — C50/C80/D50 integrate from sample 0 instead of the direct sound
- **Files**: `code/generate_database.py` (`_calculate_descriptors`),
  `code/cpp/clarity/*`, `code/cpp/definition/*`.
- **Problem**: ISO 3382-1 integrates from the arrival of the direct sound. Currently the
  50/80 ms window starts at sample 0 of the band-filtered RIR.
- **Evidence**: measured onset delay (peak position): great_hall median 33 ms (16–50),
  octagon 31 ms (14–50), classroom 17 ms (10–25). With a 33 ms delay, C50 only sees 17 ms
  after the direct sound. The causal filter also adds group delay at low frequencies.
- **Proposal**: detect the onset on the **broadband** RIR (first crossing above
  peak − 20 dB) and trim before filtering. Use the same onset for every band. See the
  `room-acoustics` skill.
- **Acceptance**: analytic tests (ideal exponential with a pre-delay → same C50 as without
  the delay); recomputed C50/D50 distributions compared against v0.

### BUG-03 — Pink noise added after the filter bank and unfiltered
- **Files**: `code/generate_database.py` (`_get_tae_with_noise`).
- **Problem**: **broadband** pink noise is added to an **already band-filtered** signal. The
  125 Hz band receives noise up to 8 kHz, and the SNR is identical in every band.
- **Proposal**: `x = speech*rir + a*noise` in broadband (SNR defined broadband) → filter
  bank → TAE. The per-band SNR becomes a derived quantity.
- **Acceptance**: the noisy training flow matches inference (real recording → filters → TAE).

### BUG-04 — Train/validation/test leakage
- **Files**: `run.py`, `code/model.py` (`normalize_descriptors`), `configs/exp*.py`.
- **Problem**:
  1. The 95th percentile is computed over train and test together.
  2. `validation_split` splits individual records: the same RIR (with another speech file
     or band) appears in both train and validation → optimistic early stopping.
  3. The test set shares **rooms** with train (great_hall/octagon/classroom positions in
     both) and **speakers** (e.g. `F1s3` in train, `F1s4` in test).
- **Proposal**: group splits (room for real RIRs, source RIR for augmented ones, speaker
  for speech). Leave-one-room-out evaluation. Normalization fitted on train only.
- **Acceptance**: split function with a test guaranteeing disjoint groups.

### BUG-05 — Silenced exceptions
- **Files**: `code/generate_database.py`, `code/parameters_calculation/drr_augmentation.py`.
- **Problem**: `except Exception: continue` throughout the generator hides real errors.
  Example: the `DRRR_buscado` typo (drr_augmentation.py, line 113) raises a `NameError`
  nobody sees. `drr_aug` returns `None` and then `None /= ...` fails. The `TAE` `assert` is
  swallowed too.
- **Proposal**: catch specific exceptions (`NoiseError`, `TrAugmentationError`,
  `ValueError`), count discards by reason and log them with `logging`. Fix the typo.
- **Acceptance**: summary of discards by reason at the end of generation.

### BUG-06 — Reproducibility
- **Files**: `code/generate_database.py`, `configs/exp*.py`.
- **Problem**: `random.sample(...)` is evaluated on every iteration of the RT sweep (it does
  not pick 5 of 29; each variant enters with p = 5/29). The seed comes from the clock.
  `os.listdir` order is not guaranteed, so with the same `seed` the split changes across
  machines. Global `np.random` is used for noise and augmentation.
- **Proposal**: `sorted(os.listdir(...))`; `np.random.default_rng(seed)` derived from
  `(seed, rir_name)` and passed explicitly; pick the DRR subset once.
- **Acceptance**: generating the same database twice yields identical results.

### BUG-07 — Direct-sound detection with `argmax(rir)` without `abs`
- **Files**: `drr_augmentation.py` (`get_DRR`, `drr_aug`), `tr_augmentation.py`
  (`temporal_decompose`).
- **Problem**: if the direct sound has negative polarity, the wrong peak is selected.
  `drr_aug` can also produce a negative index in `rir[:argmax - window]`.
- **Proposal**: a single shared `find_onset(rir, fs)` function (see BUG-02).

### BUG-08 — The `code` package shadows the stdlib `code` module
- **Files**: `code/`, `run.py`.
- **Evidence**: running `python -c "import tensorflow"` from the repo root fails with
  `ModuleNotFoundError: No module named 'filters'`, because TF indirectly imports the stdlib
  `code` module and Python loads the local package instead.
- **Proposal**: solved by CODE-04 (rename to `reverbnet`).

---

## DATA — Data and processing

### DATA-01 — Derive D50 from C50
- **Problem**: `D50 = 100 / (1 + 10^(-C50/10))`. Predicting both separately is redundant
  and can yield inconsistent values.
- **Evidence**: near C50 = 0 dB the slope is ≈ 5.8 %/dB, so a 1 dB error in C50 (1 JND)
  ≈ 5.8 % in D50 (> the 5 % JND). This quantifies the D50 spread observed in the thesis.
  In baseline v0, D50 MAE ranges from 4 to 15 %.
- **Proposal**: predict C50 and derive D50 (or predict the EDC, see RES-01).

### DATA-02 — Higher-resolution and/or dB TAE
- **Problem**: TAE at 40 Hz with a 20 Hz low-pass → ~25 ms resolution, insufficient for the
  first 50–80 ms structure that defines C50/C80/D50. Peak normalization depends on a single
  value.
- **Proposal**: try 100–200 Hz with a matching low-pass; envelope in dB (exponential decay
  becomes linear). Ablation against the current TAE.

### DATA-03 — More realistic synthetic RIRs
- **Problem**: 2900 of 3368 RIRs (86 %) are white noise × a single exponential: same decay
  in every band, no direct sound, no early reflections, no air absorption. Their "direct
  sound" (peak) lands anywhere between 0 and 80 ms.
- **Proposal**: per-band T60 with realistic profiles, explicit direct sound with controlled
  DRR, image-source early reflections (pyroomacoustics) + stochastic tail.

### DATA-04 — More real RIRs
- **Problem**: only 3 real rooms (Great Hall, Octagon, Classroom).
- **Proposal**: add ACE Challenge, BUT ReverbDB, MIT IR Survey, OpenAIR, Arni, dEchorate.
  Check licenses and sample rates.

### DATA-05 — C50/C80 augmentation
- **Problem**: augmentation controls RT and DRR, but C50/C80/D50 end up "random" (future
  work in the thesis).
- **Proposal**: generalize `drr_aug` to scale the energy of the 0–50/80 ms window after the
  direct sound with a smooth crossfade, targeting a given C50.

### DATA-06 — Speech corpus
- **Problem**: 33 training files and 8 test files, and only the first 5 s are used.
- **Proposal**: LibriSpeech / VCTK / EARS with random 5 s crops and VAD to guarantee speech
  activity. Split by speaker.

### DATA-07 — Real noise and a clean/noisy mix
- **Proposal**: besides pink noise, real noise (DEMAND, MUSAN: babble, HVAC, traffic).
  Train on a mix of clean and noisy examples (future work in the thesis).

### DATA-08 — Filter influence on descriptors
- **Problem**: causal Butterworth filters (order 4 → order-8 bandpass) bias the decay at
  low bands and short RTs. The 8 kHz band at fs = 16 kHz is a 5.66–8 kHz high-pass (half
  an octave).
- **Proposal**: for RIR descriptors, time-reversed or zero-phase filtering (check
  Jacobsen's B·T > 16 criterion). Document the 8 kHz band or raise fs.

---

## TRAIN — Model and training

### TRAIN-01 — Multi-band model
- **Proposal**: input `(T, 7)` (the 7 bands as channels) and output `7 × 4` (or per-descriptor
  heads). Low bands benefit from mid-band information.

### TRAIN-02 — Architecture
- **Problem**: ~4.6k parameters; `Flatten` fixes the input length at 200.
- **Proposal**: dilated/residual convolutions, `GlobalAveragePooling1D` (variable-length
  input), ablation against the thesis architecture.

### TRAIN-03 — Normalization and loss
- **Proposal**: z-score with train statistics saved alongside the model; Huber or Gaussian
  NLL (mean + variance) to get per-prediction uncertainty.

### TRAIN-04 — LR schedule and convergence
- **Evidence** (baseline v0): in most runs the best epoch is between 450 and 500 out of 500
  → training did not converge and early stopping (patience 100) almost never kicked in.
- **Proposal**: ReduceLROnPlateau or cosine decay, AdamW, revisit batch size and initial LR.

### TRAIN-05 — On-the-fly data generation
- **Proposal**: `tf.data` / `torch.utils.data.Dataset` combining speech, RIR, noise and SNR
  on the fly. Avoids the ~17 GB cache and gives more variety per epoch. Keep a fixed test set.

### TRAIN-06 — GPU
- **Evidence**: env `ReverbNet` (TF 2.16.1) → `tf.config.list_physical_devices('GPU')`
  returns `[]`. TF ≥ 2.11 does not support GPU on native Windows.
- **Proposal**: WSL2, or migrate to PyTorch / Keras 3 with the torch backend. Record the
  decision in `log.md`.

### TRAIN-07 — Batched prediction without rounding
- **Files**: `code/model.py` (`prediction`, `descriptors_err`).
- **Problem**: predicts one sample at a time (very slow) and rounds to 2 decimals in the
  normalized space (error up to 0.005 × p95).
- **Proposal**: `model.predict(X, batch_size=...)`; vectorized errors without rounding.

### TRAIN-08 — Metrics
- **Proposal**: MAE, RMSE, bias, Pearson and Spearman in physical units; % within JND;
  breakdown by RIR type (real/synthetic/augmented) and by room. Reusable script
  (see `scripts/summarize_results.py`).

---

## CODE — Code and engineering

### CODE-01 — Augment and compute descriptors once per RIR
- **Files**: `code/generate_database.py` (`process_single_rir`).
- **Problem**: RT augmentation (curve_fit, Lundeby, filtering) and descriptors are
  recomputed inside the speech loop (~33× per RIR). Since the augmented tail is random, the
  same variant gets different labels depending on the speech file.
- **Proposal**: generate RIR variants and their descriptors once, then convolve each speech file.

### CODE-02 — Storage
- **Problem**: pickled DataFrames holding Python lists; `read_dataset` reads the full
  4.3 GB twice per band and concatenates inside a loop (quadratic).
- **Proposal**: per-band float32 arrays (npz/HDF5/zarr) or band-partitioned parquet, with
  metadata in columns.

### CODE-03 — Future of the C++ extension
- **Problem**: without the OctaveFilterBank, the C++ code only computes RMS, C50/C80 and D50
  (1–3 lines of numpy each) and copies data into `std::vector`. It adds CMake, a submodule
  and `cp312-win` paths.
- **Options**: (a) remove it and use numpy; (b) keep it only for a real-time / plugin goal
  (RES-04). Record the decision in `log.md`.

### CODE-04 — Installable `reverbnet` package
- **Proposal**: rename `code/` → `reverbnet/`, add `pyproject.toml`, `pip install -e .`,
  consistent relative imports (currently `from filters import ...` depends on `sys.path`).
  Solves BUG-08.

### CODE-05 — De-duplicated configs and hashed cache
- **Problem**: the 4 configs repeat ~100 lines of split logic. The cache name does not
  include bands, fs, filter order, speech list or code version → risk of reusing a stale cache.
- **Proposal**: move split logic into the code; declarative configs (YAML/dataclass) holding
  only the differences; cache name = hash of the parameters that affect it.

### CODE-06 — Inference CLI
- **Proposal**: `python -m reverbnet.predict audio.wav` → 5 s windows with VAD over the whole
  recording, per-band median and uncertainty. Field evaluation currently uses only the first 5 s.

### CODE-07 — Tests and CI
- **Proposal**: `tests/` with pytest: analytic descriptor cases (ideal exponential),
  convolution alignment, disjoint splits, TAE shape. GitHub Actions.

### CODE-08 — Dependencies and README
- **Problem**: `numpy` unpinned with TF 2.16 (requires numpy < 2); dependencies unused by the
  code (numba, sounddevice, resampy, tables); outdated README (mentions the C++ filter bank
  and has a duplicated paragraph at the end).

### CODE-09 — Duplicated code
- **Problem**: `generate_database()` (serial) duplicates `process_single_rir`. The notebooks
  copy `read_dataset` and the inference functions; `get_parameters` calls
  `filter_.filtered_signals`, which does not exist.
- **Proposal**: a single implementation in the package, imported from the notebooks.

---

## RES — Research

### RES-01 — Predict the per-band EDC
Predict the energy decay curve (or multi-exponential parameters, DecayFitNet-style) and
derive T30, EDT, C50, C80 and D50 analytically. Guarantees physical consistency.

### RES-02 — Input representations
Gammatone or log-mel spectrogram with a 2D CNN; self-supervised features (WavLM).

### RES-03 — ACE Challenge benchmark
Evaluate with the ACE Challenge protocol and data (T60 and DRR) to compare with the literature.

### RES-04 — Embedded / real-time inference
The current model (~4.6k parameters) fits in a plugin or on a phone. Related to CODE-03 and
`analyze_model.py`.

### RES-05 — Field measurements with controlled noise
Thesis future work: measure with controlled external noise sources to compare models
trained with and without noise under adverse conditions.

### RES-06 — Non-shoebox geometries
Thesis future work: simulate non-rectangular rooms (ray tracing / hybrid methods) and check
whether the model generalizes.
