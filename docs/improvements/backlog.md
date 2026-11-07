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
- **Resolution** (2026-11-07, branch `fix/BUG-01-causal-convolution`): new shared helper
  `code/convolution.py::causal_convolve` (full convolution truncated to `len(signal)`), used by
  `DataBaseGenerator._process_entry` and exported from the package for inference.
  `tests/test_convolution.py` covers impulse alignment, delay, output length, equivalence with
  `np.convolve`, and a regression test showing that the output no longer depends on the RIR
  length (the old `mode='same'` fails it). The cache name now carries `_dv1` (see CODE-05) so
  the v0 caches are not reused. Smoke-tested on a real speech file with a measured and a
  synthetic RIR. **Pending**: regenerating the database and measuring the impact, deferred to
  baseline v1 (phase 3) so all P0 fixes are evaluated together.

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
- **Resolution** (2026-11-07, branch `fix/BUG-01-causal-convolution`): new
  `code/parameters_calculation/onset.py` with `find_onset` (first sample ≥ peak − 20 dB,
  polarity-independent) and `trim_to_onset`. `DataBaseGenerator._process_entry` detects the
  onset on the broadband RIR, filters the trimmed RIR and passes it to C50/C80/D50. T30 and DRR
  intentionally keep their v0 definitions (band-peak referenced) so the change is isolated.
  `DATA_PIPELINE_VERSION` → 2. `tests/test_onset.py`: onset detection cases, analytic values
  for T = 1 s with a 33 ms pre-delay (C50 ≈ −0.02 dB, C80 ≈ 3.05 dB, D50 ≈ 49.9 %), and
  generator-level invariance to the pre-delay.
- **Measured impact** (new − old, per band; mean and p5 to p95). Real RIRs (n = 468, onset
  median 24.7 ms, range 7.8–50.8 ms):

  | Band [Hz] | ΔC50 [dB] | ΔC80 [dB] | ΔD50 [%] |
  | --- | --- | --- | --- |
  | 125 | +4.80 (+0.31 to +18.32) | +2.41 (+0.18 to +6.64) | +7.64 (+0.95 to +19.83) |
  | 250 | +4.57 (+0.09 to +18.65) | +2.90 (+0.13 to +8.57) | +10.69 (+0.43 to +31.92) |
  | 500 | +2.95 (+0.10 to +12.21) | +2.08 (+0.13 to +5.50) | +8.57 (+0.49 to +26.89) |
  | 1000 | +2.43 (+0.12 to +7.21) | +1.81 (+0.19 to +4.23) | +9.79 (+0.56 to +34.37) |
  | 2000 | +2.26 (+0.08 to +6.26) | +1.80 (+0.16 to +3.84) | +10.49 (+0.38 to +34.21) |
  | 4000 | +2.08 (+0.09 to +5.59) | +1.77 (+0.26 to +3.54) | +10.30 (+0.25 to +31.03) |
  | 8000 | +1.89 (−0.08 to +5.50) | +1.72 (+0.11 to +3.44) | +9.19 (−0.21 to +28.12) |

  Synthetic RIRs (n = 194, every 15th file): onset ≤ 0.2 ms, all changes ≤ 0.03 dB / 0.13 %.
  So in v0 the real (and real-derived augmented) RIRs had C50/C80/D50 labels biased low by
  well over one JND, while synthetic labels were correct: the training set mixed two
  inconsistent label definitions. The field references (REW) are onset-referenced, so this
  also affected the thesis field comparison.
- **Pending**: the effect on model metrics is measured with baseline v1 (phase 3). The
  band-filter group delay at low frequencies is still present (DATA-08).

### BUG-03 — Pink noise added after the filter bank and unfiltered
- **Files**: `code/generate_database.py` (`_get_tae_with_noise`).
- **Problem**: **broadband** pink noise is added to an **already band-filtered** signal. The
  125 Hz band receives noise up to 8 kHz, and the SNR is identical in every band.
- **Proposal**: `x = speech*rir + a*noise` in broadband (SNR defined broadband) → filter
  bank → TAE. The per-band SNR becomes a derived quantity.
- **Acceptance**: the noisy training flow matches inference (real recording → filters → TAE).
- **Resolution** (2026-11-07, branch `fix/BUG-01-causal-convolution`): new
  `code/mixing.py::mix_at_snr` (broadband mean-power SNR) and
  `DataBaseGenerator._build_input_bands`, which mixes pink noise with the reverberant speech
  broadband and then applies the filter bank; the TAE of every band is computed from that
  single recording. One SNR draw per recording (previously one per band). Records keep `snr`
  (now the broadband SNR) and gain `snr_band` (resulting per-band SNR, diagnostic).
  `pink_noise` accepts an optional `rng`. The C++ `AudioProcessor` (rms, rms_comp) is no longer
  used (see CODE-03). `DATA_PIPELINE_VERSION` → 3. `tests/test_noise_mixing.py`: exact SNR,
  invariance to noise level, mixture-then-filter equivalence, per-band SNR, a regression test
  showing the 125 Hz band has no out-of-band noise (< −60 dB above 2 kHz), and one SNR per
  recording.
- **Measured impact**: see DATA-09. With a broadband SNR, the per-band SNR spreads widely
  (pink noise vs. the speech spectrum): +3 to +7.5 dB above the broadband value at 125–500 Hz,
  −2 dB at 1 kHz, −8 dB at 2 kHz, −14.5 dB at 4 kHz and −19 dB at 8 kHz on average.

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
- **Evidence** (2026-11-07): two training speech files are shorter than 5 s (`F5s3.wav`
  4.69 s, `M8s3.wav` 4.05 s). Their TAE has fewer than 200 samples, the `assert` fails and
  **every record built from them is silently dropped** (all RIRs, all bands, also in the v0
  databases). The thesis models were effectively trained on 28 speech files, not 30.
- **Proposal**: catch specific exceptions (`NoiseError`, `TrAugmentationError`,
  `ValueError`), count discards by reason and log them with `logging`. Fix the typo. Bring
  every speech signal to exactly 5 s with an explicit `fix_length` (zero-pad or crop) and log
  it, or reject short files up front with a clear message.
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
- **Problem**: 30 training files and 8 test files, and only the first 5 s are used.
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

### DATA-09 — Revisit the SNR range and definition
- **Files**: `configs/exp*.py` (`snr`), `code/mixing.py`, `code/generate_database.py`.
- **Problem**: since BUG-03 the SNR is broadband (mean power over the 5 s window). The range
  `[-5, 20]` dB was chosen when it applied per band. With pink noise, high bands end up far
  below the broadband value, so most 4–8 kHz TAEs of a noisy database are noise-dominated.
- **Evidence** (30 training speech files × 8 real RIRs, band SNR − broadband SNR in dB, mean
  and p5 to p95):

  | Band [Hz] | Offset [dB] | Band SNR for broadband −5…20 dB (mean) |
  | --- | --- | --- |
  | 125 | +3.1 (−8.1 to +10.1) | −1.9 to +23.1 |
  | 250 | +7.5 (+3.0 to +10.6) | +2.5 to +27.5 |
  | 500 | +7.2 (+3.1 to +10.1) | +2.2 to +27.2 |
  | 1000 | −2.3 (−8.8 to +2.4) | −7.3 to +17.7 |
  | 2000 | −8.3 (−14.6 to −0.8) | −13.3 to +11.7 |
  | 4000 | −14.5 (−20.6 to −8.6) | −19.5 to +5.5 |
  | 8000 | −19.3 (−29.3 to −11.2) | −24.3 to +0.7 |

- **Proposal**: decide the SNR range from realistic scenarios (e.g. 0–40 dB broadband),
  consider defining the SNR over active speech only (VAD) or A-weighted, and add real noise
  spectra (DATA-07). Report metrics per `snr_band` bucket. Decide before generating the noisy
  variant of baseline v1.
- **Analysis** (2026-11-07; 28 full-length training speech files × 8 real RIRs spread over the
  3 rooms; script kept outside the repo, method described here):
  1. *Definition*. Active-speech power (20 ms frames within 25 dB of the 95th-percentile
     frame) vs. whole-window power: reverberation fills the pauses (93 % of frames active), so
     `SNR_active − SNR_window` = +0.30 dB on average (p95 +0.84 dB). **A VAD-based definition
     would change nothing in practice**; the whole-window definition can stay.
  2. *Noise color dominates the per-band SNR*. Band SNR − broadband SNR (mean, dB) for noise
     with power density ∝ 1/f^β:

     | Band [Hz] | white β=0 | pink β=1 | β=1.5 | brown β=2 |
     | --- | --- | --- | --- | --- |
     | 125 | +9.5 | +2.3 | +10.0 | +21.8 |
     | 250 | +11.2 | +6.9 | +16.2 | +29.4 |
     | 500 | +9.3 | +8.1 | +18.9 | +33.6 |
     | 1000 | −3.4 | −1.7 | +10.7 | +26.9 |
     | 2000 | −12.5 | −7.7 | +6.1 | +23.8 |
     | 4000 | −21.7 | −14.0 | +1.4 | +20.5 |
     | 8000 | −28.4 | −18.5 | −2.0 | +18.1 |

     Pink noise has equal energy per octave; typical room background noise (HVAC, traffic,
     NC/NR-like spectra) falls with frequency per octave, i.e. lies between pink and brown.
     Always using pink noise is therefore a worst case for 2–8 kHz.
  3. *How much band SNR the TAE needs*. Pearson correlation between the clean and the noisy
     TAE of the same band, with noise at a controlled per-band SNR (n = 38 recordings per cell):

     | Band SNR | −5 dB | 0 dB | +5 dB | +10 dB | +15 dB | +20 dB | +30 dB |
     | --- | --- | --- | --- | --- | --- | --- | --- |
     | linear TAE (range over bands) | 0.46–0.93 | 0.75–0.97 | 0.91–0.99 | 0.97–1.00 | ≥ 0.99 | 1.00 | 1.00 |
     | TAE in dB, −60 dB floor (range over bands) | 0.45–0.50 | 0.59–0.65 | 0.75–0.79 | 0.85–0.88 | 0.91–0.93 | 0.95–0.96 | 0.99 |

     The linear correlation is dominated by speech peaks and is optimistic. In dB, which reflects
     the low-level decay that encodes T30, the TAE only resembles the clean one (r ≥ 0.9) from
     **~15 dB of band SNR**, and needs ~20 dB for r ≥ 0.95. Below ~5 dB it is mostly noise.
- **Options**:
  - (a) Keep pink noise and `[-5, 20]` dB: 4–8 kHz bands at −24…+6 dB, mostly unusable for the
    decay; a large part of the noisy database becomes noise.
  - (b) Randomize the noise color per recording (β ∈ [1, 2], pink to brown) and use a broadband
    range of `[0, 40]` dB: every band spans from hard (< 5 dB) to clean (> 30 dB) cases.
  - (c) Option (b) plus a fraction of noise-free recordings (thesis future work, DATA-07).

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
  recomputed inside the speech loop (~30× per RIR, once per training speech file). Since the augmented tail is random, the
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
- **Update** (2026-11-07, BUG-03): `AudioProcessor` (rms, snr, rms_comp) is no longer used by
  the pipeline; only `ClarityCalculator` and `DefinitionCalculator` remain in use.

### CODE-04 — Installable `reverbnet` package
- **Proposal**: rename `code/` → `reverbnet/`, add `pyproject.toml`, `pip install -e .`,
  consistent relative imports (currently `from filters import ...` depends on `sys.path`).
  Solves BUG-08.

### CODE-05 — De-duplicated configs and hashed cache
- **Problem**: the 4 configs repeat ~100 lines of split logic. The cache name does not
  include bands, fs, filter order, speech list or code version → risk of reusing a stale cache.
- **Proposal**: move split logic into the code; declarative configs (YAML/dataclass) holding
  only the differences; cache name = hash of the parameters that affect it.
- **Progress** (2026-11-07, BUG-01 branch): `DATA_PIPELINE_VERSION` in
  `code/generate_database.py`, appended to the cache name as `_dv<N>`. It must be bumped by
  every change that alters generated features or labels. Version 0 = thesis caches (no suffix).
  Config de-duplication and hashing of the remaining parameters are still pending.

### CODE-06 — Inference CLI
- **Proposal**: `python -m reverbnet.predict audio.wav` → 5 s windows with VAD over the whole
  recording, per-band median and uncertainty. Field evaluation currently uses only the first 5 s.

### CODE-07 — Tests and CI
- **Proposal**: `tests/` with pytest: analytic descriptor cases (ideal exponential),
  convolution alignment, disjoint splits, TAE shape. GitHub Actions.
- **Progress** (2026-11-07, BUG-01 branch): pytest 9.2.0 installed in the `ReverbNet` env;
  `tests/conftest.py` (sys.path workaround until CODE-04) and `tests/test_convolution.py`.
  Pending: add pytest to the dependency files, descriptor/TAE/split tests, CI.

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
