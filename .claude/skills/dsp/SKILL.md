---
name: dsp
description: Signal-processing rules for ReverbNet - octave filter bank, speech*RIR convolution, temporal amplitude envelope (TAE), Hilbert transform, resampling, adding noise at a target SNR, normalization, dB conventions and verification with test signals. Use when writing or reviewing any code that filters, convolves, resamples, mixes noise or extracts audio features.
---

# DSP in ReverbNet

## Guiding principle: training = inference

At inference time there is a real recording: `x = s * h + n` (speech convolved with the
room, plus noise), which goes through **filter bank → TAE → model**. Data generation must
reproduce exactly that chain and use **the same functions**. Any step that only exists at
training time (normalizations, crops, noise added at a different point) is a bias the model
can learn and that is absent in the field.

## Project parameters

- `fs = 16000` Hz, mono, 5 s windows (80 000 samples).
- Octave bands `[125, 250, 500, 1000, 2000, 4000, 8000]` Hz. Edges at `fc/√2` and `fc·√2`.
  The 8 kHz band is a high-pass from 5657 Hz (at fs = 16 kHz there is no full octave).
- Filter bank: `code/filters.py::BandpassFilter`, Butterworth SOS. `order=4` in
  `butter(..., btype='bandpass')` yields an **order-8** filter (two poles per order and edge).

## Filters

- Always `output='sos'` + `sosfilt`/`sosfiltfilt`; never `ba` (unstable at high orders).
- **Causal** (`sosfilt`): for signals that emulate the measurement chain (reverberant speech,
  recordings). Introduces group delay, larger at low frequencies.
- **Zero-phase or time-reversed** (`sosfiltfilt`, or filter `h[::-1]` and reverse back): to
  analyze RIR decays; reduces filter bias at short RTs (see the `room-acoustics` skill,
  DATA-08). Note that `sosfiltfilt` squares the magnitude response (effective order doubles).
- When the bank changes, check with `sosfreqz`: ~0 dB gain at each `fc`, −3 dB at the edges
  (−6 dB with `sosfiltfilt`), and an approximately flat sum of all bands.

## Convolution

- Causal system: `y = scipy.signal.oaconvolve(s, h)[:len(s)]` (or `fftconvolve`).
- **Never** `mode='same'`: it centers the output and shifts it by `(len(h)-1)//2` samples (BUG-01).
- Minimal test: with `s = δ[n]`, the output must be `h` truncated.

## Duration and cropping

- Bring signals to exactly N samples with an explicit `fix_length(x, n)` function (crop or
  zero-pad). `librosa.load(duration=5.0)` returns fewer samples if the file is shorter, which
  currently breaks the TAE `assert` silently.
- For speech: prefer random 5 s crops with speech activity (VAD) over always taking the first
  5 s (DATA-06).

## Noise and SNR

- Noise is added **broadband, before the filter bank** (BUG-03):
  `x = y + a·n`, with `a = rms(y) / (rms(n) · 10^(SNR/20))`. Use `code/mixing.py::mix_at_snr`
  and `DataBaseGenerator._build_input_bands`; do not write new mixing code.
- SNR definition: broadband mean power over the whole 5 s window (speech pauses included),
  one draw per recording. Changing it (e.g. active speech only, A-weighting) is a new
  experiment (DATA-09).
- A broadband SNR does not mean the same SNR per band: with pink noise and speech, 4 kHz sits
  ~15 dB and 8 kHz ~19 dB below the broadband value on average. The generator stores the
  resulting per-band SNR in `snr_band`.
- Noise color: `colored_noise(n, beta, rng=rng)` (power density ∝ 1/f^β; `pink_noise` = β 1).
  Room background noise lies between pink (β = 1) and brown (β = 2); pink is the worst case for
  2–8 kHz. The generator draws β per recording from `noise_beta` (DATA-09).
- Per recording, the generator draws in this fixed order: clean/noisy (`clean_fraction`) → β →
  SNR → noise realization, from a generator seeded with the project seed and the recording
  name (`DataBaseGenerator._noise_rng`). Same seed + same recording = same noisy recording.
- The TAE needs ~15 dB of SNR **per band** to resemble the clean one in the dB domain (r ≥ 0.9);
  below ~5 dB it is mostly noise (DATA-09 analysis).
- Verify the achieved SNR after mixing (`10·log10(P_y / P_an)` ≈ target ± 0.01 dB).

## Envelope (TAE)

Current implementation (`code/parameters_calculation/tae.py`): `|hilbert(x)|` → 20 Hz
Butterworth low-pass (`sosfilt`) → `librosa.resample` to 40 Hz → peak normalization →
200 samples.

- The TAE is part of the **definition of the model input**: any change (cutoff, rate, dB
  scale, normalization) is a new experiment (DATA-02) and requires regenerating the database
  and retraining.
- Envelope rate and low-pass go together: cutoff ≤ 0.5 × the new rate.
- If using dB: `20·log10(env + eps)` with a fixed floor (e.g. −80 dB re. peak) so numeric
  noise does not dominate.
- Hilbert over the whole window has edge effects at both ends; keep that in mind when
  analyzing the first/last ~50 ms.

## Resampling

- Audio signals: `librosa.load(path, sr=fs)` (soxr) or `scipy.signal.resample_poly`.
- Do not mix resampling methods between training and inference.

## Levels and dB

- Energy/power: `10·log10`; amplitude: `20·log10`. Always with an epsilon.
- Peak normalization (`x / max(|x|)`) does not change energy-ratio descriptors (C50, C80, D50,
  T30), but it does change RMS and SNR: normalize **after** mixing noise, or not at all.
- Detect peaks with `np.argmax(np.abs(x))`, never `np.argmax(x)` (BUG-07).

## Verification with test signals

Every new DSP function is tested with known-result signals:
- Unit impulse (convolution, filter delay).
- A sine at each band's `fc` (~0 dB gain) and outside the band (attenuation).
- White noise with an exponential envelope of known T (see the analytic values in the
  `room-acoustics` skill).
- Noise at a target SNR (check the measured SNR).
