---
name: room-acoustics
description: Room-acoustics definitions and rules for ReverbNet following ISO 3382-1 - Schroeder decay curve (EDC), Lundeby method, T30/T20/EDT, C50, C80, D50, DRR, direct-sound onset, dynamic range, JND, the C50-D50 relation, analytic values for tests and low-frequency limits. Use when computing, validating, augmenting or interpreting acoustic descriptors or RIRs (real, synthetic or augmented).
---

# Room acoustics in ReverbNet

Normative reference: **ISO 3382-1** (performance spaces). Current implementation:
`code/parameters_calculation/tr_lundeby.py` (T30), `code/cpp/clarity` (C50/C80),
`code/cpp/definition` (D50), `code/parameters_calculation/drr_augmentation.py` (DRR).

## Onset: where integration starts

- All parameters are computed **from the arrival of the direct sound**, not from sample 0 of
  the file. ISO 3382-1: the start is where the signal rises clearly above the background
  noise, before the direct sound, and no more than 20 dB below the maximum.
- Implementation: `code/parameters_calculation/onset.py` → `find_onset(rir)` returns the first
  index where `|h| ≥ max|h| · 10^(-20/20)` on the **broadband** RIR; `trim_to_onset(rir)` returns
  the RIR from there. Trim **before** band filtering and use the same onset for every band.
  Always reuse these functions; do not write new onset logic.
- The project's measured RIRs have 8–51 ms of pre-delay (−20 dB crossing; median 24.7 ms).
  Ignoring it biased C50 by ~2–5 dB and D50 by ~8–10 % on average (BUG-02, fixed in data
  pipeline v2).
- Current scope: C50/C80/D50 are onset-referenced; T30 (Lundeby, from the band peak) and DRR
  keep their v0 definitions.
- Detect peaks with `np.argmax(np.abs(h))` (BUG-07).

## Decay curve (Schroeder) and Lundeby

- EDC: `E(t) = ∫_t^∞ h²(τ) dτ`, normalized to 0 dB at t = onset.
- Background noise in the RIR bends the EDC upwards and lengthens the RT. **Lundeby**
  iteratively estimates the noise level and the decay/noise crossing point, truncates the
  integration there and adds a compensation `C` for the truncated energy.
- `max_ruido_dB` (configs: −60 or −45 dB) discards RIRs whose noise (last 10 %) exceeds that
  level relative to the peak → `NoiseError`. These RIRs are discarded per band, which is why
  there is less data at 125 and 250 Hz.
- Measured on the 468 real RIRs (DATA-10): at −60 dB none survive at 125/250 Hz and only
  278/312 at 500 Hz/1 kHz; at −45 dB all survive from 500 Hz up and 211 at 250 Hz; none at
  125 Hz either way. −45 dB is the ISO minimum for T30, so −60 dB is stricter than needed;
  the v1 configs use −45 dB (decision recorded in DATA-10).
- "Not enough dynamic range above the noise" is always a `NoiseError` (an expected discard),
  also when `lundeby` finds no interval 10 dB above the noise. `tr_lundeby` raises
  `ValueError` only for invalid input (empty, all-zero or non-finite RIR), which is a bug.

## Reverberation time

| Parameter | EDC regression range | Extrapolation |
| --- | --- | --- |
| EDT | 0 to −10 dB | ×6 |
| T20 | −5 to −25 dB | ×3 |
| T30 | −5 to −35 dB | ×2 |

- `T = −60 / slope` (slope in dB/s of the least-squares line).
- Dynamic range: the bottom of the evaluation range must be ≥ 10 dB above the noise → T30
  needs ≥ 45 dB between peak and noise; T20 needs ≥ 35 dB. This explains the two
  `max_ruido_dB` options (−60 conservative, −45 the minimum for T30).
- If there is not enough range for T30, do not make it up: discard, or use T20 explicitly
  (and record it).

## Clarity and definition

- `C_te = 10·log10( ∫_0^te h² / ∫_te^∞ h² )` dB, with `te` = 50 ms (C50) or 80 ms (C80),
  measured from the onset.
- `D50 = ∫_0^50ms h² / ∫_0^∞ h²`. ISO gives it as a 0–1 fraction; the project uses **%**.
- Exact relation: `C50 = 10·log10( D50 / (1 − D50) )` ⇔ `D50 = 1 / (1 + 10^(−C50/10))`.
  Near C50 = 0 dB, 1 dB of C50 ≈ 5.8 % of D50. They carry the same information: predict one
  and derive the other (DATA-01).
- Physical consistency: always **C80 ≥ C50** (the early window grows and the late one
  shrinks). Useful as a sanity check on predictions.

## DRR

- Current implementation: energy within ±2.5 ms of the direct-sound peak vs. the energy after
  it (`get_DRR`, `window_length=0.0025`).
- In low bands a 2.5 ms window is shorter than one period (125 Hz → 8 ms) and the filter
  smears the direct sound: per-band DRR at low frequencies is unreliable.

## JND (just noticeable difference)

Thesis (table 14), based on ISO 3382-1: **T30 5 % relative · C50 1 dB · C80 1 dB · D50 5 %**
(ISO gives EDT 5 %, C80 1 dB, D50 0.05, Ts 10 ms). Report the % of predictions within the JND.

## Analytic values for tests

For an ideal decaying energy `h²(t) = e^(−a·t)` with `a = 6·ln(10)/T = 13.8155/T`
(no pre-delay, no noise):

- `C_te = 10·log10( e^(a·te) − 1 )`
- `D50 = 1 − e^(−a·0.05)`
- With **T = 1 s**: C50 ≈ −0.02 dB · C80 ≈ 3.05 dB · D50 ≈ 49.9 % · T30 = 1 s
  (verified numerically at fs = 16 kHz).
- With a pre-delay (zeros before the direct sound) the values must **not** change if the
  onset is detected correctly (regression test for BUG-02).
- For amplitude-domain synthetic RIRs: `h(t) = n(t)·e^(−6.908·t/T)` (−60 dB of energy at T).
  Use a fixed-seed noise and statistical tolerances (e.g. ±5 % on T30).

## Physical limits to keep in mind

- **Low frequencies**: below the Schroeder frequency `f_s ≈ 2000·√(T/V)` the field is modal
  and the EDC stops being exponential; in small rooms the 125 Hz band gives outliers. Report
  low bands separately.
- **Filter influence**: if the bandwidth × RT product is small (Jacobsen's criterion,
  B·T > 16), the band filter biases the decay. An octave at 125 Hz has B ≈ 88 Hz → reliable
  for T > ~0.18 s. Time-reversed filtering reduces the bias (DATA-08).
- **Air absorption**: shortens the RT at 4–8 kHz; synthetic RIRs should reflect it (DATA-03).
- Truncated RIRs (the measured ones are 2.0 s long): with long RTs the tail is cut and
  Lundeby may fail or underestimate.

## Plausible ranges (sanity check)

T30 0.1–10 s · C50 −15 to +20 dB · C80 −10 to +25 dB · D50 0–100 %. Values outside these
ranges are a bug or an invalid RIR: log and discard, never clip silently.
