"""
Temporal Amplitude Envelope (TAE) feature extraction.

Given a mono audio signal, this routine computes its analytic signal via the
Hilbert transform, takes the magnitude envelope, smooths it with a provided
low‑pass filter (SOS form), downsamples the result to 40 Hz, peak‑normalizes,
and returns a 200‑sample TAE vector.

**Assumption:** Upstream code passes ~5 s signals at 16 kHz. After low‑pass
filtering and resampling to 40 Hz, the resulting envelope should contain
``5 s * 40 Hz = 200`` samples; this is enforced by an ``assert``.

Parameters
----------
data : array_like
    Input audio signal.
fs : int or float
    Sampling rate of ``data`` in Hz.
sos_lowpass_filter : ndarray
    Second‑order sections low‑pass filter (as produced by ``scipy.signal``) used
    to smooth the amplitude envelope before resampling.

Returns
-------
ndarray
    Peak‑normalized TAE vector (length 200 expected).
"""

import numpy as np
from librosa import resample
from scipy.signal import hilbert, sosfilt


# Main function: TAE per frequency band (fullband signal is pre‑banded upstream).
def TAE(data, fs, sos_lowpass_filter):
    # Compute analytic signal (Hilbert transform).
    analytic_signal = hilbert(data)

    # Magnitude envelope.
    amplitude_envelope = np.abs(analytic_signal)

    # Low‑pass filter the envelope to obtain the smoothed TAE trajectory.
    tae = sosfilt(sos_lowpass_filter, amplitude_envelope)

    # Downsample the smoothed envelope to 40 Hz.
    tae = resample(tae, orig_sr=fs, target_sr=40)

    # Peak normalize.
    tae = tae / np.max(np.abs(tae))

    # For 5 s @ 16 kHz input, the TAE should have exactly 200 samples.
    assert(tae.size==200)

    return tae