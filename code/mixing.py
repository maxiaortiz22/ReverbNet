"""
Mixing a signal with noise at a target signal-to-noise ratio.

In a real recording the background noise adds to the reverberant speech **before**
any analysis, so the mixture must be built broadband and only then passed through
the filter bank (BUG-03). The SNR is defined broadband as the ratio of mean powers
over the whole window (speech pauses included):

    SNR_dB = 10 * log10( mean(signal**2) / mean((gain * noise)**2) )
"""

from __future__ import annotations

import numpy as np


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, float]:
    """
    Add ``noise`` to ``signal`` scaled so the mixture has the requested broadband SNR.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Clean (reverberant) signal.
    noise : ndarray, shape (n_samples,)
        Noise realization of the same length; its level is irrelevant.
    snr_db : float
        Target SNR in dB (mean-power ratio over the whole window).

    Returns
    -------
    mixture : ndarray, shape (n_samples,)
        ``signal + gain * noise`` (float64).
    gain : float
        Linear gain applied to ``noise``.

    Raises
    ------
    ValueError
        If the inputs are not 1-D arrays of equal, non-zero length, if either has zero
        power, or if ``snr_db`` is not finite.
    """
    signal = np.asarray(signal, dtype=float)
    noise = np.asarray(noise, dtype=float)

    if signal.ndim != 1 or noise.ndim != 1:
        raise ValueError("`signal` and `noise` must be 1-D")
    if signal.size == 0 or signal.size != noise.size:
        raise ValueError(
            f"`signal` and `noise` must have the same non-zero length, got {signal.size} and {noise.size}"
        )
    if not np.isfinite(snr_db):
        raise ValueError(f"`snr_db` must be finite, got {snr_db}")

    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    if signal_power == 0 or noise_power == 0:
        raise ValueError("`signal` and `noise` must have non-zero power")

    gain = float(np.sqrt(signal_power / (noise_power * 10.0 ** (snr_db / 10.0))))
    return signal + gain * noise, gain
