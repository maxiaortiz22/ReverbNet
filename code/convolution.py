"""
Causal convolution helpers shared by data generation and inference.

A recording made in a room is the *causal* convolution of the source signal with
the room impulse response (RIR): the output starts with the source and the
reverberant tail extends past it. To emulate a recording window of the same
length as the dry signal, the full convolution is truncated to its first
``len(signal)`` samples.

``scipy.signal.fftconvolve(..., mode='same')`` must **not** be used for this: it
returns the central part of the full convolution, shifting the output by
``(len(rir) - 1) // 2`` samples. That drops the onset of the reverberant signal
and makes the shift depend on the RIR length, which for synthetic RIRs encodes
the reverberation time (see BUG-01 in ``docs/improvements/backlog.md``).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def causal_convolve(signal: np.ndarray, impulse_response: np.ndarray) -> np.ndarray:
    """
    Convolve ``signal`` with ``impulse_response`` and keep the first ``len(signal)`` samples.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Dry (anechoic) input signal.
    impulse_response : ndarray, shape (n_taps,)
        Room impulse response, sampled at the same rate as ``signal``.

    Returns
    -------
    ndarray, shape (n_samples,)
        Causally aligned output: sample ``n`` depends only on ``signal[:n + 1]``.
        Output dtype is float64.

    Raises
    ------
    ValueError
        If either input is not 1-D or is empty.
    """
    signal = np.asarray(signal, dtype=float)
    impulse_response = np.asarray(impulse_response, dtype=float)

    for name, arr in (("signal", signal), ("impulse_response", impulse_response)):
        if arr.ndim != 1:
            raise ValueError(f"`{name}` must be 1-D, got shape {arr.shape}")
        if arr.size == 0:
            raise ValueError(f"`{name}` must not be empty")

    return fftconvolve(signal, impulse_response)[: signal.size]
