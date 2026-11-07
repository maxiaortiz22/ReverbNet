"""
Direct-sound onset detection for room impulse responses.

ISO 3382-1 requires energy-ratio parameters (C50, C80, D50, ...) to be integrated
from the arrival of the direct sound, not from the first sample of the file:
the start of the impulse response is where the signal first rises significantly
above the background noise, before the direct sound, and no more than 20 dB
below the maximum.

Measured RIRs usually carry a pre-delay (propagation time plus leading silence);
in this project 10–50 ms (see BUG-02 in ``docs/improvements/backlog.md``).
Integrating from sample 0 shortens the effective early window and biases
C50/C80/D50 by an amount that depends on the source–receiver distance.
"""

from __future__ import annotations

import numpy as np

DEFAULT_ONSET_THRESHOLD_DB = -20.0


def find_onset(rir: np.ndarray, threshold_db: float = DEFAULT_ONSET_THRESHOLD_DB) -> int:
    """
    Return the index of the direct-sound onset of an impulse response.

    The onset is the first sample whose magnitude reaches ``threshold_db`` relative
    to the absolute peak (polarity-independent). Apply it to the **broadband** RIR
    and reuse the index for every band, so all bands share the same time origin.

    Parameters
    ----------
    rir : ndarray, shape (n_samples,)
        Room impulse response.
    threshold_db : float, default=-20.0
        Detection threshold in dB relative to the peak amplitude. Must be negative.

    Returns
    -------
    int
        Sample index of the onset (``0 <= onset <= argmax(|rir|)``).

    Raises
    ------
    ValueError
        If ``rir`` is not 1-D, is empty or all zeros, or ``threshold_db >= 0``.
    """
    rir = np.asarray(rir, dtype=float)
    if rir.ndim != 1:
        raise ValueError(f"`rir` must be 1-D, got shape {rir.shape}")
    if rir.size == 0:
        raise ValueError("`rir` must not be empty")
    if threshold_db >= 0:
        raise ValueError(f"`threshold_db` must be negative, got {threshold_db}")

    magnitude = np.abs(rir)
    peak = magnitude.max()
    if peak == 0:
        raise ValueError("`rir` is all zeros; the onset is undefined")

    threshold = peak * 10.0 ** (threshold_db / 20.0)
    return int(np.argmax(magnitude >= threshold))


def trim_to_onset(rir: np.ndarray, threshold_db: float = DEFAULT_ONSET_THRESHOLD_DB) -> np.ndarray:
    """
    Return ``rir`` starting at its direct-sound onset (see :func:`find_onset`).

    Parameters
    ----------
    rir : ndarray, shape (n_samples,)
        Room impulse response.
    threshold_db : float, default=-20.0
        Detection threshold in dB relative to the peak amplitude.

    Returns
    -------
    ndarray, shape (n_samples - onset,)
        View of ``rir`` from the onset onwards.
    """
    rir = np.asarray(rir)
    return rir[find_onset(rir, threshold_db):]
