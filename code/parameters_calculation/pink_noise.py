"""
Pink (1/f) noise generator.

This helper wraps :func:`colorednoise.powerlaw_psd_gaussian` to produce a
Gaussian pink-noise sequence of a specified length, then peak-normalizes the
result to ±1.

Parameters
----------
samples : int
    Number of samples to generate.
rng : numpy.random.Generator or int or None, default=None
    Random source passed to ``colorednoise`` as ``random_state``. ``None`` uses
    NumPy's global state (legacy behavior, see BUG-06).

Returns
-------
ndarray
    Pink-noise signal (float), peak-normalized.
"""

import colorednoise as cn
import numpy as np


def pink_noise(samples, rng=None):
    # Exponent for power-law PSD: 0=white noise; 1=pink noise; 2=red ("brownian") noise.
    beta = 1

    # Generate noise using the colorednoise helper.
    noise = cn.powerlaw_psd_gaussian(beta, samples, random_state=rng)

    # Peak normalize before returning.
    return noise / np.max(np.abs(noise))
