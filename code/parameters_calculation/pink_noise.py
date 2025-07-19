"""
Pink (1/f) noise generator.

This helper wraps :func:`colorednoise.powerlaw_psd_gaussian` to produce a
Gaussian pink-noise sequence of a specified length, then peak-normalizes the
result to ±1.

Parameters
----------
samples : int
    Number of samples to generate.

Returns
-------
ndarray
    Pink-noise signal (float), peak-normalized.
"""

import colorednoise as cn
import numpy as np


def pink_noise(samples):
    # Exponent for power-law PSD: 0=white noise; 1=pink noise; 2=red ("brownian") noise.
    beta = 1

    # Generate noise using the colorednoise helper.
    noise = cn.powerlaw_psd_gaussian(beta, samples)

    # Peak normalize before returning.
    return noise / np.max(np.abs(noise))