"""
Colored (1/f^beta) noise generators.

These helpers wrap :func:`colorednoise.powerlaw_psd_gaussian` to produce Gaussian
noise whose power spectral density falls as ``1/f**beta`` (0 = white, 1 = pink,
2 = brown/red), peak-normalized to ±1.

Room background noise (HVAC, traffic, NC/NR-like spectra) usually falls with
frequency per octave, between pink (beta = 1) and brown (beta = 2); see DATA-09
in ``docs/improvements/backlog.md``.
"""

import colorednoise as cn
import numpy as np


def colored_noise(samples, beta, rng=None):
    """
    Generate peak-normalized Gaussian noise with a ``1/f**beta`` power spectrum.

    Parameters
    ----------
    samples : int
        Number of samples to generate.
    beta : float
        Spectral exponent of the power density (0 white, 1 pink, 2 brown).
    rng : numpy.random.Generator or int or None, default=None
        Random source passed to ``colorednoise`` as ``random_state``. ``None`` uses
        NumPy's global state (legacy behavior, see BUG-06).

    Returns
    -------
    ndarray, shape (samples,)
        Noise signal, peak-normalized to ±1.
    """
    noise = cn.powerlaw_psd_gaussian(beta, samples, random_state=rng)
    return noise / np.max(np.abs(noise))


def pink_noise(samples, rng=None):
    """Peak-normalized pink (1/f) noise; see :func:`colored_noise`."""
    return colored_noise(samples, 1.0, rng=rng)
