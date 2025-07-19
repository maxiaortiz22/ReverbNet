"""
Direct-to-Reverberant Ratio (DRR) helper functions and augmentation utilities.

This module provides:

* :func:`get_DRR` – Compute the direct-to-reverberant ratio (dB) of a room
  impulse response (RIR) by splitting it into early and late segments around
  the direct sound.
* :func:`drr_aug` – Synthesize a new RIR with a *target* DRR by scaling the
  early portion (Hamming-weighted crossfade) while preserving the late tail.
* :func:`bhaskara` – Solve the quadratic that arises when determining the
  early-signal scaling factor needed to reach the desired DRR.

**Note:** Variable names and return structure are preserved in Spanish for compatibility
with existing code; only comments, docstrings, and print messages are in English.
"""

import numpy as np


def get_DRR(rir, fs, window_length=0.0025):
    """
    Calculate the direct-to-reverberant ratio (DRR) of a room impulse response.

    Parameters
    ----------
    rir : numpy.ndarray
        Room impulse response.
    fs : float
        Sampling rate in Hz.
    window_length : float, default=0.0025
        Symmetric window length (seconds) around the direct sound used to
        define the *early* portion.

    Returns
    -------
    tuple
        ``(DRR_dB, early, late)`` where ``early`` and ``late`` are the sliced
        arrays delineating the direct/early region and reverberant tail.

    Raises
    ------
    ValueError
        If the RIR is empty or if the late energy is zero. (Messages remain in Spanish.)
    """
    if len(rir) == 0:
        raise ValueError("La RIR no puede estar vacía")
    
    t_d = np.argmax(rir)  # Direct path sample index.
    t_o = int(window_length * fs)  # Window length in samples.
    init_idx = max(t_d - t_o, 0)
    final_idx = min(t_d + t_o + 1, len(rir))

    early = rir[init_idx:final_idx]
    late = rir[final_idx:]

    energia_late = np.sum(late**2)
    if energia_late == 0:
        raise ValueError("La energía de la parte tardía es cero")

    DRR = 10 * np.log10(np.sum(early**2) / energia_late)
    return DRR, early, late


def drr_aug(rir, fs, DRR_buscado, window_length=0.0025):
    """
    Generate a new RIR with a desired DRR by scaling the early portion.

    The early segment is tapered with a Hamming window and scaled via a
    quadratic solution (see :func:`bhaskara`) to achieve the requested DRR.

    Parameters
    ----------
    rir : numpy.ndarray
        Original room impulse response.
    fs : float
        Sampling rate in Hz.
    DRR_buscado : float
        Target DRR in dB.
    window_length : float, default=0.0025
        Symmetric window length (seconds) around the direct sound.

    Returns
    -------
    numpy.ndarray or None
        Normalized augmented RIR if successful; ``None`` if the requested DRR
        cannot be realized.

    Raises
    ------
    ValueError
        If the RIR is empty or if the late energy is zero.
    """
    if len(rir) == 0:
        raise ValueError("The RIR cannot be empty.")

    DRR_original, early, late = get_DRR(rir, fs, window_length)
    delay = rir[:np.argmax(rir) - int(window_length * fs)]

    w = np.hamming(len(early))  # Hamming window.
    energia_late = np.sum(late**2)
    if energia_late == 0:
        raise ValueError("The energy of the late part is zero.")

    # Solve for alpha scaling factor.
    a = np.sum((w**2) * (early**2))
    b = 2 * np.sum((1 - w) * w * (early**2))
    c = np.sum(((1 - w)**2) * (early**2)) - (10**(DRR_buscado / 10) * energia_late)
    alpha = bhaskara(a, b, c)

    # Abort if no valid alpha (no real solution).
    if alpha is None:
        print(f"Could not generate audio with DRR = {DRRR_buscado:.2f} dB. Skipping...")
        return None

    # Construct new early segment.
    new_early = (alpha * w * early) + ((1 - w) * early)
    if np.max(np.abs(new_early)) < np.max(np.abs(late)):
        print("Desired level is too low; cannot generate audio.")
        return None

    # Assemble augmented RIR and report achieved DRR.
    rir_aug = np.concatenate((delay, new_early, late)).astype(np.float32)
    DRR_obtenido = 10 * np.log10(np.sum(new_early**2) / energia_late)
    print(f"Target DRR: {DRR_buscado:.2f}, obtained DRR: {DRR_obtenido:.2f}")

    return rir_aug / np.max(np.abs(rir_aug))


def bhaskara(a, b, c):
    """
    Solve a quadratic equation ``a*x^2 + b*x + c = 0`` and return the larger root.

    Parameters
    ----------
    a, b, c : float
        Quadratic coefficients.

    Returns
    -------
    float or None
        Largest real root (alpha) if it exists; ``None`` if the discriminant
        is negative (no real solution).
    """
    r = b**2 - 4 * a * c
    if r > 0:
        x1 = ((-b) + np.sqrt(r)) / (2 * a)
        x2 = ((-b) - np.sqrt(r)) / (2 * a)
        return max(x1, x2)
    elif r == 0:
        return -b / (2 * a)
    else:
        return None  # No real solution.