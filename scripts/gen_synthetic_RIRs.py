"""
Generate synthetic room impulse responses (RIRs) and write them to disk.

The helper function :func:`syntheticRIR` builds a simple synthetic RIR whose
envelope decays exponentially according to a target reverberation time (``Rt``).
White Gaussian noise is amplitude‑shaped by this decay, producing a stochastic
impulse response realization. The result is peak‑normalized (|y| <= 1) and the
random seed used to generate the noise is returned for reproducibility.

When executed as a script, the module sweeps reverberation times from 0.2 s to
3.0 s in 0.1 s steps. For each RT value, 100 synthetic RIRs are generated and
written as WAV files into ``../data/RIRs/`` using the filename pattern:

``sintetica_Seed{seed}_Tr{TR}.wav``

Notes
-----
The output path and filename convention are relied upon by downstream scripts
that glob for ``'sintetica'`` in the filename. Do not change the path or naming
scheme unless you update all dependent code.
"""

import numpy as np
from random import randrange


def syntheticRIR(Rt, fs):
    """
    Create a synthetic RIR using an exponential Schroeder‑style decay.

    Parameters
    ----------
    Rt : float
        Target reverberation time in seconds.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    y : ndarray
        Peak‑normalized synthetic RIR signal.
    seed : int
        RNG seed used to generate the white noise component (for reproducibility).

    Notes
    -----
    The decay envelope is implemented as ``exp((-6.9 * t) / Rt)``, which
    approximates a -60 dB decay over the specified reverberation time.
    """
    # Time vector extends 0.5 s beyond Rt to allow the tail to settle.
    t = np.arange(0, Rt + 0.5, 1 / fs)

    # Exponentially decaying envelope (approx. Schroeder decay).
    y = np.e**((-6.9 * t) / Rt)

    # Random seed for reproducibility of the noise realization.
    seed = randrange(int(2**32))
    np.random.seed(seed)

    # White Gaussian noise.
    n = np.random.normal(0, 1, y.shape)

    # Apply the decay envelope to the noise.
    y = y * n

    # Peak normalize.
    return y / np.max(np.abs(y)), seed


if __name__ == '__main__':
    import soundfile as sf

    # TR values from 0.2 to 3.0 s (inclusive) in 0.1 s steps.
    TRs = np.arange(0.2, 3.1, 0.1, dtype=float)
    fs = 16000

    COUNT = 1
    for i, TR in enumerate(TRs):
        TR = np.round(TR, 2)

        for u in range(1, 101):
            # Generate a synthetic RIR.
            RIR, seed = syntheticRIR(TR, fs)

            # Write audio file.
            sf.write(f'../data/RIRs/sintetica_Seed{seed}_Tr{TR}.wav', RIR, fs)

            # Status message.
            print(f'Generated {COUNT} synthetic RIRs! sintetica_Seed{seed}_Tr{TR}.wav')
            COUNT += 1