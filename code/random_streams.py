"""
Reproducible random streams derived from the project-wide seed.

The config's ``seed`` is the only seed of the project. Each random process gets its
own stream id, and per-item generators are derived from ``(seed, stream, item)``, so:

* draws are independent across processes (noise vs. splits vs. augmentation);
* the draw for an item depends only on the seed and the item identity, not on
  processing order, multiprocessing or which other items exist.

Never reuse or renumber a stream id: that would silently change previously
reproducible databases and splits.
"""

from __future__ import annotations

import zlib

import numpy as np

RNG_STREAMS = {
    'noise': 1,       # clean/noisy decision, noise color, SNR and noise realization (DATA-09)
    'split': 2,       # per-file train/test assignment of synthetic RIRs (BUG-04)
    'validation': 3,  # per-RIR validation assignment within the training set (BUG-04)
    'augment': 4,     # choice of real RIRs to augment (BUG-04 configs)
    'drr_subset': 5,  # TR variants of a RIR that also get DRR variants (BUG-06)
    'tr_tail': 6,     # synthesized noiseless tail of a TR-augmented RIR (BUG-06)
    'training': 7,    # TensorFlow seed of each band's training run (TRAIN-09)
}


def item_rng(seed: int, stream: str, item: str) -> np.random.Generator:
    """
    Return the generator for one item of one random process.

    Parameters
    ----------
    seed : int
        Project-wide seed (the config's ``seed``).
    stream : str
        Key of ``RNG_STREAMS``.
    item : str
        Stable identity of the item (file or recording name). ``zlib.crc32`` is used
        because Python's ``hash`` is salted per process.
    """
    if stream not in RNG_STREAMS:
        raise ValueError(f"Unknown random stream {stream!r}; add it to RNG_STREAMS")
    return np.random.default_rng([int(seed), RNG_STREAMS[stream], zlib.crc32(item.encode('utf-8'))])


def training_seed(seed: int, band: int, replica: int = 0) -> int:
    """
    TensorFlow seed for training one band (TRAIN-09).

    Each band gets its own value, so its result does not depend on which bands were
    trained before it in the same process.

    Parameters
    ----------
    seed : int
        Project-wide seed (the config's ``seed``).
    band : int
        Band center frequency in Hz.
    replica : int, default=0
        Index of a repeated training on the same data, used to measure run-to-run
        variation. It selects another value of this stream; it is not a second project
        seed, and it leaves the database, the splits and the noise untouched.

    Returns
    -------
    int
        Seed in ``[0, 2**31)`` for ``tf.keras.utils.set_random_seed``.

    Raises
    ------
    ValueError
        If ``replica`` is negative.
    """
    if replica < 0:
        raise ValueError(f"`replica` must be >= 0, got {replica}")
    return int(item_rng(seed, 'training', f'{band}|{replica}').integers(2**31))
