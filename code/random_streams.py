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
