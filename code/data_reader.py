"""
Utilities for reading cached database partitions into a pandas DataFrame.

The cached database is expected to live under ``cache/{db_name}`` and contain
multiple pickle partitions (e.g., ``0.pkl``, ``1.pkl`` …) written by the
database generation pipeline. This function filters rows by frequency band
and dataset split (train/test), concatenates all matching rows from every
partition, and optionally subsamples the result.

Parameters
----------
band : int
    Frequency band (Hz) to select (must match values stored in the ``band`` column).
db_name : str
    Name of the cached database directory under ``cache/``.
sample_frac : float, default=1.0
    Fraction of matching rows to sample uniformly at random (0 < frac <= 1).
random_state : int or None, default=None
    Seed passed to ``DataFrame.sample`` for reproducibility.
type_data : {'train', 'test'}, default='train'
    Dataset split label to filter on (matches the ``type_data`` column).

Returns
-------
pandas.DataFrame
    Filtered (and possibly subsampled) view of the cached database for the
    requested band and split.
"""

import pandas as pd
from pathlib import Path
from progress.bar import IncrementalBar

# Written last by run.py when a database is complete (counts, discards by reason).
GENERATION_SUMMARY = 'generation_summary.json'


def read_dataset(band, db_name, sample_frac=1.0, random_state=None, type_data='train'):
    """Read and concatenate cached database partitions for a given band/split."""
    # Only the numbered pickle partitions, in numeric order (os.listdir order is arbitrary
    # and the folder also holds the generation summary).
    partitions = sorted(Path('cache', db_name).glob('*.pkl'), key=lambda p: int(p.stem))

    bar = IncrementalBar('Reading database', max=len(partitions))

    parts = []
    for partition in partitions:
        aux_df = pd.read_pickle(partition)
        # Filter by band and split, then accumulate (concatenated once at the end).
        parts.append(aux_df.loc[(aux_df.band == band) & (aux_df.type_data == type_data)])
        bar.next()

    db = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    db = db.sample(frac=sample_frac, random_state=random_state)
    bar.finish()

    return db