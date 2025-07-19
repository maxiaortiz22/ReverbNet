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
import os
from progress.bar import IncrementalBar


def read_dataset(band, db_name, sample_frac=1.0, random_state=None, type_data='train'):
    """Read and concatenate cached database partitions for a given band/split."""
    partitions = os.listdir(f'cache/{db_name}')

    bar = IncrementalBar('Reading database', max=len(partitions))

    db = pd.DataFrame()
    for partition in partitions:
        # Load partition.
        aux_df = pd.read_pickle(f'cache/{db_name}/{partition}')
        # Filter by band and split, then accumulate.
        filtered_df = aux_df.loc[(aux_df.band == band) & (aux_df.type_data == type_data)]
        db = pd.concat([db, filtered_df], ignore_index=True)
        bar.next()
    
    db = db.sample(frac=sample_frac, random_state=random_state)
    bar.finish()

    return db