"""
Evaluation metrics and per-record test metadata (TRAIN-08).

Everything here works in **physical units** (T30 in s, C50/C80 in dB, D50 in %): callers
de-normalize predictions and targets with the training p95 factors before calling these
functions. The functions are pure (arrays in, numbers or DataFrames out) so they can be
shared by ``run.py``, ``scripts/summarize_results.py`` and the notebooks.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .splits import SYNTHETIC, rir_group, speaker_of

DESCRIPTORS: tuple[str, ...] = ('T30', 'C50', 'C80', 'D50')
UNITS: tuple[str, ...] = ('s', 'dB', 'dB', '%')

# Just-noticeable differences (thesis table 14, ISO 3382-1): T30 is relative (5 % of the true
# value); C50/C80 are absolute in dB; D50 is absolute in percentage points.
JND: dict[str, tuple[str, float]] = {
    'T30': ('relative', 0.05),
    'C50': ('absolute', 1.0),
    'C80': ('absolute', 1.0),
    'D50': ('absolute', 5.0),
}

# RIR types of a test record.
REAL, AUGMENTED = 'real', 'augmented'
ORIGINAL_TAG = 'original'

METADATA_COLUMNS: tuple[str, ...] = (
    'recording', 'speech', 'speaker', 'rir', 'room', 'rir_type', 'augmentation',
)


def record_metadata(recording_names: Iterable[str]) -> pd.DataFrame:
    """
    Parse database record names ``'speech|rir|augmentation'`` into metadata columns.

    Parameters
    ----------
    recording_names : iterable of str
        ``ReverbedAudio`` values of the records, e.g. ``'F4s1|classroom_00x05y|original'``
        or ``'M9s2|octagon_x12y11|TR_var_0.80'``.

    Returns
    -------
    pandas.DataFrame
        One row per record (same order) with the columns of :data:`METADATA_COLUMNS`:
        ``recording`` (the input name), ``speech`` (speech file stem), ``speaker``,
        ``rir`` (source RIR stem), ``room`` (``'great_hall'``, ``'octagon'``, ``'classroom'``
        or ``'synthetic'``), ``rir_type`` (``'real'``, ``'augmented'`` or ``'synthetic'``) and
        ``augmentation`` (the augmentation tag, ``'original'`` for unmodified RIRs).

    Raises
    ------
    ValueError
        If a name does not have three ``|``-separated fields, or its RIR room or speaker
        cannot be inferred.
    """
    rows = []
    for name in recording_names:
        parts = str(name).split('|')
        if len(parts) != 3:
            raise ValueError(
                f"Unexpected recording name {name!r}: expected 'speech|rir|augmentation'")
        speech, rir, augmentation = parts
        room = rir_group(rir)
        if room == SYNTHETIC:
            rir_type = SYNTHETIC
        else:
            rir_type = REAL if augmentation == ORIGINAL_TAG else AUGMENTED
        rows.append((str(name), speech, speaker_of(speech), rir, room, rir_type, augmentation))
    return pd.DataFrame(rows, columns=list(METADATA_COLUMNS))


# Per-record numeric columns of the database copied into the test metadata when present
# (older caches lack the noise columns).
EXTRA_RECORD_COLUMNS: tuple[str, ...] = ('drr', 'snr', 'snr_band', 'noise_beta')


def build_test_metadata(records: pd.DataFrame) -> pd.DataFrame:
    """
    Metadata of the test records of one band, in the order used for ``X_test``.

    Used by ``run.py`` when saving results and by ``scripts/recover_test_metadata.py``
    for runs saved before TRAIN-08, so both produce the same columns.

    Parameters
    ----------
    records : pandas.DataFrame
        Test records as returned by :func:`code.data_reader.read_dataset` (needs the
        ``ReverbedAudio`` column).

    Returns
    -------
    pandas.DataFrame
        :func:`record_metadata` columns plus those of :data:`EXTRA_RECORD_COLUMNS` that
        exist in ``records``, with a fresh ``RangeIndex``.
    """
    meta = record_metadata(records['ReverbedAudio'])
    for column in EXTRA_RECORD_COLUMNS:
        if column in records.columns:
            meta[column] = records[column].to_numpy(dtype=float)
    return meta


def _correlation(func, x: np.ndarray, y: np.ndarray) -> float:
    """Correlation that returns NaN (instead of warning or raising) for degenerate inputs."""
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float('nan')
    return float(func(x, y)[0])


def descriptor_metrics(pred: np.ndarray, true: np.ndarray, descriptor: str) -> dict[str, float]:
    """
    Error metrics of one descriptor, in its physical unit.

    Parameters
    ----------
    pred, true : ndarray, shape (n,)
        Predicted and true values in physical units (s, dB or %).
    descriptor : {'T30', 'C50', 'C80', 'D50'}
        Selects the JND used for ``within_jnd``.

    Returns
    -------
    dict
        ``n``; ``mae``, ``rmse`` and ``bias`` (mean of ``pred - true``) in the descriptor's
        unit; ``pearson`` and ``spearman`` correlations (NaN when undefined); ``within_jnd``,
        the fraction of records whose absolute error is within the JND (relative to the true
        value for T30).

    Raises
    ------
    ValueError
        If the arrays differ in shape, are empty, or the descriptor is unknown.
    """
    if descriptor not in JND:
        raise ValueError(f"Unknown descriptor {descriptor!r}; expected one of {DESCRIPTORS}")
    pred = np.asarray(pred, dtype=float).ravel()
    true = np.asarray(true, dtype=float).ravel()
    if pred.shape != true.shape:
        raise ValueError(f"`pred` and `true` differ in shape: {pred.shape} vs {true.shape}")
    if pred.size == 0:
        raise ValueError("Cannot compute metrics of an empty set")

    err = pred - true
    kind, jnd = JND[descriptor]
    tolerance = jnd * np.abs(true) if kind == 'relative' else jnd
    return {
        'n': int(pred.size),
        'mae': float(np.mean(np.abs(err))),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'bias': float(np.mean(err)),
        'pearson': _correlation(pearsonr, pred, true),
        'spearman': _correlation(spearmanr, pred, true),
        'within_jnd': float(np.mean(np.abs(err) <= tolerance)),
    }


def denormalize(values: np.ndarray, norm: Sequence[float]) -> np.ndarray:
    """
    Bring normalized descriptors back to physical units.

    Parameters
    ----------
    values : ndarray, shape (n, 4) or (n, 4, 1)
        Normalized ``[T30, C50, C80, D50]`` values (predictions or targets).
    norm : sequence of 4 floats
        Training p95 factors, in the same order.

    Returns
    -------
    ndarray, shape (n, 4)
        Values in s, dB, dB and %.
    """
    values = np.asarray(values, dtype=float).reshape(-1, len(DESCRIPTORS))
    norm = np.asarray(norm, dtype=float).reshape(1, len(DESCRIPTORS))
    return values * norm


def metrics_by_group(pred: np.ndarray, true: np.ndarray,
                     groups: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Metrics of every descriptor, overall or per group of records.

    Parameters
    ----------
    pred, true : ndarray, shape (n, 4)
        Predictions and targets in physical units, columns ``[T30, C50, C80, D50]``.
    groups : sequence of str, length n, optional
        Group label of each record (e.g. the ``rir_type`` or ``room`` metadata column).
        ``None`` computes a single ``'all'`` group.

    Returns
    -------
    pandas.DataFrame
        One row per (group, descriptor) with the columns ``group``, ``descriptor`` and the
        keys of :func:`descriptor_metrics`. Groups are sorted by name.
    """
    pred = np.asarray(pred, dtype=float).reshape(-1, len(DESCRIPTORS))
    true = np.asarray(true, dtype=float).reshape(-1, len(DESCRIPTORS))
    if pred.shape != true.shape:
        raise ValueError(f"`pred` and `true` differ in shape: {pred.shape} vs {true.shape}")
    if groups is None:
        labels = np.full(len(pred), 'all', dtype=object)
    else:
        labels = np.asarray(groups, dtype=object)
    if len(labels) != len(pred):
        raise ValueError(f"`groups` has {len(labels)} labels for {len(pred)} records")

    rows = []
    for group in sorted(set(labels)):
        mask = labels == group
        for k, name in enumerate(DESCRIPTORS):
            rows.append({'group': group, 'descriptor': name,
                         **descriptor_metrics(pred[mask, k], true[mask, k], name)})
    return pd.DataFrame(rows)


def metadata_sidecar_path(results_dir: Path, band: int) -> Path:
    """Path of the recovered test metadata of a band (``scripts/recover_test_metadata.py``)."""
    return Path(results_dir) / f'test_metadata_{band}.pickle'
