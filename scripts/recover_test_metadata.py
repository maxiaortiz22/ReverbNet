"""
Recover the per-record test metadata of runs saved before TRAIN-08.

Results saved before TRAIN-08 store ``X_test``/``y_test`` and the predictions but not which
recording each test row is, so metrics cannot be broken down by RIR type, room or speaker.
The test set of a band is rebuilt with **the same calls as** ``run.py``
(:func:`generator_from_config` for the cache name, :func:`read_dataset` for the records and
:func:`reshape_data` for the arrays), and the rebuilt ``X_test`` and normalized ``y_test``
must match the stored ones exactly; otherwise the script stops without writing anything for
that band.

The metadata of each band is written to a new file ``results/exp<N>/test_metadata_<band>.pickle``
(a pandas DataFrame, see :func:`code.metrics.build_test_metadata`); the existing artifacts of
the run are only read. ``scripts/summarize_results.py --by ...`` picks the file up.

Usage
-----
    python scripts/recover_test_metadata.py --config configs/exp5.py [--bands 125 8000]

Run from the repository root (configs and caches use paths relative to it). Each band reads
the whole cache once (~1–2 min per band for a 5.6 GB database).
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

# Until the package is renamed and installable (BUG-08 / CODE-04), make the repository root
# and code/ importable exactly like tests/conftest.py does; the root must come before the
# standard library so that ``code`` resolves to this project's package.
ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / 'code', ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from code.data_reader import read_dataset  # noqa: E402
from code.generate_database import generator_from_config  # noqa: E402
from code.metrics import DESCRIPTORS, build_test_metadata, metadata_sidecar_path  # noqa: E402
from code.model import reshape_data  # noqa: E402
from code.utils import import_configs_objs  # noqa: E402

logger = logging.getLogger(__name__)


def recover_band(config: dict, db_name: str, band: int, results_dir: Path,
                 overwrite: bool = False) -> Path | None:
    """
    Rebuild and verify the test set of one band, then write its metadata.

    Parameters
    ----------
    config : dict
        Experiment config namespace.
    db_name : str
        Cache folder of the run (``generator_from_config(config).db_name``).
    band : int
        Band in Hz.
    results_dir : Path
        ``results/exp<N>``.
    overwrite : bool, default=False
        Replace an existing metadata file.

    Returns
    -------
    Path or None
        The written file, or ``None`` when the band was skipped (metadata already stored).

    Raises
    ------
    FileNotFoundError
        If the band's results pickle does not exist.
    RuntimeError
        If the rebuilt test set does not match the stored one.
    """
    results_path = results_dir / f'results_{band}.pickle'
    out_path = metadata_sidecar_path(results_dir, band)
    with open(results_path, 'rb') as handle:
        res = pickle.load(handle)
    if 'test_metadata' in res:
        logger.info("%s Hz: metadata already stored in %s; skipping", band, results_path.name)
        return None
    if out_path.exists() and not overwrite:
        logger.info("%s Hz: %s exists; skipping (use --overwrite)", band, out_path.name)
        return None

    db_test = read_dataset(band, db_name, config['sample_frac'], config['random_state'],
                           type_data='test')
    X_test, y_test = reshape_data(list(db_test.tae.to_numpy()),
                                  list(db_test.descriptors.to_numpy()))
    norm = np.array([res[f'{name}_perc_95'] for name in DESCRIPTORS]).reshape(1, 4, 1)

    stored_X = np.asarray(res['X_test'])
    stored_y = np.asarray(res['y_test'], dtype=float)
    if X_test.shape != stored_X.shape or not np.array_equal(X_test, stored_X):
        raise RuntimeError(f"{band} Hz: rebuilt X_test {X_test.shape} does not match the stored "
                           f"one {stored_X.shape}; the cache or the reading order changed")
    if not np.array_equal(y_test / norm, stored_y):
        raise RuntimeError(f"{band} Hz: rebuilt y_test does not match the stored one")

    meta = build_test_metadata(db_test)
    meta.to_pickle(out_path)
    logger.info("%s Hz: %d test records verified; metadata written to %s",
                band, len(meta), out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--config', required=True, help='Config of the run (e.g. configs/exp5.py).')
    parser.add_argument('--bands', nargs='+', type=int, help='Bands to recover (default: all).')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing metadata files.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"Run from the repository root ({ROOT}); "
                         "paths in configs are relative to it.")

    config = import_configs_objs(args.config)
    db_name = generator_from_config(config).db_name
    results_dir = Path('results') / f"exp{config['exp_num']}"
    for band in args.bands or config['bands']:
        recover_band(config, db_name, band, results_dir, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
