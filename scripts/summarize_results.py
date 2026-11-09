"""
Summarize saved experiment results as Markdown tables.

Reads ``results/exp<N>/results_<band>.pickle`` (written by ``save_exp_data``),
de-normalizes predictions and targets with the stored 95th-percentile factors,
and prints, per band and descriptor, Spearman's rho and the mean absolute error
in physical units, plus training convergence info.

With ``--by``, it prints instead a detailed table per band and group of test records:
N, MAE, bias (mean of prediction − target) and the percentage of records within the JND,
grouped by ``all`` (no grouping), ``rir_type`` (real / augmented / synthetic), ``room``,
``speaker`` or ``augmentation`` (TRAIN-08). The grouping needs per-record test metadata:
stored in the results pickle by runs after TRAIN-08, or recovered for older runs with
``scripts/recover_test_metadata.py`` (``test_metadata_<band>.pickle``).

The output is meant to be pasted into ``docs/improvements/experiments.md``.

Usage
-----
    python scripts/summarize_results.py --exp 1 2 3 4
    python scripts/summarize_results.py --exp 5 --by rir_type

Notes
-----
* Predictions stored before TRAIN-07 are rounded to 2 decimals in the normalized space,
  so their MAE values carry a small quantization error.
* Run from the repository root.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Until the package is renamed and installable (BUG-08 / CODE-04), make the repository root
# and code/ importable exactly like tests/conftest.py does.
ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "code", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from code.metrics import (  # noqa: E402
    DESCRIPTORS,
    UNITS,
    denormalize,
    descriptor_metrics,
    metadata_sidecar_path,
    metrics_by_group,
)

BANDS = (125, 250, 500, 1000, 2000, 4000, 8000)
GROUPINGS = ("all", "rir_type", "room", "speaker", "augmentation")


def load_band(path: Path) -> dict:
    """
    Load one ``results_<band>.pickle`` with predictions and targets in physical units.

    Returns
    -------
    dict
        The stored results plus ``pred`` and ``true`` (ndarray, shape (n, 4), physical
        units) and ``metadata`` (DataFrame or ``None`` when no metadata is available).
    """
    with open(path, "rb") as handle:
        res = pickle.load(handle)
    norm = [res[f"{name}_perc_95"] for name in DESCRIPTORS]
    res["pred"] = denormalize(res["prediction"], norm)
    res["true"] = denormalize(res["y_test"], norm)

    metadata = res.get("test_metadata")
    band = path.stem.removeprefix("results_")
    sidecar = metadata_sidecar_path(path.parent, int(band))
    if metadata is None and sidecar.exists():
        metadata = pd.read_pickle(sidecar)
    if metadata is not None and len(metadata) != len(res["true"]):
        raise ValueError(
            f"{path}: metadata has {len(metadata)} rows for {len(res['true'])} records")
    res["metadata"] = metadata
    return res


def summarize_band(path: Path) -> dict:
    """Return the headline metrics (ρ, MAE, convergence) of one band."""
    res = load_band(path)
    metrics = {
        "n": len(res["true"]),
        "best_epoch": int(res["best_epoch"]),
        "total_epochs": int(res["total_epochs"]),
        "best_val_loss": float(res["best_val_loss"]),
    }
    for k, name in enumerate(DESCRIPTORS):
        m = descriptor_metrics(res["pred"][:, k], res["true"][:, k], name)
        metrics[f"{name}_rho"] = m["spearman"]
        metrics[f"{name}_mae"] = m["mae"]
    return metrics


def markdown_table(exp_num: str, results_dir: Path) -> str:
    """Build the headline Markdown table for one experiment."""
    header = ["Band [Hz]", "N test", "Best epoch / total"]
    header += [f"{d} ρ / MAE [{u}]" for d, u in zip(DESCRIPTORS, UNITS)]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]

    for band in BANDS:
        path = results_dir / f"exp{exp_num}" / f"results_{band}.pickle"
        if not path.exists():
            continue
        m = summarize_band(path)
        row = [str(band), str(m["n"]), f"{m['best_epoch']} / {m['total_epochs']}"]
        row += [f"{m[f'{d}_rho']:.3f} / {m[f'{d}_mae']:.2f}" for d in DESCRIPTORS]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def breakdown_table(exp_num: str, results_dir: Path, by: str) -> str:
    """
    Build the detailed Markdown table (N, MAE / bias / % within JND) per band and group.

    Raises
    ------
    ValueError
        If a band has no metadata and ``by`` is not ``'all'``.
    """
    header = ["Band [Hz]", by if by != "all" else "Group", "N"]
    header += [f"{d} MAE / bias [{u}] / ≤JND" for d, u in zip(DESCRIPTORS, UNITS)]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]

    for band in BANDS:
        path = results_dir / f"exp{exp_num}" / f"results_{band}.pickle"
        if not path.exists():
            continue
        res = load_band(path)
        if by == "all":
            groups = None
        elif res["metadata"] is None:
            raise ValueError(
                f"{path} has no test metadata; run scripts/recover_test_metadata.py first")
        else:
            groups = res["metadata"][by].to_numpy()
        table = metrics_by_group(res["pred"], res["true"], groups)
        for group, rows in table.groupby("group", sort=True):
            rows = rows.set_index("descriptor")
            row = [str(band), str(group), str(int(rows["n"].iloc[0]))]
            row += [f"{rows.at[d, 'mae']:.2f} / {rows.at[d, 'bias']:+.2f} / "
                    f"{100 * rows.at[d, 'within_jnd']:.0f} %" for d in DESCRIPTORS]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--exp", nargs="+", required=True,
                        help="Experiment ids (e.g. 1 5 5_octagon).")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--by", choices=GROUPINGS,
                        help="Print the detailed table (MAE / bias / %% within JND) "
                             "grouped by this field.")
    args = parser.parse_args()

    # Windows consoles default to cp1252, which cannot encode "ρ".
    sys.stdout.reconfigure(encoding="utf-8")

    for exp_num in args.exp:
        print(f"\n#### exp{exp_num}\n")
        if args.by:
            print(breakdown_table(exp_num, args.results_dir, args.by))
        else:
            print(markdown_table(exp_num, args.results_dir))


if __name__ == "__main__":
    main()
