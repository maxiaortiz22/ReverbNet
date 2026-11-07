"""
Summarize saved experiment results as Markdown tables.

Reads ``results/exp<N>/results_<band>.pickle`` (written by ``save_exp_data``),
de-normalizes predictions and targets with the stored 95th-percentile factors,
and prints, per band and descriptor, Spearman's rho and the mean absolute error
in physical units, plus training convergence info.

The output is meant to be pasted into ``docs/improvements/experiments.md``.

Usage
-----
    python scripts/summarize_results.py --exp 1 2 3 4

Notes
-----
* Predictions stored by the original pipeline are rounded to 2 decimals in the
  normalized space (see TRAIN-07), so MAE values carry a small quantization error.
* Run from the repository root.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

DESCRIPTORS = ("T30", "C50", "C80", "D50")
UNITS = ("s", "dB", "dB", "%")
BANDS = (125, 250, 500, 1000, 2000, 4000, 8000)


def summarize_band(path: Path) -> dict:
    """Return metrics for a single ``results_<band>.pickle`` file."""
    with open(path, "rb") as handle:
        res = pickle.load(handle)

    pred = np.asarray(res["prediction"], dtype=float).reshape(-1, 4)
    true = np.asarray(res["y_test"], dtype=float).reshape(-1, 4)
    norm = np.array([res[f"{name}_perc_95"] for name in DESCRIPTORS])
    pred, true = pred * norm, true * norm

    metrics = {
        "n": len(true),
        "best_epoch": int(res["best_epoch"]),
        "total_epochs": int(res["total_epochs"]),
        "best_val_loss": float(res["best_val_loss"]),
    }
    for k, name in enumerate(DESCRIPTORS):
        metrics[f"{name}_rho"] = spearmanr(pred[:, k], true[:, k]).correlation
        metrics[f"{name}_mae"] = float(np.mean(np.abs(pred[:, k] - true[:, k])))
    return metrics


def markdown_table(exp_num: str, results_dir: Path) -> str:
    """Build the Markdown table for one experiment."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--exp", nargs="+", required=True, help="Experiment ids (e.g. 1 5 5_octagon).")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    # Windows consoles default to cp1252, which cannot encode "ρ".
    sys.stdout.reconfigure(encoding="utf-8")

    for exp_num in args.exp:
        print(f"\n#### exp{exp_num}\n")
        print(markdown_table(exp_num, args.results_dir))


if __name__ == "__main__":
    main()
