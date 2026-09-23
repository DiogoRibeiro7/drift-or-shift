"""Aggregate raw benchmark outputs into summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def _load_config(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _std_ddof0(series: pd.Series) -> float:
    return float(series.std(ddof=0))


def _load_raw_data(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw CSVs found in {raw_dir}")
    return pd.concat((pd.read_csv(path) for path in files), ignore_index=True)


def run_make_tables(config_path: Path | str) -> Path:
    config = _load_config(config_path)
    raw_dir = Path(config["outputs"]["raw"])
    df = _load_raw_data(raw_dir)
    # Select the columns the metrics need before applying. Operating on the
    # grouping columns too is deprecated in pandas 2.2, and the alternative
    # flag (include_groups=False) does not exist below that version.
    grouped = df.groupby(["estimator", "calibration", "fold"])[["y_true", "y_prob"]]
    metrics = grouped.apply(
        lambda group: pd.Series(
            {
                "brier": float(brier_score_loss(group["y_true"], group["y_prob"])),
                # No eps= here: scikit-learn deprecated it in 1.3 and removed
                # it in 1.5, and now clips probabilities internally.
                "log_loss": float(
                    log_loss(group["y_true"], group["y_prob"], labels=[0, 1])
                ),
                "roc_auc": float(roc_auc_score(group["y_true"], group["y_prob"])),
            }
        )
    )
    metrics = metrics.reset_index()
    summary = (
        metrics.groupby(["estimator", "calibration"])
        .agg(
            brier_mean=("brier", "mean"),
            brier_std=("brier", _std_ddof0),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", _std_ddof0),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", _std_ddof0),
        )
        .reset_index()
    )

    tables_dir = Path(config["outputs"]["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "summary.csv"
    summary.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create summary tables from benchmark outputs."
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to YAML config."
    )
    args = parser.parse_args()
    run_make_tables(args.config)


if __name__ == "__main__":
    main()
