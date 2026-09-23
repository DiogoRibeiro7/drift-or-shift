"""Produce reliability diagrams and other figures from benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve


def _load_config(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_raw_data(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw CSVs found in {raw_dir}")
    return pd.concat((pd.read_csv(path) for path in files), ignore_index=True)


def _parse_method(method: str) -> tuple[str, str]:
    if "|" not in method:
        raise ValueError("Method string must be of form 'estimator|calibration'.")
    estimator, calibration = method.split("|", 1)
    return estimator, calibration


def run_make_figures(config_path: Path | str) -> Path:
    config = _load_config(config_path)
    df = _load_raw_data(Path(config["outputs"]["raw"]))
    figure_dir = Path(config["outputs"]["figures"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    methods = config.get("figures", {}).get("reliability", {}).get("methods")
    if not methods:
        combos = sorted(
            {
                f"{est}|{cal}"
                for est, cal in zip(df["estimator"], df["calibration"], strict=True)
            }
        )
        methods = combos
    fig, ax = plt.subplots(figsize=(6, 5))
    for method in methods:
        estimator, calibration = _parse_method(method)
        subset = df[(df["estimator"] == estimator) & (df["calibration"] == calibration)]
        if subset.empty:
            continue
        prob_true, prob_pred = calibration_curve(
            subset["y_true"], subset["y_prob"], n_bins=10, strategy="uniform"
        )
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            label=method,
            linestyle="-",
        )
    ax.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    out_path = figure_dir / "reliability.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build figures from benchmark outputs."
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to YAML config."
    )
    args = parser.parse_args()
    run_make_figures(args.config)


if __name__ == "__main__":
    main()
