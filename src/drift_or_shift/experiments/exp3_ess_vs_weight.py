"""Experiment 3: ESS degradation as class weight alpha increases."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from drift_or_shift import (
    PI_TRAIN,
    SEEDS,
    ExperimentConfig,
    ess_fraction,
    plot_ess_vs_alpha,
    save_json,
    save_table,
    timestamped_run_dir,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 3: ESS vs class weight alpha."
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--pi", type=float, default=PI_TRAIN)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1, 5, 10, 20, 50])
    parser.add_argument("--results-dir", type=Path, default="results")
    return parser.parse_args()


def _run_experiment(
    config: ExperimentConfig, alphas: list[float], results_dir: Path
) -> None:
    rows = []
    for alpha in alphas:
        rows.append(
            {
                "alpha": alpha,
                "ess_fraction": ess_fraction(config.n_train, config.pi_train, alpha),
            }
        )

    df = pd.DataFrame(rows)
    run_dir = timestamped_run_dir(results_dir, config.exp_name)
    table_path = run_dir / "tables" / f"{config.exp_name}.csv"
    save_table(df, table_path)

    figure = plot_ess_vs_alpha(df["alpha"], df["ess_fraction"], title=config.exp_name)
    figure_path = run_dir / "figures" / f"{config.exp_name}.png"
    figure.savefig(figure_path)
    figure.clf()

    summary_path = run_dir / f"{config.exp_name}_summary.json"
    summary_data = {
        **config.metadata(),
        "table": str(table_path),
        "figure": str(figure_path),
        "alpha": df["alpha"].tolist(),
        "ess_fraction": df["ess_fraction"].tolist(),
    }
    save_json(summary_data, summary_path)


def main() -> None:
    args = _parse_args()
    config = ExperimentConfig(
        exp_name="exp3_ess_vs_weight",
        n_train=args.n,
        n_test=args.n,
        d=1,
        pi_train=args.pi,
        seeds=SEEDS,
    )
    _run_experiment(config, args.alphas, args.results_dir)


if __name__ == "__main__":
    main()
