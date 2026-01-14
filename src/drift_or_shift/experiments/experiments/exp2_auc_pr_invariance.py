"""Experiment 2: ROC/PR invariance check under label shift."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from drift_or_shift import (
    ExperimentConfig,
    PI_TRAIN,
    PI_TEST_GRID,
    SEEDS,
    aggregate_mean_std,
    fit_logistic_regression,
    make_gaussian_binary,
    plot_auc_pr_vs_prevalence,
    predict_logits,
    pr_auc,
    roc_auc,
    save_json,
    save_table,
    timestamped_run_dir,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2: AUC/PR invariance.")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--d", type=int, default=6)
    parser.add_argument("--pi-train", type=float, default=PI_TRAIN)
    parser.add_argument("--pi-tests", type=float, nargs="+", default=list(PI_TEST_GRID))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--results-dir", type=Path, default="results")
    return parser.parse_args()


def _run_experiment(config: ExperimentConfig, results_dir: Path) -> None:
    rows = []
    for seed in config.seeds:
        rng_train = np.random.default_rng(seed)
        mu0 = np.zeros(config.d)
        mu1 = np.zeros(config.d)
        mu1[: min(5, config.d)] = 1.0
        X_train, y_train = make_gaussian_binary(
            config.n_train, config.d, config.pi_train, mu0, mu1, sigma=1.0, rng=rng_train
        )
        model = fit_logistic_regression(X_train, y_train, rng=rng_train)

        for pi_test in config.pi_tests:
            rng_test = np.random.default_rng(seed + int(pi_test * 100))
            X_test, y_test = make_gaussian_binary(
                config.n_test, config.d, pi_test, mu0, mu1, sigma=1.0, rng=rng_test
            )
            scores = predict_logits(model, X_test)
            rows.append(
                {
                    "seed": seed,
                    "pi_test": pi_test,
                    "auc": roc_auc(y_test, scores),
                    "pr_auc": pr_auc(y_test, scores),
                }
            )

    df = pd.DataFrame(rows)
    summary = aggregate_mean_std(df, metrics=("auc", "pr_auc"), groupby="pi_test")

    run_dir = timestamped_run_dir(results_dir, config.exp_name)
    table_path = run_dir / "tables" / f"{config.exp_name}.csv"
    save_table(summary, table_path)

    figure = plot_auc_pr_vs_prevalence(
        summary["pi_test"],
        summary["auc_mean"],
        summary["pr_auc_mean"],
        title=config.exp_name,
    )
    figure_path = run_dir / "figures" / f"{config.exp_name}.png"
    figure.savefig(figure_path)
    figure.clf()

    summary_path = run_dir / f"{config.exp_name}_summary.json"
    summary_data = {
        **config.metadata(),
        "table": str(table_path),
        "figure": str(figure_path),
        "aggregated": summary.to_dict(orient="list"),
    }
    save_json(summary_data, summary_path)


def main() -> None:
    args = _parse_args()
    config = ExperimentConfig(
        exp_name="exp2_auc_pr_invariance",
        n_train=args.n_train,
        n_test=args.n_test,
        d=args.d,
        pi_train=args.pi_train,
        pi_tests=args.pi_tests,
        seeds=args.seeds,
        c10=1.0,
        c01=1.0,
    )
    _run_experiment(config, args.results_dir)


if __name__ == "__main__":
    main()
