"""Experiment 4: Concept drift where offset correction fails."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drift_or_shift import (
    ExperimentConfig,
    DEFAULT_COSTS,
    aggregate_mean_std,
    apply_logit_offset,
    concept_drift_mu1,
    fit_logistic_regression,
    logit_offset,
    make_gaussian_binary,
    predict_logits,
    risk_cost_sensitive,
    save_json,
    save_table,
    timestamped_run_dir,
    threshold_from_costs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 4: concept drift comparison.")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--d", type=int, default=6)
    parser.add_argument("--pi-train", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--results-dir", type=Path, default="results")
    return parser.parse_args()


def _run_experiment(config: ExperimentConfig, results_dir: Path) -> None:
    rows = []
    for seed in config.seeds:
        rng_train = np.random.default_rng(seed)
        mu0 = np.zeros(config.d)
        mu1 = np.zeros(config.d)
        mu1[: min(5, config.d)] = 1.0
        mu1_drift = concept_drift_mu1(mu1, shift_dim=min(5, config.d - 1), delta=0.5)

        X_train, y_train = make_gaussian_binary(
            config.n_train, config.d, config.pi_train, mu0, mu1, sigma=1.0, rng=rng_train
        )
        model = fit_logistic_regression(X_train, y_train, rng=rng_train)
        threshold = threshold_from_costs(config.pi_train, config.c10, config.c01)

        rng_test = np.random.default_rng(seed + 42)
        X_test, y_test = make_gaussian_binary(
            config.n_test, config.d, config.pi_train, mu0, mu1_drift, sigma=1.0, rng=rng_test
        )

        scores = predict_logits(model, X_test)
        risk_none = risk_cost_sensitive(
            y_test, (scores >= threshold).astype(int), config.c10, config.c01
        )

        offset = logit_offset(config.pi_train, config.pi_train)
        shifted_scores = apply_logit_offset(scores, offset)
        risk_offset = risk_cost_sensitive(
            y_test, (shifted_scores >= threshold).astype(int), config.c10, config.c01
        )

        rng_retrain = np.random.default_rng(seed + 99)
        X_retrain, y_retrain = make_gaussian_binary(
            config.n_train, config.d, config.pi_train, mu0, mu1_drift, sigma=1.0, rng=rng_retrain
        )
        retrained_model = fit_logistic_regression(X_retrain, y_retrain, rng=rng_retrain)
        retrained_scores = predict_logits(retrained_model, X_test)
        risk_retrain = risk_cost_sensitive(
            y_test, (retrained_scores >= threshold).astype(int), config.c10, config.c01
        )

        rows.append(
            {
                "seed": seed,
                "risk_none": risk_none,
                "risk_offset": risk_offset,
                "risk_retrain": risk_retrain,
            }
        )

    df = pd.DataFrame(rows)
    summary = aggregate_mean_std(
        df, metrics=("risk_none", "risk_offset", "risk_retrain"), groupby=()
    )

    run_dir = timestamped_run_dir(results_dir, config.exp_name)
    table_path = run_dir / "tables" / f"{config.exp_name}.csv"
    save_table(summary, table_path)

    row = summary.iloc[0]
    means = [
        row["risk_none_mean"],
        row["risk_offset_mean"],
        row["risk_retrain_mean"],
    ]
    stds = [
        row["risk_none_std"],
        row["risk_offset_std"],
        row["risk_retrain_std"],
    ]
    approaches = ["no correction", "offset", "retrained"]
    x = np.arange(len(approaches))
    fig, ax = plt.subplots()
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.set_ylabel("Risk")
    ax.set_title(config.exp_name)
    ax.grid(True, axis="y", linestyle=":")
    figure_path = run_dir / "figures" / f"{config.exp_name}.png"
    fig.savefig(figure_path)
    fig.clf()

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
        exp_name="exp4_concept_drift",
        n_train=args.n_train,
        n_test=args.n_test,
        d=args.d,
        pi_train=args.pi_train,
        seeds=args.seeds,
        c10=DEFAULT_COSTS["c10"],
        c01=DEFAULT_COSTS["c01"],
    )
    _run_experiment(config, args.results_dir)


if __name__ == "__main__":
    main()
