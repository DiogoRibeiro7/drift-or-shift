"""Experiment 5: Breast cancer dataset under label shift."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from drift_or_shift import (
    DRIFT_FEATURE_METRICS,
    PI_TEST_GRID,
    PI_TRAIN,
    SEEDS,
    ExperimentConfig,
    aggregate_mean_std,
    apply_logit_offset,
    feature_drift_metrics,
    fit_logistic_regression,
    logit_offset,
    oracle_threshold_min_risk,
    plot_risk_vs_prevalence,
    predict_logits,
    resample_to_prevalence,
    risk_cost_sensitive,
    save_json,
    save_table,
    threshold_from_costs,
    timestamped_run_dir,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 5: breast cancer label shift."
    )
    parser.add_argument("--pi-train", type=float, default=PI_TRAIN)
    parser.add_argument("--pi-tests", type=float, nargs="+", default=list(PI_TEST_GRID))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--results-dir", type=Path, default="results")
    return parser.parse_args()


def _run_experiment(config: ExperimentConfig, results_dir: Path) -> None:
    data = load_breast_cancer()
    X_full = data.data
    y_full = data.target
    rows = []
    for seed in config.seeds:
        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
            X_full,
            y_full,
            test_size=0.5,
            stratify=y_full,
            random_state=seed,
        )
        rng_train = np.random.default_rng(seed + 1)
        X_train, y_train = resample_to_prevalence(
            X_train_full, y_train_full, config.pi_train, rng_train
        )
        model = fit_logistic_regression(X_train, y_train, rng=rng_train)
        threshold = threshold_from_costs(config.pi_train, config.c10, config.c01)

        for pi_test in config.pi_tests:
            rng_test = np.random.default_rng(seed + int(pi_test * 100))
            X_test, y_test = resample_to_prevalence(
                X_test_full, y_test_full, pi_test, rng_test
            )
            scores = predict_logits(model, X_test)
            decisions_none = (scores >= threshold).astype(int)
            risk_none = risk_cost_sensitive(
                y_test, decisions_none, config.c10, config.c01
            )

            offset = logit_offset(config.pi_train, pi_test)
            shifted_scores = apply_logit_offset(scores, offset)
            decisions_offset = (shifted_scores >= threshold).astype(int)
            risk_offset = risk_cost_sensitive(
                y_test, decisions_offset, config.c10, config.c01
            )

            _, risk_oracle = oracle_threshold_min_risk(
                y_test, scores, config.c10, config.c01
            )
            drift = feature_drift_metrics(X_train, X_test)
            rows.append(
                {
                    "seed": seed,
                    "pi_test": pi_test,
                    "risk_none": risk_none,
                    "risk_offset": risk_offset,
                    "risk_oracle": risk_oracle,
                    **drift,
                }
            )

    df = pd.DataFrame(rows)
    summary = aggregate_mean_std(
        df,
        metrics=(
            "risk_none",
            "risk_offset",
            "risk_oracle",
            *DRIFT_FEATURE_METRICS,
        ),
        groupby="pi_test",
    )

    run_dir = timestamped_run_dir(results_dir, config.exp_name)
    table_path = run_dir / "tables" / f"{config.exp_name}.csv"
    save_table(summary, table_path)

    figure = plot_risk_vs_prevalence(
        summary["pi_test"],
        summary["risk_none_mean"],
        summary["risk_offset_mean"],
        summary["risk_oracle_mean"],
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
        exp_name="exp5_realdata_breast_cancer",
        n_train=0,
        n_test=0,
        d=0,
        pi_train=args.pi_train,
        pi_tests=args.pi_tests,
        seeds=args.seeds,
        c10=1.0,
        c01=1.0,
    )
    _run_experiment(config, args.results_dir)


if __name__ == "__main__":
    main()
