"""Experiment 9: label shift on the Covertype dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype

from drift_or_shift import (
    DEFAULT_COSTS,
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
    parser = argparse.ArgumentParser(description="Run Exp9: Covertype label shift.")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--pi-train", type=float, default=PI_TRAIN)
    parser.add_argument("--pi-tests", type=float, nargs="+", default=list(PI_TEST_GRID))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--c10", type=float, default=DEFAULT_COSTS["c10"])
    parser.add_argument("--c01", type=float, default=DEFAULT_COSTS["c01"])
    parser.add_argument("--results-dir", type=Path, default="results")
    return parser.parse_args()


def _load_covtype() -> tuple[np.ndarray, np.ndarray]:
    data = fetch_covtype(download_if_missing=True)
    X = data.data.astype(float)
    y = (data.target == 1).astype(int)
    return X, y


def _run_experiment(config: ExperimentConfig, results_dir: Path) -> None:
    X_full, y_full = _load_covtype()
    rows: list[dict[str, float]] = []
    required = config.n_train + config.n_test
    for seed in config.seeds:
        rng = np.random.default_rng(seed)
        if required > X_full.shape[0]:
            raise ValueError(
                "Covertype dataset does not contain enough samples for the requested N"
            )
        subset = rng.choice(X_full.shape[0], size=required, replace=False)
        pool_X = X_full[subset]
        pool_y = y_full[subset]
        X_train_raw = pool_X[: config.n_train]
        y_train_raw = pool_y[: config.n_train]
        X_train, y_train = resample_to_prevalence(
            X_train_raw, y_train_raw, config.pi_train, rng
        )
        model = fit_logistic_regression(X_train, y_train, rng=rng)
        threshold = threshold_from_costs(config.pi_train, config.c10, config.c01)

        for pi_test in config.pi_tests:
            X_test_raw = pool_X[config.n_train :]
            y_test_raw = pool_y[config.n_train :]
            X_test, y_test = resample_to_prevalence(
                X_test_raw, y_test_raw, pi_test, rng
            )
            scores = predict_logits(model, X_test)
            decisions_none = (scores >= threshold).astype(int)
            risk_none = risk_cost_sensitive(
                y_test, decisions_none, config.c10, config.c01
            )

            offset = logit_offset(config.pi_train, pi_test)
            scores_offset = apply_logit_offset(scores, offset)
            risk_offset = risk_cost_sensitive(
                y_test, (scores_offset >= threshold).astype(int), config.c10, config.c01
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
        exp_name="exp9_covtype_label_shift",
        n_train=args.n_train,
        n_test=args.n_test,
        d=54,
        pi_train=args.pi_train,
        pi_tests=args.pi_tests,
        seeds=args.seeds,
        c10=args.c10,
        c01=args.c01,
    )
    _run_experiment(config, args.results_dir)


if __name__ == "__main__":
    main()
