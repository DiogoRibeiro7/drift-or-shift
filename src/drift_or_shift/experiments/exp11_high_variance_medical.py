"""Experiment 11: high-variance medical benchmark with class-specific covariances."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
    plot_risk_vs_prevalence,
    predict_logits,
    risk_cost_sensitive,
    save_json,
    save_table,
    threshold_from_costs,
    timestamped_run_dir,
)


def _sample_with_covariances(
    n: int,
    d: int,
    pi: float,
    mu0: np.ndarray,
    mu1: np.ndarray,
    cov0: np.ndarray,
    cov1: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a binary Gaussian dataset with distinct class covariances."""
    labels = rng.choice([0, 1], size=n, p=[1 - pi, pi])
    X = np.empty((n, d), dtype=float)
    neg_idx = labels == 0
    pos_idx = labels == 1
    if np.any(neg_idx):
        X[neg_idx] = rng.multivariate_normal(mu0, cov0, size=np.sum(neg_idx))
    if np.any(pos_idx):
        X[pos_idx] = rng.multivariate_normal(mu1, cov1, size=np.sum(pos_idx))
    return X, labels


def _build_covariance(scale: float, correlation: float, d: int) -> np.ndarray:
    """Create a covariance with optional off-diagonal correlation in the first two dims."""
    cov = np.eye(d, dtype=float) * scale
    if d >= 2 and correlation != 0.0:
        cov[0, 1] = cov[1, 0] = correlation
    return cov


def _paste_non_linear_shift(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Warp positive-class features so the decision boundary becomes non-linear."""
    warped = X.copy()
    mask = y == 1
    if not np.any(mask):
        return warped
    warped[mask, 0] += 0.45 * np.sin(warped[mask, 0])
    warped[mask, 1] += 0.3 * np.cos(warped[mask, 1])
    if warped.shape[1] > 2:
        warped[mask, 2:] += 0.2 * warped[mask, 0][:, None]
    return warped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp11: high-variance covariance/feature shift benchmark."
    )
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--d", type=int, default=6)
    parser.add_argument("--pi-train", type=float, default=PI_TRAIN)
    parser.add_argument("--pi-tests", type=float, nargs="+", default=list(PI_TEST_GRID))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--c10", type=float, default=DEFAULT_COSTS["c10"])
    parser.add_argument("--c01", type=float, default=DEFAULT_COSTS["c01"])
    parser.add_argument("--results-dir", type=Path, default="results")
    return parser.parse_args()


def _run_experiment(config: ExperimentConfig, results_dir: Path) -> None:
    rows: list[dict[str, float]] = []
    mu0 = np.zeros(config.d)
    mu1 = np.zeros(config.d)
    mu1[: min(5, config.d)] = 1.0
    cov0_train = _build_covariance(scale=0.8, correlation=0.0, d=config.d)
    cov1_train = _build_covariance(scale=1.4, correlation=0.0, d=config.d)
    cov0_test = _build_covariance(scale=1.6, correlation=0.25, d=config.d)
    cov1_test = _build_covariance(scale=0.7, correlation=-0.2, d=config.d)

    for seed in config.seeds:
        rng = np.random.default_rng(seed)
        X_train, y_train = _sample_with_covariances(
            config.n_train,
            config.d,
            config.pi_train,
            mu0,
            mu1,
            cov0_train,
            cov1_train,
            rng,
        )
        model = fit_logistic_regression(X_train, y_train, rng=rng)
        threshold = threshold_from_costs(config.pi_train, config.c10, config.c01)

        for pi_test in config.pi_tests:
            rng_test = np.random.default_rng(seed + int(pi_test * 100))
            X_test, y_test = _sample_with_covariances(
                config.n_test,
                config.d,
                pi_test,
                mu0,
                mu1,
                cov0_test,
                cov1_test,
                rng=rng_test,
            )
            X_test = _paste_non_linear_shift(X_test, y_test)
            scores = predict_logits(model, X_test)
            decisions_none = (scores >= threshold).astype(int)
            risk_none = risk_cost_sensitive(
                y_test, decisions_none, config.c10, config.c01
            )

            offset = logit_offset(config.pi_train, pi_test)
            scores_offset = apply_logit_offset(scores, offset)
            risk_offset = risk_cost_sensitive(
                y_test,
                (scores_offset >= threshold).astype(int),
                config.c10,
                config.c01,
            )

            retrain_model = fit_logistic_regression(X_test, y_test, rng=rng_test)
            threshold_test = threshold_from_costs(pi_test, config.c10, config.c01)
            retrain_scores = predict_logits(retrain_model, X_test)
            risk_retrain = risk_cost_sensitive(
                y_test,
                (retrain_scores >= threshold_test).astype(int),
                config.c10,
                config.c01,
            )

            drift = feature_drift_metrics(X_train, X_test)
            rows.append(
                {
                    "seed": seed,
                    "pi_test": pi_test,
                    "risk_none": risk_none,
                    "risk_offset": risk_offset,
                    "risk_retrain": risk_retrain,
                    **drift,
                }
            )

    df = pd.DataFrame(rows)
    summary = aggregate_mean_std(
        df,
        metrics=(
            "risk_none",
            "risk_offset",
            "risk_retrain",
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
        summary["risk_retrain_mean"],
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
        exp_name="exp11_high_variance_medical",
        n_train=args.n_train,
        n_test=args.n_test,
        d=args.d,
        pi_train=args.pi_train,
        pi_tests=args.pi_tests,
        seeds=args.seeds,
        c10=args.c10,
        c01=args.c01,
    )
    _run_experiment(config, args.results_dir)


if __name__ == "__main__":
    main()
