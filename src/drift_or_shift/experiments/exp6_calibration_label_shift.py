"""Experiment 6: assess calibration plus offset under label shift."""

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
    apply_calibrator,
    apply_logit_offset,
    feature_drift_metrics,
    find_best_temperature,
    fit_isotonic_calibrator,
    fit_logistic_regression,
    logit_offset,
    make_gaussian_binary,
    plot_risk_vs_prevalence,
    predict_logits,
    risk_cost_sensitive,
    save_json,
    save_table,
    temperature_scale,
    threshold_from_costs,
    timestamped_run_dir,
)


def _prob_to_logit(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    clipped = np.clip(probs, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp6: calibration plus offset in label-shifted settings."
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
    rows = []
    for seed in config.seeds:
        rng_train = np.random.default_rng(seed)
        mu0 = np.zeros(config.d)
        mu1 = np.zeros(config.d)
        mu1[: min(5, config.d)] = 1.0
        X_train, y_train = make_gaussian_binary(
            config.n_train,
            config.d,
            config.pi_train,
            mu0,
            mu1,
            sigma=1.0,
            rng=rng_train,
        )
        model = fit_logistic_regression(X_train, y_train, rng=rng_train)
        threshold = threshold_from_costs(config.pi_train, config.c10, config.c01)
        train_logits = predict_logits(model, X_train)
        best_temp = find_best_temperature(train_logits, y_train)
        iso_calibrator = fit_isotonic_calibrator(train_logits, y_train)

        for pi_test in config.pi_tests:
            rng_test = np.random.default_rng(seed + int(pi_test * 100))
            X_test, y_test = make_gaussian_binary(
                config.n_test, config.d, pi_test, mu0, mu1, sigma=1.0, rng=rng_test
            )
            scores = predict_logits(model, X_test)
            baseline = (scores >= threshold).astype(int)
            risk_none = risk_cost_sensitive(y_test, baseline, config.c10, config.c01)

            offset = logit_offset(config.pi_train, pi_test)
            scores_offset = apply_logit_offset(scores, offset)
            risk_offset = risk_cost_sensitive(
                y_test, (scores_offset >= threshold).astype(int), config.c10, config.c01
            )

            scaled_scores = temperature_scale(scores, best_temp)
            scaled_offset = apply_logit_offset(scaled_scores, offset)
            risk_temp_offset = risk_cost_sensitive(
                y_test, (scaled_offset >= threshold).astype(int), config.c10, config.c01
            )

            iso_logits = _prob_to_logit(apply_calibrator(iso_calibrator, scores))
            iso_offset = apply_logit_offset(iso_logits, offset)
            risk_iso_offset = risk_cost_sensitive(
                y_test, (iso_offset >= threshold).astype(int), config.c10, config.c01
            )

            drift = feature_drift_metrics(X_train, X_test)
            rows.append(
                {
                    "seed": seed,
                    "pi_test": pi_test,
                    "risk_none": risk_none,
                    "risk_offset": risk_offset,
                    "risk_temp_offset": risk_temp_offset,
                    "risk_isotonic_offset": risk_iso_offset,
                    **drift,
                }
            )

    df = pd.DataFrame(rows)
    summary = aggregate_mean_std(
        df,
        metrics=(
            "risk_none",
            "risk_offset",
            "risk_temp_offset",
            "risk_isotonic_offset",
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
        summary["risk_temp_offset_mean"],
        title=config.exp_name,
    )
    ax = figure.axes[0]
    ax.plot(
        summary["pi_test"],
        summary["risk_isotonic_offset_mean"],
        label="Isotonic + offset",
        marker="d",
    )
    ax.legend()
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
        exp_name="exp6_calibration_label_shift",
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
