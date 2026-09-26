"""Experiment 7: map how different drift types break offset correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drift_or_shift import (
    DEFAULT_COSTS,
    DRIFT_FEATURE_METRICS,
    PI_TRAIN,
    SEEDS,
    ExperimentConfig,
    aggregate_mean_std,
    apply_covariance_shift,
    apply_logit_offset,
    concept_drift_mu1,
    density_ratio_shift,
    drift_score_from_ratio,
    feature_drift_metrics,
    fit_logistic_regression,
    inject_label_noise,
    logit_offset,
    make_gaussian_binary,
    predict_logits,
    risk_cost_sensitive,
    save_json,
    save_table,
    threshold_from_costs,
    timestamped_run_dir,
)
from drift_or_shift.io_utils import ensure_dir, save_figure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp7: evaluate drift types that defeat single-scalar offsets."
    )
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--d", type=int, default=6)
    parser.add_argument("--pi-train", type=float, default=PI_TRAIN)
    parser.add_argument("--c10", type=float, default=DEFAULT_COSTS["c10"])
    parser.add_argument("--c01", type=float, default=DEFAULT_COSTS["c01"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--results-dir", type=Path, default="results")
    return parser.parse_args()


def _sample_drifted_data(
    drift_type: str,
    n: int,
    d: int,
    pi_test: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    mu0 = np.zeros(d)
    mu1 = np.zeros(d)
    mu1[: min(5, d)] = 1.0

    if drift_type == "covariance_shift":
        cov = apply_covariance_shift(np.eye(d), 2.0)
        return make_gaussian_binary(n, d, pi_test, mu0, mu1, cov, rng)
    if drift_type == "feature_shift":
        shifted = concept_drift_mu1(mu1, shift_dim=min(5, d - 1), delta=0.5)
        return make_gaussian_binary(n, d, pi_test, mu0, shifted, 1.0, rng)
    if drift_type == "label_noise":
        X, y = make_gaussian_binary(n, d, pi_test, mu0, mu1, 1.0, rng)
        noisy = inject_label_noise(y, 0.3, rng)
        return X, noisy
    raise ValueError(f"Unsupported drift type: {drift_type}")


def _plot_drift_risks(summary: pd.DataFrame, path: Path) -> None:
    labels = summary["drift_type"].tolist()
    x = np.arange(len(labels))
    width = 0.3
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, summary["risk_none_mean"], width, label="No correction")
    ax.bar(x, summary["risk_offset_mean"], width, label="Offset")
    ax.bar(x + width, summary["risk_retrain_mean"], width, label="Retrain")
    ax.set_xlabel("Drift type")
    ax.set_ylabel("Mean risk")
    ax.set_title("Offset robustness across drift types")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()
    ensure_dir(path.parent)
    save_figure(fig, path)
    plt.close(fig)


def _run_experiment(
    config: ExperimentConfig, drift_types: list[str], results_dir: Path
) -> None:
    records: list[dict[str, object]] = []
    offset = logit_offset(config.pi_train, config.pi_train)
    for seed in config.seeds:
        rng_train = np.random.default_rng(seed)
        mu0 = np.zeros(config.d)
        mu1 = np.zeros(config.d)
        mu1[: min(5, config.d)] = 1.0
        X_train, y_train = make_gaussian_binary(
            config.n_train, config.d, config.pi_train, mu0, mu1, 1.0, rng_train
        )
        model = fit_logistic_regression(X_train, y_train, rng=rng_train)
        threshold = threshold_from_costs(config.pi_train, config.c10, config.c01)

        for idx, drift_type in enumerate(drift_types):
            rng_test = np.random.default_rng(seed * len(drift_types) + idx)
            X_test, y_test = _sample_drifted_data(
                drift_type, config.n_test, config.d, config.pi_train, rng_test
            )
            scores = predict_logits(model, X_test)
            risk_none = risk_cost_sensitive(
                y_test, (scores >= threshold).astype(int), config.c10, config.c01
            )
            risk_offset = risk_cost_sensitive(
                y_test,
                (apply_logit_offset(scores, offset) >= threshold).astype(int),
                config.c10,
                config.c01,
            )
            retrain_model = fit_logistic_regression(X_test, y_test, rng=rng_test)
            retrain_scores = predict_logits(retrain_model, X_test)
            risk_retrain = risk_cost_sensitive(
                y_test,
                (retrain_scores >= threshold).astype(int),
                config.c10,
                config.c01,
            )
            ratio = density_ratio_shift(X_train, X_test)
            drift_score = drift_score_from_ratio(ratio)
            drift = feature_drift_metrics(X_train, X_test)

            records.append(
                {
                    "drift_type": drift_type,
                    "risk_none": risk_none,
                    "risk_offset": risk_offset,
                    "risk_retrain": risk_retrain,
                    "drift_score": drift_score,
                    **drift,
                }
            )

    df = pd.DataFrame(records)
    summary = aggregate_mean_std(
        df,
        metrics=(
            "risk_none",
            "risk_offset",
            "risk_retrain",
            "drift_score",
            *DRIFT_FEATURE_METRICS,
        ),
        groupby="drift_type",
    )

    run_dir = timestamped_run_dir(results_dir, config.exp_name)
    table_path = run_dir / "tables" / f"{config.exp_name}.csv"
    save_table(summary, table_path)

    figure_path = run_dir / "figures" / f"{config.exp_name}.png"
    _plot_drift_risks(summary, figure_path)

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
        exp_name="exp7_drift_types",
        n_train=args.n_train,
        n_test=args.n_test,
        d=args.d,
        pi_train=args.pi_train,
        pi_tests=(args.pi_train,),
        seeds=args.seeds,
        c10=args.c10,
        c01=args.c01,
    )
    drift_types = ["covariance_shift", "feature_shift", "label_noise"]
    _run_experiment(config, drift_types, args.results_dir)


if __name__ == "__main__":
    main()
