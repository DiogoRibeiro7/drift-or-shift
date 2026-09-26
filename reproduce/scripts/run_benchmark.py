"""Run calibration benchmarks described in the replication notebook."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris

from caliblab import benchmark
from drift_or_shift.io_utils import ensure_dir, load_yaml, save_table

_DATASETS: dict[str, Any] = {
    "breast_cancer": load_breast_cancer,
    "iris": load_iris,
    "digits": load_digits,
}


def _load_config(path: Path | str) -> dict[str, Any]:
    return load_yaml(path)


def _select_dataset(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    name = spec.get("name", "breast_cancer")
    if name not in _DATASETS:
        raise ValueError(f"Dataset {name!r} is not supported.")
    data = _DATASETS[name]()
    X, y = data.data, data.target
    max_samples = spec.get("max_samples")
    if max_samples is not None:
        sample_size = min(max_samples, len(y))
        rng = np.random.default_rng(spec.get("random_state"))
        idx = rng.choice(len(y), size=sample_size, replace=False)
        X = X[idx]
        y = y[idx]
    return X, y


def run_benchmark(config_path: Path | str) -> Path:
    config = _load_config(config_path)
    X, y = _select_dataset(config["dataset"])
    result = benchmark(
        X,
        y,
        config["estimators"],
        config["calibration_methods"],
        cv_params=config["cv"],
    )
    raw_dir = Path(config["outputs"]["raw"])
    ensure_dir(raw_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = raw_dir / f"benchmark_{timestamp}.csv"
    save_table(result, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the calibration benchmark.")
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to YAML config."
    )
    args = parser.parse_args()
    run_benchmark(args.config)


if __name__ == "__main__":
    main()
