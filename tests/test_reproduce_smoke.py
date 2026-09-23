"""End-to-end smoke tests for the reproduction pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from reproduce.scripts.run_benchmark import run_benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO_ROOT / "reproduce" / "config" / "default.yaml"


def _write_config(tmp_path: Path, calibration_methods: list[str]) -> Path:
    config = {
        "dataset": {"name": "breast_cancer", "max_samples": 50, "random_state": 0},
        "estimators": [
            {
                "name": "logistic",
                "type": "logistic",
                "params": {"C": 1.0, "max_iter": 1000, "solver": "liblinear"},
            }
        ],
        "calibration_methods": calibration_methods,
        "cv": {"n_splits": 2, "random_state": 0},
        "outputs": {
            "raw": str(tmp_path / "results" / "raw"),
            "tables": str(tmp_path / "results" / "tables"),
            "figures": str(tmp_path / "results" / "figures"),
        },
        "figures": {"reliability": {"methods": ["logistic|none"]}},
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return config_path


def test_reproduce_smoke(tmp_path: Path) -> None:
    output_path = run_benchmark(_write_config(tmp_path, ["none"]))
    assert output_path.exists()


def test_reproduce_smoke_with_shipped_calibration_methods(tmp_path: Path) -> None:
    """The shipped default config asks for sigmoid + isotonic; it must run.

    This previously raised ``TypeError`` on any scikit-learn >= 1.2 while the
    suite stayed green, because the only smoke test used ``["none"]``.
    """
    methods = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))[
        "calibration_methods"
    ]
    assert "sigmoid" in methods and "isotonic" in methods, methods

    output_path = run_benchmark(_write_config(tmp_path, methods))

    assert output_path.exists()
    frame = pd.read_csv(output_path)
    assert set(frame["calibration"].unique()) == set(methods)


@pytest.mark.parametrize("missing_key", ["dataset", "estimators", "cv"])
def test_run_benchmark_rejects_incomplete_config(
    tmp_path: Path, missing_key: str
) -> None:
    config_path = _write_config(tmp_path, ["none"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del config[missing_key]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises((KeyError, TypeError, ValueError)):
        run_benchmark(config_path)
