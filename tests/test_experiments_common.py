import pandas as pd

from drift_or_shift.experiments import (
    PI_TEST_GRID,
    SEEDS,
    ExperimentConfig,
    aggregate_mean_std,
)


def test_aggregate_mean_std_returns_mean_and_std() -> None:
    df = pd.DataFrame(
        {
            "pi_test": [0.1, 0.1, 0.2],
            "risk_none": [1.0, 2.0, 4.0],
        }
    )
    summary = aggregate_mean_std(df, metrics=("risk_none",))
    assert "risk_none_mean" in summary.columns
    assert "risk_none_std" in summary.columns
    row = summary[summary["pi_test"] == 0.1].iloc[0]
    assert row["risk_none_mean"] == 1.5
    assert row["risk_none_std"] == 0.5


def test_experiment_config_metadata_contains_expected_fields() -> None:
    config = ExperimentConfig(exp_name="exp", n_train=100, n_test=100, d=3)
    metadata = config.metadata()
    assert metadata["exp_name"] == "exp"
    assert metadata["pi_tests"] == list(PI_TEST_GRID)
    assert metadata["seeds"] == list(SEEDS)
