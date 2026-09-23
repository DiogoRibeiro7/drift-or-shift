"""Tests for the drift alert summary script."""

from __future__ import annotations

from pathlib import Path

from drift_or_shift.reporting import (
    DRIFT_ALERT_THRESHOLDS,
    collect_drift_alert_records,
    load_drift_alert_thresholds,
    write_drift_alerts,
)


def test_build_records_honors_alerts() -> None:
    summaries = [
        {
            "exp_name": "expX",
            "timestamp": "2026-01-01T00:00:00",
            "aggregated": {
                "feature_max_ks": [DRIFT_ALERT_THRESHOLDS["feature_max_ks"] + 0.05],
            },
        }
    ]
    records = collect_drift_alert_records(summaries)
    assert records
    assert records[0]["exp_name"] == "expX"
    assert "alert_1_name" in records[0]


def test_build_records_empty_when_no_alerts() -> None:
    summaries = [
        {
            "exp_name": "expY",
            "aggregated": {"feature_max_ks": [0.0]},
        }
    ]
    assert not collect_drift_alert_records(summaries)


def test_write_alerts(tmp_path: Path) -> None:
    summaries = [
        {
            "exp_name": "expX",
            "timestamp": "2026-01-01T00:00:00",
            "aggregated": {
                "feature_max_ks": [DRIFT_ALERT_THRESHOLDS["feature_max_ks"] + 0.05],
            },
        }
    ]
    path = tmp_path / "alerts.csv"
    written = write_drift_alerts(summaries, path)
    assert written.exists()
    content = written.read_text()
    assert "expX" in content


def test_threshold_config(tmp_path: Path) -> None:
    summaries = [
        {
            "exp_name": "expY",
            "timestamp": "2026-01-01T00:00:00",
            "aggregated": {"feature_max_ks": [0.1]},
        }
    ]
    config_path = tmp_path / "thresholds.yaml"
    config_path.write_text("feature_max_ks: 0.05", encoding="utf-8")
    thresholds = load_drift_alert_thresholds(config_path)
    assert thresholds["feature_max_ks"] == 0.05
    path = tmp_path / "alerts.csv"
    written = write_drift_alerts(summaries, path, thresholds=thresholds)
    assert written.exists()
    assert "expY" in written.read_text()


def test_alerts_fire_on_the_key_shape_experiments_actually_write() -> None:
    """The alerting was dead code, and these tests were why.

    Every other test in this module hand-builds a summary keyed on the bare
    metric name, e.g. `feature_max_ks`. No experiment writes that: they
    aggregate through `aggregate_mean_std`, which emits `feature_max_ks_mean`.
    The thresholds therefore matched nothing in a real run, `--fail-on-alerts`
    could never trip, and `watch_results.py` watched for something that could
    not happen -- while these tests stayed green.

    This builds the summary through the real aggregation function instead of
    asserting against a hand-written shape.
    """
    import pandas as pd

    from drift_or_shift import DRIFT_FEATURE_METRICS, aggregate_mean_std
    from drift_or_shift.reporting import detect_drift_alerts

    threshold = DRIFT_ALERT_THRESHOLDS["feature_max_ks"]
    frame = pd.DataFrame(
        [
            {"pi_test": 0.1, **dict.fromkeys(DRIFT_FEATURE_METRICS, threshold + 0.2)},
            {"pi_test": 0.1, **dict.fromkeys(DRIFT_FEATURE_METRICS, threshold + 0.2)},
        ]
    )
    aggregated = aggregate_mean_std(
        frame, metrics=tuple(DRIFT_FEATURE_METRICS), groupby="pi_test"
    )
    summary = {
        "exp_name": "exp_synthetic",
        "aggregated": aggregated.to_dict(orient="list"),
    }

    # The shape really is `<metric>_mean`, not the bare name.
    assert "feature_max_ks_mean" in summary["aggregated"]
    assert "feature_max_ks" not in summary["aggregated"]

    alerts = detect_drift_alerts(summary)

    assert alerts, "no alert fired on a summary shaped the way experiments write them"
    assert any(metric == "feature_max_ks" for metric, _, _ in alerts)


def test_no_alerts_when_the_aggregated_metrics_are_quiet() -> None:
    """The mirror of the above: the same real shape must stay silent."""
    import pandas as pd

    from drift_or_shift import DRIFT_FEATURE_METRICS, aggregate_mean_std
    from drift_or_shift.reporting import detect_drift_alerts

    frame = pd.DataFrame(
        [
            {"pi_test": 0.1, **dict.fromkeys(DRIFT_FEATURE_METRICS, 0.0)},
            {"pi_test": 0.1, **dict.fromkeys(DRIFT_FEATURE_METRICS, 0.0)},
        ]
    )
    aggregated = aggregate_mean_std(
        frame, metrics=tuple(DRIFT_FEATURE_METRICS), groupby="pi_test"
    )

    assert not detect_drift_alerts(
        {"exp_name": "quiet", "aggregated": aggregated.to_dict(orient="list")}
    )
