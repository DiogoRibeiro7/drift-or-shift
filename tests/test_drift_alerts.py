"""Tests for the drift alert summary script."""

from __future__ import annotations

from pathlib import Path

from drift_or_shift.reporting import DRIFT_ALERT_THRESHOLDS, collect_drift_alert_records, write_drift_alerts


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
