"""Smoke test for aggregate_results and drift alerts integration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_aggregate_results_fails_when_drift_threshold_exceeded(tmp_path: Path) -> None:
    results_dir = tmp_path / "results" / "exp1" / "20260101_000000"
    results_dir.mkdir(parents=True)
    summary = {
        "exp_name": "exp1",
        "timestamp": "2026-01-01T00:00:00",
        "aggregated": {"feature_max_ks": [0.5]},
    }
    (results_dir / "exp1_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    threshold_config = tmp_path / "thresholds.yaml"
    threshold_config.write_text("feature_max_ks: 0.1", encoding="utf-8")

    dashboard = tmp_path / "dashboard.md"
    drift_output = tmp_path / "drift_alerts.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_results.py",
            "--results-dir",
            str(tmp_path / "results"),
            "--output",
            str(dashboard),
            "--drift-output",
            str(drift_output),
            "--fail-on-alerts",
            "--threshold-config",
            str(threshold_config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert drift_output.exists()
    assert "exp1" in drift_output.read_text()


def test_aggregate_results_succeeds_when_drift_under_threshold(tmp_path: Path) -> None:
    results_dir = tmp_path / "results" / "exp1" / "20260101_000000"
    results_dir.mkdir(parents=True)
    summary = {
        "exp_name": "exp1",
        "timestamp": "2026-01-01T00:00:00",
        "aggregated": {"feature_max_ks": [0.05]},
    }
    (results_dir / "exp1_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    threshold_config = tmp_path / "thresholds.yaml"
    threshold_config.write_text("feature_max_ks: 0.1", encoding="utf-8")

    dashboard = tmp_path / "dashboard.md"
    drift_output = tmp_path / "drift_alerts.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_results.py",
            "--results-dir",
            str(tmp_path / "results"),
            "--output",
            str(dashboard),
            "--drift-output",
            str(drift_output),
            "--threshold-config",
            str(threshold_config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert dashboard.exists()
    assert not drift_output.exists()
