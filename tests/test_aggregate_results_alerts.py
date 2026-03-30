"""Regression tests that ensure drift alerts can fail the aggregator."""

from __future__ import annotations

import json
import subprocess
import sys


def test_aggregate_results_fail_on_alerts(tmp_path) -> None:
    results_dir = tmp_path / "results"
    run_dir = results_dir / "expX" / "20260101_000000"
    run_dir.mkdir(parents=True)
    summary = {
        "exp_name": "expX",
        "timestamp": "2026-01-01T00:00:00",
        "pi_test": [0.1],
        "aggregated": {
            "risk_none_mean": [0.1],
            "risk_offset_mean": [0.05],
            "feature_max_ks": [0.3],
        },
    }
    summary_path = run_dir / "expX_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    dashboard = tmp_path / "dashboard.md"
    figure = tmp_path / "best.png"
    alerts = tmp_path / "alerts.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_results.py",
            "--results-dir",
            str(results_dir),
            "--output",
            str(dashboard),
            "--figure",
            str(figure),
            "--drift-output",
            str(alerts),
            "--fail-on-alerts",
        ],
        check=False,
    )
    assert result.returncode == 1
    assert alerts.exists()
