"""Tests for the reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path

from drift_or_shift.reporting import (
    DRIFT_ALERT_THRESHOLDS,
    collect_summary_jsons,
    detect_drift_alerts,
    select_latest_summaries,
    write_results_overview,
)


def test_collect_summary_jsons(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run_dir = results / "exp" / "20260101_000000"
    run_dir.mkdir(parents=True)
    summary = {
        "exp_name": "exp1_label_shift_synth",
        "timestamp": "2026-01-01T00:00:00",
        "table": "foo.csv",
        "figure": "foo.png",
        "aggregated": {"risk_none_mean": [0.1]},
    }
    summary_path = run_dir / "exp1_label_shift_synth_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    collected = collect_summary_jsons(results)
    assert len(collected) == 1
    assert collected[0]["exp_name"] == "exp1_label_shift_synth"


def test_write_results_overview(tmp_path: Path) -> None:
    summaries = [
        {
            "exp_name": "exp1",
            "timestamp": "2026-01-01T00:00:00",
            "table": "t.csv",
            "figure": "f.png",
            "aggregated": {"metric_mean": [0.2]},
            "_path": "results/exp1/...",
        }
    ]
    output = tmp_path / "reports" / "overview.md"
    path = write_results_overview(summaries, output)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "# Results overview" in content
    assert "exp1" in content


def test_select_latest_summaries(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    third_path = tmp_path / "third.json"
    first_path.write_text("first")
    second_path.write_text("second")
    third_path.write_text("third")

    summaries = [
        {
            "exp_name": "exp1",
            "timestamp": "2026-01-01T00:00:00",
            "_path": str(first_path),
        },
        {
            "exp_name": "exp1",
            "timestamp": "2026-01-02T00:00:00",
            "_path": str(second_path),
        },
        {
            "exp_name": "exp2",
            "timestamp": "2026-01-03T00:00:00",
            "_path": str(third_path),
        },
    ]

    latest = select_latest_summaries(summaries)
    assert len(latest) == 2
    exp_names = {entry["exp_name"] for entry in latest}
    assert exp_names == {"exp1", "exp2"}
    assert latest[0]["timestamp"] == "2026-01-02T00:00:00"
    assert latest[1]["timestamp"] == "2026-01-03T00:00:00"


def test_detect_drift_alerts_reports_threshold_exceeded() -> None:
    summary = {
        "aggregated": {
            "feature_max_ks": [DRIFT_ALERT_THRESHOLDS["feature_max_ks"] + 0.1],
            "feature_max_mean_diff": [0.0],
        }
    }
    alerts = detect_drift_alerts(summary)
    assert alerts
    assert alerts[0][0] == "feature_max_ks"
