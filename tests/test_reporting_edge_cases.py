"""Tests for the reporting layer's failure and fallback paths.

`scripts/aggregate_results.py` runs over whatever happens to be in `results/`,
so these paths meet real input: a directory that does not exist yet, a summary
truncated by an interrupted run, a timestamp that cannot be parsed. They decide
whether the dashboard degrades gracefully or takes the pipeline down with it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from dataexcept import DataLoadingError

from drift_or_shift.reporting import (
    DRIFT_ALERT_THRESHOLDS,
    collect_summary_jsons,
    format_results_overview,
    load_drift_alert_thresholds,
    select_latest_summaries,
)

# ---------------------------------------------------------------------------
# collect_summary_jsons
# ---------------------------------------------------------------------------


def test_collect_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    assert collect_summary_jsons(tmp_path / "never_created") == []


def test_collect_skips_malformed_summaries(tmp_path: Path) -> None:
    """A truncated summary from an interrupted run must not abort the sweep."""
    good = tmp_path / "exp1" / "run"
    good.mkdir(parents=True)
    (good / "exp1_summary.json").write_text(
        json.dumps({"exp_name": "exp1"}), encoding="utf-8"
    )

    bad = tmp_path / "exp2" / "run"
    bad.mkdir(parents=True)
    (bad / "exp2_summary.json").write_text('{"exp_name": "exp2"', encoding="utf-8")

    collected = collect_summary_jsons(tmp_path)

    assert [entry["exp_name"] for entry in collected] == ["exp1"]
    assert "_path" in collected[0]


# ---------------------------------------------------------------------------
# select_latest_summaries
# ---------------------------------------------------------------------------


def test_select_latest_picks_the_newest_run_per_experiment() -> None:
    older = {"exp_name": "exp1", "timestamp": "2026-01-01T00:00:00+00:00", "tag": "old"}
    newer = {"exp_name": "exp1", "timestamp": "2026-06-01T00:00:00+00:00", "tag": "new"}

    assert [s["tag"] for s in select_latest_summaries([older, newer])] == ["new"]
    assert [s["tag"] for s in select_latest_summaries([newer, older])] == ["new"]


def test_select_latest_tolerates_naive_and_aware_timestamps() -> None:
    """Legacy summaries are naive; comparing them to aware ones must not raise."""
    legacy = {"exp_name": "exp1", "timestamp": "2026-01-15T09:52:31", "tag": "legacy"}
    modern = {
        "exp_name": "exp1",
        "timestamp": "2026-09-22T10:00:00+00:00",
        "tag": "new",
    }

    assert [s["tag"] for s in select_latest_summaries([legacy, modern])] == ["new"]


def test_select_latest_skips_entries_without_a_usable_name_or_timestamp() -> None:
    usable = {"exp_name": "exp1", "timestamp": "2026-01-01T00:00:00+00:00"}
    nameless = {"timestamp": "2026-01-01T00:00:00+00:00"}
    wrongly_typed_name = {"exp_name": 42, "timestamp": "2026-01-01T00:00:00+00:00"}
    undateable = {"exp_name": "exp2", "timestamp": "not-a-timestamp"}

    selected = select_latest_summaries(
        [usable, nameless, wrongly_typed_name, undateable]
    )

    assert [entry["exp_name"] for entry in selected] == ["exp1"]


def test_select_latest_falls_back_to_file_mtime(tmp_path: Path) -> None:
    """With an unparseable timestamp, the file's mtime orders the summary."""
    path = tmp_path / "exp1_summary.json"
    path.write_text("{}", encoding="utf-8")

    payload = {"exp_name": "exp1", "timestamp": "nonsense", "_path": str(path)}

    assert [entry["exp_name"] for entry in select_latest_summaries([payload])] == [
        "exp1"
    ]


def test_select_latest_drops_summaries_whose_fallback_path_is_gone(
    tmp_path: Path,
) -> None:
    payload = {
        "exp_name": "exp1",
        "timestamp": "nonsense",
        "_path": str(tmp_path / "deleted_summary.json"),
    }

    assert select_latest_summaries([payload]) == []


def test_select_latest_returns_one_entry_per_experiment() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    summaries = [
        {"exp_name": name, "timestamp": (now + timedelta(days=offset)).isoformat()}
        for name in ("exp2", "exp1")
        for offset in (0, 1)
    ]

    selected = select_latest_summaries(summaries)

    assert [entry["exp_name"] for entry in selected] == ["exp1", "exp2"]


# ---------------------------------------------------------------------------
# load_drift_alert_thresholds
# ---------------------------------------------------------------------------


def test_thresholds_default_when_no_path_is_given() -> None:
    assert load_drift_alert_thresholds() == DRIFT_ALERT_THRESHOLDS


def test_thresholds_raise_for_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataLoadingError, match=r"absent\.yaml") as error:
        load_drift_alert_thresholds(tmp_path / "absent.yaml")
    assert isinstance(error.value.original, FileNotFoundError)


def test_thresholds_override_defaults_without_dropping_them(tmp_path: Path) -> None:
    config = tmp_path / "thresholds.yaml"
    config.write_text("feature_max_ks: 0.9\n", encoding="utf-8")

    thresholds = load_drift_alert_thresholds(config)

    assert thresholds["feature_max_ks"] == pytest.approx(0.9)
    # Keys the file did not mention must survive.
    assert thresholds["feature_max_std_diff"] == pytest.approx(
        DRIFT_ALERT_THRESHOLDS["feature_max_std_diff"]
    )


def test_thresholds_reject_a_non_mapping_document(tmp_path: Path) -> None:
    config = tmp_path / "thresholds.yaml"
    config.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping"):
        load_drift_alert_thresholds(config)


def test_thresholds_reject_non_string_keys(tmp_path: Path) -> None:
    config = tmp_path / "thresholds.yaml"
    config.write_text("1: 0.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="keys must be strings"):
        load_drift_alert_thresholds(config)


def test_thresholds_reject_non_numeric_values(tmp_path: Path) -> None:
    config = tmp_path / "thresholds.yaml"
    config.write_text("feature_max_ks: not_a_number\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_drift_alert_thresholds(config)


# ---------------------------------------------------------------------------
# format_results_overview
# ---------------------------------------------------------------------------


def test_overview_reports_when_there_is_nothing_to_show() -> None:
    assert "No experiment summaries found" in format_results_overview([])


def test_overview_includes_experiment_details() -> None:
    overview = format_results_overview(
        [
            {
                "exp_name": "exp1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "table": "tables/exp1.csv",
                "figure": "figures/exp1.png",
                "aggregated": {"risk_offset_mean": [0.1, 0.2]},
                "_path": "results/exp1/run/exp1_summary.json",
            }
        ]
    )

    assert "exp1" in overview
    assert "tables/exp1.csv" in overview
    assert "risk_offset_mean" in overview
