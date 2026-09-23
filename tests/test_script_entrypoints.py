"""Tests for the command-line wiring of the helper scripts.

Coverage previously measured only `drift_or_shift` and `caliblab`, so these
modules were invisible: the reported 89% excluded a fifth of the shipped code,
and four scripts sat at 0%. Their library functions were tested, but the layer
users actually touch -- argument parsing, defaults, the branch that reports
"nothing found", the exit status -- was not.

Every test here drives the real `main()` through `sys.argv`, the way the
documented commands in README.md and docs/CLI_USAGE.md invoke them.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import yaml


def _run_main(monkeypatch, module_name: str, argv: list[str]):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [module_name, *argv])
    return module.main()


@pytest.fixture
def results_tree(tmp_path: Path) -> Path:
    """A minimal results/ tree with one summary, as an experiment would write."""
    run = tmp_path / "results" / "exp1_label_shift_synth" / "20260101_000000"
    (run / "tables").mkdir(parents=True)
    (run / "figures").mkdir(parents=True)
    (run / "tables" / "exp1.csv").write_text("pi_test\n0.1\n", encoding="utf-8")
    (run / "exp1_label_shift_synth_summary.json").write_text(
        json.dumps(
            {
                "exp_name": "exp1_label_shift_synth",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "table": str(run / "tables" / "exp1.csv"),
                "figure": str(run / "figures" / "exp1.png"),
                "aggregated": {
                    "pi_test": [0.01, 0.5],
                    "risk_none_mean": [0.05, 0.17],
                    "risk_offset_mean": [0.01, 0.13],
                    "feature_max_ks_mean": [0.02, 0.03],
                },
            }
        ),
        encoding="utf-8",
    )
    return tmp_path / "results"


# ---------------------------------------------------------------------------
# scripts/drift_alerts.py
# ---------------------------------------------------------------------------


def test_drift_alerts_reports_when_nothing_tripped(
    monkeypatch, tmp_path, results_tree, capsys
) -> None:
    """Below threshold there is no CSV, and the script must say so rather than
    leave the user wondering whether it ran."""
    output = tmp_path / "alerts.csv"

    _run_main(
        monkeypatch,
        "drift_alerts",
        ["--results-dir", str(results_tree), "--output", str(output)],
    )

    assert "No drift alerts detected" in capsys.readouterr().out
    assert not output.exists()


def test_drift_alerts_writes_a_csv_when_thresholds_are_exceeded(
    monkeypatch, tmp_path, results_tree, capsys
) -> None:
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text("feature_max_ks: 0.001\n", encoding="utf-8")
    output = tmp_path / "alerts.csv"

    _run_main(
        monkeypatch,
        "drift_alerts",
        [
            "--results-dir",
            str(results_tree),
            "--output",
            str(output),
            "--threshold-config",
            str(thresholds),
        ],
    )

    assert output.is_file()
    assert "drift alerts written to" in capsys.readouterr().out
    assert "exp1_label_shift_synth" in output.read_text(encoding="utf-8")


def test_drift_alerts_propagates_a_missing_threshold_file(
    monkeypatch, tmp_path, results_tree
) -> None:
    with pytest.raises(FileNotFoundError):
        _run_main(
            monkeypatch,
            "drift_alerts",
            [
                "--results-dir",
                str(results_tree),
                "--output",
                str(tmp_path / "a.csv"),
                "--threshold-config",
                str(tmp_path / "absent.yaml"),
            ],
        )


# ---------------------------------------------------------------------------
# scripts/watch_results.py
# ---------------------------------------------------------------------------


def test_watch_results_runs_the_pipeline_once_and_exits(
    monkeypatch, tmp_path, results_tree, capsys
) -> None:
    """`--once` must not enter the polling loop."""
    watch = importlib.import_module("watch_results")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(watch.subprocess, "run", fake_run)

    _run_main(
        monkeypatch,
        "watch_results",
        [
            "--results-dir",
            str(results_tree),
            "--drift-output",
            str(tmp_path / "alerts.csv"),
            "--once",
        ],
    )

    assert len(calls) == 2, f"expected drift_alerts + aggregate_results, got {calls}"
    joined = " ".join(" ".join(c) for c in calls)
    assert "drift_alerts" in joined and "aggregate_results" in joined
    assert "running drift_alerts" in capsys.readouterr().out


def test_watch_results_passes_fail_on_alerts_through(
    monkeypatch, tmp_path, results_tree
) -> None:
    watch = importlib.import_module("watch_results")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        watch.subprocess,
        "run",
        lambda cmd, **kw: calls.append(cmd) or type("R", (), {"returncode": 0})(),
    )

    _run_main(
        monkeypatch,
        "watch_results",
        [
            "--results-dir",
            str(results_tree),
            "--drift-output",
            str(tmp_path / "alerts.csv"),
            "--once",
            "--fail-on-alerts",
        ],
    )

    assert any("--fail-on-alerts" in c for c in calls)


def test_watch_results_survives_a_failing_pipeline(
    monkeypatch, tmp_path, results_tree, capsys
) -> None:
    """A failed run must be reported, not raised out of the watcher."""
    import subprocess

    watch = importlib.import_module("watch_results")

    def boom(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(watch.subprocess, "run", boom)

    _run_main(
        monkeypatch,
        "watch_results",
        ["--results-dir", str(results_tree), "--once"],
    )

    assert "pipeline failed" in capsys.readouterr().out


def test_watch_results_snapshot_detects_change(tmp_path) -> None:
    watch = importlib.import_module("watch_results")
    results = tmp_path / "results"
    results.mkdir()

    empty = watch._snapshot(results)
    assert empty == {}
    assert watch._snapshot(tmp_path / "absent") == {}

    (results / "a_summary.json").write_text("{}", encoding="utf-8")
    one = watch._snapshot(results)

    assert watch._has_changes(empty, one) is True
    assert watch._has_changes(one, one) is False


# ---------------------------------------------------------------------------
# reproduce/scripts/make_tables.py and make_figures.py
# ---------------------------------------------------------------------------


@pytest.fixture
def reproduce_config(tmp_path: Path) -> Path:
    """A config plus the raw benchmark output the later stages consume."""
    from reproduce.scripts.run_benchmark import run_benchmark

    config = {
        "dataset": {"name": "breast_cancer", "max_samples": 60, "random_state": 0},
        "estimators": [
            {"name": "logistic", "type": "logistic", "params": {"solver": "liblinear"}}
        ],
        "calibration_methods": ["none", "sigmoid"],
        "cv": {"n_splits": 2, "random_state": 0},
        "outputs": {
            "raw": str(tmp_path / "raw"),
            "tables": str(tmp_path / "tables"),
            "figures": str(tmp_path / "figures"),
        },
        "figures": {"reliability": {"methods": ["logistic|none", "logistic|sigmoid"]}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_benchmark(path)
    return path


def test_make_tables_writes_a_summary_table(reproduce_config) -> None:
    from reproduce.scripts.make_tables import run_make_tables

    out = run_make_tables(reproduce_config)

    assert out.is_file() and out.stat().st_size > 0


def test_make_tables_cli(monkeypatch, reproduce_config) -> None:
    _run_main(
        monkeypatch,
        "reproduce.scripts.make_tables",
        ["--config", str(reproduce_config)],
    )


def test_make_figures_writes_a_reliability_plot(reproduce_config) -> None:
    from reproduce.scripts.make_figures import run_make_figures

    out = run_make_figures(reproduce_config)

    assert out.is_file() and out.stat().st_size > 0


def test_make_figures_cli(monkeypatch, reproduce_config) -> None:
    _run_main(
        monkeypatch,
        "reproduce.scripts.make_figures",
        ["--config", str(reproduce_config)],
    )
