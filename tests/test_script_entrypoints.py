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
from dataexcept import DataLoadingError


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
    with pytest.raises(DataLoadingError):
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


# ---------------------------------------------------------------------------
# scripts/aggregate_results.py
# ---------------------------------------------------------------------------


def test_aggregate_results_writes_a_dashboard_and_figure(
    monkeypatch, tmp_path, results_tree, capsys
) -> None:
    output = tmp_path / "dashboard.md"
    figure = tmp_path / "best_risk.png"

    _run_main(
        monkeypatch,
        "aggregate_results",
        [
            "--results-dir",
            str(results_tree),
            "--output",
            str(output),
            "--figure",
            str(figure),
            "--drift-output",
            str(tmp_path / "alerts.csv"),
        ],
    )

    assert output.is_file() and figure.is_file()
    text = output.read_text(encoding="utf-8")
    assert "exp1_label_shift_synth" in text
    assert "dashboard written to" in capsys.readouterr().out


def test_aggregate_results_reports_an_empty_tree(monkeypatch, tmp_path, capsys) -> None:
    _run_main(
        monkeypatch,
        "aggregate_results",
        [
            "--results-dir",
            str(tmp_path / "nothing"),
            "--output",
            str(tmp_path / "d.md"),
        ],
    )

    assert "no summaries found" in capsys.readouterr().out


def test_aggregate_results_fails_the_build_on_alerts(
    monkeypatch, tmp_path, results_tree
) -> None:
    """`--fail-on-alerts` is the hook CI would use; it must actually exit non-zero."""
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text("feature_max_ks: 0.001\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _run_main(
            monkeypatch,
            "aggregate_results",
            [
                "--results-dir",
                str(results_tree),
                "--output",
                str(tmp_path / "d.md"),
                "--figure",
                str(tmp_path / "f.png"),
                "--drift-output",
                str(tmp_path / "alerts.csv"),
                "--threshold-config",
                str(thresholds),
                "--fail-on-alerts",
            ],
        )

    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# scripts/monitor_drift.py
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text("x,y\n" + "".join(rows), encoding="utf-8")
    return path


def test_monitor_drift_reports_no_alert_for_identical_inputs(
    monkeypatch, tmp_path, capsys
) -> None:
    rows = ["0,0\n", "1,1\n", "2,2\n"]
    reference = _write_csv(tmp_path / "reference.csv", rows)
    target = _write_csv(tmp_path / "target.csv", rows)

    _run_main(
        monkeypatch,
        "monitor_drift",
        ["--reference", str(reference), "--target", str(target)],
    )

    out = capsys.readouterr().out
    assert "Feature drift summary" in out
    assert "No drift alerts detected" in out


def test_monitor_drift_exits_non_zero_when_drift_is_detected(
    monkeypatch, tmp_path, capsys
) -> None:
    reference = _write_csv(tmp_path / "reference.csv", ["0,0\n", "1,1\n", "2,2\n"])
    target = _write_csv(tmp_path / "target.csv", ["90,90\n", "91,91\n", "92,92\n"])

    with pytest.raises(SystemExit) as excinfo:
        _run_main(
            monkeypatch,
            "monitor_drift",
            [
                "--reference",
                str(reference),
                "--target",
                str(target),
                "--ks-threshold",
                "0.1",
            ],
        )

    assert excinfo.value.code == 1
    assert "Drift alerts" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# scripts/smoke_exp1.py
# ---------------------------------------------------------------------------


def test_smoke_exp1_produces_a_run(monkeypatch, tmp_path) -> None:
    """The fast sanity check documented in the README."""
    results = tmp_path / "results"

    _run_main(monkeypatch, "smoke_exp1", ["--results-dir", str(results)])

    summaries = list(results.rglob("*_summary.json"))
    assert len(summaries) == 1
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert Path(payload["table"]).is_file()
    assert Path(payload["figure"]).is_file()
