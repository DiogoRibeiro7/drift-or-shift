# -*- coding: utf-8 -*-
"""Watch `results/*/_summary.json` and refresh the dashboard/alerts when they change."""

from __future__ import annotations

import argparse
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun drift_alerts.py + aggregate_results.py whenever result summaries change."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--drift-output", type=Path, default=Path("reports") / "drift_alerts.csv")
    parser.add_argument("--fail-on-alerts", action="store_true")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run the pipeline once, even if no changes were detected, then exit.",
    )
    return parser.parse_args()


def _snapshot(results_dir: Path) -> Dict[Path, float]:
    if not results_dir.exists():
        return {}
    snapshot: Dict[Path, float] = {}
    for summary in results_dir.rglob("*_summary.json"):
        try:
            snapshot[summary] = summary.stat().st_mtime
        except OSError:
            continue
    return snapshot


def _has_changes(previous: Dict[Path, float], current: Dict[Path, float]) -> bool:
    if previous.keys() != current.keys():
        return True
    for path, mtime in current.items():
        if previous.get(path) != mtime:
            return True
    return False


def _run_pipeline(results_dir: Path, drift_output: Path, fail_on_alerts: bool) -> None:
    python = sys.executable
    drift_cmd = [
        python,
        "scripts/drift_alerts.py",
        "--results-dir",
        str(results_dir),
        "--output",
        str(drift_output),
    ]
    aggregate_cmd = [
        python,
        "scripts/aggregate_results.py",
        "--results-dir",
        str(results_dir),
        "--drift-output",
        str(drift_output),
    ]
    if fail_on_alerts:
        aggregate_cmd.append("--fail-on-alerts")
    print(f"[{datetime.now().isoformat()}] running drift_alerts + aggregate_results")
    subprocess.run(drift_cmd, check=True)
    subprocess.run(aggregate_cmd, check=True)


def main() -> None:
    args = _parse_args()
    previous = _snapshot(args.results_dir)
    triggered = False
    while True:
        current = _snapshot(args.results_dir)
        changed = _has_changes(previous, current)
        if changed or args.once or not triggered:
            try:
                _run_pipeline(args.results_dir, args.drift_output, args.fail_on_alerts)
            except subprocess.CalledProcessError as exc:
                print(f"pipeline failed: {exc}")
            previous = current
            triggered = True
        if args.once:
            break
        sleep = max(0.1, args.poll_interval)
        print(f"[{datetime.now().isoformat()}] sleeping {sleep:.1f}s")
        time.sleep(sleep)


if __name__ == "__main__":
    main()
