"""Summarize experiments that triggered feature-drift alerts."""

from __future__ import annotations

import argparse
from pathlib import Path

from drift_or_shift.reporting import collect_summary_jsons, load_drift_alert_thresholds, write_drift_alerts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize drift alerts across experiments.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("reports") / "drift_alerts.csv")
    parser.add_argument("--threshold-config", type=Path, default=None, help="YAML/JSON file with drift thresholds")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summaries = collect_summary_jsons(args.results_dir)
    thresholds = load_drift_alert_thresholds(args.threshold_config)
    path = write_drift_alerts(summaries, args.output, thresholds=thresholds)
    if not path.exists():
        print("No drift alerts detected.")
        return
    print(f"drift alerts written to {path}")


if __name__ == "__main__":
    main()
