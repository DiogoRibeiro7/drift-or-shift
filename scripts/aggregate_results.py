"""Generate a lightweight dashboard summarizing experiment outputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from dataexcept import DataLoadingError

from drift_or_shift.io_utils import ensure_dir, load_table, save_figure, save_text
from drift_or_shift.reporting import (
    collect_summary_jsons,
    detect_drift_alerts,
    format_results_overview,
    load_drift_alert_thresholds,
    select_latest_summaries,
    write_drift_alerts,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the latest experiment tables and identify best strategies."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figure", type=Path)
    parser.add_argument(
        "--drift-output", type=Path, default=Path("reports") / "drift_alerts.csv"
    )
    parser.add_argument(
        "--fail-on-alerts",
        action="store_true",
        help="exit with status 1 when drift alerts appear",
    )
    parser.add_argument(
        "--threshold-config",
        type=Path,
        default=None,
        help="YAML/JSON file with drift thresholds",
    )
    return parser.parse_args()


def _safe_mean(values: Sequence[Any] | None) -> float | None:
    if not values:
        return None
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _best_risk_metric(summary: dict) -> tuple[str | None, float | None]:
    aggregated = summary.get("aggregated", {}) or {}
    best_name: str | None = None
    best_value: float | None = None
    for metric, values in aggregated.items():
        if not (
            isinstance(metric, str)
            and metric.startswith("risk_")
            and metric.endswith("_mean")
        ):
            continue
        candidate = _safe_mean(values)
        if candidate is None:
            continue
        if best_value is None or candidate < best_value:
            best_name = metric
            best_value = candidate
    if best_name and best_value is not None:
        return best_name, best_value

    table_path = summary.get("table")
    if not table_path:
        return None, None
    table_file = Path(table_path)
    if not table_file.exists():
        return None, None
    try:
        df = load_table(table_file)
    except (DataLoadingError, ValueError):
        return None, None
    risk_cols = [
        col for col in df.columns if col.startswith("risk_") and col.endswith("_mean")
    ]
    if not risk_cols:
        return None, None
    best_col = min(risk_cols, key=lambda col: float(df[col].mean()))
    return best_col, float(df[best_col].mean())


def _pretty_metric_name(metric: str) -> str:
    trimmed = metric.removesuffix("_mean")
    return trimmed.replace("_", " ").strip()


def _collect_best_records(summaries: Sequence[dict]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary in summaries:
        metric, value = _best_risk_metric(summary)
        if not metric or value is None:
            continue
        records.append(
            {
                "exp_name": summary.get("exp_name", "unknown"),
                "metric": metric,
                "value": value,
                "table": summary.get("table"),
                "figure": summary.get("figure"),
                "alerts": detect_drift_alerts(summary),
            }
        )
    records.sort(key=lambda rec: rec["value"])
    return records


def _plot_best_risks(records: Sequence[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    experiments = [rec["exp_name"] for rec in records]
    values = [rec["value"] for rec in records]
    labels = [_pretty_metric_name(rec["metric"]) for rec in records]

    fig, ax = plt.subplots(figsize=(8, max(3, len(records) * 0.6)))
    bars = ax.barh(experiments, values, color="tab:blue")
    ax.set_xlabel("Mean risk (lower is better)")
    ax.set_title("Best-performing strategy per experiment")
    ax.invert_yaxis()
    for bar, label, value in zip(bars, labels, values, strict=True):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {label} ({value:.3f})",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def _build_dashboard_text(
    base_overview: str,
    records: Sequence[dict[str, Any]],
    figure_path: Path | None,
) -> str:
    lines = [base_overview.rstrip(), "", "## Dashboard highlights", ""]
    if not records:
        lines.append("No ranked risks are available yet.")
    else:
        lines.append(
            "Highlights show the lowest mean-risk strategy for each experiment."
        )
        lines.append("")
        for record in records:
            metric_label = _pretty_metric_name(record["metric"])
            lines.append(
                f"- {record['exp_name']}: `{metric_label}` (mean risk {record['value']:.4f})"
            )
            table = record.get("table")
            if table:
                lines.append(f"  - table: {table}")
            fig = record.get("figure")
            if fig:
                lines.append(f"  - figure: {fig}")
            alerts = record.get("alerts")
            if alerts:
                lines.append("  - drift alerts:")
                for metric, value, threshold in alerts:
                    lines.append(
                        f"    - {metric} = {value:.3f} (threshold {threshold:.3f})"
                    )
    if figure_path:
        lines.append("")
        lines.append(f"Best-risk figure: {figure_path}")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or Path("reports") / f"results_dashboard_{timestamp}.md"
    figure_path = args.figure or Path("reports") / f"best_risk_{timestamp}.png"

    summaries = collect_summary_jsons(args.results_dir)
    if not summaries:
        print(f"no summaries found under {args.results_dir}")
        return

    latest = select_latest_summaries(summaries)
    if not latest:
        print("no valid summaries with timestamps were found")
        return

    records = _collect_best_records(latest)
    dashboard_text = _build_dashboard_text(
        format_results_overview(latest),
        records,
        figure_path if records else None,
    )
    save_text(dashboard_text, output_path)
    print(f"dashboard written to {output_path}")

    if records:
        _plot_best_risks(records, figure_path)
        print(f"best-risk figure written to {figure_path}")

    thresholds = load_drift_alert_thresholds(args.threshold_config)
    drift_path = write_drift_alerts(latest, args.drift_output, thresholds=thresholds)
    if drift_path.exists():
        print(f"drift alerts written to {drift_path}")
        if args.fail_on_alerts and drift_path.stat().st_size > 0:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
