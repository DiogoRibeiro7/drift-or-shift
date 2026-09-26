"""I/O helpers for experiment artifact management."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dataexcept import DataLoadingError, FileWriteError
from matplotlib.figure import Figure


def ensure_dir(path: Path | str) -> Path:
    """Ensure a directory exists and return its path object."""
    path_obj = Path(path)
    try:
        path_obj.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FileWriteError(str(path_obj), exc) from exc
    return path_obj


def load_table(path: Path | str) -> pd.DataFrame:
    """Load a CSV, preserving the source path and underlying read error."""
    source = Path(path)
    try:
        return pd.read_csv(source)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DataLoadingError(str(source), exc) from exc


def load_yaml(path: Path | str) -> Any:
    """Load a YAML document with a source-aware error on read or parse failure."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DataLoadingError(str(source), exc) from exc


def save_table(df: pd.DataFrame, path: Path | str) -> None:
    """Persist a DataFrame to CSV, creating parent directories as needed."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    try:
        df.to_csv(path_obj, index=False)
    except (OSError, UnicodeError) as exc:
        raise FileWriteError(str(path_obj), exc) from exc


def save_json(obj: Any, path: Path | str, *, indent: int = 2) -> None:
    """Write an object to disk in JSON format."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    try:
        with path_obj.open("w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=indent)
    except OSError as exc:
        raise FileWriteError(str(path_obj), exc) from exc


def save_text(text: str, path: Path | str) -> None:
    """Write UTF-8 text, creating its parent directory."""
    destination = Path(path)
    ensure_dir(destination.parent)
    try:
        destination.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise FileWriteError(str(destination), exc) from exc


def save_figure(figure: Figure, path: Path | str) -> None:
    """Save a Matplotlib figure with a path-aware write error."""
    destination = Path(path)
    ensure_dir(destination.parent)
    try:
        figure.savefig(destination)
    except OSError as exc:
        raise FileWriteError(str(destination), exc) from exc


def timestamped_run_dir(base: Path | str = "results", name: str = "exp1") -> Path:
    """Create a new timestamped run directory and return its path."""
    base_path = ensure_dir(Path(base) / name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(base_path / timestamp)
    ensure_dir(run_dir / "tables")
    ensure_dir(run_dir / "figures")
    return run_dir
