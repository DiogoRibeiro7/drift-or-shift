import json
from pathlib import Path

import pandas as pd
import pytest
from dataexcept import DataLoadingError, FileWriteError

from drift_or_shift import io_utils


def test_io_helpers_create_dirs_and_files(tmp_path: Path) -> None:
    output_dir = io_utils.ensure_dir(tmp_path / "runs")
    assert output_dir.exists()

    df = pd.DataFrame({"x": [1, 2]})
    table_path = output_dir / "table.csv"
    io_utils.save_table(df, table_path)
    assert table_path.exists()
    read_back = pd.read_csv(table_path)
    assert read_back.equals(df)

    json_path = output_dir / "meta.json"
    io_utils.save_json({"exp": "test"}, json_path)
    assert json_path.exists()
    assert json.loads(json_path.read_text())["exp"] == "test"

    run_dir = io_utils.timestamped_run_dir(base=tmp_path, name="exp5")
    assert (run_dir / "tables").exists()
    assert (run_dir / "figures").exists()


def test_loaders_report_the_source_and_original_failure(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(DataLoadingError) as csv_error:
        io_utils.load_table(missing)
    assert csv_error.value.source == str(missing)
    assert isinstance(csv_error.value.original, FileNotFoundError)
    assert csv_error.value.__cause__ is csv_error.value.original

    invalid = tmp_path / "config.yaml"
    invalid.write_text("key: [unclosed", encoding="utf-8")
    with pytest.raises(DataLoadingError) as yaml_error:
        io_utils.load_yaml(invalid)
    assert yaml_error.value.source == str(invalid)
    assert yaml_error.value.__cause__ is yaml_error.value.original


def test_writers_report_the_path_and_preserve_serialization_errors(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "output.csv"
    destination.mkdir()
    with pytest.raises(FileWriteError) as error:
        io_utils.save_table(pd.DataFrame({"value": [1]}), destination)
    assert error.value.path == str(destination)
    assert isinstance(error.value.original, OSError)
    assert error.value.__cause__ is error.value.original

    with pytest.raises(TypeError):
        io_utils.save_json({"invalid": object()}, tmp_path / "invalid.json")
