"""Tests for the installable packaging contract.

The distribution was unbuildable for a long stretch (`tool.setuptools.packages`
was set to the literal string ``"find:"``), so every console script and
subpackage shipped broken. These tests assert the contract the build must keep.
"""

from __future__ import annotations

import importlib
import sys
from importlib import metadata
from pathlib import Path

import pytest

import drift_or_shift

if sys.version_info >= (3, 11):
    import tomllib
else:  # tomllib entered the standard library in 3.11
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def test_package_exposes_version() -> None:
    assert isinstance(drift_or_shift.__version__, str)
    assert drift_or_shift.__version__.count(".") >= 2


def test_declared_version_matches_installed_metadata() -> None:
    try:
        installed = metadata.version("drift-shift-pipeline")
    except metadata.PackageNotFoundError:  # pragma: no cover - not installed
        pytest.skip("drift-shift-pipeline is not installed in this environment")
    assert installed == drift_or_shift.__version__


@pytest.mark.parametrize(
    "module",
    [
        "drift_or_shift",
        "drift_or_shift.experiments",
        "drift_shift_pipeline",
        "caliblab",
    ],
)
def test_shipped_packages_are_importable(module: str) -> None:
    assert importlib.import_module(module) is not None


def test_every_console_script_target_is_importable(pyproject: dict) -> None:
    """Each `dos-expN` entry point must resolve, or the wheel ships dead scripts."""
    scripts = pyproject["project"]["scripts"]
    assert scripts, "expected console scripts to be declared"

    for name, target in scripts.items():
        module_path, _, attr = target.partition(":")
        module = importlib.import_module(module_path)
        assert callable(getattr(module, attr)), f"{name} -> {target} is not callable"


def test_py_typed_markers_are_present() -> None:
    """PEP 561: without these, downstream type checkers ignore our annotations."""
    src = PYPROJECT.parent / "src"
    for package in ("drift_or_shift", "drift_shift_pipeline", "caliblab"):
        assert (src / package / "py.typed").is_file(), f"missing py.typed: {package}"


def test_license_file_exists_for_declared_license(pyproject: dict) -> None:
    assert pyproject["project"]["license"] == "MIT"
    assert (PYPROJECT.parent / "LICENSE").is_file()
