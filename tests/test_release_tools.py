"""Tests for the scripts the release workflow depends on.

These decide whether a mistagged or undocumented release is caught before it
is published. They run in CI on every PR, so a break is visible long before
somebody pushes a tag.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import drift_or_shift

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS))

from changelog_section import extract_section  # noqa: E402
from print_version import read_version  # noqa: E402

# ---------------------------------------------------------------------------
# print_version
# ---------------------------------------------------------------------------


def test_reported_version_matches_the_package() -> None:
    """The workflow compares a tag against this; it must not drift."""
    assert read_version() == drift_or_shift.__version__


def test_version_is_read_without_importing_the_package(tmp_path: Path) -> None:
    init = tmp_path / "__init__.py"
    init.write_text(
        '"""doc."""\n\n__version__ = "9.9.9"\n\nimport nonexistent_module\n',
        encoding="utf-8",
    )

    assert read_version(init) == "9.9.9"


def test_missing_version_assignment_is_an_error(tmp_path: Path) -> None:
    init = tmp_path / "__init__.py"
    init.write_text('"""no version here."""\n', encoding="utf-8")

    with pytest.raises(SystemExit):
        read_version(init)


def test_print_version_script_runs_standalone() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOLS / "print_version.py")],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == drift_or_shift.__version__


# ---------------------------------------------------------------------------
# changelog_section
# ---------------------------------------------------------------------------


CHANGELOG = """# Changelog

## [Unreleased]

### Added

- Something pending.

## [1.2.0] - 2026-05-01

### Fixed

- A real bug.

### Added

- A feature.

## [1.1.0] - 2026-04-01

### Added

- Older work.

[Unreleased]: https://example.com/compare/v1.2.0...HEAD
[1.2.0]: https://example.com/releases/tag/v1.2.0
"""


def test_extracts_only_the_requested_version() -> None:
    section = extract_section("1.2.0", CHANGELOG)

    assert "A real bug." in section
    assert "A feature." in section
    assert "Older work." not in section, "bled into the previous release"
    assert "Something pending." not in section, "picked up Unreleased"


def test_link_definitions_are_stripped() -> None:
    """Link refs are plumbing for the file, not release notes."""
    section = extract_section("1.1.0", CHANGELOG)

    assert "Older work." in section
    assert "https://example.com" not in section
    assert "[Unreleased]:" not in section


def test_unknown_version_yields_nothing() -> None:
    assert extract_section("9.9.9", CHANGELOG) == ""


def test_section_is_stripped_of_surrounding_blank_lines() -> None:
    section = extract_section("1.1.0", CHANGELOG)

    assert section == section.strip()
    assert section.startswith("###")


def test_real_changelog_documents_the_current_version() -> None:
    """The release workflow refuses to publish a version with no notes."""
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    version = drift_or_shift.__version__

    assert extract_section(version, text), (
        f"CHANGELOG.md has no '## [{version}]' section; "
        "the release workflow would reject this tag"
    )


def test_changelog_script_exits_nonzero_for_an_unknown_version() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOLS / "changelog_section.py"), "9.9.9"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "no CHANGELOG section" in result.stderr


def test_changelog_script_requires_an_argument() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOLS / "changelog_section.py")],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "usage" in result.stderr
