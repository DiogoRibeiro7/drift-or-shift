"""Tests for the scripts the release workflow depends on.

These decide whether a mistagged or undocumented release is caught before it
is published. They run in CI on every PR, so a break is visible long before
somebody pushes a tag.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import drift_or_shift

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS))

from changelog_section import extract_section  # noqa: E402
from check_sarif import collect_results, describe  # noqa: E402
from print_version import is_prerelease, read_version  # noqa: E402

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


# ---------------------------------------------------------------------------
# check_sarif
# ---------------------------------------------------------------------------


SARIF_WITH_FINDING = {
    "runs": [
        {
            "results": [
                {
                    "ruleId": "zizmor/template-injection",
                    "message": {"text": "code injection via template expansion"},
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {"uri": "release.yml"},
                                "region": {"startLine": 36},
                            }
                        }
                    ],
                }
            ]
        }
    ]
}


def test_collect_results_spans_every_run() -> None:
    report = {"runs": [{"results": [{"ruleId": "a"}]}, {"results": [{"ruleId": "b"}]}]}

    assert [r["ruleId"] for r in collect_results(report)] == ["a", "b"]


def test_collect_results_of_a_clean_report_is_empty() -> None:
    assert collect_results({"runs": [{"results": []}]}) == []
    assert collect_results({}) == []


def test_describe_names_the_rule_and_location() -> None:
    line = describe(SARIF_WITH_FINDING["runs"][0]["results"][0])

    assert "template-injection" in line
    assert "release.yml:36" in line


def test_describe_tolerates_a_result_without_a_location() -> None:
    assert "some-rule" in describe({"ruleId": "some-rule", "message": {"text": "x"}})


def _run_check(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOLS / "check_sarif.py"), str(path)],
        capture_output=True,
        text=True,
    )


def test_check_sarif_fails_when_the_report_has_findings(tmp_path: Path) -> None:
    report = tmp_path / "z.sarif"
    report.write_text(json.dumps(SARIF_WITH_FINDING), encoding="utf-8")

    result = _run_check(report)

    assert result.returncode == 1
    assert "::error::" in result.stdout


def test_check_sarif_passes_on_a_clean_report(tmp_path: Path) -> None:
    report = tmp_path / "z.sarif"
    report.write_text(json.dumps({"runs": [{"results": []}]}), encoding="utf-8")

    assert _run_check(report).returncode == 0


def test_check_sarif_treats_a_missing_report_as_an_error(tmp_path: Path) -> None:
    """An audit that never ran must not look like an audit that passed."""
    result = _run_check(tmp_path / "absent.sarif")

    assert result.returncode == 2
    assert "did not run" in result.stderr


def test_check_sarif_treats_an_empty_report_as_an_error(tmp_path: Path) -> None:
    report = tmp_path / "z.sarif"
    report.write_text("", encoding="utf-8")

    assert _run_check(report).returncode == 2


def test_check_sarif_rejects_malformed_json(tmp_path: Path) -> None:
    report = tmp_path / "z.sarif"
    report.write_text("{not json", encoding="utf-8")

    result = _run_check(report)

    assert result.returncode == 2
    assert "not valid SARIF" in result.stderr


# ---------------------------------------------------------------------------
# prerelease detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.1.0", False),
        ("1.2.3", False),
        ("10.0.0", False),
        ("1.0.0.post1", False),
        ("0.1.0a1", True),
        ("0.1.0b2", True),
        ("0.1.0rc1", True),
        ("0.1.0.dev1", True),
        ("2.0.0alpha3", True),
        ("2.0.0beta1", True),
    ],
)
def test_prerelease_detection(version: str, expected: bool) -> None:
    """Drives whether the GitHub release is flagged as a pre-release.

    Getting this wrong in the permissive direction would publish an alpha as
    the repository's latest stable release.
    """
    assert is_prerelease(version) is expected


def test_prerelease_flag_matches_the_current_version() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOLS / "print_version.py"), "--prerelease"],
        capture_output=True,
        text=True,
        check=True,
    )

    expected = "true" if is_prerelease(drift_or_shift.__version__) else "false"
    assert result.stdout.strip() == expected


def test_print_version_rejects_unknown_arguments() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOLS / "print_version.py"), "--nope"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "usage" in result.stderr
