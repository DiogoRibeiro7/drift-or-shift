"""Keep `docs/API.md` in step with what the package actually exports.

The package ships `py.typed` and exports fifty-odd names, and none of them were
documented anywhere. A hand-written reference solves that once and then rots,
in exactly the way `RESULTS_DIGEST.md` did before its figures were enforced, so
this checks it on every run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import drift_or_shift

API_DOC = Path(__file__).resolve().parents[1] / "docs" / "API.md"

# Names whose bare mention in prose is enough; they are described in the text
# rather than given a row of their own.
PUBLIC_NAMES = sorted(n for n in drift_or_shift.__all__ if not n.startswith("__"))


@pytest.fixture(scope="module")
def api_doc() -> str:
    return API_DOC.read_text(encoding="utf-8")


def test_api_doc_exists() -> None:
    assert API_DOC.is_file(), "docs/API.md is missing"


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_every_public_name_is_documented(api_doc: str, name: str) -> None:
    """A new export must be added to the reference in the same change."""
    pattern = rf"\b{re.escape(name)}\b"

    assert re.search(pattern, api_doc), (
        f"{name} is exported from drift_or_shift but does not appear in "
        f"docs/API.md. Add it there, or drop it from __all__."
    )


def test_the_doc_does_not_describe_names_that_no_longer_exist(api_doc: str) -> None:
    """Catch the other direction: a rename leaving the doc pointing at nothing.

    Only backticked identifiers that look like our naming style are checked, so
    ordinary prose and third-party names do not trip it.
    """
    candidates = set(re.findall(r"`([a-z_][a-z0-9_]{3,})\(", api_doc))
    known = set(dir(drift_or_shift))
    # Helpers that are genuinely referenced but not exported at top level.
    allowed = {"detect_drift_alerts", "default_rng", "metadata", "use", "astype"}

    stale = sorted(name for name in candidates if name not in known | allowed)

    assert not stale, f"docs/API.md documents names that are not exported: {stale}"


def test_every_public_callable_has_a_docstring() -> None:
    """`py.typed` promises annotations; a docstring is the other half."""
    undocumented = [
        name
        for name in PUBLIC_NAMES
        if callable(getattr(drift_or_shift, name, None))
        and not (getattr(drift_or_shift, name).__doc__ or "").strip()
    ]

    assert not undocumented, f"public callables without a docstring: {undocumented}"


def test_all_exports_actually_resolve() -> None:
    """`__all__` naming something that does not exist breaks `import *`."""
    missing = [
        name for name in drift_or_shift.__all__ if not hasattr(drift_or_shift, name)
    ]

    assert not missing, f"__all__ names that do not resolve: {missing}"


def test_readme_points_at_the_api_reference() -> None:
    readme = (API_DOC.parents[1] / "README.md").read_text(encoding="utf-8")

    assert "docs/API.md" in readme, "README does not link to the API reference"
