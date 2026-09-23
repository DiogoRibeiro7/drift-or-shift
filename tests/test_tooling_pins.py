"""The dev extra and the pre-commit hooks must pin the same tool versions.

`CONTRIBUTING.md` says to bump them together, and that instruction was already
broken once: Dependabot watches `pyproject.toml` but not
`.pre-commit-config.yaml`, so a dependency PR moved `mypy` from 1.20.2 to 2.3.1
in one file and left the hook on the old version. Nothing went red, because
each file is internally consistent -- the hooks and CI simply stopped checking
the same thing.

A convention that only a human can enforce is not enforced, so this checks it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:  # tomllib entered the standard library in 3.11
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PRE_COMMIT = REPO_ROOT / ".pre-commit-config.yaml"

# hook repository -> distribution name in the dev extra
MYPY_MIRROR = "https://github.com/pre-commit/mirrors-mypy"

HOOK_REPOS = {
    "https://github.com/astral-sh/ruff-pre-commit": "ruff",
    "https://github.com/psf/black-pre-commit-mirror": "black",
    MYPY_MIRROR: "mypy",
}

PIN = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^;\s]+)")


@pytest.fixture(scope="module")
def dev_pins() -> dict[str, str]:
    """Exact `name==version` pins from the `dev` extra."""
    with PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    pins = {}
    for entry in data["project"]["optional-dependencies"]["dev"]:
        match = PIN.match(entry)
        if match:
            pins[match["name"].lower()] = match["version"]
    return pins


@pytest.fixture(scope="module")
def pre_commit_config() -> dict:
    return yaml.safe_load(PRE_COMMIT.read_text(encoding="utf-8"))


def _rev_for(config: dict, repo_url: str) -> str:
    for repo in config["repos"]:
        if repo["repo"] == repo_url:
            return str(repo["rev"])
    raise AssertionError(f"no pre-commit repo entry for {repo_url}")


def _mypy_hook_dependencies(config: dict) -> list[str]:
    """Return the mypy hook's `additional_dependencies`.

    Written as a lookup that always either returns or raises, rather than a
    `for/else` that assigns and breaks: on the fall-through path the latter
    leaves the result unbound, which CodeQL flags as
    `py/uninitialized-local-variable`. It is unreachable because `pytest.fail`
    raises, but a function that cannot express the bad state is better than one
    that relies on a reader knowing that.
    """
    for repo in config["repos"]:
        if repo["repo"] == MYPY_MIRROR:
            return list(repo["hooks"][0].get("additional_dependencies", []))
    raise AssertionError(f"no {MYPY_MIRROR} hook in {PRE_COMMIT.name}")


@pytest.mark.parametrize(("repo_url", "package"), sorted(HOOK_REPOS.items()))
def test_hook_revision_matches_the_dev_extra(
    dev_pins, pre_commit_config, repo_url: str, package: str
) -> None:
    expected = dev_pins[package]
    rev = _rev_for(pre_commit_config, repo_url)

    # Some mirrors tag `v1.2.3`, others `1.2.3`; both are accepted.
    assert rev.lstrip("v") == expected, (
        f"{package} is pinned to {expected} in pyproject.toml's dev extra but "
        f"the pre-commit hook is on {rev}. Bump both together, or the hooks "
        f"and CI stop checking the same thing."
    )


def test_mypy_hook_stubs_match_the_dev_extra(dev_pins, pre_commit_config) -> None:
    """The mypy hook installs its own stubs; they must match too.

    Different stub versions mean the hook and CI can disagree about types even
    when the mypy versions line up.
    """
    for entry in _mypy_hook_dependencies(pre_commit_config):
        match = PIN.match(entry)
        assert match, f"mypy hook dependency {entry!r} is not pinned exactly"
        name = match["name"].lower()
        if name in dev_pins:
            assert match["version"] == dev_pins[name], (
                f"{name} is {dev_pins[name]} in the dev extra but "
                f"{match['version']} in the mypy hook"
            )


def test_every_dev_dependency_is_pinned_exactly(dev_pins) -> None:
    """An unpinned dev tool means a new release can turn CI red on its own."""
    with PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)

    unpinned = [
        entry
        for entry in data["project"]["optional-dependencies"]["dev"]
        if not PIN.match(entry)
    ]

    assert not unpinned, f"dev dependencies without an exact pin: {unpinned}"


def test_dependabot_watches_pre_commit() -> None:
    """Dependabot watching only pyproject.toml is what caused the drift."""
    config = yaml.safe_load(
        (REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    )
    ecosystems = {update["package-ecosystem"] for update in config["updates"]}

    assert "pre-commit" in ecosystems, (
        "dependabot.yml does not watch pre-commit, so hook versions will drift "
        "from the dev extra again"
    )
