"""Print the packaged version, without importing the package.

Used by the release workflow to check a tag against the single source of
truth in `src/drift_or_shift/__init__.py`. It deliberately avoids importing
`drift_or_shift`, so it works before the project is installed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

INIT = Path(__file__).resolve().parents[1] / "src" / "drift_or_shift" / "__init__.py"
PATTERN = re.compile(r'^__version__ = "([^"]+)"', re.MULTILINE)

# PEP 440 pre-release and development segments, including the spellings PEP 440
# normalises away (`alpha` -> `a`, `preview` -> `rc`, and so on).
PRERELEASE = re.compile(
    r"(a|b|c|rc|alpha|beta|pre|preview|dev)[.\-_]?\d*$", re.IGNORECASE
)


def read_version(init_path: Path = INIT) -> str:
    """Return the value of ``__version__`` declared in the package."""
    match = PATTERN.search(init_path.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"no __version__ assignment found in {init_path}")
    return match.group(1)


def is_prerelease(version: str) -> bool:
    """Return True when the version carries a PEP 440 pre-release segment.

    The release workflow uses this to mark a GitHub release as a pre-release,
    so an alpha does not advertise itself as the latest stable version.
    """
    return PRERELEASE.search(version) is not None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    version = read_version()
    if args == ["--prerelease"]:
        sys.stdout.write(("true" if is_prerelease(version) else "false") + "\n")
        return 0
    if args:
        sys.stderr.write("usage: print_version.py [--prerelease]\n")
        return 2
    sys.stdout.write(version + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
