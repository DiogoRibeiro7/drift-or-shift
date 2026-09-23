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


def read_version(init_path: Path = INIT) -> str:
    """Return the value of ``__version__`` declared in the package."""
    match = PATTERN.search(init_path.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"no __version__ assignment found in {init_path}")
    return match.group(1)


def main() -> int:
    sys.stdout.write(read_version() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
