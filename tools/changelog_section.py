"""Print the CHANGELOG section for one version, for use as release notes.

The trailing link-reference definitions are markdown plumbing for the file
itself, not something a reader of the release wants, so they are dropped.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
LINK_DEFINITION = re.compile(r"^\[[^\]]+\]:\s*http")


def extract_section(version: str, text: str) -> str:
    """Return the body of the ``## [version]`` section, without link defs."""
    match = re.search(
        rf"^## \[{re.escape(version)}\].*?$(.*?)(?=^## \[|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        return ""
    kept = [
        line for line in match.group(1).splitlines() if not LINK_DEFINITION.match(line)
    ]
    return "\n".join(kept).strip()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: changelog_section.py VERSION\n")
        return 2
    version = argv[1]
    section = extract_section(version, CHANGELOG.read_text(encoding="utf-8"))
    if not section:
        sys.stderr.write(f"no CHANGELOG section found for version {version}\n")
        return 1
    sys.stdout.write(section + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
