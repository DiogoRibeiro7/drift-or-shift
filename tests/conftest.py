"""Test configuration helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# Select a non-interactive backend before anything imports pyplot. The
# experiments all render figures, and without this the suite can try to open a
# GUI window on developer machines and hang on headless CI runners.
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# `scripts/` holds top-level modules rather than a package. Putting it on the
# path lets the tests import and drive them in-process, which is also what
# makes them visible to coverage -- driving them through subprocess leaves
# them measured at 0%.
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
