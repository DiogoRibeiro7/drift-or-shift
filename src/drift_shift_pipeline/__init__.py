"""Deprecated alias for :mod:`drift_or_shift`.

The distribution was once called ``drift-shift-pipeline`` and this package
existed so that name worked as an import too. The distribution is now
``drift-or-shift``, matching the import package, so this alias is the last
remnant of the old name.

It is kept so existing local scripts keep working, and emits a
``DeprecationWarning`` on import. Import :mod:`drift_or_shift` instead.
"""

import warnings
from importlib import metadata

from drift_or_shift import *  # noqa: F403
from drift_or_shift import __all__ as __drift_or_shift_all

warnings.warn(
    "drift_shift_pipeline is a deprecated alias for drift_or_shift and will be "
    "removed in a future release; import drift_or_shift instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    __version__ = metadata.version("drift-or-shift")
except metadata.PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0"

__all__ = [*__drift_or_shift_all, "__version__"]
