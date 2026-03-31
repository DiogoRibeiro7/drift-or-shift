"""Alias package for drift_or_shift to maintain backward compatibility."""

from importlib import metadata

from drift_or_shift import *  # noqa: F401,F403
from drift_or_shift import __all__ as __drift_or_shift_all  # type: ignore[import]

try:
    __version__ = metadata.version("drift-shift-pipeline")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = list(__drift_or_shift_all) + ["__version__"]
