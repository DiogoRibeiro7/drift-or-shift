"""Alias package for drift_or_shift to maintain backward compatibility."""

from importlib import metadata

from drift_or_shift import *  # noqa: F403
from drift_or_shift import __all__ as __drift_or_shift_all

try:
    __version__ = metadata.version("drift-shift-pipeline")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [*__drift_or_shift_all, "__version__"]
