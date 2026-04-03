"""Backward-compatible shim — canonical location is corticalfields.analysis.normative."""
import warnings as _w
_w.warn(
    "corticalfields.normative is deprecated; "
    "use corticalfields.analysis.normative instead.",
    DeprecationWarning, stacklevel=2,
)
from corticalfields.analysis.normative import *  # noqa: F401,F403
