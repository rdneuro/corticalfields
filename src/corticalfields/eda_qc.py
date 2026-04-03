"""Backward-compatible shim — canonical location is corticalfields.analysis.eda_qc."""
import warnings as _w
_w.warn(
    "corticalfields.eda_qc is deprecated; "
    "use corticalfields.analysis.eda_qc instead.",
    DeprecationWarning, stacklevel=2,
)
from corticalfields.analysis.eda_qc import *  # noqa: F401,F403
