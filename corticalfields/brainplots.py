"""Backward-compatible shim — canonical location is corticalfields.viz.brainplots."""
import warnings as _w
_w.warn(
    "corticalfields.brainplots is deprecated; "
    "use corticalfields.viz.brainplots instead.",
    DeprecationWarning, stacklevel=2,
)
from corticalfields.viz.brainplots import *  # noqa: F401,F403
