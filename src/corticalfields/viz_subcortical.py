"""Backward-compatible shim — canonical location is corticalfields.viz.subcortical."""
import warnings as _w
_w.warn(
    "corticalfields.viz_subcortical is deprecated; "
    "use corticalfields.viz.subcortical instead.",
    DeprecationWarning, stacklevel=2,
)
from corticalfields.viz.subcortical import *  # noqa: F401,F403
