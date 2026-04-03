"""Backward-compatible shim — canonical location is corticalfields.viz.bayes."""
import warnings as _w
_w.warn(
    "corticalfields.bayes_viz is deprecated; "
    "use corticalfields.viz.bayes instead.",
    DeprecationWarning, stacklevel=2,
)
from corticalfields.viz.bayes import *  # noqa: F401,F403
