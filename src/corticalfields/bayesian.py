"""Backward-compatible shim — canonical location is corticalfields.analysis.bayesian."""
import warnings as _w
_w.warn(
    "corticalfields.bayesian is deprecated; "
    "use corticalfields.analysis.bayesian instead.",
    DeprecationWarning, stacklevel=2,
)
from corticalfields.analysis.bayesian import *  # noqa: F401,F403
