
# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026


"""Runtime configuration: debug flags, numerical constants and import paths."""

import os
import sys

import tensorflow as tf


# Debug flags; values > 0 enable the corresponding behavior.
_DEBUGD_DEFAULTS = {
    "print": 1,
    "print2D": 0,
    "dbg": 1,
    "tattle": 1,
    "plot": 0,
    "verify": 1,
    "pause": 2,
}

_get_run_eagerly = getattr(tf.config, "functions_run_eagerly", None)
if _get_run_eagerly is None:
    _get_run_eagerly = getattr(tf.config, "run_functions_eagerly", None)

_inside_function = tf.inside_function
_executing_eagerly = tf.executing_eagerly


def _is_graph_mode():
    if _inside_function():
        return True
    if _get_run_eagerly is not None:
        try:
            if _get_run_eagerly():
                return False
        except TypeError:
            pass
    return not _executing_eagerly()


class _DebugConfig(dict):
    """Dict of debug flags that reads as all zeros in graph (``tf.function``) mode."""

    __slots__ = ("_disabled",)

    def __init__(self, values):
        super().__init__(values)
        self._disabled = {key: 0 for key in values}

    def __getitem__(self, key):
        if _is_graph_mode():
            return self._disabled[key]
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if _is_graph_mode():
            return self._disabled.get(key, default)
        return dict.get(self, key, default)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self._disabled.setdefault(key, 0)


DEBUGD = _DebugConfig(_DEBUGD_DEFAULTS)
is_funcs_eager = not _is_graph_mode()

EPS = 1.0e-7

DATA_FORMAT = "channels_last"

# Make the vendored Sionna submodule importable.
SUBMODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../ext/sionna")
)

if SUBMODULE_PATH not in sys.path:
    sys.path.insert(0, SUBMODULE_PATH)
