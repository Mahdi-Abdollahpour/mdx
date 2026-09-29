"""Shared utility helpers."""

from .debug_utils import dbg, tattle
from .math_utils import (
    huber,
    masked_quantile,
    masked_reduce_max,
    masked_reduce_mean,
    masked_reduce_min,
    mse,
)

__all__ = [
    "COLORMAP",
    "_flatten_dims",
    "_insert_dims",
    "_squeeze",
    "_squeeze_inds_nd",
    "_squeeze_inds_nd_",
    "_stop_gradients",
    "c2ri",
    "collapse_axes",
    "dbg",
    "huber",
    "masked_quantile",
    "masked_reduce_max",
    "masked_reduce_mean",
    "masked_reduce_min",
    "mse",
    "plot_error_metric_results",
    "ri2c",
    "swap_axis",
    "tattle",
]


def __getattr__(name):
    """Lazily import the plotting and tensor helpers on first access."""
    if name in {"COLORMAP", "plot_error_metric_results"}:
        from .eval_plots import COLORMAP, plot_error_metric_results

        globals()["COLORMAP"] = COLORMAP
        globals()["plot_error_metric_results"] = plot_error_metric_results
        return globals()[name]

    if name in {
        "_flatten_dims",
        "_insert_dims",
        "_squeeze",
        "_squeeze_inds_nd",
        "_squeeze_inds_nd_",
        "_stop_gradients",
        "c2ri",
        "collapse_axes",
        "ri2c",
        "swap_axis",
    }:
        from .tensor_ops import (
            _flatten_dims,
            _insert_dims,
            _squeeze,
            _squeeze_inds_nd,
            _squeeze_inds_nd_,
            _stop_gradients,
            c2ri,
            collapse_axes,
            ri2c,
            swap_axis,
        )

        globals().update(
            {
                "_flatten_dims": _flatten_dims,
                "_insert_dims": _insert_dims,
                "_squeeze": _squeeze,
                "_squeeze_inds_nd": _squeeze_inds_nd,
                "_squeeze_inds_nd_": _squeeze_inds_nd_,
                "_stop_gradients": _stop_gradients,
                "c2ri": c2ri,
                "collapse_axes": collapse_axes,
                "ri2c": ri2c,
                "swap_axis": swap_axis,
            }
        )
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
