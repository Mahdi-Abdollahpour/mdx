# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026

"""Evaluation-result plotting utilities."""

import pickle
import warnings
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np

from ext.neural_rx.utils.parameters import Parameters

COLORMAP = [
    "#76B900",
    "#F3BD00",
    "#814B9D",
    "#5C5C5C",
    "#214B9D",
    "#e48aa7",
    "#7c4848",
    "#78A2EB",
    "#ff1397",
    "#bee7dd",
]


def _resolve_color(spec):
    """Map an int to a ``COLORMAP`` entry (modulo its length); pass other colours through."""
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        return COLORMAP[int(spec) % len(COLORMAP)]
    return spec


def plot_error_metric_results(config_name, show_ber=False, xlim=None, ylim=None,
                              sim_idx=None, num_tx_eval=None, fig=None, color_offset=0,
                              labels=None, mcs_arr_eval_idx=0, line_styles=None, axis=None,
                              show=[True, True, True, True, True, True], results_dir=None,
                              target_snr=None, target_er=None, metric=None, verbose=0,
                              metric_db=False):
    # pylint: disable=line-too-long
    """Plot stored evaluation results.

    With a single ``config_name``, plots the selected metric versus SNR for
    each stored result series. With a list of configurations, plots either the
    metric at ``target_snr`` or the SNR needed to reach ``target_er`` for each
    configuration (linearly interpolated, with a warning, if not sampled).

    Parameters
    ----------
    config_name : str or list[str]
        Configuration name(s) whose results file is loaded.
    show_ber : bool
        Plot BER instead of BLER when ``metric`` is None.
    xlim, ylim : tuple[float, float] or None
        Axis limits.
    sim_idx : int or list[int] or None
        Indices of the matching result series to plot (default: all).
    num_tx_eval : int or None
        Number of active users to select (default: the config maximum).
    fig, axis : matplotlib Figure / Axes or None
        Where to draw; a new figure is created if both are None.
    color_offset : int
        Offset into ``COLORMAP``.
    labels : list[str] or None
        Series labels; labels starting with ``-`` are not plotted.
    mcs_arr_eval_idx : int
        MCS index to select.
    line_styles : list or None
        Per-series linestyle, or a list
        ``[linestyle, color, marker, marker_size, linewidth, markevery]``
        (trailing entries optional; ``color`` is a ``COLORMAP`` index or any
        matplotlib colour).
    show : list[bool]
        ``[title, x_label, y_label, legend, x_ticks, y_ticks]`` flags.
    results_dir : tuple[str, str] or None
        Prefix and suffix around the config label in the results filename.
    target_snr, target_er : float or None
        Comparison targets for multi-config mode (use one).
    metric : {"ber", "bler", "mse", "nmse"} or None
        Metric to plot; overrides ``show_ber``.
    verbose : int
        Print the series found in each results file if > 0.
    metric_db : bool
        Plot ``mse``/``nmse`` in dB (``target_er`` is then in dB as well).

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or None when drawing only on a given ``axis``.
    """

    metric_display_names = {
        "ber": "BER",
        "bler": "BLER",
        "mse": "MSE",
        "nmse": "NMSE",
    }

    def _resolve_metric_name(metric_, show_ber_):
        if metric_ is None:
            return "ber" if show_ber_ else "bler"
        metric_ = str(metric_).lower()
        if metric_ not in metric_display_names:
            raise ValueError(
                f"Unsupported metric '{metric_}'. "
                f"Choose from {tuple(metric_display_names)}."
            )
        return metric_

    metric_name = _resolve_metric_name(metric, show_ber)
    if metric_db and metric_name not in ("mse", "nmse"):
        raise ValueError(
            "metric_db=True is only supported for 'mse' and 'nmse'."
        )

    metric_label = metric_display_names[metric_name]
    plot_metric_label = f"{metric_label} [dB]" if metric_db else metric_label
    y_axis_metric_label = f"{metric_label} (dB)" if metric_db else metric_label
    use_log_y = not metric_db

    def remove_trailing_zeros(snrs_, vals_):
        """Trim trailing zero-valued metric samples."""
        last_non_zero = len(vals_) - 1
        while last_non_zero >= 0 and vals_[last_non_zero] == 0:
            last_non_zero -= 1

        if last_non_zero >= 0:
            vals_trimmed = np.asarray(vals_[:last_non_zero + 1], dtype=float)
            snrs_trimmed = np.asarray(snrs_[:last_non_zero + 1], dtype=float)
        else:
            vals_trimmed = np.asarray([], dtype=float)
            snrs_trimmed = np.asarray([], dtype=float)
        return snrs_trimmed, vals_trimmed

    def _to_numpy_1d(values_):
        if hasattr(values_, "numpy"):
            values_ = values_.numpy()
        return np.asarray(values_, dtype=float)

    def _metric_to_plot_values(values_, cfg_name_=None, key_=None):
        vals_arr_ = np.asarray(values_, dtype=float)
        if not metric_db:
            return vals_arr_

        if np.any(vals_arr_ <= 0):
            location_ = ""
            if cfg_name_ is not None:
                location_ += f" config '{cfg_name_}'"
            if key_ is not None:
                location_ += f", key={key_}"
            raise ValueError(
                f"Cannot convert {metric_name} to dB for{location_}: "
                "all values must be strictly positive."
            )
        return 10.0 * np.log10(vals_arr_)

    def _metric_target_to_raw(target_val_):
        if not metric_db:
            return float(target_val_)
        return float(10.0 ** (float(target_val_) / 10.0))

    def _make_axis(fig_, axis_):
        """Create or reuse matplotlib figure/axis."""
        if fig_ is None and axis_ is None:
            fig_, ax_ = plt.subplots(figsize=(12, 8))
        elif fig_ is not None and axis_ is None:
            ax_ = fig_.gca()
        elif fig_ is None and axis_ is not None:
            ax_ = axis_
        else:
            fig_ = None
            ax_ = axis_
        return fig_, ax_

    def _normalize_sim_idx(sim_idx_):
        """Normalize result indices into a list or None."""
        if sim_idx_ is None:
            return None
        assert isinstance(sim_idx_, (int, list, tuple)), \
            "sim_idx must be int, list, or tuple of ints."
        return [sim_idx_] if isinstance(sim_idx_, int) else list(sim_idx_)

    def _get_filename(sys_parameters_):
        """Build results filename from config parameters."""
        filename_ = f"../results/{sys_parameters_.label}_results"
        if results_dir is not None:
            filename_ = f"{results_dir[0]}{sys_parameters_.label}{results_dir[1]}_results"
        return filename_

    def _load_parameters(config_name_):
        try:
            return Parameters(config_name_, training=False, system="dummy")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Unknown config file '{config_name_}'."
            ) from exc

    def _candidate_keys(series_map_, num_tx_eval_):
        candidates_ = []
        for key_ in series_map_:
            if not isinstance(key_, tuple) or len(key_) < 2:
                continue
            if num_tx_eval_ != key_[1]:
                continue
            if len(key_) >= 3 and mcs_arr_eval_idx != key_[2]:
                continue
            candidates_.append(key_)
        return candidates_

    def _print_available_experiments(config_name_, filename_, series_map_):
        if verbose <= 0:
            return

        keys_ = [key_ for key_ in series_map_ if isinstance(key_, tuple)]
        print(f"[plot_error_metric_results] config='{config_name_}' file='{filename_}'")
        if not keys_:
            print("  experiments: none")
            return

        print("  experiments:")
        for key_ in sorted(keys_, key=lambda item: tuple(str(part) for part in item)):
            print(f"    {key_}")

    def _get_style_for_series(plot_idx, label_idx=0, default_label=None):
        """Return label and style for one plotted series."""
        l_style = "-"
        color = COLORMAP[(plot_idx + color_offset) % len(COLORMAP)]
        marker = ""
        marker_size = 3
        linewidth = 2.0
        markevery = 1

        if labels is None:
            label = default_label
        else:
            use_idx = min(label_idx, len(labels) - 1)
            label = labels[use_idx]

        if line_styles is not None:
            use_idx = min(label_idx, len(line_styles) - 1)
            style_ = line_styles[use_idx]

            if isinstance(style_, list):
                if len(style_) > 0:
                    l_style = style_[0]
                if len(style_) > 1:
                    color = _resolve_color(style_[1])
                if len(style_) > 2:
                    marker = style_[2]
                if len(style_) > 3:
                    marker_size = style_[3]
                if len(style_) > 4:
                    linewidth = style_[4]
                if len(style_) > 5:
                    markevery = style_[5]
            else:
                l_style = style_

        return label, l_style, color, marker, marker_size, linewidth, markevery

    def _skip_empty_series(cfg_name_, key_):
        if verbose > 0:
            print(
                f"[plot_error_metric_results] skipping empty {metric_name} series "
                f"for config='{cfg_name_}', key={key_}"
            )

    def _load_result_bundle(single_config_name):
        """Load one config and return metadata plus selected SNR/ER arrays."""
        sys_parameters_ = _load_parameters(single_config_name)

        filename_ = _get_filename(sys_parameters_)

        if not exists(filename_):
            raise FileNotFoundError(
                f"No results found for config '{single_config_name}' at '{filename_}'"
            )

        with open(filename_, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict) and {"snr", "ber", "bler", "mse", "nmse"} <= set(data):
            snr_map_ = data["snr"]
            metric_map_ = data[metric_name]
            _print_available_experiments(single_config_name, filename_, metric_map_)

            if not isinstance(snr_map_, dict) or not isinstance(metric_map_, dict):
                raise ValueError(
                    f"Invalid dict-based results format in '{filename_}'."
                )

            num_tx_eval_ = sys_parameters_.max_num_tx if num_tx_eval is None else num_tx_eval
            sim_idx_list_ = _normalize_sim_idx(sim_idx)
            candidates_ = _candidate_keys(metric_map_, num_tx_eval_)

            if sim_idx_list_ is None:
                sim_idx_list_ = list(np.arange(len(candidates_)))

            selected_key_ = None
            for idx_, key_ in enumerate(candidates_):
                if idx_ in sim_idx_list_:
                    selected_key_ = key_
                    break

            if selected_key_ is None:
                raise ValueError(
                    f"No matching result series found for config '{single_config_name}' "
                    f"(metric={metric_name}, num_tx_eval={num_tx_eval_}, "
                    f"sim_idx={sim_idx_list_}, mcs_arr_eval_idx={mcs_arr_eval_idx})."
                )

            snrs_sel_ = _to_numpy_1d(snr_map_[selected_key_])
            vals_sel_ = _to_numpy_1d(metric_map_[selected_key_])
            snrs_sel_, vals_sel_ = remove_trailing_zeros(snrs_sel_, vals_sel_)

            if len(snrs_sel_) == 0 or len(vals_sel_) == 0:
                raise ValueError(
                    f"Empty SNR/{metric_name} series for config '{single_config_name}'"
                )

            return sys_parameters_, selected_key_, snrs_sel_, vals_sel_

        SNRs_ = None
        if len(data) == 3:
            snrs_, BERs_, BLERs_ = data
        elif len(data) == 7:
            snrs_, BERs_, BLERs_, _, _, _, _ = data
        elif len(data) == 8:
            snrs_, BERs_, BLERs_, _, _, _, _, SNRs_ = data
        else:
            raise ValueError(f"Unsupported results format in '{filename_}'")

        ERs_ = BERs_ if metric_name == "ber" else BLERs_
        _print_available_experiments(single_config_name, filename_, ERs_)
        num_tx_eval_ = sys_parameters_.max_num_tx if num_tx_eval is None else num_tx_eval
        sim_idx_list_ = _normalize_sim_idx(sim_idx)
        candidates_ = _candidate_keys(ERs_, num_tx_eval_)

        if sim_idx_list_ is None:
            sim_idx_list_ = list(np.arange(len(candidates_)))

        selected_key_ = None
        for idx_, key_ in enumerate(candidates_):
            if idx_ in sim_idx_list_:
                selected_key_ = key_
                break

        if selected_key_ is None:
            raise ValueError(
                f"No matching result series found for config '{single_config_name}' "
                f"(metric={metric_name}, num_tx_eval={num_tx_eval_}, sim_idx={sim_idx_list_}, "
                f"mcs_arr_eval_idx={mcs_arr_eval_idx})."
            )

        if SNRs_ is not None and selected_key_ in SNRs_:
            snrs_sel_ = SNRs_[selected_key_]
        else:
            snrs_sel_ = snrs_

        ers_sel_ = ERs_[selected_key_]
        snrs_sel_, ers_sel_ = remove_trailing_zeros(snrs_sel_, ers_sel_)

        if len(snrs_sel_) == 0 or len(ers_sel_) == 0:
            raise ValueError(f"Empty SNR/ER series for config '{single_config_name}'")

        return sys_parameters_, selected_key_, np.asarray(snrs_sel_, dtype=float), np.asarray(ers_sel_, dtype=float)

    def _interp_metric_at_snr(snrs_, vals_, target_snr_, cfg_name_):
        """Return metric at target SNR using linear interpolation if needed."""
        order = np.argsort(snrs_)
        snrs_sorted = np.asarray(snrs_[order], dtype=float)
        vals_sorted = np.asarray(vals_[order], dtype=float)

        exact = np.where(np.isclose(snrs_sorted, target_snr_))[0]
        if len(exact) > 0:
            return float(vals_sorted[exact[0]])

        if target_snr_ < snrs_sorted[0] or target_snr_ > snrs_sorted[-1]:
            raise ValueError(
                f"Target SNR={target_snr_} is outside available range "
                f"[{snrs_sorted[0]}, {snrs_sorted[-1]}] for config '{cfg_name_}'."
            )

        warnings.warn(
            f"Config '{cfg_name_}': exact SNR={target_snr_} not found. "
            f"Using linear interpolation.",
            UserWarning
        )
        return float(np.interp(target_snr_, snrs_sorted, vals_sorted))

    def _interp_snr_at_metric(snrs_, vals_, target_val_, cfg_name_):
        """Return SNR at target metric value using linear interpolation if needed."""
        vals_arr = np.asarray(vals_, dtype=float)
        snrs_arr = np.asarray(snrs_, dtype=float)

        exact = np.where(np.isclose(vals_arr, target_val_))[0]
        if len(exact) > 0:
            return float(snrs_arr[exact[0]])

        order = np.argsort(vals_arr)
        vals_sorted = vals_arr[order]
        snrs_sorted = snrs_arr[order]

        vals_unique, unique_idx = np.unique(vals_sorted, return_index=True)
        snrs_unique = snrs_sorted[unique_idx]

        if len(vals_unique) < 2:
            raise ValueError(
                f"Cannot interpolate SNR for config '{cfg_name_}': "
                f"not enough unique {metric_name} points."
            )

        if target_val_ < vals_unique[0] or target_val_ > vals_unique[-1]:
            raise ValueError(
                f"Target {metric_name}={target_val_} is outside available range "
                f"[{vals_unique[0]}, {vals_unique[-1]}] for config '{cfg_name_}'."
            )

        warnings.warn(
            f"Config '{cfg_name_}': exact {metric_name}={target_val_} not found. "
            f"Using linear interpolation.",
            UserWarning
        )
        return float(np.interp(target_val_, vals_unique, snrs_unique))

    show_title, show_x_label, show_y_label, show_legend, show_x_ticks, show_y_ticks = show

    is_multi_config = isinstance(config_name, (list, tuple))
    if is_multi_config and isinstance(config_name, tuple):
        config_name = list(config_name)

    if target_snr is not None and target_er is not None:
        raise ValueError("Use only one of target_snr or target_er.")

    fig, ax = _make_axis(fig, axis)

    if is_multi_config:
        if len(config_name) == 0:
            raise ValueError("config_name list is empty.")

        if target_snr is None and target_er is None:
            raise ValueError(
                "For multiple configs, provide either target_snr or target_er."
            )

        x_vals = np.arange(len(config_name))
        y_vals = []

        for cfg in config_name:
            _, _, snrs_, vals_ = _load_result_bundle(cfg)

            if target_snr is not None:
                y_val = _interp_metric_at_snr(snrs_, vals_, target_snr, cfg)
            else:
                y_val = _interp_snr_at_metric(
                    snrs_, vals_, _metric_target_to_raw(target_er), cfg
                )

            y_vals.append(y_val)

        if target_snr is not None:
            y_vals = _metric_to_plot_values(y_vals)
        else:
            y_vals = np.asarray(y_vals, dtype=float)

        default_label = None
        if labels is None:
            if target_snr is not None:
                default_label = f"{plot_metric_label} @ SNR={target_snr}"
            else:
                default_label = f"SNR @ {plot_metric_label}={target_er}"

        label, l_style, color, marker, marker_size, linewidth, markevery = _get_style_for_series(
            0, label_idx=0, default_label=default_label
        )

        if marker == "":
            marker = "o"

        if not label.startswith('-'):
            if target_snr is not None:
                plot_fn = ax.semilogy if use_log_y else ax.plot
                plot_fn(x_vals, y_vals,
                        label=label,
                        color=color,
                        linewidth=linewidth,
                        marker=marker,
                        ms=marker_size,
                        markevery=markevery,
                        linestyle=l_style)
            else:
                ax.plot(x_vals, y_vals,
                        label=label,
                        color=color,
                        linewidth=linewidth,
                        marker=marker,
                        ms=marker_size,
                        markevery=markevery,
                        linestyle=l_style)

        ax.grid(True, which="both")

        if show_title:
            if target_snr is not None:
                title = f"{plot_metric_label} at SNR={target_snr}"
            else:
                title = f"SNR at {plot_metric_label}={target_er}"
            ax.set_title(title, fontsize=10)

        if show_x_label:
            ax.set_xlabel("Config index", fontsize=10)

        if show_y_label:
            if target_snr is not None:
                ax.set_ylabel(y_axis_metric_label, fontsize=10)
            else:
                ax.set_ylabel("SNR [dB]", fontsize=10)

        if show_x_ticks:
            ax.set_xticks(x_vals)
            ax.set_xticklabels([str(i + 1) for i in range(len(config_name))], fontsize=10)
        else:
            ax.set_xticks([])

        if show_y_ticks:
            ax.tick_params(axis='y', labelsize=10)
        else:
            ax.set_yticks([])

        if show_legend:
            ax.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center', ncols=2)

        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim([-0.5, len(config_name) - 0.5])

        if ylim is not None:
            ax.set_ylim(ylim)

        return fig

    sys_parameters = _load_parameters(config_name)

    filename = _get_filename(sys_parameters)

    if num_tx_eval is None:
        num_tx_eval = sys_parameters.max_num_tx

    sim_idx = _normalize_sim_idx(sim_idx)

    x_min = None
    x_max = None

    if exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict) and {"snr", "ber", "bler", "mse", "nmse"} <= set(data):
            snr_map = data["snr"]
            metric_map = data[metric_name]
            _print_available_experiments(config_name, filename, metric_map)
            candidates = _candidate_keys(metric_map, num_tx_eval)

            if sim_idx is None:
                sim_idx = np.arange(len(candidates))

            l_idx = 0
            for idx, key in enumerate(candidates):
                if idx not in sim_idx:
                    continue

                default_label = key[0]
                label, l_style, color, marker, marker_size, linewidth, markevery = _get_style_for_series(
                    idx, label_idx=l_idx, default_label=default_label
                )

                snrs_ = _to_numpy_1d(snr_map[key])
                vals_ = _to_numpy_1d(metric_map[key])
                snrs_, vals_ = remove_trailing_zeros(snrs_, vals_)
                if len(snrs_) == 0 or len(vals_) == 0:
                    _skip_empty_series(config_name, key)
                    continue
                vals_plot = _metric_to_plot_values(vals_, config_name, key)
                x_min = float(np.min(snrs_)) if x_min is None else min(x_min, float(np.min(snrs_)))
                x_max = float(np.max(snrs_)) if x_max is None else max(x_max, float(np.max(snrs_)))

                if not label.startswith('-'):
                    plot_fn = ax.semilogy if use_log_y else ax.plot
                    plot_fn(snrs_, vals_plot,
                            label=label,
                            color=color,
                            linewidth=linewidth,
                            marker=marker,
                            ms=marker_size,
                            markevery=markevery,
                            linestyle=l_style)

                l_idx += 1

        else:
            SNRs = None
            if len(data) == 3:
                snrs, BERs, BLERs = data
            elif len(data) == 7:
                snrs, BERs, BLERs, _, _, _, _ = data
            elif len(data) == 8:
                snrs, BERs, BLERs, _, _, _, _, SNRs = data
            else:
                raise ValueError(f"Unsupported results format in '{filename}'")

            if metric_name not in ("ber", "bler"):
                raise ValueError(
                    f"Legacy results format in '{filename}' only supports 'ber' or 'bler', "
                    f"not '{metric_name}'."
                )

            ERs = BERs if metric_name == "ber" else BLERs
            _print_available_experiments(config_name, filename, ERs)
            candidates = _candidate_keys(ERs, num_tx_eval)

            if sim_idx is None:
                sim_idx = np.arange(len(candidates))

            l_idx = 0
            for idx, key in enumerate(candidates):
                if idx not in sim_idx:
                    continue

                default_label = key[0]
                label, l_style, color, marker, marker_size, linewidth, markevery = _get_style_for_series(
                    idx, label_idx=l_idx, default_label=default_label
                )

                if SNRs is not None and key in SNRs:
                    snrs_ = SNRs[key]
                else:
                    snrs_ = snrs

                vals_ = ERs[key]
                snrs_, vals_ = remove_trailing_zeros(snrs_, vals_)
                if len(snrs_) == 0 or len(vals_) == 0:
                    _skip_empty_series(config_name, key)
                    continue
                vals_plot = _metric_to_plot_values(vals_, config_name, key)
                x_min = float(np.min(snrs_)) if x_min is None else min(x_min, float(np.min(snrs_)))
                x_max = float(np.max(snrs_)) if x_max is None else max(x_max, float(np.max(snrs_)))

                if not label.startswith('-'):
                    plot_fn = ax.semilogy if use_log_y else ax.plot
                    plot_fn(snrs_, vals_plot,
                            label=label,
                            color=color,
                            linewidth=linewidth,
                            marker=marker,
                            ms=marker_size,
                            markevery=markevery,
                            linestyle=l_style)

                l_idx += 1

        if use_log_y:
            y_min, y_max = ax.get_ylim()
            if y_min > 0:
                decades = np.logspace(
                    np.floor(np.log10(y_min)),
                    np.ceil(np.log10(y_max)),
                    num=int(np.ceil(np.log10(y_max) - np.floor(np.log10(y_min)) + 1))
                )
                ax.set_yticks(decades)
    else:
        print("No results found")

    text_size = 10
    if show_title:
        title = f"5G NR PUSCH {num_tx_eval}x{sys_parameters.num_rx_antennas} " \
                f"MU-MIMO, {sys_parameters.channel_type}-Channel, " \
                f"MCS={sys_parameters.mcs_index[mcs_arr_eval_idx]}, " \
                f"PRBs={sys_parameters.n_size_bwp}"
        ax.set_title(title, fontsize=text_size)

    if show_x_ticks:
        ax.tick_params(axis='x', labelsize=text_size)

    if show_y_ticks:
        ax.tick_params(axis='y', labelsize=text_size)

    ax.grid(True, which="both")

    if show_x_label:
        ax.set_xlabel(r"$\mathrm{E_b/N_0}$ [dB]", fontsize=text_size)
    if show_y_label:
        ax.set_ylabel(y_axis_metric_label, fontsize=text_size)

    if show_legend:
        ax.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center', ncols=2)

    if xlim is not None:
        ax.set_xlim(xlim)
    elif x_min is not None and x_max is not None:
        ax.set_xlim([x_min, x_max])

    if ylim is not None:
        ax.set_ylim(ylim)

    return fig
