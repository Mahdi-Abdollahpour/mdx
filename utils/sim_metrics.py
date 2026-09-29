# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026

"""Monte Carlo simulation of BER/BLER and channel-estimation MSE/NMSE.

Adapted from ``sim_ber`` in Sionna (https://github.com/NVlabs/sionna,
Apache-2.0 license), extended to accumulate MSE/NMSE alongside BER/BLER.
"""

import time
import numpy as np
import tensorflow as tf


def hard_decisions(llr):
    """Transforms LLRs into hard decisions."""
    zero = tf.constant(0, dtype=llr.dtype)
    return tf.cast(tf.math.greater(llr, zero), dtype=llr.dtype)


def count_errors(b, b_hat):
    """Count element-wise bit errors."""
    errors = tf.not_equal(b, b_hat)
    return tf.reduce_sum(tf.cast(errors, tf.int64))


def count_block_errors(b, b_hat):
    """Count blocks with at least one bit error along the last axis."""
    errors = tf.reduce_any(tf.not_equal(b, b_hat), axis=-1)
    return tf.reduce_sum(tf.cast(errors, tf.int64))


def _as_complex_channel(h):
    """Convert channel tensors to complex form for representation-independent MSE."""
    if h.dtype.is_complex:
        return h

    if not h.dtype.is_floating:
        raise TypeError("Channel tensors must be complex or floating-point.")

    last_dim = tf.shape(h)[-1]
    tf.debugging.assert_equal(
        tf.math.floormod(last_dim, 2),
        0,
        message="Packed real/imag channel tensors must have an even last dimension.",
    )

    if h.dtype not in (tf.float32, tf.float64):
        h = tf.cast(h, tf.float32)

    h_real, h_imag = tf.split(h, 2, axis=-1)
    return tf.complex(h_real, h_imag)


def sim_metrics(
    mc_fun,
    ebno_dbs,
    batch_size,
    max_mc_iter,
    mode="both",
    soft_estimates=False,
    num_target_bit_errors=None,
    num_target_block_errors=None,
    target_ber=None,
    target_bler=None,
    target_mse=None,
    num_target_samples=None,
    target_squared_error=None,
    early_stop=True,
    graph_mode=None,
    distribute=None,
    verbose=True,
    forward_keyboard_interrupt=True,
    callback=None,
    dtype=tf.complex64,
):
    """Run a Monte Carlo SNR sweep for BER/BLER and/or MSE/NMSE.

    ``mode`` selects the metrics: ``"ber"`` (BER, BLER), ``"mse"`` (MSE, NMSE)
    or ``"both"``. Per-SNR stopping targets are tracked separately for the two
    metric families, but the sweep moves to the next SNR point only once all
    selected families are done; in ``"both"`` mode all metrics of an SNR point
    are computed from the same batches.

    Parameters
    ----------
    mc_fun : callable
        ``mc_fun(batch_size, ebno_db, return_channel)`` returning ``(b, b_hat)``,
        or ``(b, b_hat, h, h_hat)`` when ``return_channel`` is True (MSE modes).
        ``b``/``b_hat`` have shape ``[batch_size, ..., n_bits]`` (the last axis
        is the block for BLER). ``h``/``h_hat`` may be complex, or real with
        real and imaginary parts concatenated along the last axis.
    ebno_dbs : tf.Tensor
        1-D tensor of SNR points [dB].
    batch_size : int
        Batch size per Monte Carlo iteration.
    max_mc_iter : int
        Maximum number of iterations per SNR point.
    mode : {"ber", "mse", "both"}
        Metrics to evaluate.
    soft_estimates : bool
        If True, ``b_hat`` holds LLRs and is converted to hard decisions.
    num_target_bit_errors, num_target_block_errors : int or None
        Per-SNR stopping targets for BER/BLER.
    num_target_samples, target_squared_error : int, float or None
        Per-SNR stopping targets for MSE/NMSE (number of channel coefficients
        or accumulated squared error).
    target_ber, target_bler, target_mse : float or None
        Stop the sweep once the metric falls below this value
        (requires ``early_stop``).
    early_stop : bool
        Enable early stopping of the whole sweep.
    graph_mode : {None, "graph", "xla"}
        Execution mode used to wrap ``mc_fun``.
    distribute : None, "all", list of GPU indices, or tf.distribute.Strategy
        Multi-GPU distribution.
    verbose : bool
        Print a progress table.
    forward_keyboard_interrupt : bool
        If False, a ``KeyboardInterrupt`` returns the partial results.
    callback : callable or None
        ``callback(mc_iter, snr_idx, ebno_dbs, state)`` called after each
        iteration; may return ``sim_metrics.CALLBACK_STOP`` or
        ``sim_metrics.CALLBACK_NEXT_SNR``.
    dtype : tf.DType
        Complex dtype of the simulation.

    Returns
    -------
    dict
        Selected metrics per SNR point, their accumulated counts, runtime
        statistics and a per-SNR status code.
    """

    valid_modes = {"ber", "mse", "both"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")

    compute_ber = mode in ("ber", "both")
    compute_mse = mode in ("mse", "both")
    need_channel = compute_mse

    assert isinstance(early_stop, bool), "early_stop must be bool."
    assert isinstance(verbose, bool), "verbose must be bool."
    assert isinstance(soft_estimates, bool), "soft_estimates must be bool."
    assert dtype.is_complex, "dtype must be a complex type."

    if target_ber is None:
        target_ber = -1.0
    if target_bler is None:
        target_bler = -1.0
    if target_mse is None:
        target_mse = -1.0

    if graph_mode is None:
        graph_mode = "default"
    assert isinstance(graph_mode, str), "graph_mode must be str."

    if graph_mode == "default":
        pass
    elif graph_mode == "graph":
        if not isinstance(mc_fun, tf.types.experimental.GenericFunction):
            mc_fun = tf.function(
                mc_fun,
                jit_compile=False,
                experimental_follow_type_hints=True,
            )
    elif graph_mode == "xla":
        if not isinstance(mc_fun, tf.types.experimental.GenericFunction) or \
           not mc_fun.function_spec.jit_compile:
            mc_fun = tf.function(
                mc_fun,
                jit_compile=True,
                experimental_follow_type_hints=True,
            )
    else:
        raise TypeError("Unknown graph_mode selected.")

    @tf.function(jit_compile=False)
    def _run_distributed(strategy, mc_fun, batch_size, ebno_db, return_channel):
        outputs_rep = strategy.run(
            mc_fun,
            kwargs={
                "batch_size": batch_size,
                "ebno_db": ebno_db,
                "return_channel": return_channel,
            },
        )

        b = strategy.gather(outputs_rep[0], axis=0)
        b_hat = strategy.gather(outputs_rep[1], axis=0)

        if return_channel:
            h = strategy.gather(outputs_rep[2], axis=0)
            h_hat = strategy.gather(outputs_rep[3], axis=0)
            return b, b_hat, h, h_hat

        return b, b_hat

    if len(tf.config.list_logical_devices("GPU")) == 0:
        run_multigpu = False
        distribute = None
    elif distribute is None:
        run_multigpu = False
    elif isinstance(distribute, tf.distribute.Strategy):
        run_multigpu = True
        strategy = distribute
    else:
        run_multigpu = True
        if distribute == "all":
            gpus = tf.config.list_logical_devices("GPU")
        elif isinstance(distribute, (tuple, list)):
            gpus_avail = tf.config.list_logical_devices("GPU")
            gpus = [gpus_avail[i] for i in distribute if i < len(gpus_avail)]
        else:
            raise ValueError("Unknown value for distribute.")

        if verbose:
            print("Setting tf.debugging.set_log_device_placement to False.")
        tf.debugging.set_log_device_placement(False)
        strategy = tf.distribute.MirroredStrategy(
            gpus,
            cross_device_ops=tf.distribute.ReductionToOneDevice(
                reduce_to_device=gpus[0].name
            ),
        )

    if run_multigpu:
        num_replicas = strategy.num_replicas_in_sync
        max_mc_iter = int(np.ceil(max_mc_iter / num_replicas))
        if verbose:
            print(f"Distributing simulation across {num_replicas} devices.")
            print(f"Reducing max_mc_iter to {max_mc_iter}")

    ebno_dbs = tf.cast(ebno_dbs, dtype.real_dtype)
    batch_size = tf.cast(batch_size, tf.int32)
    num_points = tf.shape(ebno_dbs)[0]

    bit_errors = tf.Variable(tf.zeros([num_points], dtype=tf.int64), dtype=tf.int64)
    block_errors = tf.Variable(tf.zeros([num_points], dtype=tf.int64), dtype=tf.int64)
    nb_bits = tf.Variable(tf.zeros([num_points], dtype=tf.int64), dtype=tf.int64)
    nb_blocks = tf.Variable(tf.zeros([num_points], dtype=tf.int64), dtype=tf.int64)

    mse_num = tf.Variable(tf.zeros([num_points], dtype=tf.float64), dtype=tf.float64)
    mse_den = tf.Variable(tf.zeros([num_points], dtype=tf.float64), dtype=tf.float64)
    nmse_den = tf.Variable(tf.zeros([num_points], dtype=tf.float64), dtype=tf.float64)

    status = np.zeros(num_points, dtype=np.int32)
    runtime = np.zeros(num_points, dtype=np.float64)
    runtime_samples = np.zeros(num_points, dtype=np.int64)

    ber_done = np.full(num_points, not compute_ber, dtype=bool)
    mse_done = np.full(num_points, not compute_mse, dtype=bool)
    joint_mode = compute_ber and compute_mse

    if num_target_bit_errors is not None:
        num_target_bit_errors = tf.cast(num_target_bit_errors, tf.int64)
    if num_target_block_errors is not None:
        num_target_block_errors = tf.cast(num_target_block_errors, tf.int64)
    if num_target_samples is not None:
        num_target_samples = tf.cast(num_target_samples, tf.float64)
    if target_squared_error is not None:
        target_squared_error = tf.cast(target_squared_error, tf.float64)

    column_defs = [("EbNo[dB]", 8)]
    if compute_ber:
        column_defs += [
            ("BER", 9),
            ("BLER", 9),
            ("bErr", 9),
            ("nBit", 9),
            ("blkE", 9),
            ("nBlk", 9),
        ]
    if compute_mse:
        column_defs += [
            ("MSE", 9),
            ("NMSE", 9),
            ("sqErr", 9),
            ("nSmp", 9),
            ("refPwr", 9),
        ]
    column_defs += [("rt[s]", 7), ("rt/smp[s]", 9), ("stat", 16)]

    def _format_fixed(value):
        return f"{float(np.nan_to_num(value)):.2f}"

    def _format_sci(value):
        return f"{float(np.nan_to_num(value)):.2e}"

    def _runtime_per_sample(idx_snr):
        idx_snr = int(idx_snr)
        samples = runtime_samples[idx_snr]
        if samples == 0:
            return 0.0
        return runtime[idx_snr] / samples

    def _avg_runtime_per_sample():
        avg_runtime = np.zeros_like(runtime, dtype=np.float64)
        np.divide(
            runtime,
            runtime_samples,
            out=avg_runtime,
            where=runtime_samples != 0,
        )
        return avg_runtime

    def _completion_targets(idx_snr):
        hits = []

        if compute_ber:
            if num_target_bit_errors is not None:
                target = int(num_target_bit_errors.numpy())
                if int(bit_errors[idx_snr].numpy()) >= target:
                    hits.append("bitE")
            if num_target_block_errors is not None:
                target = int(num_target_block_errors.numpy())
                if int(block_errors[idx_snr].numpy()) >= target:
                    hits.append("blkE")

        if compute_mse:
            if num_target_samples is not None:
                target = float(num_target_samples.numpy())
                if float(mse_den[idx_snr].numpy()) >= target:
                    hits.append("nSmp")
            if target_squared_error is not None:
                target = float(target_squared_error.numpy())
                if float(mse_num[idx_snr].numpy()) >= target:
                    hits.append("sqErr")

        return hits

    def _early_stop_targets(idx_snr):
        hits = []

        if compute_ber:
            bit_err = float(bit_errors[idx_snr].numpy())
            blk_err = float(block_errors[idx_snr].numpy())
            n_bit = float(nb_bits[idx_snr].numpy())
            n_blk = float(nb_blocks[idx_snr].numpy())
            ber_np = 0.0 if n_bit == 0.0 else bit_err / n_bit
            bler_np = 0.0 if n_blk == 0.0 else blk_err / n_blk

            if blk_err == 0.0:
                hits.append("noErr")
            if target_ber >= 0.0 and ber_np < target_ber:
                hits.append("BER")
            if target_bler >= 0.0 and bler_np < target_bler:
                hits.append("BLER")

        if compute_mse:
            mse_den_np = float(mse_den[idx_snr].numpy())
            mse_np = 0.0 if mse_den_np == 0.0 else float(mse_num[idx_snr].numpy()) / mse_den_np
            if target_mse >= 0.0 and mse_np < target_mse:
                hits.append("MSE")

        return hits

    def _status_text(idx_snr, idx_it=None):
        code = int(status[idx_snr])
        if code == 0 and idx_it is not None:
            return f"it:{int(idx_it) + 1}/{int(max_mc_iter)}"
        if code == 1:
            return "max_iter"
        if code == 2:
            hits = _completion_targets(idx_snr)
            return f"tgt:{'+'.join(hits)}" if hits else "target"
        if code == 3:
            hits = _early_stop_targets(idx_snr)
            return f"tgt:{'+'.join(hits)}" if hits else "target"
        if code == 4:
            return "callback"
        return "unknown"

    def _format_metric_values(idx_snr, idx_it=None):
        vals = [_format_fixed(ebno_dbs[idx_snr].numpy())]

        if compute_ber:
            ber_np = tf.math.divide_no_nan(
                tf.cast(bit_errors[idx_snr], tf.float64),
                tf.cast(nb_bits[idx_snr], tf.float64),
            ).numpy()
            bler_np = tf.math.divide_no_nan(
                tf.cast(block_errors[idx_snr], tf.float64),
                tf.cast(nb_blocks[idx_snr], tf.float64),
            ).numpy()
            vals += [
                _format_sci(ber_np),
                _format_sci(bler_np),
                _format_sci(bit_errors[idx_snr].numpy()),
                _format_sci(nb_bits[idx_snr].numpy()),
                _format_sci(block_errors[idx_snr].numpy()),
                _format_sci(nb_blocks[idx_snr].numpy()),
            ]

        if compute_mse:
            mse_np = tf.math.divide_no_nan(
                tf.cast(mse_num[idx_snr], tf.float64),
                tf.cast(mse_den[idx_snr], tf.float64),
            ).numpy()
            nmse_np = tf.math.divide_no_nan(
                tf.cast(mse_num[idx_snr], tf.float64),
                tf.cast(nmse_den[idx_snr], tf.float64),
            ).numpy()
            vals += [
                _format_sci(mse_np),
                _format_sci(nmse_np),
                _format_sci(mse_num[idx_snr].numpy()),
                _format_sci(mse_den[idx_snr].numpy()),
                _format_sci(nmse_den[idx_snr].numpy()),
            ]

        vals += [
            _format_fixed(runtime[idx_snr]),
            _format_sci(_runtime_per_sample(idx_snr)),
            _status_text(idx_snr, idx_it),
        ]
        return vals

    def _render_row(values):
        return " | ".join(
            f"{value:>{width}}" for value, (_, width) in zip(values, column_defs)
        )

    def _print_header():
        header_line = _render_row([name for name, _ in column_defs])
        print(header_line)
        print("-" * len(header_line))

    def _print_progress(idx_snr, idx_it=None, is_final=False):
        row = _render_row(_format_metric_values(idx_snr, idx_it))
        print(row, end="\n" if is_final else "\r", flush=not is_final)

    def _ber_finished(i):
        if not compute_ber:
            return True
        if num_target_bit_errors is not None and bit_errors[i] >= num_target_bit_errors:
            return True
        if num_target_block_errors is not None and block_errors[i] >= num_target_block_errors:
            return True
        return False

    def _mse_finished(i):
        if not compute_mse:
            return True
        if num_target_samples is not None and mse_den[i] >= num_target_samples:
            return True
        if target_squared_error is not None and mse_num[i] >= target_squared_error:
            return True
        return False

    if verbose:
        _print_header()

    try:
        for i in tf.range(num_points):
            snr_start = time.perf_counter()
            cb_state = sim_metrics.CALLBACK_CONTINUE

            for ii in tf.range(max_mc_iter):
                if run_multigpu:
                    outputs = _run_distributed(
                        strategy,
                        mc_fun,
                        batch_size,
                        ebno_dbs[i],
                        need_channel,
                    )
                else:
                    outputs = mc_fun(
                        batch_size=batch_size,
                        ebno_db=ebno_dbs[i],
                        return_channel=need_channel,
                    )

                if need_channel:
                    b, b_hat, h, h_hat = outputs[:4]
                else:
                    b, b_hat = outputs[:2]
                    h = None
                    h_hat = None

                runtime_samples[i] += int(tf.shape(b)[0].numpy())

                track_ber = compute_ber and (joint_mode or not ber_done[i])
                if track_ber:
                    b_hat_eff = hard_decisions(b_hat) if soft_estimates else b_hat
                    bit_e = count_errors(b, b_hat_eff)
                    block_e = count_block_errors(b, b_hat_eff)
                    bit_n = tf.size(b)
                    block_n = tf.size(b[..., -1])

                    bit_errors.assign(tf.tensor_scatter_nd_add(
                        bit_errors, [[i]], tf.cast([bit_e], tf.int64)
                    ))
                    block_errors.assign(tf.tensor_scatter_nd_add(
                        block_errors, [[i]], tf.cast([block_e], tf.int64)
                    ))
                    nb_bits.assign(tf.tensor_scatter_nd_add(
                        nb_bits, [[i]], tf.cast([bit_n], tf.int64)
                    ))
                    nb_blocks.assign(tf.tensor_scatter_nd_add(
                        nb_blocks, [[i]], tf.cast([block_n], tf.int64)
                    ))

                    if _ber_finished(i):
                        ber_done[i] = True

                track_mse = compute_mse and (joint_mode or not mse_done[i])
                if track_mse:
                    h_eff = _as_complex_channel(h)
                    h_hat_eff = _as_complex_channel(h_hat)
                    tf.debugging.assert_equal(
                        tf.shape(h_eff),
                        tf.shape(h_hat_eff),
                        message="h and h_hat must match after channel conversion.",
                    )
                    err = h_eff - h_hat_eff
                    se = tf.reduce_sum(tf.square(tf.abs(err)))
                    ref_power = tf.reduce_sum(tf.square(tf.abs(h_eff)))
                    n = tf.cast(tf.size(h_eff), tf.float64)

                    mse_num.assign(tf.tensor_scatter_nd_add(
                        mse_num, [[i]], [tf.cast(se, tf.float64)]
                    ))
                    mse_den.assign(tf.tensor_scatter_nd_add(
                        mse_den, [[i]], [n]
                    ))
                    nmse_den.assign(tf.tensor_scatter_nd_add(
                        nmse_den, [[i]], [tf.cast(ref_power, tf.float64)]
                    ))

                    if _mse_finished(i):
                        mse_done[i] = True

                runtime[i] = time.perf_counter() - snr_start

                state = {
                    "bit_errors": bit_errors,
                    "block_errors": block_errors,
                    "nb_bits": nb_bits,
                    "nb_blocks": nb_blocks,
                    "mse_num": mse_num,
                    "mse_den": mse_den,
                    "nmse_den": nmse_den,
                    "runtime_samples": runtime_samples.copy(),
                    "avg_runtime_per_sample": _avg_runtime_per_sample(),
                    "ber_done": ber_done.copy(),
                    "mse_done": mse_done.copy(),
                    "mode": mode,
                }

                if callback is not None:
                    cb_state = callback(ii, i, ebno_dbs, state)
                    if cb_state in (
                        sim_metrics.CALLBACK_STOP,
                        sim_metrics.CALLBACK_NEXT_SNR,
                    ):
                        status[i] = 4
                        break

                metrics_done = bool(ber_done[i] and mse_done[i])
                if metrics_done:
                    status[i] = 2

                if verbose:
                    _print_progress(i, idx_it=ii, is_final=False)

                if metrics_done:
                    break

                if ii == max_mc_iter - 1:
                    status[i] = 1

            stop_message = None
            if early_stop:
                stop_whole_sim = False

                if compute_ber:
                    ber_true = tf.math.divide_no_nan(
                        tf.cast(bit_errors[i], tf.float64),
                        tf.cast(nb_bits[i], tf.float64),
                    )
                    bler_true = tf.math.divide_no_nan(
                        tf.cast(block_errors[i], tf.float64),
                        tf.cast(nb_blocks[i], tf.float64),
                    )
                    ber_early = (
                        block_errors[i] == 0 or
                        (target_ber >= 0.0 and ber_true < target_ber) or
                        (target_bler >= 0.0 and bler_true < target_bler)
                    )
                else:
                    ber_early = False

                if compute_mse:
                    mse_true = tf.math.divide_no_nan(mse_num[i], mse_den[i])
                    mse_early = (target_mse >= 0.0 and mse_true < target_mse)
                else:
                    mse_early = False

                selected_early_conditions = []
                if compute_ber:
                    selected_early_conditions.append(bool(ber_early))
                if compute_mse:
                    selected_early_conditions.append(bool(mse_early))

                if selected_early_conditions and all(selected_early_conditions):
                    stop_whole_sim = True

                if stop_whole_sim:
                    status[i] = 3
                    stop_message = (
                        f"\nSimulation stopped by early-stop condition "
                        f"@ EbNo = {ebno_dbs[i].numpy():.1f} dB.\n"
                    )

            if cb_state == sim_metrics.CALLBACK_STOP:
                status[i] = 4
                stop_message = (
                    f"\nSimulation stopped by callback "
                    f"@ EbNo = {ebno_dbs[i].numpy():.1f} dB.\n"
                )

            if verbose:
                _print_progress(i, is_final=True)

            if stop_message is not None:
                if verbose:
                    print(stop_message)
                break

    except KeyboardInterrupt as e:
        if forward_keyboard_interrupt:
            raise e
        print(f"\nSimulation stopped by the user @ EbNo = {ebno_dbs[i].numpy()} dB.")

    results = {"mode": mode, "ebno_dbs": tf.cast(ebno_dbs, tf.float64)}

    if compute_ber:
        ber = tf.math.divide_no_nan(
            tf.cast(bit_errors, tf.float64),
            tf.cast(nb_bits, tf.float64),
        )
        bler = tf.math.divide_no_nan(
            tf.cast(block_errors, tf.float64),
            tf.cast(nb_blocks, tf.float64),
        )
        ber = tf.where(tf.math.is_nan(ber), tf.zeros_like(ber), ber)
        bler = tf.where(tf.math.is_nan(bler), tf.zeros_like(bler), bler)

        results["ber"] = ber
        results["bler"] = bler
        results["bit_errors"] = tf.cast(bit_errors, tf.float64)
        results["block_errors"] = tf.cast(block_errors, tf.float64)
        results["nb_bits"] = tf.cast(nb_bits, tf.float64)
        results["nb_blocks"] = tf.cast(nb_blocks, tf.float64)

    if compute_mse:
        mse = tf.math.divide_no_nan(mse_num, mse_den)
        nmse = tf.math.divide_no_nan(mse_num, nmse_den)

        results["mse"] = mse
        results["nmse"] = nmse
        results["mse_num"] = mse_num
        results["mse_den"] = mse_den
        results["nmse_den"] = nmse_den

    results["status"] = status
    results["runtime"] = runtime
    results["avg_runtime_per_sample"] = _avg_runtime_per_sample()

    return results


sim_metrics.CALLBACK_CONTINUE = None
sim_metrics.CALLBACK_STOP = 2
sim_metrics.CALLBACK_NEXT_SNR = 1
