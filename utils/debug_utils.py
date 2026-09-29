# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026

"""Debug-printing and tensor-inspection helpers."""

import inspect
import os

import tensorflow as tf

from core.runtime import DEBUGD
from .math_utils import masked_quantile


def tattle(x, verbose=0, names=None, expected_shapes=None, n_samples=8, max_width=120,
           mask=None, masked_stats=False, masked_only=False):
    """Print a debugging report for one or more tensors.

    For each tensor, prints the call site, dtype, static shape and NaN/Inf
    counts, followed by statistics computed on finite values only. If exactly
    two tensors with equal static shape and dtype are given, their difference
    is reported as well. Disabled when ``DEBUGD["tattle"]`` is 0.

    Args:
        x: Tensor or list/tuple of tensors.
        verbose: 0 = header only, 1 = min/max/mean, 2 = + q10/q50/q90,
            3 = + first/last ``n_samples`` values, 4 = split real/imaginary
            parts instead of magnitudes, 5 = + dense quantiles.
        names: Label or list of labels (used when a tensor has no name).
        expected_shapes: Optional expected-shape strings, one per tensor.
        n_samples: Number of samples printed at ``verbose >= 3``.
        max_width: Unused; kept for API compatibility.
        mask: Mask, or list of masks (one per tensor), for masked statistics.
            If None, ``t != 0`` is used.
        masked_stats: Print masked statistics in addition to unmasked ones.
        masked_only: Print only masked statistics.
    """
    do_run = True
    try:
        do_run = bool(DEBUGD.get("tattle", 1) > 0)
    except Exception:
        do_run = True
    if not do_run:
        return 0

    if isinstance(names, str):
        _names_seq = [names]
    elif isinstance(names, (list, tuple)):
        _names_seq = list(names)
    elif names is None:
        _names_seq = None
    else:
        _names_seq = [str(names)]

    def _ri_views(t):
        if t.dtype.is_complex:
            r = tf.math.real(t); i = tf.math.imag(t)
        else:
            r = tf.cast(t, tf.float32); i = tf.zeros_like(r, dtype=tf.float32)
        return tf.cast(r, tf.float32), tf.cast(i, tf.float32)

    def _abs_view(t):
        return tf.abs(t) if t.dtype.is_complex else tf.cast(t, tf.float32)

    def _flatten_finite(xf):
        xf = tf.cast(xf, tf.float32)
        flat = tf.reshape(xf, [-1])
        maskf = tf.math.is_finite(flat)
        return tf.boolean_mask(flat, maskf)

    def _nonfinite_counts(t):
        r, im = _ri_views(t)
        isnan = tf.logical_or(tf.math.is_nan(r), tf.math.is_nan(im))
        isinf = tf.logical_or(tf.math.is_inf(r), tf.math.is_inf(im))
        n_nan = tf.reduce_sum(tf.cast(isnan, tf.int64))
        n_inf = tf.reduce_sum(tf.cast(isinf, tf.int64))
        total = tf.size(r, out_type=tf.int64)
        n_fin = total - n_nan - n_inf
        return n_nan, n_inf, n_fin, total

    @tf.function(jit_compile=False)
    def _percentiles(vals, qs):
        vals = _flatten_finite(vals)
        n = tf.size(vals)
        n_eff = tf.maximum(n, 1)
        vals_pad = tf.concat([vals, tf.zeros([1], vals.dtype)], axis=0)
        vals_eff = vals_pad[:n_eff]
        sorted_vals = tf.sort(vals_eff, axis=0)
        n_f = tf.cast(n_eff - 1, tf.float32)
        pos = tf.clip_by_value(qs * n_f, 0.0, n_f)
        lo = tf.cast(tf.math.floor(pos), tf.int32)
        hi = tf.cast(tf.math.ceil(pos),  tf.int32)
        w  = pos - tf.cast(lo, tf.float32)
        v_lo = tf.gather(sorted_vals, lo)
        v_hi = tf.gather(sorted_vals, hi)
        out = (1.0 - w) * v_lo + w * v_hi
        nan_vec = tf.fill([tf.size(qs)], tf.constant(float('nan'), out.dtype))
        return tf.where(tf.cast(tf.equal(n, 0), tf.bool), nan_vec, out)

    @tf.function(jit_compile=False)
    def _min_max_mean(vals):
        vals = _flatten_finite(vals)
        n = tf.size(vals)
        n_eff = tf.maximum(n, 1)
        vals_pad = tf.concat([vals, tf.zeros([1], vals.dtype)], axis=0)
        vals_eff = vals_pad[:n_eff]
        vmin = tf.reduce_min(vals_eff)
        vmax = tf.reduce_max(vals_eff)
        vmean = tf.reduce_mean(vals_eff)
        nan = tf.constant(float('nan'), vals.dtype)
        vmin = tf.where(n > 0, vmin, nan)
        vmax = tf.where(n > 0, vmax, nan)
        vmean = tf.where(n > 0, vmean, nan)
        return vmin, vmax, vmean

    @tf.function(jit_compile=False)
    def _mse_rmse(vals):
        vals = _flatten_finite(vals)
        n = tf.size(vals)
        n_eff = tf.maximum(n, 1)
        vals_pad = tf.concat([vals, tf.zeros([1], vals.dtype)], axis=0)
        vals_eff = vals_pad[:n_eff]
        sq = tf.square(vals_eff)
        mse = tf.reduce_mean(sq)
        nan = tf.constant(float('nan'), vals.dtype)
        mse = tf.where(n > 0, mse, nan)
        rmse = tf.sqrt(mse)
        return mse, rmse

    @tf.function(jit_compile=False)
    def _first_samples(vals, k):
        flat = _flatten_finite(vals)
        n = tf.size(flat)
        k = tf.minimum(tf.cast(k, tf.int32), tf.cast(n, tf.int32))
        return tf.gather(flat, tf.range(k))

    @tf.function(jit_compile=False)
    def _last_samples(vals, k):
        flat = _flatten_finite(vals)
        n = tf.size(flat)
        k = tf.minimum(tf.cast(k, tf.int32), tf.cast(n, tf.int32))
        start = tf.maximum(n - k, 0)
        return tf.gather(flat, tf.range(start, n))

    @tf.function(jit_compile=False)
    def _masked_percentiles(vals, qs, m):
        def _one(q):
            return masked_quantile(vals, m, q, axis=None, keepdims=False, return_counts=False)
        return tf.map_fn(_one, qs, fn_output_signature=tf.cast(vals, tf.float32).dtype)

    @tf.function(jit_compile=False)
    def _masked_min_max_mean(vals, m):
        v = tf.cast(vals, tf.float32)
        flat = tf.reshape(v, [-1])
        finite = tf.math.is_finite(flat)
        if m is None:
            sel = finite
        else:
            mf = tf.reshape(tf.cast(m, tf.bool), [-1])
            sel = tf.logical_and(mf, finite)
        sel_vals = tf.boolean_mask(flat, sel)
        n = tf.size(sel_vals)
        n_eff = tf.maximum(n, 1)
        pad = tf.concat([sel_vals, tf.zeros([1], sel_vals.dtype)], axis=0)[:n_eff]
        vmin = tf.reduce_min(pad)
        vmax = tf.reduce_max(pad)
        vmean = tf.reduce_mean(pad)
        nan = tf.constant(float('nan'), sel_vals.dtype)
        vmin  = tf.where(n > 0, vmin,  nan)
        vmax  = tf.where(n > 0, vmax,  nan)
        vmean = tf.where(n > 0, vmean, nan)
        return vmin, vmax, vmean

    @tf.function(jit_compile=False)
    def _first_samples_masked(vals, m, k):
        flat = tf.reshape(tf.cast(vals, tf.float32), [-1])
        mf   = tf.reshape(tf.cast(m, tf.bool), [-1])
        finite = tf.math.is_finite(flat)
        sel = tf.boolean_mask(flat, tf.logical_and(mf, finite))
        n = tf.size(sel)
        k = tf.minimum(tf.cast(k, tf.int32), tf.cast(n, tf.int32))
        return tf.gather(sel, tf.range(k))

    @tf.function(jit_compile=False)
    def _last_samples_masked(vals, m, k):
        flat = tf.reshape(tf.cast(vals, tf.float32), [-1])
        mf   = tf.reshape(tf.cast(m, tf.bool), [-1])
        finite = tf.math.is_finite(flat)
        sel = tf.boolean_mask(flat, tf.logical_and(mf, finite))
        n = tf.size(sel)
        k = tf.minimum(tf.cast(k, tf.int32), tf.cast(n, tf.int32))
        start = tf.maximum(n - k, 0)
        return tf.gather(sel, tf.range(start, n))

    def _tensor_label(t, idx):
        name = getattr(t, "name", None)
        if name:
            return name.split(":")[0]
        if _names_seq is not None and idx < len(_names_seq):
            return str(_names_seq[idx])
        return f"tensor_{idx}"

    def _expected_for(i):
        if expected_shapes is None:
            return None
        if i == "diff":
            if len(expected_shapes) >= 3 and expected_shapes[2]:
                return expected_shapes[2]
            return None
        if isinstance(i, int) and 0 <= i < len(expected_shapes):
            return expected_shapes[i]
        return None

    def _mask_for(i, t):
        if mask is None:
            return None
        if isinstance(mask, (tuple, list)):
            return mask[i] if i < len(mask) else None
        return mask

    def _mask_hint_and_auto(i, t, m):
        total = tf.size(t, out_type=tf.int64)
        if m is None:
            auto = tf.not_equal(tf.cast(t, t.dtype), tf.cast(0, t.dtype))
            nnz  = tf.reduce_sum(tf.cast(auto, tf.int64))
            tf.print("  [mask] auto (t!=0): nnz/total =", nnz, "/", total)
            return tf.cast(auto, tf.bool)
        else:
            m_bool = tf.cast(m, tf.bool)
            truec  = tf.reduce_sum(tf.cast(m_bool, tf.int64))
            tf.print("  [mask] provided: true/total =", truec, "/", total)
            return m_bool

    try:
        frame = inspect.stack()[1]
        filename = os.path.basename(frame.filename); lineno = frame.lineno
        funcname = frame.function
        clsname  = frame.frame.f_locals['self'].__class__.__name__ if 'self' in frame.frame.f_locals else None
        caller = f"{clsname}.{funcname}" if clsname else funcname
        print("__________________________________________")
        print(f"[tattle] callsite: {filename}:{lineno} in {caller}")
        print("------------------------------------------")
    except Exception as e:
        print("__________________________________________")
        print(f"[tattle] callsite: <unavailable> ({e})")
        print("------------------------------------------")

    tensors = x if isinstance(x, (tuple, list)) else (x,)
    tensors = tuple(tensors)

    split_ri    = (verbose >= 4)
    want_stats  = (verbose >= 1)
    want_q3     = (verbose >= 2)
    want_smpl   = (verbose >= 3)
    want_qdense = (verbose >= 5)

    q_basic = tf.constant([0.1, 0.5, 0.9], tf.float32)
    q_dense = tf.constant([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.98], tf.float32)

    def _print_header(i, t, label):
        static_shape = tuple(t.shape.as_list())
        tf.print("\n[tattle]", i, "name:", label)
        tf.print("  dtype =", str(t.dtype))
        exp = _expected_for(i)
        if exp:
            tf.print("  expected =", exp)
        tf.print("  shape(static) =", str(static_shape))

        # Only print the NaN/Inf lines if any are present (branchless, no tf.cond)
        nan_c, inf_c, fin_c, tot_c = _nonfinite_counts(t)
        has_nf = tf.greater(nan_c + inf_c, tf.constant(0, tf.int64))
        line1 = tf.strings.format(
            "  [nonfinite] NaNs= {}  Infs= {}  finite/total= {} / {}\n",
            (nan_c, inf_c, fin_c, tot_c)
        )
        line2 = tf.constant("  [hint] Statistics below are computed on FINITE values only (NaN/Inf discarded).\n")
        extra = tf.where(has_nf, tf.strings.join([line1, line2]),
                         tf.constant("", dtype=tf.string))
        tf.print(extra, end="")

    for i, t in enumerate(tensors):
        label = _tensor_label(t, i)
        _print_header(i, t, label)

        if not (want_stats or want_q3 or want_smpl or want_qdense):
            continue

        m_user =_mask_for(i, t) if (masked_stats or masked_only) else None
        m_use  = None
        if (masked_stats or masked_only):
            m_use = _mask_hint_and_auto(i, t, m_user)

        do_unmasked = not masked_only
        do_masked   = (masked_stats or masked_only)

        if split_ri:
            r, im = _ri_views(t)

            if do_unmasked:
                rmin, rmax, rmean = _min_max_mean(r)
                imin, imax, imean = _min_max_mean(im)
                tf.print("  Re: min=", rmin, " max=", rmax, " mean=", rmean)
                tf.print("  Im: min=", imin, " max=", imax, " mean=", imean)
                if want_q3:
                    rq = _percentiles(r, q_basic); iq = _percentiles(im, q_basic)
                    tf.print("  Re q10,q50,q90 =", rq)
                    tf.print("  Im q10,q50,q90 =", iq)
                if want_qdense:
                    rq = _percentiles(r, q_dense); iq = _percentiles(im, q_dense)
                    tf.print("  Re quantiles", q_dense, "=", rq)
                    tf.print("  Im quantiles", q_dense, "=", iq)
                if want_smpl:
                    rs_f = _first_samples(r, tf.cast(n_samples, tf.int32))
                    is_f = _first_samples(im, tf.cast(n_samples, tf.int32))
                    rs_l = _last_samples(r,  tf.cast(n_samples, tf.int32))
                    is_l = _last_samples(im, tf.cast(n_samples, tf.int32))
                    tf.print("  samples Re (first", n_samples, "):", rs_f)
                    tf.print("  samples Im (first", n_samples, "):", is_f)
                    tf.print("  samples Re (last",  n_samples, "):", rs_l)
                    tf.print("  samples Im (last",  n_samples, "):", is_l)

            if do_masked:
                rmin, rmax, rmean = _masked_min_max_mean(r, m_use)
                imin, imax, imean = _masked_min_max_mean(im, m_use)
                tf.print("  Re(masked): min=", rmin, " max=", rmax, " mean=", rmean)
                tf.print("  Im(masked): min=", imin, " max=", imax, " mean=", imean)
                if want_q3:
                    rq = _masked_percentiles(r, q_basic, m_use); iq = _masked_percentiles(im, q_basic, m_use)
                    tf.print("  Re(masked) q10,q50,q90 =", rq)
                    tf.print("  Im(masked) q10,q50,q90 =", iq)
                if want_qdense:
                    rq = _masked_percentiles(r, q_dense, m_use); iq = _masked_percentiles(im, q_dense, m_use)
                    tf.print("  Re(masked) quantiles", q_dense, "=", rq)
                    tf.print("  Im(masked) quantiles", q_dense, "=", iq)
                if want_smpl:
                    rs_f = _first_samples_masked(r, m_use, tf.cast(n_samples, tf.int32))
                    is_f = _first_samples_masked(im, m_use, tf.cast(n_samples, tf.int32))
                    rs_l = _last_samples_masked(r,  m_use, tf.cast(n_samples, tf.int32))
                    is_l = _last_samples_masked(im, m_use, tf.cast(n_samples, tf.int32))
                    tf.print("  samples Re(masked) (first", n_samples, "):", rs_f)
                    tf.print("  samples Im(masked) (first", n_samples, "):", is_f)
                    tf.print("  samples Re(masked) (last",  n_samples, "):", rs_l)
                    tf.print("  samples Im(masked) (last",  n_samples, "):", is_l)
        else:
            a = _abs_view(t)
            label_mag = "|·|" if t.dtype.is_complex else "value"

            if do_unmasked:
                amin, amax, amean = _min_max_mean(a)
                tf.print(" ", label_mag, ": min=", amin, " max=", amax, " mean=", amean)
                if want_q3:
                    aq = _percentiles(a, q_basic)
                    tf.print(" ", label_mag, " q10,q50,q90 =", aq)
                if want_smpl:
                    sa_f = _first_samples(a, tf.cast(n_samples, tf.int32))
                    sa_l = _last_samples(a,  tf.cast(n_samples, tf.int32))
                    tf.print("  samples", label_mag, "(first", n_samples, "):", sa_f)
                    tf.print("  samples", label_mag, "(last",  n_samples, "):", sa_l)

            if do_masked:
                amin, amax, amean = _masked_min_max_mean(a, m_use)
                tf.print(" ", label_mag, "(masked): min=", amin, " max=", amax, " mean=", amean)
                if want_q3:
                    aq = _masked_percentiles(a, q_basic, m_use)
                    tf.print(" ", label_mag, "(masked) q10,q50,q90 =", aq)
                if want_smpl:
                    sa_f = _first_samples_masked(a, m_use, tf.cast(n_samples, tf.int32))
                    sa_l = _last_samples_masked(a,  m_use, tf.cast(n_samples, tf.int32))
                    tf.print("  samples", label_mag, "(masked) (first", n_samples, "):", sa_f)
                    tf.print("  samples", label_mag, "(masked) (last",  n_samples, "):", sa_l)

    if len(tensors) == 2:
        t0, t1 = tensors
        same_dtype = (t0.dtype == t1.dtype)
        s0 = t0.shape.as_list(); s1 = t1.shape.as_list()
        static_known = (None not in s0) and (None not in s1)
        same_shape_static = static_known and (tuple(s0) == tuple(s1))
        if same_dtype and same_shape_static:
            diff = t0 - t1
            if _names_seq and len(_names_seq) >= 3 and _names_seq[2]:
                label = str(_names_seq[2])
            else:
                label = "diff"
            _print_header("diff", diff, label)
            diff_mse, diff_rmse = _mse_rmse(tf.abs(diff))
            tf.print(" diff stats: mse=", diff_mse, " rmse=", diff_rmse)

            m0 = _mask_for(0, t0) if (masked_stats or masked_only) else None
            m1 = _mask_for(1, t1) if (masked_stats or masked_only) else None
            m_diff = None
            if (masked_stats or masked_only):
                if (m0 is not None) and (m1 is not None):
                    m_diff = tf.logical_and(tf.cast(m0, tf.bool), tf.cast(m1, tf.bool))
                else:
                    m_diff = m0 if m0 is not None else m1
                m_diff = _mask_hint_and_auto("diff", diff, m_diff)

            split_ri_local = (verbose >= 4)
            want_stats_local  = (verbose >= 1)
            want_q3_local     = (verbose >= 2)
            want_smpl_local   = (verbose >= 3)
            want_qdense_local = (verbose >= 5)

            do_unmasked = not masked_only
            do_masked   = (masked_stats or masked_only)

            if split_ri_local:
                r, im = _ri_views(diff)
                if do_unmasked and want_stats_local:
                    rmin, rmax, rmean = _min_max_mean(r)
                    imin, imax, imean = _min_max_mean(im)
                    tf.print("  Re: min=", rmin, " max=", rmax, " mean=", rmean)
                    tf.print("  Im: min=", imin, " max=", imax, " mean=", imean)
                    if want_q3_local:
                        rq = _percentiles(r, q_basic); iq = _percentiles(im, q_basic)
                        tf.print("  Re q10,q50,q90 =", rq)
                        tf.print("  Im q10,q50,q90 =", iq)
                    if want_qdense_local:
                        rq = _percentiles(r, q_dense); iq = _percentiles(im, q_dense)
                        tf.print("  Re quantiles", q_dense, "=", rq)
                        tf.print("  Im quantiles", q_dense, "=", iq)
                    if want_smpl_local:
                        rs_f = _first_samples(r, tf.cast(n_samples, tf.int32))
                        is_f = _first_samples(im, tf.cast(n_samples, tf.int32))
                        rs_l = _last_samples(r,  tf.cast(n_samples, tf.int32))
                        is_l = _last_samples(im, tf.cast(n_samples, tf.int32))
                        tf.print("  samples Re (first", n_samples, "):", rs_f)
                        tf.print("  samples Im (first", n_samples, "):", is_f)
                        tf.print("  samples Re (last",  n_samples, "):", rs_l)
                        tf.print("  samples Im (last",  n_samples, "):", is_l)
                if do_masked and want_stats_local:
                    rmin, rmax, rmean = _masked_min_max_mean(r, m_diff)
                    imin, imax, imean = _masked_min_max_mean(im, m_diff)
                    tf.print("  Re(masked): min=", rmin, " max=", rmax, " mean=", rmean)
                    tf.print("  Im(masked): min=", imin, " max=", imax, " mean=", imean)
                    if want_q3_local:
                        rq = _masked_percentiles(r, q_basic, m_diff); iq = _masked_percentiles(im, q_basic, m_diff)
                        tf.print("  Re(masked) q10,q50,q90 =", rq)
                        tf.print("  Im(masked) q10,q50,q90 =", iq)
                    if want_qdense_local:
                        rq = _masked_percentiles(r, q_dense, m_diff); iq = _masked_percentiles(im, q_dense, m_diff)
                        tf.print("  Re(masked) quantiles", q_dense, "=", rq)
                        tf.print("  Im(masked) quantiles", q_dense, "=", iq)
                    if want_smpl_local:
                        rs_f = _first_samples_masked(r, m_diff, tf.cast(n_samples, tf.int32))
                        is_f = _first_samples_masked(im, m_diff, tf.cast(n_samples, tf.int32))
                        rs_l = _last_samples_masked(r,  m_diff, tf.cast(n_samples, tf.int32))
                        is_l = _last_samples_masked(im, m_diff, tf.cast(n_samples, tf.int32))
                        tf.print("  samples Re(masked) (first", n_samples, "):", rs_f)
                        tf.print("  samples Im(masked) (first", n_samples, "):", is_f)
                        tf.print("  samples Re(masked) (last",  n_samples, "):", rs_l)
                        tf.print("  samples Im(masked) (last",  n_samples, "):", is_l)
            else:
                a = _abs_view(diff)
                label_mag = "|·|" if diff.dtype.is_complex else "value"
                if do_unmasked and want_stats_local:
                    amin, amax, amean = _min_max_mean(a)
                    tf.print(" ", label_mag, ": min=", amin, " max=", amax, " mean=", amean)
                    if want_q3_local:
                        aq = _percentiles(a, q_basic)
                        tf.print(" ", label_mag, " q10,q50,q90 =", aq)
                    if want_smpl_local:
                        sa_f = _first_samples(a, tf.cast(n_samples, tf.int32))
                        sa_l = _last_samples(a,  tf.cast(n_samples, tf.int32))
                        tf.print("  samples", label_mag, "(first", n_samples, "):", sa_f)
                        tf.print("  samples", label_mag, "(last",  n_samples, "):", sa_l)
                if do_masked and want_stats_local:
                    amin, amax, amean = _masked_min_max_mean(a, m_diff)
                    tf.print(" ", label_mag, "(masked): min=", amin, " max=", amax, " mean=", amean)
                    if want_q3_local:
                        aq = _masked_percentiles(a, q_basic, m_diff)
                        tf.print(" ", label_mag, "(masked) q10,q50,q90 =", aq)
                    if want_smpl_local:
                        sa_f = _first_samples_masked(a, m_diff, tf.cast(n_samples, tf.int32))
                        sa_l = _last_samples_masked(a,  m_diff, tf.cast(n_samples, tf.int32))
                        tf.print("  samples", label_mag, "(masked) (first", n_samples, "):", sa_f)
                        tf.print("  samples", label_mag, "(masked) (last",  n_samples, "):", sa_l)
        else:
            tf.print("[tattle] diff skipped: static shapes or dtypes differ:",
                     str(t0.dtype), tuple(s0), "vs", str(t1.dtype), tuple(s1))

    print("\n-------------------------end tattle\n")
    return 0


def dbg(msg=""):
    """Print ``msg`` prefixed with the caller's file, class.method and line number.

    Uses Python ``inspect`` (eager code only). Disabled when ``DEBUGD["dbg"]`` is 0.
    """
    if DEBUGD["dbg"] > 0:
        try:
            frame = inspect.currentframe().f_back
            info = inspect.getframeinfo(frame)

            filename = os.path.basename(info.filename) if info and info.filename else "<unknown>"
            func = frame.f_code.co_name if frame else "<unknown>"
            cls = None
            if frame and 'self' in frame.f_locals:
                cls = frame.f_locals['self'].__class__.__name__

            location = f"{cls+'.' if cls else ''}{func}"
            line = info.lineno if info and info.lineno else -1

            print(f"[DEBUG] {filename}:{location}({line}) — {msg}")
        except Exception as e:
            print(f"[DEBUG] <unknown>:<unknown>(-1) — {msg} (err: {e})")

    return 0
