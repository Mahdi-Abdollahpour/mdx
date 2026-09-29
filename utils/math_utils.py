# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026

"""Math and masked-statistics helpers."""

import tensorflow as tf


def masked_reduce_mean(p, mask, axis=None, keepdims=False):
    """Mean of ``p`` over entries where ``mask == 1``; 0 where no entry is valid."""
    p = tf.convert_to_tensor(p)
    mask = tf.convert_to_tensor(mask)

    m_bool = tf.equal(mask, tf.cast(1, mask.dtype))
    m_real = tf.cast(m_bool, p.dtype.real_dtype)

    num = tf.reduce_sum(p * tf.cast(m_real, p.dtype), axis=axis, keepdims=keepdims)
    den = tf.reduce_sum(m_real, axis=axis, keepdims=keepdims)

    return tf.math.divide_no_nan(num, tf.cast(den, p.dtype))


def masked_reduce_max(p, mask, axis=None, keepdims=False):
    """Maximum of ``p`` over entries where ``mask`` is true."""
    p = tf.convert_to_tensor(p)
    mask_bool = tf.cast(mask, tf.bool)
    neg_inf = tf.constant(tf.reduce_min(p), dtype=p.dtype.real_dtype)
    p_masked = tf.where(mask_bool, p, tf.cast(neg_inf, p.dtype))
    return tf.reduce_max(p_masked, axis=axis, keepdims=keepdims)


def masked_reduce_min(p, mask, axis=None, keepdims=False):
    """Minimum of ``p`` over entries where ``mask`` is true."""
    p = tf.convert_to_tensor(p)
    mask_bool = tf.cast(mask, tf.bool)
    pos_inf = tf.constant(tf.reduce_max(p), dtype=p.dtype.real_dtype)
    p_masked = tf.where(mask_bool, p, tf.cast(pos_inf, p.dtype))
    return tf.reduce_min(p_masked, axis=axis, keepdims=keepdims)


def masked_quantile(p, mask, q, axis=None, keepdims=False, return_counts=False, atol=None):
    """Global ``q``-quantile of ``p`` over entries where ``mask`` is true.

    Uses linear interpolation between order statistics; complex inputs are
    ordered by their real part. Only ``axis=None`` is supported and
    ``keepdims`` is ignored.

    Args:
        p: Values tensor.
        mask: Boolean or {0, 1} mask with the same shape as ``p``.
        q: Quantile in [0, 1].
        return_counts: If True, also return ``(n_le, n_lt, n_eq, n_valid)``
            as int32 counts of valid entries that are <=, <, == the quantile
            (equality within ``atol`` if given), and the number of valid entries.
        atol: Absolute tolerance for the equality count.

    Returns:
        Scalar quantile with the dtype of ``p``, optionally followed by the counts.
    """
    p = tf.convert_to_tensor(p)
    p_real = tf.math.real(p) if p.dtype.is_complex else p
    m_bool = tf.cast(mask, tf.bool)

    if axis is not None:
        raise NotImplementedError("masked_quantile currently supports axis=None (global) only.")

    p_flat = tf.reshape(p_real, [-1])       # [M]
    m_flat = tf.reshape(m_bool, [-1])       # [M]

    # Invalid entries are set to +inf so that they sort to the end
    pos_inf = tf.constant(float('inf'), dtype=p_flat.dtype)
    p_masked = tf.where(m_flat, p_flat, pos_inf)

    p_sorted = tf.sort(p_masked, axis=-1, direction="ASCENDING")

    k_valid = tf.reduce_sum(tf.cast(m_flat, tf.int32))
    k_safe  = tf.maximum(k_valid, 1)

    r  = tf.cast(q, tf.float32) * tf.cast(k_safe - 1, tf.float32)
    r0 = tf.floor(r)
    r1 = tf.minimum(r0 + 1.0, tf.cast(k_safe - 1, tf.float32))
    w  = r - r0

    i0 = tf.cast(r0, tf.int32)
    i1 = tf.cast(r1, tf.int32)
    v0 = tf.gather(p_sorted, i0)
    v1 = tf.gather(p_sorted, i1)
    qv_real = v0 + (v1 - v0) * tf.cast(w, p_sorted.dtype)
    qv = tf.cast(qv_real, p.dtype)

    if not return_counts:
        return qv

    qv_cast = tf.cast(qv_real, p_flat.dtype)
    less   = tf.logical_and(m_flat, p_flat <  qv_cast)
    if atol is None:
        equal  = tf.logical_and(m_flat, p_flat == qv_cast)
    else:
        equal  = tf.logical_and(m_flat, tf.abs(p_flat - qv_cast) <= tf.cast(atol, p_flat.dtype))
    n_valid = k_valid
    n_lt    = tf.reduce_sum(tf.cast(less,  tf.int32))
    n_eq    = tf.reduce_sum(tf.cast(equal, tf.int32))
    n_le    = n_lt + n_eq

    return qv, n_le, n_lt, n_eq, n_valid


def huber(x, delta, keepdims=True, keep_size=False):
    """Huber loss of the error ``x``, summed over the last axis unless ``keep_size``."""
    abs_x = tf.abs(x)
    l = tf.where(abs_x <= delta, 0.5 * tf.square(abs_x), delta * (abs_x - 0.5 * delta))
    if not keep_size:
        return tf.reduce_sum(l,axis=-1,keepdims=keepdims)
    else:
        return l


def mse(x, keepdims=True, keep_size=False):
    """Squared error ``|x|^2``, summed over the last axis unless ``keep_size``."""
    abs_x = tf.abs(x)
    l = tf.square(abs_x)
    if not keep_size:
        return tf.reduce_sum(l,axis=-1,keepdims=keepdims)
    else:
        return l
