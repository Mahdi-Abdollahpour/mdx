# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026

"""Tensor-shape, complex-conversion, and padding helpers."""

import tensorflow as tf
from sionna.utils import flatten_dims, insert_dims


def _squeeze_inds_nd_(inds, axis):
    """Remove the columns listed in ``axis`` from an index tensor of shape [N, rank]."""
    cols = tf.unstack(inds, axis=1)

    if isinstance(axis, int):
        axis = {axis}
    elif isinstance(axis, (list, tuple)):
        axis = {int(ci) for ci in axis}

    a = [col for j, col in enumerate(cols) if j not in axis]
    return tf.stack(a,1)


def _squeeze_inds_nd(inds, axis):
    """Apply ``_squeeze_inds_nd_`` to every tensor in a nested list/tuple/dict."""
    if tf.is_tensor(inds):
        return _squeeze_inds_nd_(inds, axis)
    if isinstance(inds, (list, tuple)):
        return type(inds)(_squeeze_inds_nd(x, axis) for x in inds)
    if isinstance(inds, dict):
        return {k: _squeeze_inds_nd(v, axis) for k, v in inds.items()}
    return inds


def _insert_dims(x, num_dims=1, axis=0):
    """Apply ``insert_dims`` to every tensor in a nested list/tuple/dict."""
    if tf.is_tensor(x):
        return insert_dims(x, num_dims=num_dims, axis=axis)
    if isinstance(x, (list, tuple)):
        return type(x)(_insert_dims(y, num_dims=num_dims, axis=axis) for y in x)
    if isinstance(x, dict):
        return {k: _insert_dims(v, num_dims=num_dims,axis=axis) for k, v in x.items()}
    return x


def _squeeze(x, axis=None):
    """Apply ``tf.squeeze`` to every tensor in a nested list/tuple/dict.

    ``axis`` may be None, an int or a sequence of ints. Container types are
    preserved and non-tensor leaves are returned unchanged.
    """
    if tf.is_tensor(x):
        if axis is None:
            return tf.squeeze(x)
        if isinstance(axis, int):
            return tf.squeeze(x, axis=[axis])
        return tf.squeeze(x, axis=list(axis))
    if isinstance(x, (list, tuple)):
        return type(x)(_squeeze(y, axis) for y in x)
    if isinstance(x, dict):
        return {k: _squeeze(v, axis) for k, v in x.items()}
    return x


def _stop_gradients(x):
    """Apply ``tf.stop_gradient`` to every tensor in a nested list/tuple/dict."""
    if tf.is_tensor(x):
        return tf.stop_gradient(x)
    if isinstance(x, (list, tuple)):
        return type(x)(_stop_gradients(y) for y in x)
    if isinstance(x, dict):
        return {k: _stop_gradients(v) for k, v in x.items()}
    return x


def swap_axis(x: tf.Tensor, swap_axis=(0, 1)) -> tf.Tensor:
    """Swap two axes of a tensor (graph/XLA compatible).

    Args:
        x: Tensor of rank >= 2.
        swap_axis: Pair ``(i, j)`` of axes to swap. Negative indices require
            a known static rank.

    Returns:
        Tensor of the same dtype with dimensions ``i`` and ``j`` exchanged.
    """
    x = tf.convert_to_tensor(x)
    st = x.shape
    r_static = st.rank

    if not (isinstance(swap_axis, (tuple, list)) and len(swap_axis) == 2):
        raise ValueError("`swap_axis` must be a tuple/list of length 2.")
    i, j = int(swap_axis[0]), int(swap_axis[1])
    if i == j:
        return tf.identity(x)

    if r_static is not None:
        i = i % r_static
        j = j % r_static
        if i == j:
            return tf.identity(x)
        if not (0 <= i < r_static and 0 <= j < r_static):
            raise ValueError(f"`swap_axis` out of range for rank {r_static}.")
    else:
        if i < 0 or j < 0:
            raise ValueError("Negative axes require known static rank.")
        r_dyn = tf.rank(x)
        tf.debugging.assert_less(tf.constant(i, tf.int32), r_dyn, message="axis i out of range")
        tf.debugging.assert_less(tf.constant(j, tf.int32), r_dyn, message="axis j out of range")

    # Identity permutation with entries i and j exchanged
    r_dyn = tf.rank(x)
    perm = tf.range(r_dyn)
    i32 = tf.constant(i, dtype=tf.int32)
    j32 = tf.constant(j, dtype=tf.int32)
    idx = tf.stack([tf.reshape(i32, [1]), tf.reshape(j32, [1])], axis=0)
    vals = tf.stack([j32, i32], axis=0)
    perm = tf.tensor_scatter_nd_update(perm, idx, vals)

    y = tf.transpose(x, perm=perm)

    if r_static is not None:
        dims = list(st)
        dims[i], dims[j] = dims[j], dims[i]
        y.set_shape(tf.TensorShape(dims))

    return y


def _flatten_dims(x, num_dims=2, axis=0):
    """Apply ``flatten_dims`` to every tensor in a nested list/tuple/dict."""
    if tf.is_tensor(x):
        return flatten_dims(x, num_dims=num_dims, axis=axis)
    if isinstance(x, (list, tuple)):
        return type(x)(_flatten_dims(y, num_dims, axis) for y in x)
    if isinstance(x, dict):
        return {k: _flatten_dims(v, num_dims, axis) for k, v in x.items()}
    return x


def collapse_axes(a: tf.Tensor, axis=(0, 1)) -> tf.Tensor:
    """Merge two (not necessarily adjacent) axes of a tensor into one.

    The merged axis of size ``s_p * s_q`` is placed at ``min(p, q)``; the
    relative order of the remaining axes is preserved. Graph/XLA compatible.

    Args:
        a: Tensor of rank r >= 2 with known static rank.
        axis: Pair ``(p, q)`` of distinct axes; negative indices are allowed.

    Returns:
        Tensor of rank r - 1 with the same dtype.
    """
    a = tf.convert_to_tensor(a)
    st = a.shape
    r_static = st.rank

    if not (isinstance(axis, (tuple, list)) and len(axis) == 2):
        raise ValueError("`axis` must be a tuple/list of length 2.")
    i, j = int(axis[0]), int(axis[1])
    if i == j:
        raise ValueError("`axis` must refer to two distinct axis.")

    i = i % r_static
    j = j % r_static
    if i == j:
        raise ValueError("`axis` must refer to two distinct axis after normalization.")
    if not (0 <= i < r_static and 0 <= j < r_static):
        raise ValueError(f"`axis` must be in [0, {r_static}).")

    if j < i:
        i, j = j, i

    # Permutation [0..i-1, i, j, i+1..j-1, j+1..r-1] makes the two axes adjacent
    r_dyn = tf.rank(a)
    before  = tf.range(0, i, dtype=tf.int32)
    between = tf.range(i + 1, j, dtype=tf.int32)
    after   = tf.range(j + 1, r_dyn, dtype=tf.int32)
    perm = tf.concat([before,
                      tf.constant([i, j], dtype=tf.int32),
                      between,
                      after], axis=0)

    a_t = tf.transpose(a, perm=perm)

    s = tf.shape(a_t)
    merged = s[i] * s[i + 1]
    new_shape = tf.concat([s[:i], tf.expand_dims(merged, 0), s[i + 2:]], axis=0)
    out = tf.reshape(a_t, new_shape)

    if r_static is not None:
        dims = list(st)
        di = dims[i]
        dj = dims[j]
        merged_dim = (di * dj) if (di is not None and dj is not None) else None
        del dims[j]
        dims[i] = merged_dim
        out.set_shape(tf.TensorShape(dims))

    return out


def ri2c(r):
    """Convert a real tensor ``[..., 2]`` of (real, imag) pairs to a complex tensor ``[...]``."""
    return tf.complex(r[...,0], r[...,1])


def c2ri(c):
    """Convert a complex tensor ``[...]`` to a real tensor ``[..., 2]`` of (real, imag) pairs."""
    return tf.stack([tf.math.real(c), tf.math.imag(c)], axis=-1)

