# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026

"""CNN-style feature extraction and fusion blocks."""


from core.runtime import DATA_FORMAT

import tensorflow as tf
from tensorflow.keras import activations, layers
from utils import tattle
from .registry import register_block


def _canonicalize_norm(norm, *, layer_name):
    """Normalize config-friendly norm aliases to the internal representation."""
    if isinstance(norm, str):
        norm_key = norm.strip().lower()
        if norm_key in {"", "false", "none", "null", "off", "no"}:
            return None
        if norm_key in {"true", "bn", "batchnorm", "batch_norm"}:
            return "bn"
        if norm_key in {"ln", "layernorm", "layer_norm"}:
            return "ln"
    elif norm is True:
        return "bn"
    elif norm is False or norm is None:
        return None

    raise ValueError(
        f"[{layer_name}] Unsupported norm {norm!r}. "
        "Supported norms: True/'bn', 'ln', or False/None."
    )


@register_block
class Conv(layers.Layer):
    """Conv2D (grouped or depthwise-separable) followed by optional norm and activation.

    Args:
        filters_in (int | None): Kept for config compatibility; the input channel
            count is always inferred in `build()`.
        filters (int): Output channels.
        kernel_size (int | tuple): Kernel size.
        strides (int | tuple): Strides.
        p (str | int): Padding. A string is passed to Conv2D ("same"/"valid");
            an int selects "same" when > 0 or when `kernel_size` > 1, else "valid".
        groups (int): Group count; `-1` selects `SeparableConv2D`.
        dilation (int | tuple): Dilation rate.
        act (str | tf.keras.layers.Layer | bool): Activation, or False/None for linear.
        norm (bool | str | None): `True`/`'bn'` for BatchNorm, `'ln'` for
            LayerNorm, `False`/`None` for none. The conv bias is disabled when a
            norm layer is used.
        kernel_init: Kernel initializer (default: truncated normal, std 0.02).
        use_bias (bool): Conv bias when no norm layer is used.

    Input shape: `[B, H, W, C_in]`. Output shape: `[B, H', W', filters]`.
    """

    def __init__(
        self,
        filters_in=None,
        filters=4,
        kernel_size=1,
        strides=1,
        p="same",
        groups=1,
        dilation=1,
        act="silu",
        norm='bn',
        kernel_init=None,
        use_bias=True,
        name="Conv",
        dtype=tf.float32,
        **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.p = p
        self.groups = groups
        self.dilation = dilation
        self.act = act
        self.norm = _canonicalize_norm(norm, layer_name=name)

        self._name = name
        if isinstance(p, str):
            padding = p
        else:
            padding = "same" if p > 0 or kernel_size > 1 else "valid"

        if self.norm is None:
            self.normalization = None
        elif self.norm == 'bn':
            self.normalization = layers.BatchNormalization(name="bn", axis=-1, dtype=dtype)
        elif self.norm == 'ln':
            self.normalization = layers.LayerNormalization(name="ln", axis=-1, epsilon=1e-5, dtype=dtype)

        if self.normalization is None:
            self.use_bias = use_bias
        else:
            self.use_bias = False

        self.kernel_init = kernel_init or tf.keras.initializers.TruncatedNormal(stddev=0.02)

        self._padding = padding
        self._input_shape = None

        if act is False or act is None:
            self.activation = None
        elif isinstance(act, str):
            self.activation = activations.get(act)
        elif isinstance(act, layers.Layer):
            self.activation = act
        else:
            self.activation = layers.Activation(tf.nn.silu)

    def build(self, input_shape):
        self.filters_in = int(input_shape[-1])  # channels-last

        if self.groups == -1:
            self.sep_conv = layers.SeparableConv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self._padding,
                use_bias=self.use_bias,
                dilation_rate=self.dilation,
                data_format=DATA_FORMAT,
                depthwise_initializer=self.kernel_init,
                pointwise_initializer=self.kernel_init,
                dtype=self.dtype,
                name="SeparableConv2D",
            )
        else:
            self.conv = layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self._padding,
                groups=self.groups,
                use_bias=self.use_bias,
                dilation_rate=self.dilation,
                data_format=DATA_FORMAT,
                kernel_initializer=self.kernel_init,
                dtype=self.dtype,
                name="Conv2D",
            )
        super().build(input_shape)

    def tat(self, x, msg=""):
        if self.name == "conv0":
            tattle(x, 3, f"conv0-{msg}:x")

    def call(self, x, mask=None, training=False):
        # mask: [N, 1, 1, 1] active batch samples
        if self.groups != -1:
            x = self.conv(x)
        else:
            x = self.sep_conv(x)

        if self.normalization is not None:
            if mask is not None:
                mask = tf.cast(mask, tf.bool)
            if self.norm == 'bn':
                x = self.normalization(x, training=training)
            else:
                x = self.normalization(x)

        if self.activation is not None:
            x = self.activation(x)
        return x

    def call_fuse(self, x):
        """Apply the convolution and activation, skipping normalization."""
        if self.groups != -1:
            x = self.conv(x)
        else:
            x = self.sep_conv(x)

        if self.activation is not None:
            x = self.activation(x)
        return x

    def get_config(self):
        cfg = super().get_config()

        cfg.update({
            "filters_in": self.filters_in,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "p": self.p,
            "groups": self.groups,
            "dilation": self.dilation,
            "act": self.act,
            "norm": self.norm,
            "kernel_init": self.kernel_init,
        })
        return cfg


@register_block
class ActBlock(layers.Layer):
    """Optional normalization followed by optional activation.

    Args:
        norm (bool | str | None): `True`/`'bn'`, `'ln'`, or `False`/`None`.
        act (str | tf.keras.layers.Layer | bool): Activation name/layer,
            `True` defaults to `'silu'`, or `False`/`None` for linear.
        name (str): Layer name.
        dtype: Compute dtype.
    """

    def __init__(
        self,
        norm="bn",
        act="silu",
        name="ActBlock",
        dtype=tf.float32,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

        self.norm = _canonicalize_norm(norm, layer_name=name)
        if act is True:
            act = "silu"
        self.act = act

        if self.norm is None:
            self.normalization = None
        elif self.norm == "bn":
            self.normalization = layers.BatchNormalization(
                axis=-1, name="bn", dtype=dtype
            )
        elif self.norm == "ln":
            self.normalization = layers.LayerNormalization(
                axis=-1, epsilon=1e-5, name="ln", dtype=dtype
            )

        if act is False or act is None:
            self.activation = None
        elif isinstance(act, str):
            try:
                self.activation = activations.get(act)
            except Exception as exc:
                raise ValueError(
                    f"[{self.name}] Unsupported activation '{act}'. "
                    "Expected a valid Keras activation name, a tf.keras.layers.Layer, or None/False."
                ) from exc
        elif isinstance(act, layers.Layer):
            self.activation = act
        else:
            raise ValueError(
                f"[{self.name}] Unsupported activation specification {act!r} "
                f"(type: {type(act).__name__}). Expected a valid Keras activation name, "
                "a tf.keras.layers.Layer, or None/False."
            )

    def call(self, x, mask=None, training=False):
        if self.normalization is not None:
            if self.norm == "bn":
                x = self.normalization(x, training=training)
            else:
                x = self.normalization(x)

        if self.activation is not None:
            x = self.activation(x)

        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "norm": self.norm,
            "act": self.act,
        })
        return cfg


@register_block
class Bottleneck(layers.Layer):
    """Two-conv bottleneck with an optional residual shortcut (YOLO-style).

    The shortcut is used only when `shortcut` is set and `filters_in == filters`.
    """

    def __init__(self, filters_in, filters, shortcut=True, groups=1, kernel_size=(3, 3), expansion=0.5,
                 name="Bottleneck", dtype=tf.float32):
        super().__init__(name=name, dtype=dtype)
        c_ = int(filters * expansion)  # hidden channels
        self.cv1 = Conv(filters_in, c_, kernel_size[0], strides=1, name="cv1", dtype=dtype)
        self.cv2 = Conv(c_, filters, kernel_size[1], strides=1, groups=groups, name="cv2", dtype=dtype)
        self.add = shortcut and filters_in == filters

    def call(self, x, mask=None, training=False):
        y = self.cv1(x, mask=mask, training=training)
        y = self.cv2(y, mask=mask, training=training)
        return x + y if self.add else y

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "filters_in": self.cv1.filters_in,
            "filters": self.cv2.filters,
            "shortcut": self.add,
            "groups": self.cv2.groups,
            "kernel_size": (self.cv1.kernel_size, self.cv2.kernel_size),
            "expansion": self.cv1.filters / self.cv2.filters,
        })
        return cfg


@register_block
class C2f(layers.Layer):
    """CSP bottleneck with two convolutions (YOLOv8-style C2f).

    `cv1` expands the input to `2 * c` channels (`c = filters * expansion`),
    `repeat` bottlenecks are chained on the second half, and all intermediate
    outputs are concatenated and fused by the 1x1 conv `cv2`. With
    `split=False`, the bottlenecks run on the full `cv1` output instead.

    Args:
        filters (int): Output channels.
        repeat (int): Number of bottlenecks.
        shortcut (bool): Residual shortcut inside each bottleneck.
        groups (int): Group count for the grouped convs.
        expansion (float): Hidden-width ratio.
        split (bool): Split the `cv1` output into two halves.
        cv1asmlp (bool): Use a 1x1 `cv1`; otherwise a 3x3 grouped conv.
    """

    def __init__(self, filters=8, repeat=1, shortcut=False, groups=1, expansion=0.5, 
                split=True, cv1asmlp=True, 
                name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.filters = filters
        self.repeat = repeat
        self.shortcut = shortcut
        self.groups = groups
        self.expansion = expansion
        self.split = split

        self.c = int(filters * expansion)  # hidden channels

        if not cv1asmlp:
            self.cv1 = Conv(filters=2 * self.c, kernel_size=3, strides=1, groups=groups, name=f"cv1", dtype=dtype)
        else:
            self.cv1 = Conv(filters=2 * self.c, kernel_size=1, strides=1, name=f"cv1", dtype=dtype)

        self.cv2 = Conv( filters=filters, kernel_size=1, name=f"cv2", dtype=dtype)

        self.m = [
            Bottleneck(self.c, self.c, shortcut, groups, kernel_size=((3, 3), (3, 3)), expansion=1.0,
                       name=f"bottleneck_{i}", dtype=dtype)
            for i in range(repeat)
        ]

    def call(self, x, mask=None, training=False):
        if not self.split:
            y0 = self.cv1(x, mask=mask, training=training)
            y = [y0]
            z = y0
            for block in self.m:
                z = block(z, mask=mask, training=training)
                y.append(z)
            out = self.cv2(tf.concat(y, axis=-1), mask=mask, training=training)
            return out

        else:
            y1, y2 = tf.split(self.cv1(x, mask=mask, training=training), num_or_size_splits=2, axis=-1)
            y = [y1, y2]
            z = y2
            for block in self.m:
                z = block(z, mask=mask, training=training)
                y.append(z)
            out = self.cv2(tf.concat(y, axis=-1), mask=mask, training=training)
            return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "repeat": self.repeat,
            "shortcut": self.shortcut,
            "groups": self.groups,
            "expansion": self.expansion,
        })
        return config


@register_block
class MDELAN(C2f):
    """Multi-dilated ELAN-style aggregation block (MDELAN).

    A stem conv (`cv1`) is followed by a chain of dilated convolutions; the stem
    output and every intermediate output are concatenated and fused by a 1x1
    conv (`cv2`). Introduced in arXiv:2607.15127 (SPAWC 2026).

    Args:
        filters (int): Output channels.
        min_filters (int): Lower bound on automatically derived conv widths.
        dilation (tuple[int, ...]): Dilation rate of each conv in the chain.
        m_filters (Sequence[int] | None): Explicit width of each conv; its length
            must match `dilation`. Defaults to `max(filters // d, min_filters)`.
        expansion (float): Hidden-width ratio of the stem.
        groups (int): Group count of the chained convs; `-1` selects separable
            convs with `kernel_size`, otherwise 3x3 kernels are used.
        kernel_size (int): Kernel size of the separable convs.
        split (bool): See `C2f`.
        cv1asmlp (bool): See `C2f`.
        shortcut, pre_act, cv2asProj, shortcut_weighted, pos_prbs: Accepted for
            config compatibility; unused.

    Input shape: `[B, H, W, C]`. Output shape: `[B, H, W, filters]`.
    """

    def __init__(
        self,
        filters,
        min_filters=4,
        dilation=(2, 4, 8),
        m_filters=None,
        expansion=0.5,
        groups=1,
        kernel_size=3,
        split=False,
        cv1asmlp=False,
        shortcut=False,
        pre_act=False,
        cv2asProj=False,
        shortcut_weighted=False,
        pos_prbs=0,
        name="MDELAN",
        dtype=tf.float32,
        **kwargs,
    ):
        repeat = len(dilation)
        super().__init__(
            filters=filters,
            repeat=repeat,
            groups=groups,
            expansion=expansion,
            split=split,
            cv1asmlp=cv1asmlp,
            name=name,
            dtype=dtype,
            **kwargs,
        )
        print(f"{self.name}, filters={filters}, min_filters={min_filters}, dilation={dilation}, \n \
              \t m_filters={m_filters}, expansion={expansion}, groups={groups}, kernel_size={kernel_size}, split={split}, cv1asmlp={cv1asmlp}")
        self.dilation = tuple(dilation)
        self.min_filters = min_filters
        self.kernel_size = kernel_size
        self.m_filters = None if m_filters is None else tuple(m_filters)

        if self.m_filters is not None and len(self.m_filters) != len(self.dilation):
            raise ValueError(
                "`m_filters` must have the same length as `dilation`. "
                f"Got {len(self.m_filters)} and {len(self.dilation)}."
            )

        block_filters = (
            list(self.m_filters)
            if self.m_filters is not None
            else [max(filters // d, self.min_filters) for d in self.dilation]
        )

        self.m = []
        for i, (d, out_c) in enumerate(zip(self.dilation, block_filters)):
            if groups == -1:
                block = Conv(
                    filters=out_c,
                    kernel_size=kernel_size,
                    dilation=d,
                    groups=-1,
                    name=f"SepConv{i}",
                    dtype=dtype,
                )
            else:
                block = Conv(
                    filters=out_c,
                    kernel_size=3,
                    dilation=d,
                    groups=groups,
                    name=f"Conv{i}",
                    dtype=dtype,
                )
            self.m.append(block)

        self.cv2 = Conv(filters=filters, kernel_size=1, name="cv2", dtype=dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dilation": self.dilation,
                "min_filters": self.min_filters,
                "kernel_size": self.kernel_size,
                "m_filters": self.m_filters,
            }
        )
        return config


__all__ = [
    "ActBlock",
    "Bottleneck",
    "C2f",
    "Conv",
    "MDELAN",
]
