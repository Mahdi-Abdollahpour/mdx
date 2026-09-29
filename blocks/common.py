# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026

"""Common block utilities and small composition layers."""

import copy

from core.runtime import DATA_FORMAT

import tensorflow as tf
from tensorflow.keras import layers
from utils import _stop_gradients, tattle
from .registry import BLOCK_REGISTRY, register_alias, register_block


@register_block(needs=('sys',))
class MCSAware(layers.Layer):
    """Wrap a block with one independent copy per MCS and mix their outputs per UE.

    All copies are applied to the same input ``x`` [B*T*RA, H, W, C]; the
    outputs are combined with the one-hot ``mcs_masks["mcs_ue_mask"]``
    [B, T, num_mcs] so that each UE uses the copy of its MCS.

    Args:
        block_type: Registered name of the wrapped block.
        sys: System parameters; ``sys["num_mcss_supported"]`` is used.
        block_cfg: Keyword arguments of the wrapped block.
    """

    def __init__(
        self,
        block_type,
        sys,
        block_cfg,
        name="MCSAware",
        dtype=tf.float32,
        **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

        if block_type not in BLOCK_REGISTRY:
            raise ValueError(f"Unknown block_type '{block_type}'")

        self.block_type = block_type
        self.num_mcs = int( sys["num_mcss_supported"])
        self.block_cfg = dict(block_cfg)
        self.block_cls = BLOCK_REGISTRY[block_type]

        base_name = self.block_cfg.get("name", block_type)

        self.blocks = []
        for i in range(self.num_mcs):
            cfg_i = copy.deepcopy(self.block_cfg)
            cfg_i["name"] = f"{base_name}_mcs{i}"
            self.blocks.append(self.block_cls(**cfg_i))

    def _broadcast_mask(self, mask_bt, y):
        """Expand ``mask_bt`` [B, T] to [B, T, 1, ..., 1] to broadcast against ``y``."""
        mask_bt = tf.cast(mask_bt, y.dtype)

        rank = y.shape.rank
        if rank is None:
            extra = tf.rank(y) - 2
            new_shape = tf.concat(
                [tf.shape(mask_bt), tf.ones([extra], dtype=tf.int32)],
                axis=0
            )
            return tf.reshape(mask_bt, new_shape)

        for _ in range(rank - 2):
            mask_bt = tf.expand_dims(mask_bt, axis=-1)
        return mask_bt

    def _unflatten_bt_ra(self, y, s):
        """Reshape [B*T*RA, H, W, C] to [B, T, RA, H, W, C]."""
        dyn = tf.shape(y)
        H, W, C = dyn[1], dyn[2], dyn[3]
        return tf.reshape(y, [s.B, s.T, s.RA, H, W, C])

    def _flatten_bt_ra(self, y, s):
        """Reshape [B, T, RA, H, W, C] to [B*T*RA, H, W, C]."""
        dyn = tf.shape(y)
        H, W, C = dyn[3], dyn[4], dyn[5]
        return tf.reshape(y, [s.B * s.T * s.RA, H, W, C])

    def _mix_outputs(self, ys, mcs_ue_mask, s):
        if self.num_mcs == 1:
            return ys[0]

        if mcs_ue_mask is None:
            raise ValueError("mcs_ue_mask is required when num_mcs > 1")

        for i, y in enumerate(ys):
            ys[i] =  self._unflatten_bt_ra(ys[i], s)

        out = tf.zeros_like(ys[0])
        for i, y in enumerate(ys):
            mask_i = self._broadcast_mask(mcs_ue_mask[:, :, i], y)
            out = out + mask_i * y
        return self._flatten_bt_ra(out, s)

    def call(self, x, mcs_masks=None, shapes=None, training=False, **kwargs):
        mcs_ue_mask = mcs_masks["mcs_ue_mask"] if mcs_masks is not None else None

        ys = []
        for block in self.blocks:
            y = block(x, training=training, **kwargs)
            ys.append(y)

        out = self._mix_outputs(ys, mcs_ue_mask, shapes)

        return out

    def call_fuse(self, x, mcs_masks=None, shapes=None, **kwargs):
        """Inference path using each wrapped block's ``call_fuse`` when available."""
        mcs_ue_mask = mcs_masks["mcs_ue_mask"] if mcs_masks is not None else None

        ys = []
        for block in self.blocks:
            if hasattr(block, "call_fuse"):
                y = block.call_fuse(x, **kwargs)
            else:
                y = block(x, training=False, **kwargs)
            ys.append(y)
        return self._mix_outputs(ys, mcs_ue_mask, shapes)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "block_type": self.block_type,
            "num_mcs": self.num_mcs,
            "block_cfg": self.block_cfg,
        })
        return cfg


@register_block
class SumFuse(layers.Layer):
    """Sum-fuse multiple same-shaped tensors, with optional learnable weights.

    Inputs:
        list/tuple of tensors with identical shape, typically NHWC.

    Output:
        Tensor with the same shape as each input.

    Args:
        weight_mode: False/None, 'scalar', 'channel', 'prb-scalar', or
            'prb-channel'.
            - False or None: no learnable coefficients.
            - 'scalar': one coefficient per residual input.
            - 'channel': one coefficient per residual input and channel.
            - 'prb-scalar': one coefficient per residual input and
              `(12, 14)` PRB element.
            - 'prb-channel': one coefficient per residual input,
              `(12, 14)` PRB element, and channel.
        init_value: initializer value for the coefficients.

    Behavior:
        The first input is never weighted. Only inputs `1..N-1` are weighted
        and summed into the first input.
    """

    def __init__(
        self,
        weight_mode=False,
        init_value=1.e-5,
        name="SumFuse",
        dtype=tf.float32,
    ):
        super().__init__(name=name, dtype=dtype)
        if weight_mode is None or weight_mode is False:
            self.weight_mode = False
        else:
            self.weight_mode = str(weight_mode).lower()
        self.init_value = float(init_value)

        if self.weight_mode is not False and self.weight_mode not in (
            "scalar",
            "channel",
            "prb-scalar",
            "prb-channel",
        ):
            raise ValueError(
                f"`weight_mode` must be None, False, 'scalar', 'channel', "
                f"'prb-scalar', or 'prb-channel', got {weight_mode!r}."
            )

        self.num_inputs = None
        self.num_residual_inputs = None
        self.height = None
        self.width = None
        self.channels = None
        self.alpha = None

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) == 0:
            raise ValueError(
                "`SumFuse` expects a non-empty list/tuple of input shapes."
            )

        self.num_inputs = len(input_shape)
        self.num_residual_inputs = max(self.num_inputs - 1, 0)

        ref_shape = tf.TensorShape(input_shape[0])
        if ref_shape.rank is None:
            raise ValueError("Input rank must be known at build time.")

        for i, shp in enumerate(input_shape[1:], start=1):
            shp = tf.TensorShape(shp)

            if shp.rank is None:
                raise ValueError(f"Input {i} rank must be known at build time.")

            if ref_shape.rank != shp.rank:
                raise ValueError(
                    f"All inputs must have the same rank. "
                    f"Input 0 rank={ref_shape.rank}, input {i} rank={shp.rank}."
                )

            for d, (d0, di) in enumerate(zip(ref_shape, shp)):
                if d0 is not None and di is not None and d0 != di:
                    raise ValueError(
                        f"All inputs must have the same shape. "
                        f"Mismatch at dim {d}: input 0 has {d0}, input {i} has {di}. "
                        f"Shapes: {ref_shape} vs {shp}."
                    )

        self.height = ref_shape[1]
        self.width = ref_shape[2]
        self.channels = ref_shape[3]

        if self.weight_mode in ("prb-scalar", "prb-channel") and ref_shape.rank != 4:
            raise ValueError(
                "For `weight_mode` in {'prb-scalar', 'prb-channel'}, "
                "`SumFuse` expects NHWC inputs with rank 4."
            )
        if self.weight_mode in ("prb-scalar", "prb-channel"):
            if self.height is None or self.width is None or self.channels is None:
                raise ValueError(
                    "For `weight_mode` in {'prb-scalar', 'prb-channel'}, "
                    "H, W, and C must be known at build time."
                )

        if self.weight_mode is not False and self.num_residual_inputs > 0:
            if self.weight_mode == "scalar":
                alpha_shape = (self.num_residual_inputs,)
            elif self.weight_mode == "channel":
                if self.channels is None:
                    raise ValueError(
                        "For `weight_mode='channel'`, the channel dimension must be known."
                    )
                alpha_shape = (self.num_residual_inputs, int(self.channels))
            elif self.weight_mode == "prb-scalar":
                alpha_shape = (self.num_residual_inputs, 12, 14)
            else:
                if self.channels is None:
                    raise ValueError(
                        "For `weight_mode='prb-channel'`, the channel dimension "
                        "must be known."
                    )
                alpha_shape = (self.num_residual_inputs, 12, 14, int(self.channels))

            self.alpha = self.add_weight(
                name="alpha",
                shape=alpha_shape,
                initializer=tf.keras.initializers.Constant(self.init_value),
                trainable=True,
                dtype=self.dtype,
            )

        super().build(input_shape)

    def _apply_prb_weight(self, x, weight, channelwise=False):
        n = tf.shape(x)[0]
        h = int(self.height)
        w = int(self.width)
        c = int(self.channels)

        tf.debugging.assert_equal(
            tf.math.floormod(h, 12), 0,
            message="SumFuse expects H to be divisible by 12 for PRB weighting."
        )
        tf.debugging.assert_equal(
            tf.math.floormod(w, 14), 0,
            message="SumFuse expects W to be divisible by 14 for PRB weighting."
        )

        x = tf.reshape(x, [n, h // 12, 12, w // 14, 14, c])
        if channelwise:
            wgt = tf.reshape(weight, [1, 1, 12, 1, 14, c])
        else:
            wgt = tf.reshape(weight, [1, 1, 12, 1, 14, 1])
        x = x * wgt
        return tf.reshape(x, [n, h, w, c])

    def call(self, inputs):
        xs = [tf.cast(x, self.compute_dtype) for x in inputs]

        if len(xs) == 1:
            return xs[0]

        x0 = xs[0]
        residuals = xs[1:]

        if self.weight_mode is False:
            weighted_residuals = residuals
        elif self.weight_mode == "scalar":
            weighted_residuals = [
                self.alpha[i] * x
                for i, x in enumerate(residuals)
            ]
        elif self.weight_mode == "channel":
            weighted_residuals = [
                tf.reshape(self.alpha[i], [1, 1, 1, -1]) * x
                for i, x in enumerate(residuals)
            ]
        elif self.weight_mode == "prb-scalar":
            weighted_residuals = [
                self._apply_prb_weight(x, self.alpha[i], channelwise=False)
                for i, x in enumerate(residuals)
            ]
        else:
            weighted_residuals = [
                self._apply_prb_weight(x, self.alpha[i], channelwise=True)
                for i, x in enumerate(residuals)
            ]

        return x0 + tf.add_n(weighted_residuals)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "weight_mode": self.weight_mode,
                "init_value": self.init_value,
            }
        )
        return config


@register_block
class UpSampleAndConcat(layers.Layer):
    """Upsample feature maps to the largest spatial size and concatenate them.

    Every input (NHWC) is upsampled by integer factors to the largest height
    and width among the inputs, then all are concatenated along
    ``concat_axis``. Nearest-neighbour upsampling uses reshape + broadcast
    (XLA-friendly); other interpolation modes use ``UpSampling2D``.

    Args:
        concat_axis: Axis along which to concatenate (default: -1).
        data_format: Must be `"channels_last"`.
        interpolation: "nearest" (XLA-safe path) or other modes passed to
            `UpSampling2D` (e.g. "bilinear").
        name: Layer name.
        dtype: Tensor dtype.
    """

    def __init__(
        self,
        concat_axis=-1,
        data_format=DATA_FORMAT,
        interpolation="nearest",
        name="upsample_concat",
        dtype=tf.float32,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

        if data_format != "channels_last":
            raise ValueError(
                "UpSampleAndConcat currently supports only 'channels_last' data_format; "
                f"got {data_format!r}."
            )

        self.data_format = "channels_last"
        self.interpolation = str(interpolation or "nearest").lower()
        self._concat_axis = concat_axis

        self._target_size = None
        self._scales = []
        self._upsamplers = []

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)):
            input_shape = [input_shape]

        shapes = [tf.TensorShape(s).as_list() for s in input_shape]

        h_idx, w_idx = 1, 2

        heights = []
        widths = []
        for s in shapes:
            h, w = s[h_idx], s[w_idx]
            if h is None or w is None:
                raise ValueError(
                    "UpSampleAndConcat requires known spatial dimensions at build time; "
                    f"got shape {s} with H={h}, W={w}."
                )
            heights.append(h)
            widths.append(w)

        target_h = max(heights)
        target_w = max(widths)
        self._target_size = (target_h, target_w)

        self._scales = []
        self._upsamplers = []

        for i, (h, w) in enumerate(zip(heights, widths)):
            if h == target_h and w == target_w:
                self._scales.append(None)
                self._upsamplers.append(None)
                continue

            if target_h % h != 0 or target_w % w != 0:
                raise ValueError(
                    "UpSampleAndConcat only supports integer upsampling factors; "
                    f"got input spatial size ({h}, {w}) and target ({target_h}, {target_w})."
                )

            scale_h = target_h // h
            scale_w = target_w // w
            self._scales.append((scale_h, scale_w))

            if self.interpolation == "nearest":
                self._upsamplers.append(None)
            else:
                up = layers.UpSampling2D(
                    size=(scale_h, scale_w),
                    data_format=self.data_format,
                    interpolation=self.interpolation,
                    dtype=self.dtype,
                    name=f"upsampling{i}_{h}x{w}_to_{target_h}x{target_w}",
                )
                self._upsamplers.append(up)

        super().build(input_shape)

    def call(self, inputs, mask=None, training=False):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        tattle(inputs, 3, f"{self.name}")

        if len(inputs) != len(self._scales):
            raise ValueError(
                "Number of inputs at call-time does not match build-time: "
                f"{len(inputs)} vs {len(self._scales)}."
            )

        upsampled = []
        for x, scale, up in zip(inputs, self._scales, self._upsamplers):
            if scale is None:
                upsampled.append(x)
                continue

            scale_h, scale_w = scale
            if self.interpolation == "nearest":
                x_up =nearest_upsample_2d(x, scale_h, scale_w)
            else:
                x_up = up(x)
            upsampled.append(x_up)

        tattle(upsampled, 3, f"{self.name}")
        return tf.concat(upsampled, axis=self._concat_axis)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "data_format": self.data_format,
                "interpolation": self.interpolation,
                "concat_axis": self._concat_axis,
            }
        )
        return cfg


@register_block
class StopGradients (layers.Layer):
    """Identity layer that stops gradients for every tensor in a (nested) input."""

    def __init__(self, name="StopGradients", dtype=tf.float32):
        super().__init__(name=name, dtype=dtype)

    def call(self, x, mask=None, training=False):
        return _stop_gradients(x)


@register_block
class UpSampling(layers.Layer):
    """Registered wrapper around ``tf.keras.layers.UpSampling2D``."""

    def __init__(self, size=(2,2), data_format=DATA_FORMAT,
                interpolation="nearest", name="upsampling", dtype=tf.float32):
        super().__init__(name=name, dtype=dtype)
        self.size = size
        self.data_format = data_format
        self.interpolation = interpolation

        self.upsampling = layers.UpSampling2D(size,data_format,interpolation, name="upsampling", dtype=dtype)

    def call(self, x, mask=None, training=False):
        return self.upsampling(x)

    def get_config(self):
        cfg = super().get_config()

        cfg.update({
            "size": self.size,
            "data_format": self.data_format,
            "interpolation": self.interpolation,
        })
        return cfg


def nearest_upsample_2d(x, scale_h, scale_w):
    """Nearest-neighbour upsampling of an NHWC tensor by integer factors.

    Uses reshape + ``tf.broadcast_to`` instead of resize/tile ops (XLA-friendly).
    """
    x_shape = tf.shape(x)
    b, h, w, c = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

    out_h = h * scale_h
    out_w = w * scale_w

    x = tf.reshape(x, [b, h, 1, w, 1, c])

    target = tf.stack([b, h, scale_h, w, scale_w, c])
    x = tf.broadcast_to(x, target)

    x = tf.reshape(x, [b, out_h, out_w, c])
    return x


@register_block
class Shapes(layers.Layer):
    """Container of runtime shape scalars shared by the blocks of a receiver.

    Stores ``batch_size``, ``num_tx``, ``num_ant_rx``, ``fft_size``,
    ``num_ant_tx`` and ``num_rx`` as 0-D tensors of ``dtype`` in the
    attributes ``B``, ``T``, ``RA``, ``F``, ``TA`` and ``R``.
    """

    def __init__(self, batch_size, num_tx, num_ant_rx,
                 fft_size,
                 num_ant_tx=1, num_rx=1,
                 name="Shapes", dtype=tf.int32):
        super().__init__(name=name, dtype=dtype)

        def _to_dtype_scalar(x, label):
            if x is None:
                return None
            t = tf.convert_to_tensor(x)
            if t.dtype.is_floating and tf.as_dtype(self.dtype).is_integer:
                t = tf.math.round(t)

            tf.debugging.assert_equal(
                tf.size(t), 1,
                message=f"[Shapes] `{label}` must be a scalar (size==1)."
            )
            t = tf.reshape(t, [])
            return tf.cast(t, self.dtype)

        self.B  = _to_dtype_scalar(batch_size, "B")
        self.T  = _to_dtype_scalar(num_tx, "T")
        self.RA = _to_dtype_scalar(num_ant_rx, "RA")
        self.F  = _to_dtype_scalar(fft_size, "F")

        self.TA = _to_dtype_scalar(num_ant_tx, "TA")
        self.R  = _to_dtype_scalar(num_rx, "R")

    def print(self):
        tf.print(
            "[Shapes]",
            "B=", self.B, "T=", self.T, "R=", self.R,
            "RA=", self.RA, "TA=", self.TA, "F=", self.F,
            sep=" "
        )
        return tf.constant(0, dtype=self.dtype)


@register_block
class PRBDropMask(layers.Layer):
    """
    Generate a random NHWC validity mask with a vertical active prefix.

    The active height is sampled independently per batch item, shared across
    all TX and RX antennas of that batch item, and quantized in PRB-sized
    steps. The mask always starts at row 0 and extends downward. Outside
    training the mask is all ones.

    Inputs:
        x: Tensor of shape [B*T*RA, H, W, C]. Only H and W are used.
        shapes: Runtime Shapes object with scalar attributes B, T, and RA.

    Output (return_mask_only=True, default):
        Boolean mask of shape [B*T*RA, H, W, 1].
    Output (return_mask_only=False):
        Tuple (masked_x, mask) where masked_x has dropped elements zeroed.
    """

    def __init__(
        self,
        max_len=None,
        min_len=12,
        prb_size=12,
        return_mask_only=True,
        name="PRBDropMask",
        dtype=tf.float32,
    ):
        super().__init__(name=name, dtype=dtype)

        self.min_len = int(min_len)
        self.max_len = None if max_len is None else int(max_len)
        self.prb_size = int(prb_size)
        self.return_mask_only = bool(return_mask_only)

        if self.prb_size <= 0:
            raise ValueError(f"[{name}] prb_size must be > 0, got {self.prb_size}")
        if self.min_len <= 0 or (self.max_len is not None and self.max_len <= 0):
            raise ValueError(
                f"[{name}] min_len and max_len must be > 0, got "
                f"{self.min_len}, {self.max_len}"
            )
        if self.max_len is not None and self.min_len > self.max_len:
            raise ValueError(
                f"[{name}] min_len={self.min_len} must be <= max_len={self.max_len}"
            )
        if (
            self.min_len % self.prb_size != 0
            or (self.max_len is not None and self.max_len % self.prb_size != 0)
        ):
            raise ValueError(
                f"[{name}] min_len and max_len must be multiples of prb_size="
                f"{self.prb_size}"
            )

        self.min_prbs = self.min_len // self.prb_size
        self.max_prbs = None if self.max_len is None else self.max_len // self.prb_size
        self._height = None
        self._width = None
        self._height_t = None
        self._width_t = None
        self._min_prbs_t = None
        self._max_prbs_t = None
        self._row_idx = None

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        if input_shape.rank is not None and input_shape.rank < 3:
            raise ValueError(
                f"[{self.name}] expected input rank >= 3, got {input_shape.rank}"
            )

        H = input_shape[1]
        W = input_shape[2]

        if H is not None:
            self._height = int(H)
            self._height_t = tf.constant(self._height, dtype=tf.int32)
            self._row_idx = tf.reshape(
                tf.range(self._height, dtype=tf.int32),
                [1, self._height, 1, 1],
            )

            max_prbs_from_h = self._height // self.prb_size
            if max_prbs_from_h <= 0:
                raise ValueError(
                    f"[{self.name}] input height must contain at least one PRB"
                )

            min_prbs = min(self.min_prbs, max_prbs_from_h)
            if self.max_prbs is None:
                max_prbs = max_prbs_from_h
            else:
                max_prbs = min(self.max_prbs, max_prbs_from_h)
            min_prbs = min(min_prbs, max_prbs)

            self._min_prbs_t = tf.constant(min_prbs, dtype=tf.int32)
            self._max_prbs_t = tf.constant(max_prbs, dtype=tf.int32)

        if W is not None:
            self._width = int(W)
            self._width_t = tf.constant(self._width, dtype=tf.int32)

        super().build(input_shape)

    def call(self, x, shapes=None, training=False):
        if shapes is None:
            raise ValueError(f"[{self.name}] `shapes` is required.")

        B = tf.cast(shapes.B, tf.int32)
        T = tf.cast(shapes.T, tf.int32)
        RA = tf.cast(shapes.RA, tf.int32)

        x_shape = tf.shape(x)
        N = x_shape[0]
        H = self._height_t if self._height_t is not None else x_shape[1]
        W = self._width_t if self._width_t is not None else x_shape[2]

        expected_n = B * T * RA
        tf.debugging.assert_equal(
            N,
            expected_n,
            message=f"[{self.name}] expected x.shape[0] == B*T*RA",
        )

        if not training:
            mask = tf.ones(tf.stack([expected_n, H, W, 1]), dtype=tf.bool)
            if not self.return_mask_only:
                return x, mask
            return mask

        prb_size = tf.constant(self.prb_size, dtype=tf.int32)
        if self._max_prbs_t is None:
            max_prbs_from_h = H // prb_size
            tf.debugging.assert_positive(
                max_prbs_from_h,
                message=f"[{self.name}] input height must contain at least one PRB",
            )

            min_prbs = tf.minimum(
                tf.constant(self.min_prbs, dtype=tf.int32), max_prbs_from_h
            )
            if self.max_prbs is None:
                max_prbs = max_prbs_from_h
            else:
                max_prbs = tf.minimum(
                    tf.constant(self.max_prbs, dtype=tf.int32), max_prbs_from_h
                )
            min_prbs = tf.minimum(min_prbs, max_prbs)
        else:
            min_prbs = self._min_prbs_t
            max_prbs = self._max_prbs_t

        # High-biased triangular sampling: max(U1, U2) has density 2x on [0, 1].
        u1 = tf.random.uniform(tf.reshape(B, [1]), dtype=self.compute_dtype)
        u2 = tf.random.uniform(tf.reshape(B, [1]), dtype=self.compute_dtype)
        frac = tf.maximum(u1, u2)

        span = tf.cast(max_prbs - min_prbs + 1, frac.dtype)
        prb_counts = min_prbs + tf.cast(tf.floor(frac * span), tf.int32)
        prb_counts = tf.minimum(prb_counts, max_prbs)
        active_len = prb_counts * prb_size

        one = tf.constant(1, dtype=tf.int32)
        row_idx = self._row_idx
        if row_idx is None:
            row_idx = tf.reshape(tf.range(H, dtype=tf.int32), tf.stack([one, H, one, one]))
        active_len = tf.reshape(active_len, tf.stack([B, one, one, one]))

        mask_b = row_idx < active_len
        mask_b = tf.broadcast_to(mask_b, tf.stack([B, H, W, one]))

        # Share the same batch-sample mask across all TX and RX antenna copies.
        mask = tf.reshape(mask_b, tf.stack([B, one, one, H, W, one]))
        mask = tf.broadcast_to(mask, tf.stack([B, T, RA, H, W, one]))
        mask = tf.reshape(mask, tf.stack([expected_n, H, W, one]))
        if not self.return_mask_only:
            masked_x = x * tf.cast(mask, x.dtype)
            return masked_x, mask
        return mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "min_len": self.min_len,
                "max_len": self.max_len,
                "prb_size": self.prb_size,
                "return_mask_only": self.return_mask_only,
            }
        )
        return cfg


register_alias("PRBdrop", "PRBDropMask")
register_alias("PRBDrop", "PRBDropMask")


@register_block
class HFreqNormalizer(layers.Layer):
    """Per-link RMS normalization of channel frequency responses.

    ``x`` [B*T*RA, F, S, 2] (real/imag) is divided by the RMS of ``y_true``
    (same shape), computed per (batch, TX) link over the RX antennas,
    subcarriers and OFDM symbols. ``x`` may also be a list/tuple whose first
    element is the channel and whose last element is a mask of the same shape
    (1 = active); the RMS is then taken over active positions only, and all
    elements but the first are returned unchanged.

    Args:
        normalize_only_in_training: If True, return ``x`` unchanged when not
            training.
        dtype: Complex dtype used for the computation.

    Call args:
        x: Channel tensor, or list/tuple as described above.
        y_true: Reference channel [B*T*RA, F, S, 2] defining the scale.
        shapes: ``Shapes`` object with B, T, RA.
        training: Keras training flag.

    Returns:
        The normalized channel [B*T*RA, F, S, 2], or a tuple
        ``(normalized, *x[1:])`` for list/tuple inputs.
    """

    def __init__(
        self,
        normalize_only_in_training=True,
        name="HFreqNormalizer",
        dtype=tf.complex64,
    ):
        super().__init__(name=name, dtype=dtype)
        self.normalize_only_in_training = bool(normalize_only_in_training)

    def build(self, input_shape):
        self._is_multi_input = (
            isinstance(input_shape, (list, tuple))
            and len(input_shape) > 0
            and isinstance(input_shape[0], tf.TensorShape)
        )
        self._has_mask = self._is_multi_input
        super().build(input_shape)

    def call(self, x, y_true, shapes=None, training=False):
        if self.normalize_only_in_training and not training:
            return x

        if self._is_multi_input:
            extra = x[1:]
            mask_input = x[-1]
            x = x[0]

        if shapes is None:
            raise ValueError(f"[{self.name}] `shapes` is required.")

        B  = tf.cast(shapes.B,  tf.int32)
        T  = tf.cast(shapes.T,  tf.int32)
        RA = tf.cast(shapes.RA, tf.int32)

        x_shape = tf.shape(x)
        N = x_shape[0]
        tf.debugging.assert_equal(
            N, B * T * RA,
            message=f"[{self.name}] expected x.shape[0] == B*T*RA",
        )

        F = x_shape[1]
        S = x_shape[2]

        # [B*T*RA, F, S, 2] -> [B, 1, RA, T, 1, S, F], complex
        h = tf.complex(x[..., 0], x[..., 1])
        h = tf.reshape(h, tf.stack([B, T, RA, F, S]))
        h = tf.transpose(h, perm=[0, 2, 1, 4, 3])
        h = tf.expand_dims(h, axis=1)
        h = tf.expand_dims(h, axis=4)
        h = tf.cast(h, self.compute_dtype)

        # same for y_true
        h_true = tf.complex(y_true[..., 0], y_true[..., 1])
        h_true = tf.reshape(h_true, tf.stack([B, T, RA, F, S]))
        h_true = tf.transpose(h_true, perm=[0, 2, 1, 4, 3])
        h_true = tf.expand_dims(h_true, axis=1)
        h_true = tf.expand_dims(h_true, axis=4)
        h_true = tf.cast(h_true, self.compute_dtype)

        # mask: [B*T*RA, F, S, 2] -> [B, 1, RA, T, 1, S, F]
        if self._has_mask:
            mask_h = tf.cast(mask_input[..., 0], tf.float32)
            mask_h = tf.reshape(mask_h, tf.stack([B, T, RA, F, S]))
            mask_h = tf.transpose(mask_h, perm=[0, 2, 1, 4, 3])
            mask_h = tf.expand_dims(mask_h, axis=1)
            mask_h = tf.expand_dims(mask_h, axis=4)

        # RMS over RA, TA, S, F per (B, R, T) link; scale: [B, 1, 1, T, 1, 1, 1]
        reduce_axes = (2, 4, 5, 6)
        if self._has_mask:
            src_sq        = tf.square(tf.abs(h_true))
            masked_sq_sum = tf.reduce_sum(src_sq * mask_h, axis=reduce_axes, keepdims=True)
            mask_count    = tf.reduce_sum(mask_h,           axis=reduce_axes, keepdims=True)
            scale = tf.cast(
                tf.sqrt(tf.math.divide_no_nan(masked_sq_sum, mask_count)), h.dtype
            )
        else:
            scale = tf.cast(
                tf.sqrt(tf.reduce_mean(tf.square(tf.abs(h_true)), axis=reduce_axes, keepdims=True)),
                h.dtype,
            )
        h = tf.math.divide_no_nan(h, scale)

        # back to [B*T*RA, F, S, 2]
        h = tf.squeeze(h, axis=[1, 4])
        h = tf.transpose(h, perm=[0, 2, 1, 4, 3])
        h = tf.reshape(h, tf.stack([B * T * RA, F, S]))
        result = tf.stack([tf.math.real(h), tf.math.imag(h)], axis=-1)
        if self._is_multi_input:
            return tuple( [result] + list(extra) )
        return result

    def get_config(self):
        cfg = super().get_config()
        cfg["normalize_only_in_training"] = self.normalize_only_in_training
        return cfg


__all__ = [
    "HFreqNormalizer",
    "MCSAware",
    "PRBDropMask",
    "Shapes",
    "StopGradients",
    "SumFuse",
    "UpSampleAndConcat",
    "UpSampling",
    "nearest_upsample_2d",
]
