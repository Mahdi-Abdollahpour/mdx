# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026

"""Residual composition blocks."""

import math
import tensorflow as tf
from tensorflow.keras import layers
from .cnn import Bottleneck, Conv, MDELAN
from .registry import register_block


@register_block
class ResBlock(layers.Layer):
    """Pre-activation residual block with optional positional encodings.

    For an input ``x`` [N, F, W, C] (F subcarriers, W OFDM symbols), computes
    ``x + w * head(trunk(pos(act(bn(x)))))``, where ``trunk`` is a ``Conv``,
    ``MDELAN`` or ``Bottleneck`` block, ``head`` a 3x3 convolution back to C
    channels and ``w`` an optional learnable per-PRB residual weight of shape
    [12, W]. The pre-activation (BN + SiLU) can be disabled per call.

    Positional encodings, concatenated as extra channels before the trunk:
        - ``pos_prbs`` > 0: fixed vertical and horizontal coordinates, the
          vertical one repeating every ``12 * pos_prbs`` subcarriers. With
          ``learnable_pos``, a learnable vertical encoding of the same period
          (``learnable_pos_c`` channels if > 0) is used instead.
        - ``pos_prbs_v`` > 0: learnable vertical encoding of period
          ``12 * pos_prbs_v`` with ``learnable_pos_v_c`` channels.
        - ``learnable_pos_h_c`` > 0: learnable horizontal encoding with that
          many channels.

    Args:
        block_type: ``'conv'``, ``'mdelan'`` or ``'bottleneck'``.
        block_config: Keyword arguments of the trunk block.
        use_res_weight: Enable the per-PRB residual weight ``w``.
        proj_norm: Normalization of the head convolution.
    """

    def __init__(
        self,
        name="ResBlock",
        block_type="conv",
        block_config=None,
        use_res_weight=False,
        pos_prbs=3,
        learnable_pos=False,
        learnable_pos_c=0,
        pos_prbs_v=0,
        learnable_pos_v_c=1,
        learnable_pos_h_c=0,
        proj_norm=False,
        dtype=tf.float32,
        **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

        self.block_type = block_type
        self.block_config = dict(block_config or {})
        self.use_res_weight = use_res_weight
        self.pos_prbs = int(pos_prbs)
        self.learnable_pos = bool(learnable_pos)
        self.learnable_pos_c = int(learnable_pos_c)
        self.pos_prbs_v = int(pos_prbs_v)
        self.learnable_pos_v_c = int(learnable_pos_v_c)
        self.learnable_pos_h_c = int(learnable_pos_h_c)

        self.has_two_inputs = False
        self.in_channels = None
        self.proj_norm=proj_norm

        self.bn = layers.BatchNormalization(axis=-1, dtype=dtype, name="bn")
        self.act = layers.Activation("silu", name="silu")

        self.head_a = None
        self.head_b = None
        self.res_weight = None

        self._real_dtype = tf.as_dtype(dtype).real_dtype
        self._pos_enc = None
        self._f_static = None
        self._w_static = None
        self.pos_enc_weight = None
        self.pos_enc_v_weight = None
        self.pos_enc_h_weight = None
        self._pos_v_num_repeats = None

        self.trunk = self._make_block(name=f"{self.name}_trunk")

    def _make_block(self, name):
        bt = self.block_type
        cfg = dict(self.block_config)

        if isinstance(bt, str):
            bt_key = bt.lower()
        else:
            bt_key = bt

        if bt_key in ("conv", Conv):
            return Conv(
                filters_in=None,
                filters=cfg.get("filters", 8),
                kernel_size=cfg.get("kernel_size", 3),
                strides=cfg.get("strides", cfg.get("stride", 1)),
                p=cfg.get("p", "same"),
                groups=cfg.get("groups", 1),
                dilation=cfg.get("dilation", 1),
                act=cfg.get("act", "relu"),
                norm=cfg.get("norm", None),
                kernel_init=cfg.get("kernel_init", None),
                name=name,
                dtype=self.dtype,
            )

        elif bt_key in ("mdelan", MDELAN):
            light = cfg.get("light", None)
            return MDELAN(
                filters=cfg.get("filters", 8),
                min_filters=cfg.get("min_filters", 4),
                dilation=cfg.get("dilation", (2, 4, 8)),
                m_filters=cfg.get("m_filters", None),
                expansion=cfg.get("expansion", 0.5),
                groups=cfg.get("groups", 1),
                kernel_size=cfg.get("kernel_size", 3),
                split=cfg.get("split", (not light) if light is not None else False),
                cv1asmlp=cfg.get("cv1asmlp", (not light) if light is not None else False),
                shortcut=cfg.get("shortcut", False),
                pre_act=cfg.get("pre_act", False),
                cv2asProj=cfg.get("cv2asProj", False),
                shortcut_weighted=cfg.get("shortcut_weighted", False),
                pos_prbs=cfg.get("pos_prbs", 0),
                name=name,
                dtype=self.dtype,
            )

        elif "Bottleneck" in globals() and bt_key in ("bottleneck", Bottleneck):
            k = cfg.get("kernel_size", 3)
            ks_tuple = ((k, k), (k, k)) if not isinstance(k, (tuple, list)) else k
            return Bottleneck(
                None,
                cfg.get("filters", 8),
                shortcut=cfg.get("shortcut", False),
                groups=cfg.get("groups", 1),
                kernel_size=ks_tuple,
                expansion=cfg.get("expansion", 1.0),
                name=name,
                dtype=self.dtype,
            )

        else:
            raise ValueError(f"Unsupported block type: {self.block_type}")

    def _make_pos_encoding(self, f_dim, w_dim):
        period = max(12 * self.pos_prbs, 1)
        period_f = tf.cast(period, self._real_dtype)
        denom_f = tf.cast(max(period - 1, 1), self._real_dtype)
        denom_w = tf.cast(max(w_dim - 1, 1), self._real_dtype)

        f_idx = tf.cast(tf.range(f_dim), self._real_dtype)
        w_idx = tf.cast(tf.range(w_dim), self._real_dtype)

        pos_v = tf.math.floormod(f_idx, period_f) / denom_f
        pos_h = w_idx / denom_w

        pos_v = tf.reshape(pos_v, [1, f_dim, 1, 1])
        pos_h = tf.reshape(pos_h, [1, 1, w_dim, 1])

        pos_v = tf.broadcast_to(pos_v, [1, f_dim, w_dim, 1])
        pos_h = tf.broadcast_to(pos_h, [1, f_dim, w_dim, 1])

        return tf.concat([pos_v, pos_h], axis=-1)

    def _concat_learnable_pos(self, x):
        n = tf.shape(x)[0]
        f, w = self._f_static, self._w_static

        pos_v = tf.cast(self.pos_enc_weight, x.dtype)

        if self.learnable_pos_c > 0:
            pos_v = tf.tile(pos_v, [self._pos_num_repeats, 1])[:f, :]
            pos_v = tf.reshape(pos_v, [1, f, 1, self.learnable_pos_c])
            pos_v = tf.broadcast_to(pos_v, [n, f, w, self.learnable_pos_c])
        else:
            pos_v = tf.tile(pos_v, [self._pos_num_repeats])[:f]
            pos_v = tf.reshape(pos_v, [1, f, 1, 1])
            pos_v = tf.broadcast_to(pos_v, [n, f, w, 1])

        return tf.concat([x, pos_v], axis=-1)

    def _concat_pos_v(self, x):
        n = tf.shape(x)[0]
        f, w = self._f_static, self._w_static
        c_v = max(self.learnable_pos_v_c, 1)

        w_v = tf.cast(self.pos_enc_v_weight, x.dtype)
        w_v = tf.tile(w_v, [self._pos_v_num_repeats, 1])[:f, :]
        w_v = tf.reshape(w_v, [1, f, 1, c_v])
        w_v = tf.broadcast_to(w_v, [n, f, w, c_v])

        return tf.concat([x, w_v], axis=-1)

    def _concat_pos_h(self, x):
        n = tf.shape(x)[0]
        f, w = self._f_static, self._w_static

        w_h = tf.cast(self.pos_enc_h_weight, x.dtype)
        w_h = tf.reshape(w_h, [1, 1, w, self.learnable_pos_h_c])
        w_h = tf.broadcast_to(w_h, [n, f, w, self.learnable_pos_h_c])

        return tf.concat([x, w_h], axis=-1)

    def _concat_pos(self, x):
        if self.learnable_pos:
            return self._concat_learnable_pos(x)

        x_shape = tf.shape(x)
        n, f, w = x_shape[0], x_shape[1], x_shape[2]

        if self._pos_enc is not None:
            pos = tf.broadcast_to(self._pos_enc, [n, self._f_static, self._w_static, 2])
        else:
            period = tf.cast(max(12 * self.pos_prbs, 1), self._real_dtype)
            denom_f = tf.cast(max(12 * self.pos_prbs - 1, 1), self._real_dtype)
            denom_w = tf.cast(tf.maximum(w - 1, 1), self._real_dtype)

            f_idx = tf.cast(tf.range(f), self._real_dtype)
            w_idx = tf.cast(tf.range(w), self._real_dtype)

            pos_v = tf.math.floormod(f_idx, period) / denom_f
            pos_h = w_idx / denom_w

            pos_v = tf.reshape(pos_v, [1, f, 1, 1])
            pos_h = tf.reshape(pos_h, [1, 1, w, 1])

            pos_v = tf.broadcast_to(pos_v, [n, f, w, 1])
            pos_h = tf.broadcast_to(pos_h, [n, f, w, 1])

            pos = tf.concat([pos_v, pos_h], axis=-1)

        pos = tf.cast(pos, x.dtype)
        return tf.concat([x, pos], axis=-1)

    def build(self, input_shape):
        x_shape = input_shape
        self.in_channels = int(x_shape[-1])
        f_dim = x_shape[-3]
        w_dim = x_shape[-2]

        self.head_a = Conv(
            filters_in=None,
            filters=self.in_channels,
            kernel_size=3,
            strides=1,
            p="same",
            groups=-1,
            act=False,
            norm=self.proj_norm,
            name="head_a",
            dtype=self.dtype,
        )

        if self.use_res_weight:
            self.res_weight = self.add_weight(
                name="res_weight",
                shape=(12, int(w_dim)),
                initializer=tf.keras.initializers.Constant(1.e-5),
                trainable=True,
                dtype=self.dtype,
            )

        if f_dim is not None and w_dim is not None:
            self._f_static = int(f_dim)
            self._w_static = int(w_dim)
            self._pos_enc = self._make_pos_encoding(self._f_static, self._w_static)

        if self.learnable_pos and self.pos_prbs > 0:
            period = max(12 * self.pos_prbs, 1)
            if self.learnable_pos_c > 0:
                self.pos_enc_weight = self.add_weight(
                    name="pos_enc_weight",
                    shape=(period, self.learnable_pos_c),
                    initializer="zeros",
                    trainable=True,
                    dtype=self._real_dtype,
                )
            else:
                self.pos_enc_weight = self.add_weight(
                    name="pos_enc_weight",
                    shape=(period,),
                    initializer="zeros",
                    trainable=True,
                    dtype=self._real_dtype,
                )
            if f_dim is not None:
                self._pos_num_repeats = math.ceil(self._f_static / period)

        if self.pos_prbs_v > 0:
            period_v = 12 * self.pos_prbs_v
            c_v = max(self.learnable_pos_v_c, 1)
            self.pos_enc_v_weight = self.add_weight(
                name="pos_enc_v_weight",
                shape=(period_v, c_v),
                initializer="zeros",
                trainable=True,
                dtype=self._real_dtype,
            )
            if f_dim is not None:
                self._pos_v_num_repeats = math.ceil(self._f_static / period_v)

        if self.learnable_pos_h_c > 0 and w_dim is not None:
            self.pos_enc_h_weight = self.add_weight(
                name="pos_enc_h_weight",
                shape=(int(w_dim), self.learnable_pos_h_c),
                initializer="zeros",
                trainable=True,
                dtype=self._real_dtype,
            )

        super().build(input_shape)

    def _apply_res_weight(self, x):
        if self.res_weight is None:
            return x

        x_shape = tf.shape(x)
        n, f, w, c = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        tf.debugging.assert_equal(
            tf.math.floormod(f, 12), 0,
            message="ResBlock expects F to be divisible by 12 when use_res_weight=True."
        )

        x = tf.reshape(x, [n, f // 12, 12, w, c])
        wgt = tf.reshape(self.res_weight, [1, 1, 12, w, 1])
        x = x * wgt
        x = tf.reshape(x, [n, f, w, c])
        return x

    def call(self, inputs, mask=None, training=False, pre_act=True):
        x0 = inputs

        if pre_act:
            x = self.bn(x0, training=training)
            x = self.act(x)
        else:
            x = x0

        if self.pos_prbs > 0:
            x = self._concat_pos(x)
        if self.pos_prbs_v > 0:
            x = self._concat_pos_v(x)
        if self.learnable_pos_h_c > 0:
            x = self._concat_pos_h(x)

        x = self.trunk(x, mask=mask, training=training)
        dx = self.head_a(x, mask=mask, training=training)
        dx = self._apply_res_weight(dx)

        return x0 + dx

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "block_type": self.block_type,
            "block_config": self.block_config,
            "use_res_weight": self.use_res_weight,
            "pos_prbs": self.pos_prbs,
            "learnable_pos": self.learnable_pos,
            "learnable_pos_c": self.learnable_pos_c,
            "pos_prbs_v": self.pos_prbs_v,
            "learnable_pos_v_c": self.learnable_pos_v_c,
            "learnable_pos_h_c": self.learnable_pos_h_c,
        })
        return cfg


@register_block
class ResNet(layers.Layer):
    """Stack of ``ResBlock`` layers.

    ``block_type`` and ``block_config`` can be shared by all blocks or given
    per block. The remaining positional-encoding arguments and ``proj_norm``
    are passed to every ``ResBlock``.

    Args:
        num_blocks: Number of residual blocks in the stack.
        block_type: Shared block type for all blocks, or a list/tuple with one
            entry per block.
        block_config: Shared block configuration dictionary, or a list/tuple with
            one dictionary per block.
        use_res_weight: Enables the learnable per-PRB residual weights.
        pos_prbs: Positional encoding period control passed to each block.
        pre_act: Pre-activation flag used by blocks after the first one.
        pre_act_first_block: Pre-activation flag used by the first block.
    """

    def __init__(
        self,
        num_blocks=2,
        block_type="conv",
        block_config=None,
        use_res_weight=True,
        pos_prbs=3,
        learnable_pos=False,
        learnable_pos_c=0,
        pos_prbs_v=0,
        learnable_pos_v_c=1,
        learnable_pos_h_c=0,
        pre_act=True,
        pre_act_first_block=False,
        proj_norm=False,
        name="ResNet",
        dtype=tf.float32,
        **kwargs
    ):
        super().__init__(name=name, dtype=dtype , **kwargs)
        self._name = name

        self.num_blocks = int(num_blocks)
        self.use_res_weight = use_res_weight
        self.pos_prbs = int(pos_prbs)
        self.learnable_pos = bool(learnable_pos)
        self.learnable_pos_c = int(learnable_pos_c)
        self.pos_prbs_v = int(pos_prbs_v)
        self.learnable_pos_v_c = int(learnable_pos_v_c)
        self.learnable_pos_h_c = int(learnable_pos_h_c)
        self.pre_act = pre_act
        self.pre_act_first_block = pre_act_first_block

        if block_config is None:
            block_config = {
                "filters": 8,
                "kernel_size": 3,
                "groups": -1,
                "act": "relu",
                "norm": None,
            }

        self.block_type = block_type
        self.block_config = block_config

        self.block_types = self._normalize_per_block_arg(
            value=block_type,
            name="block_type",
        )
        self.block_configs = self._normalize_per_block_arg(
            value=block_config,
            name="block_config",
            item_cast=lambda x: dict(x or {}),
        )

        self.blocks = [
            ResBlock(
                name=f"block_{i}",
                block_type=self.block_types[i],
                block_config=self.block_configs[i],
                use_res_weight=self.use_res_weight,
                pos_prbs=self.pos_prbs,
                learnable_pos=self.learnable_pos,
                learnable_pos_c=self.learnable_pos_c,
                pos_prbs_v=self.pos_prbs_v,
                learnable_pos_v_c=self.learnable_pos_v_c,
                learnable_pos_h_c=self.learnable_pos_h_c,
                proj_norm=proj_norm,
                dtype=dtype,
            )
            for i in range(self.num_blocks)
        ]

    def _normalize_per_block_arg(self, value, name, item_cast=None):
        if isinstance(value, (list, tuple)):
            if len(value) != self.num_blocks:
                raise ValueError(
                    f"{name} must have length {self.num_blocks}, got {len(value)}."
                )
            items = list(value)
        else:
            items = [value for _ in range(self.num_blocks)]

        if item_cast is not None:
            items = [item_cast(v) for v in items]

        return items

    def build(self, input_shape):
        x_shape = input_shape
        self.in_channels = int(x_shape[-1])

        super().build(input_shape)

    def call(self, inputs, mask=None, training=False):
        x = inputs

        for i, block in enumerate(self.blocks):
            pre_act_ = self.pre_act_first_block if i == 0 else self.pre_act
            x = block(x, mask=mask, training=training, pre_act=pre_act_)

        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_blocks": self.num_blocks,
            "block_type": self.block_type,
            "block_config": self.block_config,
            "use_res_weight": self.use_res_weight,
            "pos_prbs": self.pos_prbs,
            "learnable_pos": self.learnable_pos,
            "learnable_pos_c": self.learnable_pos_c,
            "pos_prbs_v": self.pos_prbs_v,
            "learnable_pos_v_c": self.learnable_pos_v_c,
            "learnable_pos_h_c": self.learnable_pos_h_c,
            "pre_act": self.pre_act,
            "pre_act_first_block": self.pre_act_first_block,
        })
        return cfg


__all__ = [
    "ResBlock",
    "ResNet",
]
