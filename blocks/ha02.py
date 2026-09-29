"""
HA02: hybrid attention channel estimator (baseline).

TensorFlow re-implementation of the attention-aided channel estimator "HA02"
of D. Luan and J. Thompson (2022), following the authors' MATLAB reference
code. It is used as a baseline in the CHEA paper (arXiv:2607.16462).

The LS pilot estimates (every other subcarrier on two pilot OFDM symbols) are
reshaped to [B, F, 2, 1], where F is the number of pilot values and the
length-2 axis holds [real, imag]. A transformer encoder treats real/imag as a
length-2 sequence of F-dimensional features. A small CNN decoder (2x2 convs
with 2 channels and one residual block), a dense expansion from F to F*S along
the frequency axis and an output conv then produce the full-grid estimate
[B, F, S, 2].

For F=72 and S=14 the model has 20F^2 + 26F + 55 = 105,607 trainable
parameters and ~375K MACs per sample.
"""

from __future__ import annotations
import math
import tensorflow as tf
from tensorflow.keras import layers, Model

from .registry import register_block

# Defaults of the reference implementation (72 subcarriers, 14 OFDM symbols).
N_PILOT    = 2
NF         = 72
F          = N_PILOT * NF // 2
SEQ        = 2
H_HEADS    = 2
HEAD_DIM   = F // H_HEADS
ENC_LAYERS = 1
DEC_LAYERS = 1
FILTERS    = 2
KERNEL     = (2, 2)
T_FRAME    = 14
O          = NF * T_FRAME

LR_INIT         = 2e-3
LR_DROP_EPOCHS  = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # 1-indexed epochs
LR_DROP_RATE    = 0.5
L2_REG          = 1e-10
BATCH_SIZE      = 128
NUM_EPOCHS      = 100
HUBER_DELTA     = 1.0


class MatlabGlorot(tf.keras.initializers.Initializer):
    """Glorot-uniform initializer with explicit fan values, as in the reference code.

    Samples from U(-b, b) with b = sqrt(6 / (num_in + num_out)).
    """

    def __init__(self, num_out: int, num_in: int):
        self.num_out = int(num_out)
        self.num_in  = int(num_in)

    def __call__(self, shape, dtype=None):
        dtype = tf.as_dtype(dtype or tf.float32)
        bound = math.sqrt(6.0 / (self.num_in + self.num_out))
        return tf.random.uniform(shape, minval=-bound, maxval=bound,
                                 dtype=dtype)

    def get_config(self):
        return {"num_out": self.num_out, "num_in": self.num_in}


def make_layer_norm(norm_dim: int, axis: int, name: str) -> layers.LayerNormalization:
    """LayerNorm with epsilon 1e-5 and a Glorot-initialized gamma, as in the reference code."""
    return layers.LayerNormalization(
        axis=axis,
        epsilon=1e-5,
        gamma_initializer=MatlabGlorot(norm_dim, norm_dim),
        beta_initializer="zeros",
        name=name,
    )


class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention on [B, seq, d_model] with a fused QKV projection."""

    def __init__(self, d_model: int = F, num_heads: int = H_HEADS, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0
        self.d_model   = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim  = d_model // num_heads

        self.qkv_proj = layers.Dense(
            3 * d_model, use_bias=True,
            kernel_initializer=MatlabGlorot(3 * d_model * d_model,
                                            d_model * d_model),
            bias_initializer="zeros", name="qkv_proj")

        self.out_proj = layers.Dense(
            d_model, use_bias=True,
            kernel_initializer=MatlabGlorot(d_model * d_model,
                                            d_model * d_model),
            bias_initializer="zeros", name="out_proj")

    def call(self, x):
        # x: [B, seq, F]
        B = tf.shape(x)[0]
        S = tf.shape(x)[1]
        H = self.num_heads
        D = self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = tf.split(qkv, 3, axis=-1)                            # each [B, seq, F]

        def split_heads(t):
            t = tf.reshape(t, [B, S, H, D])
            return tf.transpose(t, [0, 2, 1, 3])                        # [B, H, seq, D]

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scores  = tf.matmul(q, k, transpose_b=True)                     # [B, H, seq, seq]
        scores  = scores / tf.cast(D, tf.float32) ** 0.5
        weights = tf.nn.softmax(scores, axis=-1)
        attn    = tf.matmul(weights, v)                                  # [B, H, seq, D]

        attn = tf.transpose(attn, [0, 2, 1, 3])
        attn = tf.reshape(attn, [B, S, self.d_model])                   # [B, seq, F]

        return self.out_proj(attn)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return cfg


class FeedforwardNN(layers.Layer):
    """Feed-forward network: Dense(F) -> tanh-approximate GELU -> Dense(F)."""

    def __init__(self, d_model: int = F, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        init = MatlabGlorot(d_model * d_model, d_model * d_model)

        self.fc1 = layers.Dense(d_model, use_bias=True,
                                kernel_initializer=init,
                                bias_initializer="zeros", name="fc1")
        self.fc2 = layers.Dense(d_model, use_bias=True,
                                kernel_initializer=init,
                                bias_initializer="zeros", name="fc2")

    def call(self, x):
        x = self.fc1(x)
        x = tf.nn.gelu(x, approximate=True)
        return self.fc2(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model})
        return cfg


class EncoderBlock(layers.Layer):
    """Post-LN transformer encoder block on [B, seq, d_model]."""

    def __init__(self, d_model: int = F, num_heads: int = H_HEADS, **kwargs):
        super().__init__(**kwargs)
        self.d_model   = int(d_model)
        self.num_heads = int(num_heads)

        self.attn  = MultiHeadSelfAttention(d_model, num_heads, name="attn")
        self.ffn   = FeedforwardNN(d_model, name="ffn")
        self.norm1 = make_layer_norm(d_model, axis=-1, name="norm1")
        self.norm2 = make_layer_norm(d_model, axis=-1, name="norm2")

    def call(self, x):
        a = self.attn(x) + x
        a = self.norm1(a)
        z = self.ffn(a) + a
        return self.norm2(z)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return cfg


class DecoderBlock(layers.Layer):
    """Residual CNN block: conv, ReLU, conv, residual, LayerNorm over the frequency axis.

    Input / output shape: [B, F, seq, filters].
    """

    def __init__(self, height: int = F, filters: int = FILTERS,
                 kernel: tuple = KERNEL, **kwargs):
        super().__init__(**kwargs)
        self.height  = int(height)
        self.filters = int(filters)
        self.kernel  = kernel

        area = kernel[0] * kernel[1]
        init = MatlabGlorot(area * filters, area * filters)

        self.conv1 = layers.Conv2D(filters, kernel, padding="same",
                                   use_bias=True, kernel_initializer=init,
                                   bias_initializer="zeros", name="conv1")
        self.conv2 = layers.Conv2D(filters, kernel, padding="same",
                                   use_bias=True, kernel_initializer=init,
                                   bias_initializer="zeros", name="conv2")
        self.norm  = make_layer_norm(height, axis=1, name="norm")

    def call(self, x):
        y = self.conv1(x)
        y = tf.nn.relu(y)
        y = self.conv2(y)
        y = y + x
        return self.norm(y)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"height": self.height, "filters": self.filters,
                    "kernel": self.kernel})
        return cfg


@register_block
class HA02(layers.Layer):
    """HA02 attention-aided channel estimator (Luan & Thompson, 2022).

    Args:
        d_model: feature size; set in build() to the input frequency dimension F.
        num_heads: attention heads (default 2).
        enc_layers: transformer encoder blocks (default 1).
        dec_layers: residual decoder blocks (default 1).
        filters: decoder channels (default 2).
        kernel: decoder conv kernel (default (2, 2)).
        output_size: size of the dense expansion; set in build() to F*S.

    Input / output shape: [B, F, S, 2] (LS estimate on the full grid ->
    channel estimate). Pilots are taken from every other subcarrier on OFDM
    symbols 2 and 11.
    """

    def __init__(self,
                 d_model:     int   = F,
                 num_heads:   int   = H_HEADS,
                 enc_layers:  int   = ENC_LAYERS,
                 dec_layers:  int   = DEC_LAYERS,
                 filters:     int   = FILTERS,
                 kernel:      tuple = KERNEL,
                 output_size: int   = O,
                 name: str = "ha02",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_model     = None if d_model is None else int(d_model)
        self.num_heads   = int(num_heads)
        self.enc_layers  = int(enc_layers)
        self.dec_layers  = int(dec_layers)
        self.filters     = int(filters)
        self.kernel      = kernel
        self.output_size = int(output_size)

        area = kernel[0] * kernel[1]

        self.encoder_blocks = []

        self.dec_init_conv = layers.Conv2D(
            filters, kernel, padding="same", use_bias=True,
            kernel_initializer=MatlabGlorot(area * filters, area * filters),
            bias_initializer="zeros", name="dec_init_conv")

        self.decoder_blocks = []
        self.fc_expand = None

        self.final_conv = layers.Conv2D(
            1, kernel, padding="same", use_bias=True,
            kernel_initializer=MatlabGlorot(area * 1, area * 1),
            bias_initializer="zeros", name="final_conv")

    def build(self, input_shape):
        full_freq_dim = input_shape[1]
        self.d_model = int(full_freq_dim)
        assert self.d_model % self.num_heads == 0
        self.sym_dim = int(input_shape[2])
        self._full_freq_dim = int(full_freq_dim)
        self.output_size = self._full_freq_dim * self.sym_dim
        self._pilot_subcarrier_idx = tf.range(0, self._full_freq_dim, delta=2)
        self._pilot_symbol_idx = tf.constant([2, 11], dtype=tf.int32)
        self._pilot_reshape = (-1, self._full_freq_dim, 2, 1)

        self.encoder_blocks = [
            EncoderBlock(self.d_model, self.num_heads, name=f"encoder_{i}")
            for i in range(self.enc_layers)
        ]

        self.decoder_blocks = [
            DecoderBlock(self.d_model, self.filters, self.kernel, name=f"decoder_{i}")
            for i in range(self.dec_layers)
        ]

        self.fc_expand = layers.Dense(
            self.output_size, use_bias=True,
            kernel_initializer=MatlabGlorot(self.output_size * self.d_model,
                                            self.output_size * self.d_model),
            bias_initializer="zeros", name="fc_expand")
        super().build(input_shape)

    def _from_grid(self,x):
        # x: [B, F, S, 2]
        x = tf.gather(x, self._pilot_subcarrier_idx, axis=1)
        x = tf.gather(x, self._pilot_symbol_idx, axis=2)                # [B, F/2, 2, 2]
        x = tf.reshape(x, self._pilot_reshape)                          # [B, F, 2, 1]
        return x

    def _to_grid(self, x):
        # [B, F*S, 2, 1] -> [B, F, S, 2]
        x = tf.reshape(x, [-1, self._full_freq_dim, self.sym_dim, 2])
        return x

    def call(self, x):
        x = self._from_grid(x)

        z = tf.squeeze(x, axis=-1)
        z = tf.transpose(z, [0, 2, 1])                                   # [B, 2, F]: real/imag as sequence

        for block in self.encoder_blocks:
            z = block(z)

        z = tf.transpose(z, [0, 2, 1])
        z = tf.expand_dims(z, axis=-1)

        z = self.dec_init_conv(z)                                        # [B, F, 2, filters]

        for block in self.decoder_blocks:
            z = block(z)

        # dense expansion along the frequency axis: F -> F*S
        z = tf.transpose(z, [0, 2, 3, 1])
        z = self.fc_expand(z)                                            # [B, 2, filters, F*S]
        z = tf.transpose(z, [0, 3, 1, 2])                                # [B, F*S, 2, filters]

        z = self.final_conv(z)                                           # [B, F*S, 2, 1]
        z = self._to_grid(z)
        return z

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "output_size": self.output_size})
        return cfg
