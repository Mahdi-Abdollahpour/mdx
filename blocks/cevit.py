"""
CEViT: channel-estimation vision transformer for OFDM systems (baseline).

TensorFlow re-implementation of the CEViT network described in Section III.A of
  Liu, Fangyu, et al. "PD-CEViT: A novel pilot pattern design and channel
  estimation network for OFDM systems." IEEE Transactions on Communications
  73.6 (2024): 4363-4377.
It is used as a baseline in the CHEA paper (arXiv:2607.16462).

The input is the bilinearly interpolated LS estimate `H_interp` of shape
[B, Nf, Nt, 2] (last axis = [real, imag]); the output is the refined estimate
with the same shape. The grid is cut into Nh x Nw patches (real and imaginary
parts as separate patches), embedded, passed through one pre-LN transformer
block and projected back to the grid. Optional SNR / Doppler / delay-spread
tokens are used when a [B, 3] channel-info tensor is passed as second input.

Without channel info and with the defaults (Nh=12, Nw=7, d_model=128, 4 heads)
the block has 203,476 + 128*Nf/3 trainable parameters (206,548 for Nf=72) and
256*Nf^2/9 + 201,728*Nf/3 MACs per sample (~5.0M for Nf=72).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .registry import register_block


def _extract_patches(x, Nf, Nt, Nh, Nw):
    """[B, Nf, Nt, 2] -> [B, p, Nh*Nw] with p = (Nf//Nh) * (Nt//Nw) * 2.

    Real and imaginary patches are interleaved in the patch index.
    """
    B   = tf.shape(x)[0]
    p_f = Nf // Nh
    p_t = Nt // Nw
    x = tf.reshape(x, [B, p_f, Nh, p_t, Nw, 2])
    x = tf.transpose(x, [0, 1, 3, 5, 2, 4])          # [B, p_f, p_t, 2, Nh, Nw]
    x = tf.reshape(x, [B, p_f * p_t * 2, Nh * Nw])
    return x


def _merge_patches(x, Nf, Nt, Nh, Nw):
    """[B, p, Nh*Nw] -> [B, Nf, Nt, 2]; inverse of `_extract_patches`."""
    B   = tf.shape(x)[0]
    p_f = Nf // Nh
    p_t = Nt // Nw
    x = tf.reshape(x, [B, p_f, p_t, 2, Nh, Nw])
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])  # [B, p_f, Nh, p_t, Nw, 2]
    x = tf.reshape(x, [B, Nf, Nt, 2])
    return x


class PosEncoding(layers.Layer):
    """Learnable additive positional encoding for a sequence [B, p, d]."""

    def __init__(self, p, d_model, **kwargs):
        super().__init__(**kwargs)
        self._p      = p
        self._d      = d_model

    def build(self, input_shape):
        self.embed = self.add_weight(
            name='embed',
            shape=(1, self._p, self._d),
            initializer='zeros',
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.embed


class MultiHeadSelfAttention(layers.Layer):
    """Multi-head scaled dot-product self-attention on [B, p, d_model]."""

    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = tf.cast(self.d_head, tf.float32) ** -0.5
        self.qkv     = layers.Dense(3 * d_model)

    def call(self, x):
        B = tf.shape(x)[0]
        p = tf.shape(x)[1]

        qkv = self.qkv(x)
        q, k, v = tf.split(qkv, 3, axis=-1)                 # each [B, p, d]

        def split_heads(t):
            t = tf.reshape(t, [B, p, self.n_heads, self.d_head])
            return tf.transpose(t, [0, 2, 1, 3])             # [B, H, p, d_head]

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale  # [B, H, p, p]
        attn = tf.nn.softmax(attn, axis=-1)
        out  = tf.matmul(attn, v)

        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, p, self.d_model])           # [B, p, d]
        return out


class FFN(layers.Layer):
    """Three-layer feed-forward network: d -> 2d (GELU) -> 2d (GELU) -> d."""

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(2 * d_model)
        self.fc2 = layers.Dense(2 * d_model)
        self.fc3 = layers.Dense(d_model)

    def call(self, x):
        x = tf.keras.activations.gelu(self.fc1(x), approximate=True)
        x = tf.keras.activations.gelu(self.fc2(x), approximate=True)
        return self.fc3(x)


class TransformerBlock(layers.Layer):
    """Pre-LN transformer encoder block on [B, p, d_model].

    x = x + Attention(LayerNorm(x)); x = x + FFN(LayerNorm(x)).
    """

    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__(**kwargs)
        self.ln1  = layers.LayerNormalization()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2  = layers.LayerNormalization()
        self.ffn  = FFN(d_model)

    def call(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


@register_block
class CEViT(tf.keras.Model):
    """Channel-estimation vision transformer (CEViT baseline, Liu et al. 2024).

    Inputs:
        H_interp     : float32 [B, Nf, Nt, 2], bilinearly interpolated LS estimate,
                       given alone or as `(H_interp,)`.
        channel_info : optional float32 [B, 3], [SNR, max_doppler, delay_spread];
                       pass `(H_interp, channel_info)` to enable the token module.

    Output:
        H_hat        : float32 [B, Nf, Nt, 2], refined channel estimate.

    Args:
        Nh      : patch height; must divide Nf (default 12).
        Nw      : patch width; must divide Nt (default 7).
        d_model : transformer embedding dimension (default 128).
        n_heads : number of attention heads (default 4).
    """

    def __init__(
        self,
        Nh      = 12,
        Nw      = 7,
        d_model = 128,
        n_heads = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.Nf      = None
        self.Nt      = None
        self.Nh      = Nh
        self.Nw      = Nw
        self.d_model = d_model
        self.n_heads = n_heads
        self.p         = None
        self.patch_size = Nh * Nw

        self.snr_fc      = None
        self.doppler_fc  = None
        self.delay_fc    = None
        self.input_proj  = None
        self.pos_enc     = None
        self.transformer = TransformerBlock(d_model, n_heads, name='transformer')
        self.output_proj = layers.Dense(self.patch_size, name='output_proj')
        self._use_channel_info = None

    @staticmethod
    def _has_channel_info_input(input_shape):
        return (
            isinstance(input_shape, (list, tuple))
            and len(input_shape) == 2
            and isinstance(input_shape[0], (list, tuple, tf.TensorShape))
            and isinstance(input_shape[1], (list, tuple, tf.TensorShape))
            and tf.TensorShape(input_shape[1]).rank is not None
        )

    def build(self, input_shape):
        self._use_channel_info = self._has_channel_info_input(input_shape)
        if self._use_channel_info:
            h_shape = tf.TensorShape(input_shape[0])
        elif (
            isinstance(input_shape, (list, tuple))
            and len(input_shape) == 1
            and isinstance(input_shape[0], (list, tuple, tf.TensorShape))
        ):
            h_shape = tf.TensorShape(input_shape[0])
        else:
            h_shape = tf.TensorShape(input_shape)

        if h_shape.rank != 4:
            raise ValueError(f"H_interp must have rank 4 [B, Nf, Nt, 2], got {h_shape}")

        Nf, Nt, channels = h_shape[1], h_shape[2], h_shape[3]
        if Nf is None or Nt is None:
            raise ValueError(f"H_interp Nf and Nt must be statically known, got {h_shape}")
        if channels is not None and channels != 2:
            raise ValueError(f"H_interp last dimension must be 2, got {channels}")

        Nf = int(Nf)
        Nt = int(Nt)
        if Nf % self.Nh != 0:
            raise ValueError(f"Nf={Nf} must be divisible by Nh={self.Nh}")
        if Nt % self.Nw != 0:
            raise ValueError(f"Nt={Nt} must be divisible by Nw={self.Nw}")

        self.Nf = Nf
        self.Nt = Nt
        self.p = (Nf // self.Nh) * (Nt // self.Nw) * 2

        if self._use_channel_info:
            self.snr_fc     = layers.Dense(self.p * 2, name='snr_fc')
            self.doppler_fc = layers.Dense(self.p * 2, name='doppler_fc')
            self.delay_fc   = layers.Dense(self.p * 2, name='delay_fc')
        self.input_proj = layers.Dense(self.d_model, name='input_proj')
        self.pos_enc = PosEncoding(self.p, self.d_model, name='pos_enc')

        super().build(input_shape)

    def call(self, inputs, training=False):
        if self._use_channel_info:
            H_interp, channel_info = inputs
        else:
            H_interp = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        B = tf.shape(H_interp)[0]

        patches = _extract_patches(H_interp, self.Nf, self.Nt, self.Nh, self.Nw)  # [B, p, Nh*Nw]

        if self._use_channel_info:
            # each channel parameter: [B, 1] -> Dense(2p) -> [B, p, 2]
            snr_tok     = tf.reshape(self.snr_fc(channel_info[:, 0:1]),     [B, self.p, 2])
            doppler_tok = tf.reshape(self.doppler_fc(channel_info[:, 1:2]), [B, self.p, 2])
            delay_tok   = tf.reshape(self.delay_fc(channel_info[:, 2:3]),   [B, self.p, 2])
            tokens = tf.concat([snr_tok, doppler_tok, delay_tok], axis=-1)  # [B, p, 6]

            x = tf.concat([patches, tokens], axis=-1)  # [B, p, Nh*Nw + 6]
        else:
            x = patches

        x = self.input_proj(x)   # [B, p, d_model]
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.output_proj(x)  # [B, p, Nh*Nw]

        H_hat = _merge_patches(x, self.Nf, self.Nt, self.Nh, self.Nw)  # [B, Nf, Nt, 2]
        return H_hat
