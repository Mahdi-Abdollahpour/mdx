# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026

"""CHEA: Channel Estimation Attention.

Scalable windowed multi-resolution attention channel estimator for 5G NR
PUSCH (arXiv 2607.16462).

Input and output are resource grids of shape [B, F, S, 2] (F subcarriers,
S=14 OFDM symbols, real/imag). With the default configuration
(``decoder_type='lr_cross_attn'``, ``upsample_method='I'``):

1. Pilot extraction: the pilot OFDM symbols ``pilot_syms`` are gathered,
   giving a pilot grid of shape [B, F, 2, 2].
2. Tokenization: one token per PRB (12 subcarriers) per (pilot symbol, re/im).
3. High-resolution encoder: self-attention within windows of ``hr_prbs`` PRBs.
4. Low-resolution encoder: masked average pooling over groups of
   ``pool_factor`` PRBs, then self-attention within windows of ``lr_prbs`` PRBs.
5. Local cross-attention decoder: queries are the high-resolution tokens of a
   PRB group, keys/values the low-resolution tokens of the same group.
6. Upsampling: a per-PRB linear map from the pilot symbols to the full slot.

``CHEAStack`` chains several pilot-grid refinement stages (``CHEAStage``) with
trainable residual scales and a single final upsampling.

F must be a multiple of 12; other sizes are zero-padded internally to a whole
number of windows and cropped on output. An optional PRB mask restricts the
estimator to a contiguous prefix of valid PRBs.
"""

from __future__ import annotations

import math

import tensorflow as tf
from tensorflow.keras import layers

from .cnn import Conv
from .registry import register_block


def _ceil_to_multiple(value: int, multiple: int) -> int:
    """Round ``value`` up to the nearest positive multiple."""
    assert multiple > 0, f"multiple must be positive, got {multiple}"
    return ((int(value) + multiple - 1) // multiple) * multiple


def _lcm(a: int, b: int) -> int:
    """Return the least common multiple of two positive integers."""
    a, b = int(a), int(b)
    assert a > 0 and b > 0, f"lcm inputs must be positive, got {a}, {b}"
    return abs(a * b) // math.gcd(a, b)


class PosEncoding(layers.Layer):
    """Learnable additive positional encoding of shape [1, seq_len, d_model].

    The embedding broadcasts over the leading axis, so it is shared across the
    batch and across all windows when applied to [B*N_win, seq_len, d_model].
    """

    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self._seq = seq_len
        self._d   = d_model

    def build(self, input_shape):
        self.embed = self.add_weight(
            name="embed",
            shape=(1, self._seq, self._d),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.embed


class _MHA(layers.Layer):
    """Scaled dot-product multi-head attention (self- or cross-attention)."""

    def __init__(self, d_model: int, n_heads: int, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = float(self.d_head) ** -0.5
        self.q_proj  = layers.Dense(d_model, use_bias=True, name="q")
        self.k_proj  = layers.Dense(d_model, use_bias=True, name="k")
        self.v_proj  = layers.Dense(d_model, use_bias=True, name="v")
        self.out     = layers.Dense(d_model, use_bias=True, name="out")

    def call(self, x, kv=None, q_mask=None, kv_mask=None):
        """Attend from ``x`` [B, Sq, d] to ``kv`` [B, Sk, d], or to ``x`` if ``kv`` is None.

        ``q_mask`` [B, Sq] and ``kv_mask`` [B, Sk] are optional 0/1 token masks.
        """
        src = x if kv is None else kv
        if kv is None and kv_mask is None:
            kv_mask = q_mask

        B   = tf.shape(x)[0]
        Sq  = tf.shape(x)[1]
        Sk  = tf.shape(src)[1]
        H, D = self.n_heads, self.d_head

        def split(t, S):
            t = tf.reshape(t, [B, S, H, D])
            return tf.transpose(t, [0, 2, 1, 3])      # [B, H, S, D]

        q = split(self.q_proj(x),   Sq)
        k = split(self.k_proj(src), Sk)
        v = split(self.v_proj(src), Sk)

        scores = tf.matmul(q, k, transpose_b=True) * self.scale  # [B, H, Sq, Sk]
        if kv_mask is not None:
            kv_mask_f = tf.cast(kv_mask[:, None, None, :], scores.dtype)
            scores = scores + (1.0 - kv_mask_f) * tf.cast(-1e9, scores.dtype)

        weights = tf.nn.softmax(scores, axis=-1)
        attn    = tf.matmul(weights, v)

        attn = tf.transpose(attn, [0, 2, 1, 3])
        attn = tf.reshape(attn, [B, Sq, self.d_model])
        out = self.out(attn)
        return _apply_token_mask(out, q_mask)


def _apply_token_mask(x, mask):
    """Zero masked tokens while keeping static tensor shapes unchanged."""
    if mask is None:
        return x
    return x * tf.cast(mask[..., None], x.dtype)


class _FFN(layers.Layer):
    """Position-wise feed-forward network d_model -> ffn_dim -> ... -> d_model (GELU)."""

    def __init__(self, d_model: int, ffn_dim: int, depth: int = 2, **kwargs):
        super().__init__(**kwargs)
        assert depth >= 2, f"ffn_depth must be >= 2, got {depth}"
        self._fcs = (
            [layers.Dense(ffn_dim, use_bias=True, name=f"fc{i+1}") for i in range(depth - 1)]
            + [layers.Dense(d_model, use_bias=True, name=f"fc{depth}")]
        )

    def call(self, x):
        for fc in self._fcs[:-1]:
            x = tf.keras.activations.gelu(fc(x), approximate=True)
        return self._fcs[-1](x)


class _EncoderBlock(layers.Layer):
    """Pre-LN transformer encoder block.

    x → LN → MHA(self) → + x
      → LN → FFN        → + x
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, ffn_depth: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.ln1  = layers.LayerNormalization(epsilon=1e-5, name="ln1")
        self.attn = _MHA(d_model, n_heads, name="attn")
        self.ln2  = layers.LayerNormalization(epsilon=1e-5, name="ln2")
        self.ffn  = _FFN(d_model, ffn_dim, depth=ffn_depth, name="ffn")

    def call(self, x, token_mask=None):
        x = _apply_token_mask(x, token_mask)
        x = x + self.attn(self.ln1(x), q_mask=token_mask, kv_mask=token_mask)

        x = _apply_token_mask(x, token_mask)
        x = x + self.ffn(self.ln2(x))
        return _apply_token_mask(x, token_mask)


class _DecoderBlock(layers.Layer):
    """Pre-LN transformer decoder block.

    x     → LN → Self-Attn(Q=K=V=x)        → + x
    (x,m) → LN → Cross-Attn(Q=x, K=V=m)   → + x
    x     → LN → FFN                        → + x
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, ffn_depth: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.ln1   = layers.LayerNormalization(epsilon=1e-5, name="ln1")
        self.self_attn  = _MHA(d_model, n_heads, name="self_attn")
        self.ln2   = layers.LayerNormalization(epsilon=1e-5, name="ln2")
        self.cross_attn = _MHA(d_model, n_heads, name="cross_attn")
        self.ln3   = layers.LayerNormalization(epsilon=1e-5, name="ln3")
        self.ffn   = _FFN(d_model, ffn_dim, depth=ffn_depth, name="ffn")

    def call(self, x, memory, q_mask=None, memory_mask=None):
        x = _apply_token_mask(x, q_mask)
        x = x + self.self_attn(self.ln1(x), q_mask=q_mask, kv_mask=q_mask)

        x = _apply_token_mask(x, q_mask)
        x = x + self.cross_attn(
            self.ln2(x), kv=memory, q_mask=q_mask, kv_mask=memory_mask
        )

        x = _apply_token_mask(x, q_mask)
        x = x + self.ffn(self.ln3(x))
        return _apply_token_mask(x, q_mask)


class TokenProcessor(layers.Layer):
    """Optional MLP applied to each flattened pilot patch before tokenization.

    Input and output have shape [B, N_win, N_p, P, 2, 2] (P = patch_size pilot
    subcarriers, 2 pilot symbols, re/im); the last three axes are flattened
    for the MLP.
    """

    def __init__(
        self,
        hidden_dims=(),
        activation: str = "gelu",
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hidden_dims is None:
            hidden_dims = ()
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        assert all(dim > 0 for dim in self.hidden_dims), (
            f"token_processor_hidden_dims must contain positive integers, got {hidden_dims}"
        )
        self.activation = activation
        self.use_bias = bool(use_bias)
        self._hidden = []
        self._out = None
        self._flat_dim = None

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self._flat_dim = int(input_shape[-3]) * int(input_shape[-2]) * int(input_shape[-1])
        self._hidden = [
            layers.Dense(dim, use_bias=self.use_bias, name=f"fc{i+1}")
            for i, dim in enumerate(self.hidden_dims)
        ]
        self._out = layers.Dense(self._flat_dim, use_bias=self.use_bias, name="out")
        super().build(input_shape)

    def call(self, x):
        input_shape = tf.shape(x)
        y = tf.reshape(x, [input_shape[0], input_shape[1], input_shape[2], self._flat_dim])
        for fc in self._hidden:
            y = tf.keras.activations.get(self.activation)(fc(y))
        y = self._out(y)
        return tf.reshape(y, input_shape)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "use_bias": self.use_bias,
        })
        return cfg


PILOT_PER_PRB = 6


@register_block
class CHEA(layers.Layer):
    """CHEA channel estimator: windowed multi-resolution attention.

    Maps a resource grid [B, F, S, 2] (e.g. an LS estimate) to a refined
    channel estimate of the same shape. The layer can also be called with
    ``(x, prb_mask)``, where ``prb_mask`` [B, F, S, 2] marks a contiguous
    prefix of valid PRBs.

    Args:
        d_model: Transformer embedding dimension.
        n_heads: Number of attention heads.
        enc_layers: Encoder blocks per branch (high and low resolution).
        dec_layers: Decoder blocks (unused for ``decoder_type='fc3'``).
        decoder_type: ``'lr_cross_attn'`` (queries from the high-resolution
            encoder, keys/values from the low-resolution encoder),
            ``'cross_attn'`` (queries from the fused pilot grid, keys/values
            from the high-resolution encoder) or ``'fc3'`` (MLP decoder).
        fuse_mode: ``'sum'`` or ``'concat'`` (1x1 convolution); fusion of the
            two branches for ``'cross_attn'`` and ``'fc3'``.
        upsample_method: ``'I'`` (per-PRB dense map) or ``'II'`` (per-token
            dense map).
        hr_prbs: PRBs per high-resolution window.
        lr_prbs: PRBs per low-resolution window.
        pool_factor: Average-pooling factor of the low-resolution branch.
        ffn_mult: FFN hidden width as a multiple of ``d_model``.
        ffn_depth: Number of dense layers per FFN (>= 2).
        pilot_stride: Subcarrier stride of the pilot extraction.
        patch_size: Pilot subcarriers per token; both upsampling methods
            assume ``pilot_stride * patch_size == 12``.
        pilot_syms: Indices of the pilot OFDM symbols.
        enable_token_processor: Apply a ``TokenProcessor`` MLP to each patch
            instead of a linear token projection.
        token_processor_hidden_dims: Hidden widths of the ``TokenProcessor``;
            a non-empty value enables it.
        decoder_out_proj: If False, the decoder returns ``d_model``-wide
            tokens instead of projecting back to ``patch_size``.
        enable_upsampling: If False, return the refined pilot grid instead of
            the full resource grid.
    """

    def __init__(
        self,
        d_model:         int   = 16,
        n_heads:         int   = 2,
        enc_layers:      int   = 1,
        dec_layers:      int   = 1,
        decoder_type:    str   = "lr_cross_attn",
        fuse_mode:       str   = "sum",
        upsample_method: str   = "I",
        hr_prbs:         int   = 6,
        lr_prbs:         int   = 24,
        pool_factor:     int   = 4,
        ffn_mult:        int   = 2,
        ffn_depth:       int   = 2,
        pilot_stride:    int   = 1,
        patch_size:      int   = 12,
        pilot_syms:      tuple = (2, 11),
        enable_token_processor: bool = False,
        token_processor_hidden_dims: tuple = (),
        decoder_out_proj: bool  = True,
        enable_upsampling: bool  = True,
        name:            str   = "chea",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        assert decoder_type     in ("cross_attn", "lr_cross_attn", "fc3"), \
            f"decoder_type must be 'cross_attn', 'lr_cross_attn', or 'fc3'"
        assert fuse_mode        in ("sum", "concat"),        f"fuse_mode must be 'sum' or 'concat'"
        assert upsample_method  in ("I", "II"),              f"upsample_method must be 'I' or 'II'"
        assert d_model % n_heads == 0,                       f"d_model={d_model} must be divisible by n_heads={n_heads}"
        assert int(pilot_stride) >= 1,                       f"pilot_stride must be >= 1, got {pilot_stride}"
        assert int(patch_size)   >= 1,                       f"patch_size must be >= 1, got {patch_size}"
        assert int(pool_factor)  >= 1,                       f"pool_factor must be >= 1, got {pool_factor}"
        assert int(ffn_depth)    >= 2,                       f"ffn_depth must be >= 2, got {ffn_depth}"
        assert int(lr_prbs) % int(pool_factor) == 0, (
            f"lr_prbs must be divisible by pool_factor, got {lr_prbs} and {pool_factor}"
        )

        self.d_model          = int(d_model)
        self.n_heads          = int(n_heads)
        self.enc_layers       = int(enc_layers)
        self.dec_layers       = int(dec_layers)
        self.decoder_type     = decoder_type
        self.fuse_mode        = fuse_mode
        self.upsample_method  = upsample_method
        self.hr_prbs          = int(hr_prbs)
        self.lr_prbs          = int(lr_prbs)
        self.pool_factor      = int(pool_factor)
        self.ffn_mult         = int(ffn_mult)
        self.ffn_depth        = int(ffn_depth)
        self.pilot_stride     = int(pilot_stride)
        self.patch_size       = int(patch_size)
        self.pilot_syms       = list(pilot_syms)
        self.decoder_out_proj  = bool(decoder_out_proj)
        self.enable_upsampling = bool(enable_upsampling)
        if token_processor_hidden_dims is None:
            token_processor_hidden_dims = ()
        if isinstance(token_processor_hidden_dims, int):
            token_processor_hidden_dims = (token_processor_hidden_dims,)
        self.token_processor_hidden_dims = tuple(
            int(dim) for dim in token_processor_hidden_dims
        )
        assert all(dim > 0 for dim in self.token_processor_hidden_dims), (
            "token_processor_hidden_dims must contain positive integers"
        )
        self.enable_token_processor = bool(
            enable_token_processor or len(self.token_processor_hidden_dims) > 0
        )

        if not self.decoder_out_proj:
            assert self.d_model % self.patch_size == 0, (
                "decoder_out_proj=False requires d_model to be divisible by patch_size"
            )

        # window widths in pilot subcarriers
        self._W_h = hr_prbs * self.patch_size
        self._W_l = lr_prbs * self.patch_size
        self._N_ph = hr_prbs
        self._N_pl = lr_prbs // pool_factor

        ffn_dim = d_model * ffn_mult

        if self.enable_token_processor:
            self._hr_token_processor = TokenProcessor(
                self.token_processor_hidden_dims, name="hr_token_processor"
            )
            self._lr_token_processor = TokenProcessor(
                self.token_processor_hidden_dims, name="lr_token_processor"
            )
        else:
            self._hr_token_processor = None
            self._lr_token_processor = None

        if not self.enable_token_processor:
            self._hr_proj = layers.Dense(d_model, use_bias=True, name="hr_proj")
        self._hr_pos  = PosEncoding(self._N_ph * 4, d_model, name="hr_pos")
        self._hr_enc  = [
            _EncoderBlock(d_model, n_heads, ffn_dim, ffn_depth=self.ffn_depth, name=f"hr_enc_{i}")
            for i in range(enc_layers)
        ]
        self._hr_out  = layers.Dense(self.patch_size, use_bias=True, name="hr_out")

        if not self.enable_token_processor:
            self._lr_proj = layers.Dense(d_model, use_bias=True, name="lr_proj")
        self._lr_pos  = PosEncoding(self._N_pl * 4, d_model, name="lr_pos")
        self._lr_enc  = [
            _EncoderBlock(d_model, n_heads, ffn_dim, ffn_depth=self.ffn_depth, name=f"lr_enc_{i}")
            for i in range(enc_layers)
        ]
        self._lr_out  = layers.Dense(self.patch_size, use_bias=True, name="lr_out")

        if fuse_mode == "concat":
            self._fuse_conv = layers.Conv2D(
                2, (1, 1), padding="same", use_bias=True, name="fuse_conv"
            )

        self._dec_in_proj  = layers.Dense(d_model, use_bias=True, name="dec_in_proj")
        if self.decoder_out_proj:
            self._dec_skip_proj = layers.Dense(self.patch_size, use_bias=True, name="dec_skip_proj")
            self._dec_out_proj  = layers.Dense(self.patch_size, use_bias=True, name="dec_out_proj")
        else:
            self._dec_skip_proj = None

        if decoder_type in ("cross_attn", "lr_cross_attn"):
            if decoder_type == "lr_cross_attn":
                _dec_pos_len = pool_factor * 4          # f_p HR patches x 4 tokens
            else:
                _dec_pos_len = hr_prbs * 4              # N_ph HR patches x 4 tokens
            self._dec_pos    = PosEncoding(_dec_pos_len, d_model, name="dec_pos")
            self._dec_blocks = [
                _DecoderBlock(d_model, n_heads, ffn_dim, ffn_depth=self.ffn_depth, name=f"dec_{i}")
                for i in range(dec_layers)
            ]
            if decoder_type == "lr_cross_attn":
                self._hr_skip_proj = layers.Dense(self.patch_size, use_bias=True, name="hr_skip_proj")
        else:  # fc3
            self._fc3_1 = layers.Dense(d_model, use_bias=True, name="fc3_1")
            self._fc3_2 = layers.Dense(d_model, use_bias=True, name="fc3_2")

        if self.enable_upsampling:
            if upsample_method == "I":
                self._ups_fc = layers.Dense(12 * 14 * 2, use_bias=True, name="ups_fc")
            else:  # II
                self._ups_fc = layers.Dense(12 * 7, use_bias=True, name="ups_fc")
        else:
            self._ups_fc = None

        self._input_F    = None
        self._input_S    = None
        self._input_F_half = None
        self._input_N_prbs = None
        self._F          = None
        self._S          = None
        self._F_half     = None
        self._pad_subcarriers = 0
        self._pad_pilot_subcarriers = 0
        self._window_pilot_multiple = None
        self._N_wh       = None
        self._N_wl       = None
        self._N_p_hr     = None
        self._N_p_lr     = None
        self._N_prbs     = None
        self._pilot_sub_idx  = None
        self._pilot_sym_idx  = None
        self._input_has_prb_mask = False

    def _configure_shape_state(self, raw_F: int, raw_S: int, context: str = "CHEA"):
        """Compute internal padded dimensions and window counts."""
        raw_F = int(raw_F)
        raw_S = int(raw_S)
        assert raw_F > 0, f"{context}: F must be positive, got {raw_F}"
        assert raw_F % self.pilot_stride == 0, (
            f"{context}: F={raw_F} must be divisible by pilot_stride={self.pilot_stride}"
        )
        assert raw_F % 12 == 0, (
            f"{context}: F={raw_F} must contain a whole number of 12-subcarrier PRBs"
        )

        raw_F_half = raw_F // self.pilot_stride
        window_multiple = _lcm(self._W_h, self._W_l)
        padded_F_half = _ceil_to_multiple(raw_F_half, window_multiple)
        padded_F = padded_F_half * self.pilot_stride

        assert padded_F % 12 == 0, (
            f"{context}: padded F={padded_F} must contain whole 12-subcarrier PRBs; "
            "check pilot_stride, patch_size, hr_prbs, and lr_prbs"
        )

        self._input_F = raw_F
        self._input_S = raw_S
        self._input_F_half = raw_F_half
        self._input_N_prbs = raw_F // 12
        self._F = padded_F
        self._S = raw_S
        self._F_half = padded_F_half
        self._pad_subcarriers = padded_F - raw_F
        self._pad_pilot_subcarriers = padded_F_half - raw_F_half
        self._window_pilot_multiple = window_multiple

        self._N_wh   = padded_F_half // self._W_h
        self._N_wl   = padded_F_half // self._W_l
        self._N_p_hr = self._N_wh * self._N_ph * 4
        self._N_p_lr = self._N_wl * self._N_pl * 4
        self._N_prbs = padded_F // 12

        assert self._N_wh >= 1 and self._N_wl >= 1, (
            f"{context}: internal padding failed to create at least one HR and LR window"
        )
        assert padded_F_half % self._N_prbs == 0, (
            f"{context}: padded F_half={padded_F_half} must be divisible by "
            f"N_prbs={self._N_prbs}; check pilot_stride and patch_size"
        )

        self._pilot_sub_idx = tf.range(0, self._F, delta=self.pilot_stride)
        self._pilot_sym_idx = tf.constant(self.pilot_syms, dtype=tf.int32)

    def _pad_full_grid_inputs(self, x, prb_mask=None):
        """Pad full-grid inputs on the subcarrier axis to the build-time size."""
        if self._pad_subcarriers <= 0:
            return x, prb_mask
        paddings = [[0, 0], [0, self._pad_subcarriers], [0, 0], [0, 0]]
        x = tf.pad(x, paddings)
        if prb_mask is not None:
            prb_mask = tf.pad(tf.cast(prb_mask, tf.bool), paddings)
        return x, prb_mask

    def _pad_pilot_grid(self, pilots):
        """Pad a pilot grid on the pilot-subcarrier axis to the internal size."""
        if self._pad_pilot_subcarriers <= 0:
            return pilots
        return tf.pad(pilots, [[0, 0], [0, self._pad_pilot_subcarriers], [0, 0], [0, 0]])

    def _pad_prb_mask(self, prb_mask):
        """Pad an optional full-grid PRB mask to the internal full-grid size."""
        if prb_mask is None or self._pad_subcarriers <= 0:
            return prb_mask
        return tf.pad(
            tf.cast(prb_mask, tf.bool),
            [[0, 0], [0, self._pad_subcarriers], [0, 0], [0, 0]],
        )

    def _crop_full_grid(self, y):
        """Remove internal full-grid padding before returning a result."""
        if self._pad_subcarriers <= 0:
            return y
        return y[:, :self._input_F, :, :]

    def _crop_pilot_grid(self, y):
        """Remove internal pilot-grid padding before returning a result."""
        if self._pad_pilot_subcarriers <= 0:
            return y
        padded_F_half = self._F_half
        height = tf.shape(y)[1]
        tf.debugging.assert_equal(
            height % padded_F_half,
            0,
            message="pilot-grid output height must be a multiple of padded F//pilot_stride",
        )
        expansion = height // padded_F_half
        return y[:, : self._input_F_half * expansion, :, :]

    def build(self, input_shape):
        """Create padded shape-dependent state from the input shape.

        Args:
            input_shape: [B, F, S, 2] or ([B, F, S, 2], prb_mask).
        """
        try:
            self._input_has_prb_mask = (
                isinstance(input_shape, (list, tuple))
                and len(input_shape) == 2
                and tf.TensorShape(input_shape[0]).rank == 4
                and tf.TensorShape(input_shape[1]).rank == 4
            )
        except (TypeError, ValueError):
            self._input_has_prb_mask = False
        x_shape = tf.TensorShape(input_shape[0] if self._input_has_prb_mask else input_shape)
        self._configure_shape_state(int(x_shape[1]), int(x_shape[2]), context="CHEA")
        super().build(input_shape)

    def _extract_pilots(self, x):
        """Gather pilot subcarriers and pilot OFDM symbols.

        Args:
            x: [B, F, S, 2].

        Returns:
            [B, F//pilot_stride, len(pilot_syms), 2]; with default pilot_syms,
            [B, F//pilot_stride, 2, 2].
        """
        x = tf.gather(x, self._pilot_sub_idx, axis=1)
        x = tf.gather(x, self._pilot_sym_idx, axis=2)
        return x

    def _make_validity_masks(self, prb_mask, B):
        """Build fixed-shape masks for PRB-prefix training.

        Args:
            prb_mask: optional bool-like mask with shape [B, F, S, 2].
                Valid subcarriers are read from prb_mask[:, :, 0, 0].
            B: dynamic batch size scalar.

        Returns:
            Dict of masks:
                output: None or [B, F, S, 2]
                pilot_valid: [B, F_half]
                pilot_grid: [B, F_half, 1, 1]
                hr_global: [B, N_p_hr]
                hr_win: [B*N_wh, N_ph*4]
                lr_global: [B, N_p_lr]
                lr_win: [B*N_wl, N_pl*4]
        """

        F_half = self._F_half
        pilots_per_prb = F_half // self._N_prbs

        if prb_mask is None:
            prb_valid = tf.range(self._N_prbs)[None, :] < self._input_N_prbs  # [1, N_prbs]
            prb_valid = tf.broadcast_to(prb_valid, [B, self._N_prbs])
            output_mask = None
        else:
            prb_mask = tf.cast(prb_mask, tf.bool)
            subcarrier_valid = prb_mask[:, :, 0, 0]             # [B, F]
            n_valid_subcarriers = tf.reduce_sum(
                tf.cast(subcarrier_valid, tf.int32), axis=1
            )
            tf.debugging.assert_equal(
                n_valid_subcarriers % 12,
                tf.zeros_like(n_valid_subcarriers),
                message="prb_mask must mark a whole number of PRBs",
            )
            n_valid_prbs = n_valid_subcarriers // 12
            prb_valid = tf.range(self._N_prbs)[None, :] < n_valid_prbs[:, None]  # [B, N_prbs]
            output_mask = prb_mask

        # Pilot-grid validity: [B, F_half].
        pilot_valid = tf.repeat(prb_valid, repeats=pilots_per_prb, axis=1)
        pilot_valid = pilot_valid[:, :F_half]

        return self._make_masks_from_pilot_valid(pilot_valid, B, output_mask)

    def _make_masks_from_pilot_valid(self, pilot_valid, B, output_mask=None):
        """Build token masks from pilot-grid validity.

        Args:
            pilot_valid: bool-like tensor [B, F//pilot_stride].
            B: dynamic batch size scalar.
            output_mask: optional full-grid mask [B, F, S, 2].

        Returns:
            Mask dict used by the HR/LR branches and decoder.
        """
        pilot_valid = tf.cast(pilot_valid, tf.bool)

        # HR tokens are per-PRB when pilot_stride * patch_size == 12.
        hr_patch_valid = tf.reshape(pilot_valid, [B, self._N_wh * self._N_ph, self.patch_size])
        hr_patch_valid = tf.reduce_any(hr_patch_valid, axis=-1)
        hr_token_valid = tf.repeat(hr_patch_valid, repeats=4, axis=1)        # [B, N_p_hr]

        # LR pooled patches cover pool_factor PRBs. A pooled token is valid if
        # at least one contributing pilot subcarrier is valid.
        W_pooled = self._W_l // self.pool_factor
        lr_pool_valid = tf.reshape(pilot_valid, [B * self._N_wl, self._W_l, 1, 1])
        lr_pool_count = tf.nn.avg_pool2d(
            tf.cast(lr_pool_valid, tf.float32),
            ksize=[self.pool_factor, 1],
            strides=[self.pool_factor, 1],
            padding="VALID",
            data_format="NHWC",
        ) * float(self.pool_factor)
        lr_pool_valid = tf.reshape(lr_pool_count > 0.0, [B, self._N_wl, W_pooled])
        lr_patch_valid = tf.reshape(lr_pool_valid, [B, self._N_wl * self._N_pl, self.patch_size])
        lr_patch_valid = tf.reduce_any(lr_patch_valid, axis=-1)
        lr_token_valid = tf.repeat(lr_patch_valid, repeats=4, axis=1)        # [B, N_p_lr]

        return {
            "output": output_mask,
            "pilot_valid": pilot_valid,
            "pilot_grid": tf.cast(pilot_valid[:, :, None, None], tf.float32),
            "hr_global": hr_token_valid,
            "hr_win": tf.reshape(hr_token_valid, [B * self._N_wh, self._N_ph * 4]),
            "lr_global": lr_token_valid,
            "lr_win": tf.reshape(lr_token_valid, [B * self._N_wl, self._N_pl * 4]),
        }

    def _reverse_pilot_masks(self, masks, B):
        """Reverse pilot-axis masks for right-aligned stage windowing."""
        pilot_valid = tf.reverse(masks["pilot_valid"], axis=[1])
        return self._make_masks_from_pilot_valid(
            pilot_valid, B, output_mask=masks.get("output")
        )

    def _masked_lr_avg_pool(self, x, pilot_valid, B):
        """Average-pool LR windows using only valid pilot subcarriers.

        Args:
            x: [B, N_wl, W_l, 2, 2] or [B*N_wl, W_l, 2, 2].
            pilot_valid: [B, F_half].
            B: dynamic batch size scalar.

        Returns:
            [B*N_wl, W_l//pool_factor, 2, 2].
        """
        N_wl, W_l, f_p = self._N_wl, self._W_l, self.pool_factor
        x_dtype = x.dtype
        x = tf.reshape(x, [B * N_wl, W_l, 2, 2])
        mask = tf.reshape(pilot_valid, [B * N_wl, W_l, 1, 1])
        mask = tf.cast(mask, x.dtype)

        x_sum = tf.nn.avg_pool2d(
            x * mask,
            ksize=[f_p, 1], strides=[f_p, 1], padding="VALID",
            data_format="NHWC",
        ) * tf.cast(f_p, x.dtype)
        count = tf.nn.avg_pool2d(
            mask,
            ksize=[f_p, 1], strides=[f_p, 1], padding="VALID",
            data_format="NHWC",
        ) * tf.cast(f_p, x.dtype)

        pooled = x_sum / tf.maximum(count, tf.cast(1.0, x.dtype))
        pooled = pooled * tf.cast(count > 0.0, pooled.dtype)
        return tf.cast(pooled, x_dtype)

    def _to_tokens(self, pilots, B, N_win, N_p, W,  token_processor=None):
        """Convert a windowed pilot tensor to patch tokens.

        Args:
            pilots: [B, N_win, W, 2, 2] (W pilot subcarriers, 2 symbols, re/im).
            B: dynamic batch size scalar.
            N_win: number of windows (static int).
            N_p: patches per window, W // patch_size (static int).
            W: window width in pilot subcarriers (static int).
            token_processor: optional ``TokenProcessor`` applied per patch.

        Returns:
            [B*N_win, N_p*4, patch_size].
        """
        P = self.patch_size
        x = tf.reshape(pilots, [B, N_win, N_p, P, 2, 2])
        if token_processor is not None:
            x = token_processor(x)
        # [B, N_win, N_p, 2, 2, P]
        x = tf.transpose(x, [0, 1, 2, 4, 5, 3])
        x = tf.reshape(x, [B * N_win, N_p * 4, P])
        return x

    def _from_tokens(self, tokens, B, N_win, N_p, F_half):
        """Reconstruct pilot grid from patch tokens.

        Args:
            tokens: [B*N_win, N_p*4, token_width], token_width = patch_size * M.
            B: dynamic batch size scalar.
            N_win: number of windows (static int).
            N_p: patches per window (static int).
            F_half: F // pilot_stride.

        Returns:
            [B, F_half * M, 2, 2].
        """
        P = self.patch_size
        token_width = tf.shape(tokens)[-1]
        tf.debugging.assert_equal(
            token_width % P, 0,
            message="decoder token width must be a multiple of patch_size"
        )
        M = token_width // P
        x = tf.reshape(tokens, [B, N_win, N_p, 4, token_width])
        x = tf.reshape(x, [B, N_win, N_p, 2, 2, token_width])
        # [B, N_win, N_p, token_width, 2, 2]
        x = tf.transpose(x, [0, 1, 2, 5, 3, 4])
        x = tf.reshape(x, [B, F_half * M, 2, 2])
        return x

    def _hr_branch(self, pilots, B, masks):
        """Run the high-resolution pilot encoder branch.

        Args:
            pilots: [B, F//pilot_stride, 2, 2].
            B: dynamic batch size scalar.
            masks: dict from _make_validity_masks.

        Returns:
            If decoder_type == "lr_cross_attn":
                (None, memory [B, N_p_hr, d_model]).
            Otherwise:
                (pilot_grid [B, F//pilot_stride, 2, 2],
                 memory [B, N_p_hr, d_model]).
        """
        F_half = self._F_half
        N_wh, N_ph, W_h = self._N_wh, self._N_ph, self._W_h
        token_mask = masks["hr_win"]

        x = tf.reshape(pilots, [B, N_wh, W_h, 2, 2])

        tokens = self._to_tokens(x, B, N_wh, N_ph, W_h, self._hr_token_processor)  # [B*N_wh, N_ph*4, patch_size]
        tokens = _apply_token_mask(tokens, token_mask)

        if not self.enable_token_processor:
            tokens = self._hr_proj(tokens)                  # [B*N_wh, N_ph*4, d_model]
        tokens = self._hr_pos(tokens)
        tokens = _apply_token_mask(tokens, token_mask)

        for blk in self._hr_enc:
            tokens = blk(tokens, token_mask=token_mask)

        memory = tf.reshape(tokens, [B, N_wh, N_ph*4, self.d_model])
        memory = tf.reshape(memory, [B, self._N_p_hr, self.d_model])
        memory = _apply_token_mask(memory, masks["hr_global"])

        if self.decoder_type == "lr_cross_attn":
            return None, memory

        out = self._hr_out(tokens)
        out = _apply_token_mask(out, token_mask)

        pilot_grid = self._from_tokens(out, B, N_wh, N_ph, F_half)  # [B, F_half, 2, 2]
        pilot_grid = pilot_grid * tf.cast(masks["pilot_grid"], pilot_grid.dtype)

        return pilot_grid, memory

    def _lr_branch(self, pilots, B, masks):
        """Run the low-resolution pilot encoder branch.

        Args:
            pilots: [B, F//pilot_stride, 2, 2].
            B: dynamic batch size scalar.
            masks: dict from _make_validity_masks.

        Returns:
            If decoder_type == "lr_cross_attn":
                (None, lr_tokens [B, N_p_lr, d_model]).
            Otherwise:
                (pilot_grid [B, F//pilot_stride, 2, 2], None).

        For ``'lr_cross_attn'`` the encoder output is returned directly as the
        decoder keys/values; otherwise it is projected back and bilinearly
        upsampled to the pilot grid.
        """
        F_half = self._F_half
        N_wl, N_pl = self._N_wl, self._N_pl
        W_l = self._W_l
        f_p = self.pool_factor
        token_mask = masks["lr_win"]

        x = tf.reshape(pilots, [B, N_wl, W_l, 2, 2])

        # Masked pooling averages over valid contributors only, so partially
        # valid LR tokens keep their scale.
        x = self._masked_lr_avg_pool(x, masks["pilot_valid"], B)  # [B*N_wl, W_l//f_p, 2, 2]
        W_pooled = W_l // f_p
        x = tf.reshape(x, [B, N_wl, W_pooled, 2, 2])

        tokens = self._to_tokens(x, B, N_wl, N_pl, W_pooled, self._lr_token_processor)  # [B*N_wl, N_pl*4, patch_size]
        tokens = _apply_token_mask(tokens, token_mask)

        if not self.enable_token_processor:
            tokens = self._lr_proj(tokens)                  # [B*N_wl, N_pl*4, d_model]
        tokens = self._lr_pos(tokens)
        tokens = _apply_token_mask(tokens, token_mask)

        for blk in self._lr_enc:
            tokens = blk(tokens, token_mask=token_mask)

        if self.decoder_type == "lr_cross_attn":
            lr_tokens = tf.reshape(tokens, [B, N_wl, N_pl*4, self.d_model])
            lr_tokens = tf.reshape(lr_tokens, [B, self._N_p_lr, self.d_model])
            lr_tokens = _apply_token_mask(lr_tokens, masks["lr_global"])
            return None, lr_tokens

        out = self._lr_out(tokens)
        out = _apply_token_mask(out, token_mask)

        # Bilinear upsampling of each patch along subcarriers: P -> P*f_p.
        P  = self.patch_size
        BN = B * N_wl
        Nseq = N_pl * 4
        out = tf.reshape(out, [BN * Nseq, P, 1, 1])
        out = tf.image.resize(
            out, [P * f_p, 1], method="bilinear"
        )
        out = tf.reshape(out, [BN, Nseq, P * f_p])

        out = tf.reshape(out, [B, N_wl, N_pl, 4, P * f_p])
        out = tf.reshape(out, [B, N_wl, N_pl, 2, 2, P * f_p])
        out = tf.transpose(out, [0, 1, 2, 5, 3, 4])                  # [B, N_wl, N_pl, P*f_p, 2, 2]
        out = tf.reshape(out, [B, F_half, 2, 2])
        out = out * tf.cast(masks["pilot_grid"], out.dtype)

        return out, None

    def _decoder(self, fused, hr_memory, lr_memory, B, masks):
        """Decode branch outputs back to the pilot grid.

        Args:
            fused: [B, F//pilot_stride, 2, 2], used by "cross_attn"/"fc3";
                None for "lr_cross_attn".
            hr_memory: [B, N_p_hr, d_model].
            lr_memory: [B, N_p_lr, d_model], or None unless "lr_cross_attn".
            B: dynamic batch size scalar.
            masks: dict from _make_validity_masks.

        Returns:
            [B, F//pilot_stride * M, 2, 2], where M=1 when
            decoder_out_proj=True and M=d_model//patch_size otherwise.
        """
        F_half = self._F_half
        N_wh, N_ph = self._N_wh, self._N_ph

        if self.decoder_type == "lr_cross_attn":
            # Local cross-attention: each group of f_p*4 HR tokens (queries) attends
            # to the 4 LR tokens (keys/values) of the corresponding LR patch.
            f_p = self.pool_factor
            N_groups = B * self._N_wl * self._N_pl

            q_mask_global = masks["hr_global"]
            q_mask = tf.reshape(q_mask_global, [N_groups, f_p * 4])
            kv_mask = tf.reshape(masks["lr_global"], [N_groups, 4])

            x = self._dec_in_proj(hr_memory)
            x = _apply_token_mask(x, q_mask_global)
            x = tf.reshape(x, [N_groups, f_p * 4, self.d_model])
            x = self._dec_pos(x)
            x = _apply_token_mask(x, q_mask)

            lr_loc = tf.reshape(lr_memory, [N_groups, 4, self.d_model])
            for blk in self._dec_blocks:
                x = blk(x, lr_loc, q_mask=q_mask, memory_mask=kv_mask)

            x = tf.reshape(x, [B, self._N_p_hr, self.d_model])
            x = _apply_token_mask(x, q_mask_global)
            if self._dec_out_proj is not None:
                out_tok = self._dec_out_proj(x)
                out_tok = out_tok + self._hr_skip_proj(hr_memory)
                out_tok = _apply_token_mask(out_tok, q_mask_global)
            else:
                out_tok = x

            out = self._from_tokens(out_tok, B, N_wh, N_ph, F_half)
            return out * tf.cast(masks["pilot_grid"], out.dtype)

        # cross_attn / fc3: tokenize the fused pilot grid.
        P = self.patch_size
        token_mask_global = masks["hr_global"]

        # [B, F_half, 2, 2] -> [B, N_wh, N_ph, 2, 2, P] -> [B, N_p_hr, P]
        fused = fused * tf.cast(masks["pilot_grid"], fused.dtype)
        fused_r   = tf.reshape(fused, [B, N_wh, N_ph, P, 2, 2])
        fused_tok = tf.transpose(fused_r, [0, 1, 2, 4, 5, 3])
        fused_tok = tf.reshape(fused_tok, [B, self._N_p_hr, P])
        fused_tok = _apply_token_mask(fused_tok, token_mask_global)

        skip = self._dec_skip_proj(fused_tok) if self._dec_out_proj is not None else None

        if self.decoder_type == "cross_attn":
            # Local cross-attention: one group per HR window.
            N_groups = B * N_wh
            local_mask = tf.reshape(token_mask_global, [N_groups, N_ph * 4])

            x = self._dec_in_proj(fused_tok)
            x = _apply_token_mask(x, token_mask_global)
            x = tf.reshape(x, [N_groups, N_ph * 4, self.d_model])
            x = self._dec_pos(x)
            x = _apply_token_mask(x, local_mask)

            hr_loc = tf.reshape(hr_memory, [N_groups, N_ph * 4, self.d_model])
            for blk in self._dec_blocks:
                x = blk(x, hr_loc, q_mask=local_mask, memory_mask=local_mask)

            x = tf.reshape(x, [B, self._N_p_hr, self.d_model])
            x = _apply_token_mask(x, token_mask_global)
            out_tok = self._dec_out_proj(x) if self._dec_out_proj is not None else x
        else:  # fc3
            x = tf.keras.activations.gelu(
                self._dec_in_proj(fused_tok), approximate=True
            )
            x = _apply_token_mask(x, token_mask_global)
            x = tf.keras.activations.gelu(
                self._fc3_2(x), approximate=True
            )
            x = _apply_token_mask(x, token_mask_global)
            out_tok = self._dec_out_proj(x) if self._dec_out_proj is not None else x

        if self._dec_out_proj is not None:
            out_tok = out_tok + skip
            out_tok = _apply_token_mask(out_tok, token_mask_global)
        out = self._from_tokens(out_tok, B, N_wh, N_ph, F_half)
        return out * tf.cast(masks["pilot_grid"], out.dtype)

    def _refine_pilots(self, pilots, B, masks):
        """Refine an extracted pilot grid without final upsampling.

        Args:
            pilots: [B, F//pilot_stride, len(pilot_syms), 2].
            B: dynamic batch size scalar.
            masks: dict from _make_validity_masks.

        Returns:
            Refined pilot grid [B, F//pilot_stride * M, len(pilot_syms), 2].
        """
        pilots = pilots * tf.cast(masks["pilot_grid"], pilots.dtype)

        hr_out, hr_memory = self._hr_branch(pilots, B, masks)
        lr_pilot, lr_memory = self._lr_branch(pilots, B, masks)

        if self.decoder_type == "lr_cross_attn":
            return self._decoder(None, hr_memory, lr_memory, B, masks)

        if self.fuse_mode == "sum":
            fused = hr_out + lr_pilot
        else:
            concat = tf.concat([hr_out, lr_pilot], axis=-1)
            fused  = self._fuse_conv(concat)
            fused  = fused * tf.cast(masks["pilot_grid"], fused.dtype)

        return self._decoder(fused, hr_memory, None, B, masks)

    def _upsample(self, x, B):
        """Upsample the decoded pilot grid to the full resource grid.

        Args:
            x: [B, F//pilot_stride * M, 2, 2].
            B: dynamic batch size scalar.

        Returns:
            [B, F, 14, 2].
        """
        N_prbs = self._N_prbs
        N_wh, N_ph = self._N_wh, self._N_ph
        F_half = self._F_half

        token_height = tf.shape(x)[1]
        tf.debugging.assert_equal(
            token_height % F_half, 0,
            message="decoder output height must be a multiple of F//pilot_stride"
        )
        M = token_height // F_half

        if self.upsample_method == "I":
            # Per-PRB dense map from the PRB's pilot values to its 12 x 14 x 2 grid.
            pilots_per_prb = F_half // N_prbs

            x = tf.reshape(x, [B, N_prbs, pilots_per_prb * M, 2, 2])
            x = tf.reshape(x, [B, N_prbs, pilots_per_prb * M * 4])
            # [B, N_prbs, 12 * 14 * 2]
            x = self._ups_fc(x)
            x = tf.reshape(x, [B, N_prbs, 12, 14, 2])
            x = tf.reshape(x, [B, N_prbs * 12, 14, 2])
        else:  # II
            assert self.pilot_stride * self.patch_size == 12, (
                f"upsample_method='II' requires pilot_stride * patch_size == 12 "
                f"(got {self.pilot_stride} * {self.patch_size} = "
                f"{self.pilot_stride * self.patch_size})"
            )
            P = self.patch_size
            x = tf.reshape(x, [B, N_wh, N_ph, P * M, 2, 2])
            x = tf.transpose(x, [0, 1, 2, 4, 5, 3])
            x = tf.reshape(x, [B, N_prbs * 4, P * M])
            # [B, N_prbs*4, 12 * 7]
            x = self._ups_fc(x)
            x = tf.reshape(x, [B, N_prbs * 4, 12, 7])
            x = tf.reshape(x, [B, N_prbs, 2, 2, 12, 7])
            x = tf.transpose(x, perm=[0,1,4,5,2,3]) # [B, N_prbs, 12, 7, 2, 2]
            x = tf.reshape(x, [B, N_prbs, 12, 14, 2])
            x = tf.reshape(x, [B, N_prbs*12, 14, 2])

        return x

    def call(self, inputs, training=False):
        """Run CHEA channel estimation.

        Args:
            inputs: [B, F, S, 2], or ``(x, prb_mask)`` where ``prb_mask`` is a
                [B, F, S, 2] bool/0-1 mask of a contiguous valid PRB prefix
                starting at PRB 0.
            training: unused; kept for Keras call compatibility.

        Returns:
            [B, F, S, 2], or the refined pilot grid if upsampling is disabled.
        """
        if self._input_has_prb_mask:
            x, prb_mask = inputs
        else:
            x = inputs
            prb_mask = None

        x, prb_mask = self._pad_full_grid_inputs(x, prb_mask)
        B = tf.shape(x)[0]
        masks = self._make_validity_masks(prb_mask, B)

        # Zero inactive pilots before any bias, pooling or attention can leak them.
        pilots = self._extract_pilots(x)
        pilots = pilots * tf.cast(masks["pilot_grid"], pilots.dtype)

        refined = self._refine_pilots(pilots, B, masks)

        if not self.enable_upsampling:
            return self._crop_pilot_grid(refined)

        y = self._upsample(refined, B)                  # [B, padded_F, 14, 2]
        if masks["output"] is not None:
            y = y * tf.cast(masks["output"], y.dtype)
        return self._crop_full_grid(y)

    def get_config(self):
        """Return a serializable layer config."""
        cfg = super().get_config()
        cfg.update({
            "d_model":          self.d_model,
            "n_heads":          self.n_heads,
            "enc_layers":       self.enc_layers,
            "dec_layers":       self.dec_layers,
            "decoder_type":     self.decoder_type,
            "fuse_mode":        self.fuse_mode,
            "upsample_method":  self.upsample_method,
            "hr_prbs":          self.hr_prbs,
            "lr_prbs":          self.lr_prbs,
            "pool_factor":      self.pool_factor,
            "ffn_mult":         self.ffn_mult,
            "ffn_depth":        self.ffn_depth,
            "pilot_stride":     self.pilot_stride,
            "patch_size":       self.patch_size,
            "pilot_syms":       self.pilot_syms,
            "decoder_out_proj": self.decoder_out_proj,
            "enable_upsampling": self.enable_upsampling,
        })
        return cfg


class CHEAStage(CHEA):
    """Single CHEA refinement stage operating on the pilot grid (no upsampling).

    Takes and returns a pilot grid [B, F//pilot_stride, len(pilot_syms), 2].
    With ``residual_refine``, the output is ``x + alpha * (CHEA(x) - x)`` with
    a trainable scalar ``alpha`` initialized to ``residual_scale_init``.

    Args:
        residual_refine: Blend the refinement into the input with ``alpha``.
        residual_scale_init: Initial value of ``alpha``.
        trainable_residual_scale: Whether ``alpha`` is trainable.
        window_alignment: ``'left'`` aligns windows to the first subcarrier,
            ``'right'`` to the last one.
        **kwargs: ``CHEA`` arguments.
    """

    def __init__(
        self,
        *args,
        residual_refine: bool = True,
        residual_scale_init: float = .001,
        trainable_residual_scale: bool = True,
        window_alignment: str = "left",
        name: str = "chea_stage",
        **kwargs,
    ):
        assert window_alignment in ("left", "right"), (
            f"window_alignment must be 'left' or 'right', got {window_alignment!r}"
        )
        kwargs["enable_upsampling"] = False
        kwargs["decoder_out_proj"] = True
        super().__init__(*args, name=name, **kwargs)
        self.window_alignment = window_alignment
        self.residual_refine = bool(residual_refine)
        self.residual_scale_init = float(residual_scale_init)
        self.trainable_residual_scale = bool(trainable_residual_scale)
        self._residual_scale = None

    def build(self, input_shape):
        """Create padded shape state from a pilot-grid input shape.

        Args:
            input_shape: [B, F//pilot_stride, len(pilot_syms), 2].
        """
        pilot_shape = input_shape[0] if (
            isinstance(input_shape, (list, tuple))
            and len(input_shape) == 2
            and tf.TensorShape(input_shape[0]).rank == 4
        ) else input_shape

        pilot_shape = tf.TensorShape(pilot_shape)
        self._configure_shape_state(
            int(pilot_shape[1]) * self.pilot_stride,
            14,
            context="CHEAStage",
        )
        self._input_has_prb_mask = False

        if self.residual_refine:
            self._residual_scale = self.add_weight(
                name="residual_scale",
                shape=(),
                initializer=tf.keras.initializers.Constant(self.residual_scale_init),
                trainable=self.trainable_residual_scale,
            )

        super(CHEA, self).build(input_shape)

    def call(self, inputs, validity_masks=None, training=False):
        """Refine a pilot grid.

        Args:
            inputs: [B, F//pilot_stride, len(pilot_syms), 2], or
                (pilot_grid, prb_mask).
            validity_masks: optional mask dict from CHEA._make_validity_masks.
            training: unused; included for Keras call compatibility.

        Returns:
            [B, F//pilot_stride, len(pilot_syms), 2].
        """
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            pilots, prb_mask = inputs
        else:
            pilots, prb_mask = inputs, None

        if validity_masks is None:
            pilots = self._pad_pilot_grid(pilots)
            prb_mask = self._pad_prb_mask(prb_mask)

        B = tf.shape(pilots)[0]
        masks = validity_masks
        if masks is None:
            masks = self._make_validity_masks(prb_mask, B)

        original_masks = masks
        if self.window_alignment == "right":
            # Process the grid in reversed pilot order so that windows are aligned
            # to the high-frequency end.
            pilots = tf.reverse(pilots, axis=[1])
            masks = self._reverse_pilot_masks(masks, B)

        pilots = pilots * tf.cast(masks["pilot_grid"], pilots.dtype)
        refined = self._refine_pilots(pilots, B, masks)

        if self.residual_refine:
            refined = pilots + tf.cast(self._residual_scale, refined.dtype) * (refined - pilots)
            refined = refined * tf.cast(masks["pilot_grid"], refined.dtype)

        if self.window_alignment == "right":
            refined = tf.reverse(refined, axis=[1])
            refined = refined * tf.cast(original_masks["pilot_grid"], refined.dtype)

        return self._crop_pilot_grid(refined)

    def get_config(self):
        """Return a serializable layer config."""
        cfg = super().get_config()
        cfg.update({
            "residual_refine": self.residual_refine,
            "residual_scale_init": self.residual_scale_init,
            "trainable_residual_scale": self.trainable_residual_scale,
            "window_alignment": self.window_alignment,
        })
        return cfg


@register_block
class CHEAStack(CHEA):
    """Stack of ``num_stages`` independent ``CHEAStage`` refiners with one final upsampling.

    Input and output are resource grids [B, F, S, 2] (optionally with a PRB
    mask, as for ``CHEA``). The pilot grid is refined by each stage in turn
    and then upsampled once to the full slot.

    Args:
        num_stages: Number of refinement stages.
        residual_refine: Use trainable residual scales in every stage.
        residual_scale_init: Initial value of the residual scales.
        trainable_residual_scale: Whether the residual scales are trainable.
        stage_window_alignment: ``'left'``, ``'right'``, ``'alternate'``
            (left/right alternating across stages) or a sequence of
            ``'left'``/``'right'`` values cycled over the stages.
        **kwargs: ``CHEA`` arguments shared by all stages.
    """

    def __init__(
        self,
        num_stages: int = 2,
        residual_refine: bool = True,
        residual_scale_init: float = 0.001,
        trainable_residual_scale: bool = True,
        stage_window_alignment="alternate",
        name: str = "chea_stack",
        **kwargs,
    ):
        kwargs["enable_upsampling"] = True
        kwargs["decoder_out_proj"] = True
        super().__init__(name=name, **kwargs)
        assert int(num_stages) >= 1, f"num_stages must be >= 1, got {num_stages}"
        self.num_stages = int(num_stages)
        self.residual_refine = bool(residual_refine)
        self.residual_scale_init = float(residual_scale_init)
        self.trainable_residual_scale = bool(trainable_residual_scale)
        self.stage_window_alignment = self._normalize_stage_window_alignment(
            stage_window_alignment
        )

        stage_cfg = {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "enc_layers": self.enc_layers,
            "dec_layers": self.dec_layers,
            "decoder_type": self.decoder_type,
            "fuse_mode": self.fuse_mode,
            "upsample_method": self.upsample_method,
            "hr_prbs": self.hr_prbs,
            "lr_prbs": self.lr_prbs,
            "pool_factor": self.pool_factor,
            "ffn_mult": self.ffn_mult,
            "ffn_depth": self.ffn_depth,
            "pilot_stride": self.pilot_stride,
            "patch_size": self.patch_size,
            "pilot_syms": tuple(self.pilot_syms),
            "residual_refine": self.residual_refine,
            "residual_scale_init": self.residual_scale_init,
            "trainable_residual_scale": self.trainable_residual_scale,
        }
        self._stages = [
            CHEAStage(
                **stage_cfg,
                window_alignment=self._stage_window_alignment_for_index(i),
                name=f"stage_{i}",
            )
            for i in range(self.num_stages)
        ]

    @staticmethod
    def _normalize_stage_window_alignment(stage_window_alignment):
        """Normalize stack window alignment to a string or tuple."""
        if isinstance(stage_window_alignment, str):
            assert stage_window_alignment in ("left", "right", "alternate"), (
                "stage_window_alignment must be 'left', 'right', 'alternate', "
                "or a sequence of 'left'/'right' values"
            )
            return stage_window_alignment

        alignment = tuple(stage_window_alignment)
        assert alignment, "stage_window_alignment sequence must not be empty"
        assert all(a in ("left", "right") for a in alignment), (
            "stage_window_alignment sequence values must be 'left' or 'right'"
        )
        return alignment

    def _stage_window_alignment_for_index(self, index: int) -> str:
        """Return the configured window alignment for one stack stage."""
        alignment = self.stage_window_alignment
        if alignment == "alternate":
            return "right" if index % 2 else "left"
        if isinstance(alignment, str):
            return alignment
        return alignment[index % len(alignment)]

    def call(self, inputs, training=False):
        """Run repeated pilot-grid refinement and one final upsampling.

        Args:
            inputs: [B, F, S, 2], or ``(x, prb_mask)`` with a [B, F, S, 2]
                bool/0-1 PRB mask.
            training: Keras training flag, passed to the stages.

        Returns:
            [B, F, S, 2].
        """
        if self._input_has_prb_mask:
            x, prb_mask = inputs
        else:
            x = inputs
            prb_mask = None

        x, prb_mask = self._pad_full_grid_inputs(x, prb_mask)
        B = tf.shape(x)[0]
        masks = self._make_validity_masks(prb_mask, B)

        pilots = self._extract_pilots(x)
        pilots = pilots * tf.cast(masks["pilot_grid"], pilots.dtype)

        refined = pilots
        for stage in self._stages:
            refined = stage(refined, validity_masks=masks, training=training)

        y = self._upsample(refined, B)                 # [B, padded_F, 14, 2]
        if masks["output"] is not None:
            y = y * tf.cast(masks["output"], y.dtype)
        return self._crop_full_grid(y)

    def get_config(self):
        """Return a serializable layer config."""
        cfg = super().get_config()
        cfg.update({
            "num_stages": self.num_stages,
            "residual_refine": self.residual_refine,
            "residual_scale_init": self.residual_scale_init,
            "trainable_residual_scale": self.trainable_residual_scale,
            "stage_window_alignment": self.stage_window_alignment,
        })
        return cfg

