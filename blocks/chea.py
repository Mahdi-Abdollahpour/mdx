# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026

"""CHEA (Channel Estimation Attention) for 5G NR PUSCH, arXiv:2607.16462.

Resource grids use shape [B, F, S, 2]: batch, subcarriers, OFDM symbols
(S=14), and real/imaginary parts. The default estimator extracts pilots,
encodes them at high resolution (HR) and pooled low resolution (LR), then
uses local cross-attention to refine them. A dense map per PRB expands the
pilots to the full slot.

CHEAStack runs several CHEAStage refiners with residual scaling, then
upsamples once. F must be a multiple of 12. Incomplete windows are padded
internally and cropped from the output. An optional PRB mask selects a
contiguous prefix of valid PRBs.
"""

from __future__ import annotations

import math

import tensorflow as tf
from tensorflow.keras import layers

from .cnn import Conv
from .registry import register_block


def _ceil_to_multiple(value: int, multiple: int) -> int:
    """Round value up to a multiple of the given size."""
    assert multiple > 0, f"multiple must be positive, got {multiple}"
    return ((int(value) + multiple - 1) // multiple) * multiple


def _lcm(a: int, b: int) -> int:
    """Return the least common multiple of positive integers a and b."""
    a, b = int(a), int(b)
    assert a > 0 and b > 0, f"lcm inputs must be positive, got {a}, {b}"
    return abs(a * b) // math.gcd(a, b)


def _block_span(start: int, stop: int, block: int):
    """Return the block range [b0, b1) and edge padding for [start, stop)."""
    b0 = start // block
    b1 = -(-stop // block)
    return b0, b1, start - b0 * block, b1 * block - stop


def _pad_axis1(x, front: int, back: int):
    """Add front and back zeros along axis 1."""
    if front == 0 and back == 0:
        return x
    pads = [[0, 0]] * len(x.shape)
    pads[1] = [front, back]
    return tf.pad(x, pads)


def _periodic_pos(pos_layer, start: int, n: int):
    """Get positional embeddings for n tokens, wrapping from start."""
    if not pos_layer.built:
        pos_layer.build((None, pos_layer._seq, pos_layer._d))
    idx = [(start + t) % pos_layer._seq for t in range(n)]
    return tf.gather(pos_layer.embed, idx, axis=1)  # [1, n, d]


class PosEncoding(layers.Layer):
    """Add learned positions shared across batches and windows.

    The embedding has shape [1, seq_len, d_model].
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
    """Multi-head self- or cross-attention with scaled dot products."""

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
        """Use x [B, Sq, d] as queries and kv [B, Sk, d] as memory.

        With kv=None, attend to x itself. Optional 0/1 masks q_mask [B, Sq]
        and kv_mask [B, Sk] mark valid query and memory tokens.
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
    """Zero invalid tokens without changing the tensor shape."""
    if mask is None:
        return x
    return x * tf.cast(mask[..., None], x.dtype)


class _FFN(layers.Layer):
    """Per-token MLP with ffn_dim-wide GELU layers and d_model-wide output."""

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
    """Self-attention and an FFN with pre-normalization and residual connections."""

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
    """Pre-normalized self-attention, cross-attention, and FFN with residuals."""

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
    """Process each pilot patch with an MLP before creating tokens.

    Keeps shape [B, N_win, N_p, P, 2, 2], where P=patch_size. The MLP
    flattens the last three axes: pilot subcarriers, symbols, and real/imag.
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
    """Refine a channel grid [B, F, S, 2] using HR and LR windowed attention.

    Accepts x or (x, prb_mask), where the mask has the same shape as x and
    marks a contiguous prefix of valid PRBs. Returns the full grid by
    default, or the refined pilot grid when upsampling is disabled.

    Args:
        d_model: Token embedding width.
        n_heads: Number of attention heads.
        enc_layers: Encoder blocks in each HR and LR branch.
        dec_layers: Decoder blocks; ignored by 'fc3'.
        decoder_type: 'lr_cross_attn' uses HR queries and LR memory;
            'cross_attn' uses fused pilot queries and HR memory;
            'fc3' uses an MLP.
        fuse_mode: Merge branches by 'sum' or 'concat' with a 1x1 convolution.
            Used by 'cross_attn' and 'fc3'.
        upsample_method: Dense map per PRB ('I') or per token ('II').
        hr_prbs: PRBs per HR window.
        lr_prbs: PRBs per LR window.
        pool_factor: LR average-pooling factor.
        ffn_mult: FFN hidden width as a multiple of d_model.
        ffn_depth: Dense layers per FFN, at least 2.
        pilot_stride: Subcarrier step when extracting pilots.
        patch_size: Pilot subcarriers per token. For one patch per PRB,
            pilot_stride * patch_size must equal 12.
        pilot_syms: Pilot OFDM symbol indices.
        enable_token_processor: Use a patch MLP instead of a linear projection.
        token_processor_hidden_dims: Patch MLP hidden widths; non-empty
            values also enable the MLP.
        decoder_out_proj: Project decoder tokens to patch_size. If False,
            keep d_model-wide tokens.
        enable_upsampling: Expand pilots to the full grid; otherwise return pilots.
        drop_pad_tokens: Skip window-padding tokens during inference when
            no PRB mask is supplied and decoder_out_proj is True.
        single_window_attn: For one partial window with drop_pad_tokens,
            'keep_pads' keeps padding, 'drop_pad_q' removes padded queries,
            and 'drop_pad_qk' removes padded queries and keys.
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
        drop_pad_tokens: bool  = True,
        single_window_attn: str = "keep_pads",
        name:            str   = "chea",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert single_window_attn in ("keep_pads", "drop_pad_q", "drop_pad_qk"), \
            f"single_window_attn must be 'keep_pads', 'drop_pad_q' or 'drop_pad_qk'"
        self.drop_pad_tokens    = bool(drop_pad_tokens)
        self.single_window_attn = single_window_attn

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

        # Window widths are measured in pilot subcarriers.
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
                # Four tokens per HR patch in the pooling group.
                _dec_pos_len = pool_factor * 4
            else:
                # Four tokens per HR patch in the window.
                _dec_pos_len = hr_prbs * 4
            self._dec_pos    = PosEncoding(_dec_pos_len, d_model, name="dec_pos")
            self._dec_blocks = [
                _DecoderBlock(d_model, n_heads, ffn_dim, ffn_depth=self.ffn_depth, name=f"dec_{i}")
                for i in range(dec_layers)
            ]
            if decoder_type == "lr_cross_attn":
                self._hr_skip_proj = layers.Dense(self.patch_size, use_bias=True, name="hr_skip_proj")
        else:  # MLP decoder.
            self._fc3_1 = layers.Dense(d_model, use_bias=True, name="fc3_1")
            self._fc3_2 = layers.Dense(d_model, use_bias=True, name="fc3_2")

        if self.enable_upsampling:
            if upsample_method == "I":
                self._ups_fc = layers.Dense(12 * 14 * 2, use_bias=True, name="ups_fc")
            else:  # Per-token upsampling.
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
        """Set grid dimensions and window counts, padding incomplete windows."""
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
        """Pad x and its optional mask to the internal subcarrier count."""
        if self._pad_subcarriers <= 0:
            return x, prb_mask
        paddings = [[0, 0], [0, self._pad_subcarriers], [0, 0], [0, 0]]
        x = tf.pad(x, paddings)
        if prb_mask is not None:
            prb_mask = tf.pad(tf.cast(prb_mask, tf.bool), paddings)
        return x, prb_mask

    def _pad_pilot_grid(self, pilots):
        """Pad pilots to the internal pilot-subcarrier count."""
        if self._pad_pilot_subcarriers <= 0:
            return pilots
        return tf.pad(pilots, [[0, 0], [0, self._pad_pilot_subcarriers], [0, 0], [0, 0]])

    def _pad_prb_mask(self, prb_mask):
        """Pad the optional PRB mask to the internal grid size."""
        if prb_mask is None or self._pad_subcarriers <= 0:
            return prb_mask
        return tf.pad(
            tf.cast(prb_mask, tf.bool),
            [[0, 0], [0, self._pad_subcarriers], [0, 0], [0, 0]],
        )

    def _crop_full_grid(self, y):
        """Crop the full grid to the input subcarrier count."""
        if self._pad_subcarriers <= 0:
            return y
        return y[:, :self._input_F, :, :]

    def _crop_pilot_grid(self, y):
        """Remove padded pilots, accounting for expanded decoder tokens."""
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
        """Set up padding for [B, F, S, 2], with an optional matching mask shape."""
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
        """Extract pilots from x [B, F, S, 2].

        Returns [B, F//pilot_stride, len(pilot_syms), 2].
        """
        x = tf.gather(x, self._pilot_sub_idx, axis=1)
        x = tf.gather(x, self._pilot_sym_idx, axis=2)
        return x

    def _make_validity_masks(self, prb_mask, B):
        """Build grid and token masks for a batch of size B.

        Read valid subcarriers from prb_mask[:, :, 0, 0], if supplied.
        Otherwise, mark the input PRBs valid and exclude internal padding.

        Returns a dict of masks:
            output: [B, F, S, 2], or None.
            pilot_valid: [B, F_half].
            pilot_grid: [B, F_half, 1, 1].
            hr_global: [B, N_p_hr].
            hr_win: [B*N_wh, N_ph*4].
            lr_global: [B, N_p_lr].
            lr_win: [B*N_wl, N_pl*4].
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

        # Each valid PRB contributes its pilot subcarriers.
        pilot_valid = tf.repeat(prb_valid, repeats=pilots_per_prb, axis=1)
        pilot_valid = pilot_valid[:, :F_half]

        return self._make_masks_from_pilot_valid(pilot_valid, B, output_mask)

    def _make_masks_from_pilot_valid(self, pilot_valid, B, output_mask=None):
        """Convert pilot_valid [B, F//pilot_stride] to HR and LR token masks.

        Include the optional output_mask [B, F, S, 2] in the returned dict.
        """
        pilot_valid = tf.cast(pilot_valid, tf.bool)

        # One HR patch covers a PRB when pilot_stride * patch_size == 12.
        hr_patch_valid = tf.reshape(pilot_valid, [B, self._N_wh * self._N_ph, self.patch_size])
        hr_patch_valid = tf.reduce_any(hr_patch_valid, axis=-1)
        hr_token_valid = tf.repeat(hr_patch_valid, repeats=4, axis=1)        # [B, N_p_hr]

        # Keep an LR token if any contributing pilot is valid.
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
        """Reverse pilot validity and rebuild masks for right-aligned windows."""
        pilot_valid = tf.reverse(masks["pilot_valid"], axis=[1])
        return self._make_masks_from_pilot_valid(
            pilot_valid, B, output_mask=masks.get("output")
        )

    def _masked_lr_avg_pool(self, x, pilot_valid, B):
        """Pool LR windows, averaging only pilots marked in pilot_valid [B, F_half].

        Accepts x [B, N_wl, W_l, 2, 2] or [B*N_wl, W_l, 2, 2].
        Returns [B*N_wl, W_l//pool_factor, 2, 2].
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
        """Turn pilots [B, N_win, W, 2, 2] into [B*N_win, N_p*4, patch_size] tokens.

        Each window has N_p=W//patch_size patches, with four tokens per
        patch for the two pilot symbols and real/imag parts. Apply
        token_processor first, if supplied.
        """
        P = self.patch_size
        x = tf.reshape(pilots, [B, N_win, N_p, P, 2, 2])
        if token_processor is not None:
            x = token_processor(x)
        # Put pilot subcarriers last to form each token's features.
        x = tf.transpose(x, [0, 1, 2, 4, 5, 3])
        x = tf.reshape(x, [B * N_win, N_p * 4, P])
        return x

    def _from_tokens(self, tokens, B, N_win, N_p, F_half):
        """Restore a pilot grid from tokens [B*N_win, N_p*4, token_width].

        Returns [B, F_half*M, 2, 2], where M=token_width//patch_size and
        F_half=F//pilot_stride.
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
        # Restore subcarrier, pilot-symbol, and real/imag axes.
        x = tf.transpose(x, [0, 1, 2, 5, 3, 4])
        x = tf.reshape(x, [B, F_half * M, 2, 2])
        return x

    def _hr_branch(self, pilots, B, masks):
        """Encode pilots [B, F//pilot_stride, 2, 2] in HR windows using masks.

        Returns (pilot_grid, memory [B, N_p_hr, d_model]). The grid matches
        the pilot shape, or is None for 'lr_cross_attn'.
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
        """Pool and encode pilots [B, F//pilot_stride, 2, 2] using masks.

        For 'lr_cross_attn', return (None, memory [B, N_p_lr, d_model]).
        Otherwise, project and bilinearly resize to the input pilot shape,
        returning (pilot_grid, None).
        """
        F_half = self._F_half
        N_wl, N_pl = self._N_wl, self._N_pl
        W_l = self._W_l
        f_p = self.pool_factor
        token_mask = masks["lr_win"]

        x = tf.reshape(pilots, [B, N_wl, W_l, 2, 2])

        # Exclude invalid pilots from the average to preserve scale at the edges.
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

        # Expand each pooled patch from P to P*f_p subcarriers.
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
        """Decode HR and LR features into a refined pilot grid.

        Args:
            fused: Pilot grid [B, F//pilot_stride, 2, 2]; None for 'lr_cross_attn'.
            hr_memory: HR tokens [B, N_p_hr, d_model].
            lr_memory: LR tokens [B, N_p_lr, d_model], used by 'lr_cross_attn'.
            B: Batch size.
            masks: Grid and token masks from _make_validity_masks.

        Returns:
            [B, F//pilot_stride*M, 2, 2]. M is 1 with decoder_out_proj,
            otherwise d_model//patch_size.
        """
        F_half = self._F_half
        N_wh, N_ph = self._N_wh, self._N_ph

        if self.decoder_type == "lr_cross_attn":
            # Each f_p*4 HR query group attends to its matching four LR tokens.
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

        # The other decoders use the fused pilot grid as input.
        P = self.patch_size
        token_mask_global = masks["hr_global"]

        # Make four tokens per patch, one per pilot symbol and real/imag part.
        fused = fused * tf.cast(masks["pilot_grid"], fused.dtype)
        fused_r   = tf.reshape(fused, [B, N_wh, N_ph, P, 2, 2])
        fused_tok = tf.transpose(fused_r, [0, 1, 2, 4, 5, 3])
        fused_tok = tf.reshape(fused_tok, [B, self._N_p_hr, P])
        fused_tok = _apply_token_mask(fused_tok, token_mask_global)

        skip = self._dec_skip_proj(fused_tok) if self._dec_out_proj is not None else None

        if self.decoder_type == "cross_attn":
            # Each query window attends to the matching HR memory window.
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
        else:  # MLP decoder.
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
        """Encode and decode pilots [B, F//pilot_stride, len(pilot_syms), 2].

        Returns a refined pilot grid without upsampling. Its subcarrier
        axis grows by d_model//patch_size if decoder_out_proj is False.
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
        """Expand decoded pilots [B, F//pilot_stride*M, 2, 2] to [B, F, 14, 2].

        M is 1 with decoder_out_proj, otherwise d_model//patch_size.
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
            # Map each PRB's pilots to its full 12 x 14 x 2 grid.
            pilots_per_prb = F_half // N_prbs

            x = tf.reshape(x, [B, N_prbs, pilots_per_prb * M, 2, 2])
            x = tf.reshape(x, [B, N_prbs, pilots_per_prb * M * 4])
            # Predict all subcarriers, symbols, and real/imag parts for each PRB.
            x = self._ups_fc(x)
            x = tf.reshape(x, [B, N_prbs, 12, 14, 2])
            x = tf.reshape(x, [B, N_prbs * 12, 14, 2])
        else:  # Per-token upsampling.
            assert self.pilot_stride * self.patch_size == 12, (
                f"upsample_method='II' requires pilot_stride * patch_size == 12 "
                f"(got {self.pilot_stride} * {self.patch_size} = "
                f"{self.pilot_stride * self.patch_size})"
            )
            P = self.patch_size
            x = tf.reshape(x, [B, N_wh, N_ph, P * M, 2, 2])
            x = tf.transpose(x, [0, 1, 2, 4, 5, 3])
            x = tf.reshape(x, [B, N_prbs * 4, P * M])
            # Each token predicts 12 subcarriers across 7 symbols.
            x = self._ups_fc(x)
            x = tf.reshape(x, [B, N_prbs * 4, 12, 7])
            x = tf.reshape(x, [B, N_prbs, 2, 2, 12, 7])
            x = tf.transpose(x, perm=[0,1,4,5,2,3]) # [B, N_prbs, 12, 7, 2, 2]
            x = tf.reshape(x, [B, N_prbs, 12, 14, 2])
            x = tf.reshape(x, [B, N_prbs*12, 14, 2])

        return x

    def call(self, inputs, training=False):
        """Refine a resource grid, optionally restricting it to valid PRBs.

        Args:
            inputs: Grid [B, F, S, 2] or (x, prb_mask). The bool/0-1 mask
                has the same shape and marks valid PRBs starting at PRB 0.
            training: If True, use the padded path for training.

        Returns:
            Refined grid [B, F, S, 2], or pilots if upsampling is disabled.
        """
        if self._input_has_prb_mask:
            x, prb_mask = inputs
        else:
            x = inputs
            prb_mask = None

        if self._use_pad_free(prb_mask, training):
            refined = self._pf_refine(self._pf_extract_pilots(x), 0)
            if not self.enable_upsampling:
                return refined
            return self._pf_upsample(refined)

        x, prb_mask = self._pad_full_grid_inputs(x, prb_mask)
        B = tf.shape(x)[0]
        masks = self._make_validity_masks(prb_mask, B)

        # Clear invalid pilots before projection, pooling, or attention.
        pilots = self._extract_pilots(x)
        pilots = pilots * tf.cast(masks["pilot_grid"], pilots.dtype)

        refined = self._refine_pilots(pilots, B, masks)

        if not self.enable_upsampling:
            return self._crop_pilot_grid(refined)

        y = self._upsample(refined, B)                  # [B, padded_F, 14, 2]
        if masks["output"] is not None:
            y = y * tf.cast(masks["output"], y.dtype)
        return self._crop_full_grid(y)

    # Inference with drop_pad_tokens projects only valid tokens. The valid
    # pilots occupy [o, o + Fv) in the padded grid's coordinates.

    def _use_pad_free(self, prb_mask, training):
        return (self.drop_pad_tokens and prb_mask is None and not training
                and self.decoder_out_proj)

    def _pf_extract_pilots(self, x):
        """Extract [B, F//pilot_stride, len(pilot_syms), 2] pilots without padding."""
        x = tf.gather(x, tf.range(0, x.shape[1], delta=self.pilot_stride), axis=1)
        return tf.gather(x, tf.constant(self.pilot_syms, dtype=tf.int32), axis=2)

    def _pf_to_tokens(self, pilots, token_processor=None):
        """Tokenize [B, n_patch*P, 2, 2] as [B, n_patch*4, P], as in _to_tokens."""
        P = self.patch_size
        n_patch = pilots.shape[1] // P
        x = tf.reshape(pilots, [-1, 1, n_patch, P, 2, 2])
        if token_processor is not None:
            x = token_processor(x)
        x = tf.transpose(x, [0, 1, 2, 4, 5, 3])
        return tf.reshape(x, [-1, n_patch * 4, P])

    def _pf_from_tokens(self, tokens):
        """Restore [B, n_patch*W, 2, 2] from [B, n_patch*4, W] tokens."""
        n_patch, W = tokens.shape[1] // 4, tokens.shape[2]
        x = tf.reshape(tokens, [-1, n_patch, 2, 2, W])
        x = tf.transpose(x, [0, 1, 4, 2, 3])
        return tf.reshape(x, [-1, n_patch * W, 2, 2])

    def _pf_mha(self, mha, h, start, win, kv=None, kv_groups=None):
        """Run windowed attention, projecting only valid tokens.

        h [B, n, d] holds normalized queries at [start, start+n). Windows
        contain win tokens and start at global position 0. kv is aligned
        memory [B, n, d], or None for self-attention. kv_groups supplies
        valid memory [B*n_win, S_k, d] for each window instead.

        Partial windows get zero rows and masked keys after projection.
        single_window_attn controls whether those rows are kept when only
        one partial window is present.
        """
        n = h.shape[1]
        d, H, D = mha.d_model, mha.n_heads, mha.d_head
        w0, w1, front, back = _block_span(start, start + n, win)
        n_win = w1 - w0
        has_pads = front > 0 or back > 0
        single = n_win == 1 and has_pads
        drop_q = single and self.single_window_attn in ("drop_pad_q", "drop_pad_qk")
        drop_k = single and self.single_window_attn == "drop_pad_qk" and kv_groups is None

        q = mha.q_proj(h)
        if kv_groups is None:
            src = h if kv is None else kv
            k, v = mha.k_proj(src), mha.v_proj(src)
        else:
            k, v = mha.k_proj(kv_groups), mha.v_proj(kv_groups)

        def windows(t):
            return tf.reshape(_pad_axis1(t, front, back), [-1, win, d])

        q_w = q if drop_q else windows(q)
        kv_mask = None
        if kv_groups is None:
            k_w, v_w = (k, v) if drop_k else (windows(k), windows(v))
            if has_pads and not drop_k:
                valid = tf.constant([0.0] * front + [1.0] * n + [0.0] * back)
                valid = tf.reshape(valid, [1, n_win, win])
                kv_mask = tf.reshape(tf.tile(valid, [tf.shape(h)[0], 1, 1]), [-1, win])
        else:
            k_w, v_w = k, v

        def split(t):
            t = tf.reshape(t, [tf.shape(t)[0], t.shape[1], H, D])
            return tf.transpose(t, [0, 2, 1, 3])

        qh, kh, vh = split(q_w), split(k_w), split(v_w)
        scores = tf.matmul(qh, kh, transpose_b=True) * mha.scale
        if kv_mask is not None:
            kv_mask_f = tf.cast(kv_mask[:, None, None, :], scores.dtype)
            scores = scores + (1.0 - kv_mask_f) * tf.cast(-1e9, scores.dtype)
        weights = tf.nn.softmax(scores, axis=-1)
        attn = tf.matmul(weights, vh)
        attn = tf.transpose(attn, [0, 2, 1, 3])
        attn = tf.reshape(attn, [-1, q_w.shape[1], d])
        if not drop_q:
            attn = tf.reshape(attn, [-1, n_win * win, d])[:, front:front + n]
        return mha.out(attn)  # [B, n, d]

    def _pf_encoder_block(self, blk, x, start, win):
        x = x + self._pf_mha(blk.attn, blk.ln1(x), start, win)
        return x + blk.ffn(blk.ln2(x))

    def _pf_decoder_block(self, blk, x, start, win, memory=None, memory_groups=None):
        x = x + self._pf_mha(blk.self_attn, blk.ln1(x), start, win)
        x = x + self._pf_mha(blk.cross_attn, blk.ln2(x), start, win,
                             kv=memory, kv_groups=memory_groups)
        return x + blk.ffn(blk.ln3(x))

    def _pf_hr_branch(self, pilots, o):
        """Encode valid HR pilots; return (pilot_grid, memory, start).

        pilot_grid is [B, Fv, 2, 2], or None for 'lr_cross_attn'.
        memory is [B, n_hr, d]; start is its global token offset.
        """
        P, Fv = self.patch_size, pilots.shape[1]
        p0, p1, fr, bk = _block_span(o, o + Fv, P)
        tokens = self._pf_to_tokens(_pad_axis1(pilots, fr, bk), self._hr_token_processor)
        start = 4 * p0
        if not self.enable_token_processor:
            tokens = self._hr_proj(tokens)
        tokens = tokens + _periodic_pos(self._hr_pos, start, tokens.shape[1])
        for blk in self._hr_enc:
            tokens = self._pf_encoder_block(blk, tokens, start, self._N_ph * 4)
        if self.decoder_type == "lr_cross_attn":
            return None, tokens, start
        out = self._pf_from_tokens(self._hr_out(tokens))[:, fr:fr + Fv]
        return out, tokens, start

    def _pf_lr_branch(self, pilots, o):
        """Pool and encode valid LR pilots.

        Return (None, memory [B, n_lr, d]) for 'lr_cross_attn', or
        (pilot_grid [B, Fv, 2, 2], None) for the other decoders.
        """
        P, f, Fv = self.patch_size, self.pool_factor, pilots.shape[1]
        dtype = pilots.dtype

        g0, g1, fr, bk = _block_span(o, o + Fv, f)
        x = _pad_axis1(pilots, fr, bk)
        mask = tf.constant([0.0] * fr + [1.0] * Fv + [0.0] * bk, dtype=dtype)
        mask = tf.reshape(mask, [1, -1, 1, 1])
        x_sum = tf.nn.avg_pool2d(
            x * mask, ksize=[f, 1], strides=[f, 1], padding="VALID", data_format="NHWC",
        ) * tf.cast(f, dtype)
        count = tf.nn.avg_pool2d(
            mask, ksize=[f, 1], strides=[f, 1], padding="VALID", data_format="NHWC",
        ) * tf.cast(f, dtype)
        pooled = x_sum / tf.maximum(count, tf.cast(1.0, dtype))
        pooled = pooled * tf.cast(count > 0.0, pooled.dtype)

        l0, l1, lfr, lbk = _block_span(g0, g1, P)
        tokens = self._pf_to_tokens(_pad_axis1(pooled, lfr, lbk), self._lr_token_processor)
        start = 4 * l0
        if not self.enable_token_processor:
            tokens = self._lr_proj(tokens)
        tokens = tokens + _periodic_pos(self._lr_pos, start, tokens.shape[1])
        for blk in self._lr_enc:
            tokens = self._pf_encoder_block(blk, tokens, start, self._N_pl * 4)
        if self.decoder_type == "lr_cross_attn":
            return None, tokens

        out = self._lr_out(tokens)
        n_tok = out.shape[1]
        out = tf.reshape(out, [-1, P, 1, 1])
        out = tf.image.resize(out, [P * f, 1], method="bilinear")
        out = tf.reshape(out, [-1, n_tok, P * f])
        out = self._pf_from_tokens(out)
        s = o - l0 * P * f
        return out[:, s:s + Fv], None

    def _pf_refine(self, pilots, o):
        """Refine pilots [B, Fv, 2, 2] at offset o, skipping window-padding tokens."""
        P, Fv = self.patch_size, pilots.shape[1]
        hr_out, hr_mem, hr_start = self._pf_hr_branch(pilots, o)
        lr_out, lr_mem = self._pf_lr_branch(pilots, o)

        if self.decoder_type == "lr_cross_attn":
            Tg = self.pool_factor * 4
            x = self._dec_in_proj(hr_mem)
            x = x + _periodic_pos(self._dec_pos, hr_start, hr_mem.shape[1])
            lr_groups = tf.reshape(lr_mem, [-1, 4, self.d_model])
            for blk in self._dec_blocks:
                x = self._pf_decoder_block(blk, x, hr_start, Tg, memory_groups=lr_groups)
            out_tok = self._dec_out_proj(x) + self._hr_skip_proj(hr_mem)
            s = o - (hr_start // 4) * P
            return self._pf_from_tokens(out_tok)[:, s:s + Fv]

        if self.fuse_mode == "sum":
            fused = hr_out + lr_out
        else:
            fused = self._fuse_conv(tf.concat([hr_out, lr_out], axis=-1))

        p0, p1, fr, bk = _block_span(o, o + Fv, P)
        fused_tok = self._pf_to_tokens(_pad_axis1(fused, fr, bk))
        skip = self._dec_skip_proj(fused_tok)
        if self.decoder_type == "cross_attn":
            x = self._dec_in_proj(fused_tok)
            x = x + _periodic_pos(self._dec_pos, hr_start, fused_tok.shape[1])
            for blk in self._dec_blocks:
                x = self._pf_decoder_block(blk, x, hr_start, self._N_ph * 4, memory=hr_mem)
            out_tok = self._dec_out_proj(x)
        else:
            x = tf.keras.activations.gelu(self._dec_in_proj(fused_tok), approximate=True)
            x = tf.keras.activations.gelu(self._fc3_2(x), approximate=True)
            out_tok = self._dec_out_proj(x)
        out_tok = out_tok + skip
        return self._pf_from_tokens(out_tok)[:, fr:fr + Fv]

    def _pf_upsample(self, refined):
        """Expand valid pilots [B, F//pilot_stride, 2, 2] to [B, F, 14, 2]."""
        ppp = 12 // self.pilot_stride
        n_prb = refined.shape[1] // ppp
        if self.upsample_method == "I":
            x = tf.reshape(refined, [-1, n_prb, ppp * 4])
            x = self._ups_fc(x)
            x = tf.reshape(x, [-1, n_prb, 12, 14, 2])
            return tf.reshape(x, [-1, n_prb * 12, 14, 2])
        assert self.pilot_stride * self.patch_size == 12
        P = self.patch_size
        x = tf.reshape(refined, [-1, n_prb, P, 2, 2])
        x = tf.transpose(x, [0, 1, 3, 4, 2])
        x = tf.reshape(x, [-1, n_prb * 4, P])
        x = self._ups_fc(x)
        x = tf.reshape(x, [-1, n_prb, 2, 2, 12, 7])
        x = tf.transpose(x, perm=[0, 1, 4, 5, 2, 3])
        x = tf.reshape(x, [-1, n_prb, 12, 14, 2])
        return tf.reshape(x, [-1, n_prb * 12, 14, 2])

    def get_config(self):
        """Return the layer settings for serialization."""
        cfg = super().get_config()
        cfg.update({
            "drop_pad_tokens":    self.drop_pad_tokens,
            "single_window_attn": self.single_window_attn,
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
    """Refine a pilot grid without upsampling.

    Input and output use shape [B, F//pilot_stride, len(pilot_syms), 2].
    With residual_refine, return x + alpha * (refined - x). Alpha starts
    at residual_scale_init and can be learned or fixed.

    Args:
        residual_refine: Blend the refinement into the input using alpha.
        residual_scale_init: Starting value of alpha.
        trainable_residual_scale: Learn alpha during training.
        window_alignment: Align windows to the first ('left') or last
            ('right') subcarrier.
        **kwargs: Options passed to CHEA.
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
        """Set up padding and residual scaling for the input pilot grid."""
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
        """Refine pilots, with optional masking and residual scaling.

        Args:
            inputs: Pilot grid [B, F//pilot_stride, len(pilot_syms), 2], or
                (pilot_grid, prb_mask).
            validity_masks: Prebuilt masks from CHEA._make_validity_masks.
            training: If True, use the padded path for training.

        Returns:
            Refined pilots with the same shape as the input grid.
        """
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            pilots, prb_mask = inputs
        else:
            pilots, prb_mask = inputs, None

        if validity_masks is None and self._use_pad_free(prb_mask, training):
            return self._pf_stage(pilots)

        if validity_masks is None:
            pilots = self._pad_pilot_grid(pilots)
            prb_mask = self._pad_prb_mask(prb_mask)

        B = tf.shape(pilots)[0]
        masks = validity_masks
        if masks is None:
            masks = self._make_validity_masks(prb_mask, B)

        original_masks = masks
        if self.window_alignment == "right":
            # Reverse pilots so windows start at the high-frequency edge.
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

    def _pf_stage(self, pilots):
        """Refine unpadded pilots [B, Fv, len(pilot_syms), 2].

        For right alignment, reverse the pilots and use offset
        o=padded_F_half-Fv to match the padded grid's window positions.
        """
        Fv = pilots.shape[1]
        right = self.window_alignment == "right"
        p = tf.reverse(pilots, axis=[1]) if right else pilots
        o = _ceil_to_multiple(Fv, _lcm(self._W_h, self._W_l)) - Fv if right else 0
        refined = self._pf_refine(p, o)
        if self.residual_refine:
            refined = p + tf.cast(self._residual_scale, refined.dtype) * (refined - p)
        return tf.reverse(refined, axis=[1]) if right else refined

    def get_config(self):
        """Return the stage settings for serialization."""
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
    """Refine pilots through independent CHEAStage layers, then upsample once.

    Accepts and returns a resource grid [B, F, S, 2], with an optional PRB
    mask as in CHEA.

    Args:
        num_stages: Number of refinement stages.
        residual_refine: Blend each stage's refinement using a residual scale.
        residual_scale_init: Starting value of each residual scale.
        trainable_residual_scale: Learn the residual scales during training.
        stage_window_alignment: 'left', 'right', or 'alternate' (left, then
            right). A sequence of 'left'/'right' values cycles over stages.
        **kwargs: CHEA options shared by all stages.
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
            "drop_pad_tokens": self.drop_pad_tokens,
            "single_window_attn": self.single_window_attn,
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
        """Validate the alignment setting and convert sequences to tuples."""
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
        """Choose a stage's alignment from the configured pattern."""
        alignment = self.stage_window_alignment
        if alignment == "alternate":
            return "right" if index % 2 else "left"
        if isinstance(alignment, str):
            return alignment
        return alignment[index % len(alignment)]

    def call(self, inputs, training=False):
        """Run all pilot refinement stages, then expand to the full grid.

        Args:
            inputs: Grid [B, F, S, 2] or (x, prb_mask). The bool/0-1 mask
                has the same shape and marks valid PRBs starting at PRB 0.
            training: Training flag passed to each stage.

        Returns:
            Refined grid [B, F, S, 2].
        """
        if self._input_has_prb_mask:
            x, prb_mask = inputs
        else:
            x = inputs
            prb_mask = None

        if self._use_pad_free(prb_mask, training):
            refined = self._pf_extract_pilots(x)
            for stage in self._stages:
                refined = stage(refined, training=training)
            return self._pf_upsample(refined)

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
        """Return the stack settings for serialization."""
        cfg = super().get_config()
        cfg.update({
            "num_stages": self.num_stages,
            "residual_refine": self.residual_refine,
            "residual_scale_init": self.residual_scale_init,
            "trainable_residual_scale": self.trainable_residual_scale,
            "stage_window_alignment": self.stage_window_alignment,
        })
        return cfg
