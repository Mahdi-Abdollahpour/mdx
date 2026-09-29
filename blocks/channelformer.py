"""
Channelformer: attention-based channel estimator (baseline).

TensorFlow re-implementation of the offline Channelformer of
  D. Luan and J. Thompson, "Channelformer: Attention based neural solution for
  wireless channel estimation and effective online training," IEEE Trans.
  Wireless Commun., 2023,
following the authors' MATLAB reference code. It is used as a baseline in the
CHEA paper (arXiv:2607.16462).

Architecture: one transformer encoder block on the LS pilot estimates, a 5x5
conv, three residual CNN decoder blocks, a dense expansion from the F pilot
values to the F*S grid positions and a 5x5 output conv.

Tensors are NHWC; the last axis of input and output holds [real, imag]:
  input  [B, F, S, 2]  LS estimate on the full grid (pilots extracted internally)
  output [B, F, S, 2]  estimated channel

With the defaults (2 heads, E=5 encoder FFN channels, D=12 decoder channels),
F=72 and S=14, the model has 117,709 trainable parameters (63% in the dense
expansion) and ~5.56M MACs per sample.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from .registry import register_block


class EncoderBlock(layers.Layer):
    """Post-LN transformer encoder block (Channelformer encoder).

    The real/imag axis forms a length-2 sequence and the F pilot values the
    features: multi-head self-attention on [B, 2, F] with residual and
    LayerNorm, then a two-layer 3x3 conv FFN (tanh-approximate GELU) on
    [B, F, 2, 1] with residual and LayerNorm over F.

    Args:
        d_model: F, feature dimension (default 72).
        num_heads: number of attention heads (default 2).
        ffn_filters: E, intermediate channels of the conv FFN (default 5).
        l2: L2 regularization of kernels and LayerNorm gammas (default 1e-10).

    Input / output shape: [B, F, 2, 1].
    """

    def __init__(self, d_model=72, num_heads=2, ffn_filters=5, l2=1e-10, **kwargs):
        super().__init__(**kwargs)
        l2r = regularizers.L2(l2)
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        self.attn_qkv = layers.Dense(
            3 * d_model, use_bias=True,
            kernel_regularizer=l2r,
        )

        self.attn_proj = layers.Dense(
            d_model, use_bias=True,
            kernel_regularizer=l2r,
        )

        self.ln1 = layers.LayerNormalization(
            axis=-1, epsilon=1e-5,
            gamma_regularizer=l2r,
        )

        self.ffn_conv1 = layers.Conv2D(
            ffn_filters, (3, 3), padding='same', use_bias=True,
            kernel_regularizer=l2r,
        )

        self.ffn_conv2 = layers.Conv2D(
            1, (3, 3), padding='same', use_bias=True,
            kernel_regularizer=l2r,
        )

        # applied to [B, F, 2, 1]: normalize over F
        self.ln2 = layers.LayerNormalization(
            axis=1, epsilon=1e-5,
            gamma_regularizer=l2r,
        )

    def call(self, x, training=False):
        # x: [B, F, 2, 1]
        x_seq = tf.squeeze(x, axis=-1)
        x_seq = tf.transpose(x_seq, [0, 2, 1])      # [B, 2, F]

        c = self.attn_qkv(x_seq, training=training)  # [B, 2, 3F]
        q, k, v = tf.split(c, 3, axis=-1)

        B = tf.shape(x_seq)[0]
        q = tf.reshape(q, [B, 2, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])           # [B, H, 2, head_dim]
        k = tf.reshape(k, [B, 2, self.num_heads, self.head_dim])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.reshape(v, [B, 2, self.num_heads, self.head_dim])
        v = tf.transpose(v, [0, 2, 1, 3])

        scale = tf.sqrt(tf.cast(self.head_dim, tf.float32))
        scores = tf.matmul(q, k, transpose_b=True)  # [B, H, 2, 2]
        scores = scores / scale
        weights = tf.nn.softmax(scores, axis=-1)
        attn_out = tf.matmul(weights, v)             # [B, H, 2, head_dim]

        attn_out = tf.transpose(attn_out, [0, 2, 1, 3])
        attn_out = tf.reshape(attn_out, [B, 2, self.d_model])  # [B, 2, F]
        attn_out = self.attn_proj(attn_out, training=training)
        a = self.ln1(attn_out + x_seq)

        a = tf.transpose(a, [0, 2, 1])
        a = tf.expand_dims(a, axis=-1)              # [B, F, 2, 1]

        z = self.ffn_conv1(a)                       # [B, F, 2, E]
        z = tf.keras.activations.gelu(z, approximate=True)
        z = self.ffn_conv2(z)                       # [B, F, 2, 1]
        z = self.ln2(z + a)

        return z


class DecoderBlock(layers.Layer):
    """Residual CNN decoder block: 5x5 conv, ReLU, 5x5 conv, residual, LayerNorm over F.

    Args:
        filters: D, number of channels (default 12).
        l2: L2 regularization of kernels and LayerNorm gamma (default 1e-10).

    Input / output shape: [B, F, 2, D].
    """

    def __init__(self, filters=12, l2=1e-10, **kwargs):
        super().__init__(**kwargs)
        l2r = regularizers.L2(l2)

        self.conv1 = layers.Conv2D(
            filters, (5, 5), padding='same', use_bias=True,
            kernel_regularizer=l2r,
        )

        self.conv2 = layers.Conv2D(
            filters, (5, 5), padding='same', use_bias=True,
            kernel_regularizer=l2r,
        )

        self.ln = layers.LayerNormalization(
            axis=1, epsilon=1e-5,
            gamma_regularizer=l2r,
        )

    def call(self, x, training=False):
        y = self.conv1(x)
        y = tf.nn.relu(y)
        y = self.conv2(y)
        y = y + x
        y = self.ln(y)

        return y


@register_block
class Channelformer(tf.keras.Model):
    """Offline Channelformer channel estimator (Luan & Thompson, IEEE TWC 2023).

    Extracts the LS pilot estimates (every other subcarrier, OFDM symbols 2 and
    11), then applies the encoder blocks, a 5x5 conv to D channels, the
    residual decoder blocks, a dense expansion from F to F*S along the
    frequency axis and a 5x5 conv back to one channel, and finally reshapes
    the result to the resource grid.

    Args:
        num_heads: attention heads (default 2).
        encoder_layers: number of encoder blocks (default 1).
        decoder_layers: number of decoder blocks (default 3).
        encoder_ffn_filters: E, encoder FFN channels (default 5).
        decoder_filters: D, decoder channels (default 12).
        l2: L2 regularization (default 1e-10).

    Input / output shape: [B, F, S, 2]; F and S must be static.
    """

    def __init__(
        self,
        num_heads=2,
        encoder_layers=1,
        decoder_layers=3,
        encoder_ffn_filters=5,
        decoder_filters=12,
        l2=1e-10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        l2r = regularizers.L2(l2)
        self.num_heads = int(num_heads)
        self.encoder_layers = int(encoder_layers)
        self.encoder_ffn_filters = int(encoder_ffn_filters)
        self.l2 = l2
        self.num_subcarriers = None
        self.output_length = None
        self.S = None
        self._pilot_subcarrier_idx = None
        self._pilot_symbol_idx = tf.constant([2, 11], dtype=tf.int32)
        self._pilot_reshape = None
        self._output_reshape = None

        self.encoder_blocks = []

        self.dec_input_conv = layers.Conv2D(
            decoder_filters, (5, 5), padding='same', use_bias=True,
            kernel_regularizer=l2r, name='dec_input_conv',
        )

        self.decoder_blocks = [
            DecoderBlock(decoder_filters, l2=l2, name=f'decoder_{j}')
            for j in range(decoder_layers)
        ]

        self.fc_expand = None

        self.dec_output_conv = layers.Conv2D(
            1, (5, 5), padding='same', use_bias=True,
            kernel_regularizer=l2r, name='dec_output_conv',
        )

    def build(self, input_shape):
        if input_shape[1] is None or input_shape[2] is None:
            raise ValueError('Channelformer requires static subcarrier and OFDM symbol dimensions')

        self.num_subcarriers = int(input_shape[1])
        self.S = int(input_shape[2])
        self.output_length = self.num_subcarriers * self.S
        self._pilot_subcarrier_idx = tf.range(0, self.num_subcarriers, delta=2)
        self._pilot_reshape = (-1, self.num_subcarriers, 2, 1)
        self._output_reshape = (-1, self.num_subcarriers, self.S, 2)

        self.encoder_blocks = [
            EncoderBlock(
                d_model=self.num_subcarriers,
                num_heads=self.num_heads,
                ffn_filters=self.encoder_ffn_filters,
                l2=self.l2,
                name=f'encoder_{i}',
            )
            for i in range(self.encoder_layers)
        ]
        self.fc_expand = layers.Dense(
            self.output_length, use_bias=True,
            kernel_regularizer=regularizers.L2(self.l2), name='fc_expand',
        )
        super().build(input_shape)

    def call(self, x, training=False):
        x = tf.gather(x, self._pilot_subcarrier_idx, axis=1)
        x = tf.gather(x, self._pilot_symbol_idx, axis=2)  # [B, F/2, 2, 2]
        x = tf.reshape(x, self._pilot_reshape)       # [B, F, 2, 1]

        for enc in self.encoder_blocks:
            x = enc(x, training=training)

        x = self.dec_input_conv(x)                  # [B, F, 2, D]

        for dec in self.decoder_blocks:
            x = dec(x, training=training)

        # dense expansion along the frequency axis: F -> F*S
        x = tf.transpose(x, [0, 2, 3, 1])
        x = self.fc_expand(x)                        # [B, 2, D, F*S]
        x = tf.transpose(x, [0, 3, 1, 2])            # [B, F*S, 2, D]

        x = self.dec_output_conv(x)
        x = tf.reshape(x, self._output_reshape)       # [B, F, S, 2]
        return x


def build_model(**kwargs):
    """Return a Channelformer built from the given constructor arguments."""
    return Channelformer(**kwargs)


def train_offline(
    X_train,
    Y_train,
    X_val,
    Y_val,
    epochs=100,
    batch_size=128,
    initial_lr=2e-3,
    drop_epoch=50,
    drop_rate=0.5,
    l2=1e-10,
    **model_kwargs,
):
    """Train an offline Channelformer on pre-generated data (original recipe).

    Adam with Huber loss (delta=1), learning rate multiplied by `drop_rate` at
    `drop_epoch`, L2 regularization `l2`, shuffling every epoch.

    Args:
        X_train, X_val: float32 [N, F, S, 2], LS estimates.
        Y_train, Y_val: float32 [N, F, S, 2], reference channels.
        epochs, batch_size, initial_lr, drop_epoch, drop_rate, l2: training
            hyper-parameters.
        **model_kwargs: additional Channelformer constructor arguments.

    Returns:
        (model, history): trained Channelformer and Keras History object.
    """
    model = build_model(l2=l2, **model_kwargs)

    def lr_schedule(epoch, current_lr):
        if epoch == drop_epoch:
            return float(current_lr * drop_rate)
        return float(current_lr)

    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    loss_fn   = tf.keras.losses.Huber(delta=1.0)

    model.compile(optimizer=optimizer, loss=loss_fn)

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1),
        ],
    )

    return model, history

