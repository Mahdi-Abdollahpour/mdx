"""
Residual CNN channel estimators (InterpolateNet baselines).

TensorFlow re-implementations of the residual channel estimators in the MATLAB
reference code of D. Luan and J. Thompson (InterpolateNet, 2021), used as
baselines in the CHEA paper (arXiv:2607.16462):

  InterpolationResNet    InterpolateNet: residual CNN on the LS pilot estimates
                         followed by bilinear upsampling to the full grid.
  ResidualTransposedNet  ReEsNet-style variant with transposed-conv upsampling.

Tensors are NHWC with the last axis holding [real, imag]. InterpolationResNet
takes the full grid [B, F, 14, 2], keeps the pilots on every other subcarrier
and on the pilot OFDM symbols ([B, F/2, 2, 2]) and returns the full-grid
estimate [B, F, 14, 2]. It has 5,410 + 56F trainable parameters (9,442 for
F=72) and ~4.45M MACs per sample for F=72, dominated by the output conv.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers

from .registry import register_block


@register_block
class InterpolationResNet(tf.keras.Model):
    """
    InterpolateNet: residual CNN with bilinear-interpolation upsampling.

    `conv_1` is followed by four residual blocks (conv -> ReLU -> conv, plus
    skip) and `conv_10`. The outputs of `conv_1`, the four residual blocks and
    `conv_10` are summed, bilinearly resized to the full [F, 14] grid and
    mapped to two channels by an (F/2 x 7) output convolution.

    Args:
        filters: width of the 3x3 convolutions (default 8).
        pilot_sym_idx: OFDM symbol indices that carry pilots (default (2, 11)).
        l2: L2 kernel regularization (default 1e-11).

    Input / output shape: [B, F, 14, 2] -> [B, F, 14, 2]; F is inferred in build().
    """

    def __init__(
        self,
        filters=8,
        pilot_sym_idx=(2, 11),
        l2=1e-11,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._l2 = l2
        self._filters = filters
        self._pilot_sym_idx = tf.constant(list(pilot_sym_idx), dtype=tf.int32)
        l2r = regularizers.L2(l2)

        def _conv(name):
            return layers.Conv2D(
                filters, (3, 3),
                padding='same', use_bias=True,
                kernel_regularizer=l2r, name=name,
            )

        self.conv_1  = _conv('conv_1')

        self.conv_2  = _conv('conv_2')
        self.conv_3  = _conv('conv_3')

        self.conv_4  = _conv('conv_4')
        self.conv_5  = _conv('conv_5')

        self.conv_6  = _conv('conv_6')
        self.conv_7  = _conv('conv_7')

        self.conv_8  = _conv('conv_8')
        self.conv_9  = _conv('conv_9')

        self.conv_10 = _conv('conv_10')

    def build(self, input_shape):
        F = input_shape[1]
        self._F = F
        self._output_size = (F, 14)
        self._pilot_sc_idx = tf.cast(tf.range(0, F, 2), tf.int32)
        # Output kernel (F//2, 7) matches the original [36, 7] for F=72.
        self.conv_out = layers.Conv2D(
            2, (F // 2, 7), padding='same', use_bias=True,
            kernel_regularizer=regularizers.L2(self._l2), name='conv_out',
        )
        super().build(input_shape)

    def call(self, x, training=False):
        # x : [B, F, 14, 2]
        x = tf.gather(x, self._pilot_sc_idx, axis=1)
        x = tf.gather(x, self._pilot_sym_idx, axis=2)  # [B, F/2, 2,  2]

        c1 = self.conv_1(x)                          # [B, F/2, 2, filters]

        y  = tf.nn.relu(self.conv_2(c1))
        a1 = c1 + self.conv_3(y)

        y  = tf.nn.relu(self.conv_4(a1))
        a2 = a1 + self.conv_5(y)

        y  = tf.nn.relu(self.conv_6(a2))
        a3 = a2 + self.conv_7(y)

        y  = tf.nn.relu(self.conv_8(a3))
        a4 = a3 + self.conv_9(y)

        c10 = self.conv_10(a4)

        agg = c10 + c1 + a1 + a2 + a3 + a4

        # tf.image.resize uses half-pixel centres, as the original bilinear resize.
        agg = tf.image.resize(agg, self._output_size, method='bilinear')  # [B, F, 14, filters]

        return self.conv_out(agg)                    # [B, F, 14, 2]


class ResidualTransposedNet(tf.keras.Model):
    """
    ReEsNet-style residual CNN with transposed-convolution upsampling.

    Same residual trunk as `InterpolationResNet`, but only the `conv_1` and
    `conv_10` outputs are summed. An 11x11 transposed conv with stride (2, 7)
    upsamples [36, 2] to [72, 14] and a 3x3 conv maps to two channels. The
    input is the pilot estimate, not the full grid.

    Input / output shape: [B, 36, 2, 2] -> [B, 72, 14, 2]
    """

    def __init__(
        self,
        filters=16,
        l2=1e-11,
        **kwargs,
    ):
        super().__init__(**kwargs)
        l2r = regularizers.L2(l2)

        def _conv(name):
            return layers.Conv2D(
                filters, (3, 3), padding='same', use_bias=True,
                kernel_regularizer=l2r, name=name,
            )

        self.conv_1  = _conv('conv_1')

        self.conv_2  = _conv('conv_2')
        self.conv_3  = _conv('conv_3')

        self.conv_4  = _conv('conv_4')
        self.conv_5  = _conv('conv_5')

        self.conv_6  = _conv('conv_6')
        self.conv_7  = _conv('conv_7')

        self.conv_8  = _conv('conv_8')
        self.conv_9  = _conv('conv_9')

        self.conv_10 = _conv('conv_10')

        self.transposed_conv = layers.Conv2DTranspose(
            filters, (11, 11), strides=(2, 7), padding='same', use_bias=True,
            kernel_regularizer=l2r, name='transposed_conv',
        )

        self.conv_11 = layers.Conv2D(
            2, (3, 3), padding='same', use_bias=True,
            kernel_regularizer=l2r, name='conv_11',
        )

    def call(self, x, training=False):
        # x : [B, 36, 2, 2]

        c1 = self.conv_1(x)                         # [B, 36, 2, 16]

        y  = tf.nn.relu(self.conv_2(c1))
        a1 = c1 + self.conv_3(y)

        y  = tf.nn.relu(self.conv_4(a1))
        a2 = a1 + self.conv_5(y)

        y  = tf.nn.relu(self.conv_6(a2))
        a3 = a2 + self.conv_7(y)

        y  = tf.nn.relu(self.conv_8(a3))
        a4 = a3 + self.conv_9(y)

        c10 = self.conv_10(a4)

        agg = c10 + c1

        up  = self.transposed_conv(agg)             # [B, 72, 14, 16]

        return self.conv_11(up)                     # [B, 72, 14, 2]


def train_resnet(
    model,
    X_train,
    Y_train,
    X_val,
    Y_val,
    epochs=100,
    batch_size=128,
    initial_lr=1e-3,
    drop_epoch=100,
    drop_rate=0.5,
):
    """
    Train a residual channel estimation network with the original recipe.

    Adam (lr 1e-3), MSE loss, learning rate multiplied by `drop_rate` at
    `drop_epoch`, shuffling every epoch. L2 regularization is set on the model.

    Args:
        model    : InterpolationResNet or ResidualTransposedNet instance
        X_train  : float32 model input, e.g. [N_train, 72, 14, 2] or [N_train, 36, 2, 2]
        Y_train  : float32 [N_train, 72, 14, 2], reference channel
        X_val    : float32 validation input
        Y_val    : float32 [N_val, 72, 14, 2]

    Returns:
        Keras History object.
    """
    def lr_schedule(epoch, current_lr):
        if epoch == drop_epoch:
            return float(current_lr * drop_rate)
        return float(current_lr)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss=tf.keras.losses.MeanSquaredError(),
    )

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

    return history


if __name__ == '__main__':

    dummy_irn = tf.zeros([4, 72, 14, 2], dtype=tf.float32)
    dummy_rtn = tf.zeros([4, 36,  2, 2], dtype=tf.float32)

    print('=== InterpolationResNet shape check ===')
    interp_net = InterpolationResNet(name='InterpolationResNet')
    out = interp_net(dummy_irn, training=False)
    assert out.shape == (4, 72, 14, 2), f'Bad shape: {out.shape}'
    print(f'  Input  : {tuple(dummy_irn.shape)}')
    print(f'  Output : {tuple(out.shape)}')
    print(f'  Params : {interp_net.count_params():,}')

    print('\n=== ResidualTransposedNet shape check ===')
    transposed_net = ResidualTransposedNet(name='ResidualTransposedNet')
    out = transposed_net(dummy_rtn, training=False)
    assert out.shape == (4, 72, 14, 2), f'Bad shape: {out.shape}'
    print(f'  Input  : {tuple(dummy_rtn.shape)}')
    print(f'  Output : {tuple(out.shape)}')
    print(f'  Params : {transposed_net.count_params():,}')

    print('\n=== Smoke-test (2 epochs, random data) ===')
    rng = np.random.default_rng(0)
    N   = 200
    Xi  = rng.standard_normal((N, 72, 14, 2)).astype(np.float32)
    Xr  = rng.standard_normal((N, 36,  2, 2)).astype(np.float32)
    Yd  = rng.standard_normal((N, 72, 14, 2)).astype(np.float32)

    for name, net, Xd in [('InterpolationResNet', interp_net, Xi),
                           ('ResidualTransposedNet', transposed_net, Xr)]:
        print(f'\n  [{name}]')
        hist = train_resnet(
            net,
            Xd[:160], Yd[:160],
            Xd[160:], Yd[160:],
            epochs=2,
            batch_size=32,
        )
        print(f'    Epoch 1 loss: {hist.history["loss"][0]:.4f}')
        print(f'    Epoch 2 loss: {hist.history["loss"][1]:.4f}')

    print('\nAll checks passed.')
