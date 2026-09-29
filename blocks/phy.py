# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026

"""PHY blocks for channel estimation, equalization, demapping, and related losses."""

import tensorflow as tf
from tensorflow.keras import layers
from sionna.mapping import Demapper, Constellation
from sionna.nr import LayerDemapper
from sionna.ofdm import ResourceGridDemapper
from sionna.utils import expand_to_rank, flatten_last_dims
from utils import huber, mse
from .registry import register_block
from .common import HFreqNormalizer


@register_block(needs=('sys',))
class LMMSE(layers.Layer):
    r"""Learnable LMMSE equalizer for multi-user MIMO uplink.

    For every resource element, with received vector :math:`\mathbf{y}`,
    channel estimate :math:`\mathbf{H}` and noise plus channel-estimation
    error covariance :math:`\mathbf{S}`, it computes
    :math:`\mathbf{W} = \mathbf{H}^\mathrm{H}(\mathbf{H}\mathbf{H}^\mathrm{H} + \mathbf{S})^{-1}`,
    the unbiased symbol estimate
    :math:`\hat{\mathbf{x}} = \operatorname{diag}(\mathbf{W}\mathbf{H})^{-1}\mathbf{W}\mathbf{y}`
    and the post-equalization error variance
    :math:`\sigma^2_\mathrm{eff} = \operatorname{diag}(\mathbf{W}\mathbf{H})^{-1} - 1`.

    Trainable multipliers scale :math:`\mathbf{S}` (``noise_inx``) and
    :math:`\sigma^2_\mathrm{eff}` (``noise_outx``), either with a scalar (1)
    or with a 12 x 14 matrix shared by all PRBs (2). With ``mcs_x=1`` and
    several MCSs, :math:`\sigma^2_\mathrm{eff}` is additionally scaled by one
    learnable scalar per MCS.

    Args:
        sys: System parameters; ``sys["num_mcss_supported"]`` is used.
        noise_inx: Input-noise scaling: 0 (off), 1 (scalar), 2 (per-PRB matrix).
        noise_outx: Output-variance scaling: 0 (off), 1 (scalar), 2 (per-PRB matrix).
        mcs_x: 1 enables the per-MCS output-variance scaling.
    """

    def __init__(   self,
                    sys,
                    noise_inx=1,
                    noise_outx=1,
                    mcs_x=1,
                    name="LMMSE",
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._cdtype = tf.complex64
        self._real_dtype = tf.as_dtype(dtype).real_dtype

        self._num_mcs = sys["num_mcss_supported"]
        self._noise_inx = noise_inx
        self._noise_outx = noise_outx
        self._mcs_x = mcs_x

    def build(self, input_shape):
        shape = None
        if self._noise_inx==2:
            shape=(12,14)
            name = f"in_noise_multiplier_mat"
        if self._noise_inx==1:
            shape=()
            name = f"in_noise_multiplier"
        if shape is not None:
            self._gamma = self.add_weight(
                name=name,
                shape=shape,
                initializer=tf.keras.initializers.Constant(1.),
                trainable=True,
                dtype = self._real_dtype
            )

        shape = None
        if self._noise_outx==2:
            shape=(12,14)
            name = f"out_noise_multiplier_mat"
        if self._noise_outx==1:
            shape=()
            name = f"out_noise_multiplier"
        if shape is not None:
            self._theta = self.add_weight(
                name=name,
                shape=shape,
                initializer=tf.keras.initializers.Constant(1.),
                trainable=True,
                dtype = self._real_dtype
            )

        if self._mcs_x == 1:
            self._mcs_mul = []
            if self._num_mcs>1:
                for i in range(self._num_mcs):
                    name = f"err_mcs_mul_{i}"
                    mcs_mul_ = self.add_weight(
                        name=name,
                        shape=(),
                        initializer=tf.keras.initializers.Constant(1.),
                        trainable=True,
                        dtype = self._real_dtype
                    )
                    self._mcs_mul.append(mcs_mul_)

        if isinstance(input_shape, (list, tuple)):
            self._multiple_ins = True
            self._num_ins = len(input_shape)
        else:
            self._multiple_ins = False
            self._num_ins = 1

        super().build(input_shape)

    def cholesky_inverse(self, matrix):
        """Invert a batch of Hermitian positive-definite matrices via Cholesky."""
        L = tf.linalg.cholesky(matrix)
        identity = tf.eye(tf.shape(L)[-1], dtype=L.dtype)
        matrix_inv_ = tf.linalg.cholesky_solve(L, identity)

        return matrix_inv_

    def lmmse(self, y, h, s):
        """Per-RE LMMSE equalization.

        Args:
            y: [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant], complex.
            h: [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant, num_tx], complex.
            s: noise/error covariance, broadcastable to
                [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant, num_rx_ant].

        Returns:
            (x_hat, no_eff), each [batch_size, num_subcarriers, num_ofdm_symbols, num_tx].
        """
        shape = tf.shape(y)
        num_prbs = shape[1]//12
        num_subcarriers = shape[1]
        num_ofdm_symbols = shape[2]
        num_rx_ant = shape[3]

        e = tf.constant(0.000001,self._real_dtype)
        s = tf.nn.relu( tf.math.real(s) ) + e
        s = tf.stop_gradient(s)

        if self._noise_inx==1:
            s = self._gamma * s
        if self._noise_inx==2:
            gamma = tf.expand_dims(self._gamma,axis=0)
            gamma = tf.expand_dims(gamma,axis=0)
            gamma = tf.expand_dims(gamma,axis=-1)
            gamma = tf.expand_dims(gamma,axis=-1)

            gamma = tf.broadcast_to(gamma, [1, num_prbs, 12, 14, num_rx_ant, num_rx_ant])
            gamma = tf.reshape(gamma, [1, num_subcarriers, 14, num_rx_ant, num_rx_ant])
            s = gamma * s

        s =  tf.complex(s,tf.zeros_like(s))

        g = tf.matmul(h, h, adjoint_b=True) + s           # [Nr,Nr]
        g_inv = self.cholesky_inverse(g)
        g = tf.matmul(h, g_inv, adjoint_a=True)     # [Nt,Nr]

        y = tf.expand_dims(y, -1)
        gy = tf.squeeze(tf.matmul(g, y), axis=-1)
        gh = tf.matmul(g, h)                        # [Nt,Nt]
        d = tf.linalg.diag_part(gh)
        x_hat = tf.math.divide_no_nan(gy,d)

        # post-equalization error variance: 1/diag(GH) - 1
        d = tf.math.real(d)
        one = tf.constant(1.,dtype=self._real_dtype, shape=())
        d = tf.cast(d, dtype=self._real_dtype)
        no_eff = tf.math.divide_no_nan(one,d) - one

        num_tx = tf.shape(x_hat)[-1]
        if self._noise_outx==1:
            no_eff = tf.cast(self._theta, self._real_dtype) * no_eff

        if self._noise_outx==2:
            theta = tf.expand_dims(self._theta,axis=0)
            theta = tf.expand_dims(theta,axis=0)
            theta = tf.expand_dims(theta,axis=-1)
            theta = tf.broadcast_to(theta, [1, num_prbs, 12, 14, num_tx])
            theta = tf.reshape(theta, [1, num_subcarriers, 14, num_tx])
            no_eff = tf.cast(theta, self._real_dtype) * no_eff

        no_eff =  tf.nn.relu( no_eff ) + e

        return x_hat, no_eff

    def call(self, x, rx_signal, noise, mcs_masks, shapes, training=False):
        """Equalize the received signal.

        Args:
            x: channel estimate, either [B*T*RA, F, 14, 2] (real/imag) or a
                tuple ``(h_ft, h_ft14, mask)`` with ``h_ft14`` [B*T, RA, 14, F]
                (complex).
            rx_signal: dict with ``"y"`` [B, num_rx, RA, 14, F] (complex).
            noise: dict with the error covariance ``"s"`` (single input), or
                with ``"no"`` [B] and ``"err_var"`` [B, 1, T, F] (tuple input).
            mcs_masks: dict with ``"mcs_ue_mask"`` [B, T, num_mcs].
            shapes: ``Shapes`` object with B, T, RA, F.

        Returns:
            (x_hat, no_eff), each [B, T, F, 14].
        """
        s = shapes

        if self._num_ins>1:
            h_ft, h_ft14, mask = x
            no = noise["no"]
            err_var = noise["err_var"]

            err_var = tf.reduce_sum(err_var, axis=2)

            # [batch_size, num_subcarriers, 1, 1, 1]
            err_var = tf.transpose(err_var, perm=[0,2,1])
            err_var = tf.expand_dims(tf.expand_dims(err_var,2),-1)
            if not isinstance(no, tf.Tensor):
                no = tf.constant(no, dtype=tf.float32)

            no_ = tf.reshape(no, [-1, 1, 1, 1, 1])
            err_var = tf.cast(err_var + no_, dtype=self._cdtype)

        if self._num_ins==1:
            h_ft14 = x
            h_ft14 = tf.reshape(h_ft14, [s.B*s.T, s.RA, s.F, 14, 2])
            # [B*T, num_rx_ant, 14, fft], complex
            h_ft14 = tf.transpose(h_ft14, perm=[0,1,3,2,4])
            h_ft14 = tf.complex(h_ft14[...,0], h_ft14[...,1])

            err_var = noise["s"]

        y = rx_signal["y"]
        mcs_ue_mask = mcs_masks["mcs_ue_mask"]

        # [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant]
        y = tf.transpose(y, perm=[0,1,4,3,2])
        y = y[:,0,...]

        # [batch_size, num_rx_ant, num_tx, num_ofdm_symbols, fft]
        num_time_steps = tf.shape(h_ft14)[2]
        h_hat = tf.reshape(h_ft14, [s.B, s.T, s.RA, num_time_steps, s.F])
        h_hat = tf.transpose(h_hat, perm=[0,2,1,3,4])

        # [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant, num_tx], tf.complex
        h_hat = tf.transpose(h_hat, perm=[0,4,3,1,2])

        # [batch_size, num_subcarriers, num_ofdm_symbols, num_tx]
        x_hat, no_eff = self.lmmse(y, h_hat, err_var)

        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols]
        x_hat = tf.transpose(x_hat, perm=[0,3,1,2])
        no_eff = tf.transpose(no_eff, perm=[0,3,1,2])

        # per-MCS scaling of the error variance
        if self._num_mcs>1 and self._mcs_x==1:
            no_eff_ = tf.zeros_like(no_eff)
            for i in range(self._num_mcs):
                mask_i = tf.expand_dims(mcs_ue_mask[:, :, i], axis=-1)
                mask_i = tf.expand_dims(mask_i, axis=-1)
                no_eff_ = no_eff_ + self._mcs_mul[i]* mask_i * no_eff
            no_eff = no_eff_

        return (x_hat, no_eff)


@register_block
class BDemapper_(layers.Layer):
    """Soft-output QAM demapper for one modulation order.

    Maps ``(x, no)``, each [batch_size, num_tx, num_subcarriers, num_ofdm_symbols],
    to LLRs [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_bits_per_symbol].
    """

    def __init__(   self,
                    num_bits_per_symbol,
                    demapping_type,
                    name="BD_",
                    dtype=tf.complex64,
                    **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._real_dtype = tf.as_dtype(dtype).real_dtype

        constellation_ = Constellation.create_or_check_constellation(
                                    "qam",
                                    num_bits_per_symbol,
                                    constellation=None,
                                    dtype=self._dtype)
        self._demapper = Demapper(demapping_type,
                                constellation=constellation_,
                                hard_out=False,
                                dtype=tf.as_dtype(self.dtype))

    def call(self,inputs):
        x, no = inputs

        x_shape = tf.shape(x)
        # [batch_size, num_tx, num_subcarriers* num_ofdm_symbols], complex
        x = flatten_last_dims(x,2)
        no = flatten_last_dims(no,2)

        llrs = self._demapper([x,no])

        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_bits_per_symbol]
        llrs = tf.reshape(llrs, tf.concat([x_shape, [-1]], axis=0) )

        return llrs


@register_block(needs=('sys',))
class BDemapper(layers.Layer):
    """Per-MCS demapping of equalized symbols to per-UE codeword LLRs.

    For each supported MCS, computes LLRs with a ``BDemapper_``, then applies
    resource-grid demapping (data symbols only) and NR layer demapping.
    """

    def __init__(   self,
                    sys,
                    name="BDemapper",
                    dtype=tf.complex64,
                    **kwargs):
        super().__init__(name=name, dtype=dtype,**kwargs)
        self._real_dtype = tf.as_dtype(dtype).real_dtype

        self._rg_demapper = ResourceGridDemapper(sys["rg"], sys["sm"])
        self._num_mcss_supported = sys["num_mcss_supported"]

        self._layer_demappers = []
        for mcs_list_idx in range(self._num_mcss_supported):
                self._layer_demappers.append(
                    LayerDemapper(
                            sys["transmitters"][mcs_list_idx]._layer_mapper,
                            sys["transmitters"][mcs_list_idx]._num_bits_per_symbol))

        demapping_type = sys["demapping_type"]
        self._num_bits_per_symbol=sys["num_bits_per_symbol"]
        self._bdemapper=[]
        for mcs_list_idx in range(self._num_mcss_supported):
            num_bits_per_symbol_ = self._num_bits_per_symbol[mcs_list_idx]

            bd = BDemapper_(num_bits_per_symbol_, demapping_type, dtype=dtype, name=f"BD_{mcs_list_idx}")
            self._bdemapper.append(bd)
        self._num_data_symbols = sys["rg"].pilot_pattern.num_data_symbols

    def call(self, x, mcs_masks, training=False):
        """Compute LLRs.

        Args:
            x: ``(x_hat, no_eff)``, each [batch_size, num_tx, num_subcarriers, num_ofdm_symbols].
            mcs_masks: dict with ``"mcs_arr_eval"``, the MCS indices to evaluate.
            training: If True, return one LLR tensor per evaluated MCS.

        Returns:
            LLRs [batch_size, num_tx, num_data_symbols*num_bits_per_symbol] of
            the first evaluated MCS, or a list of them if ``training``.
        """
        x_hat, no_eff = x

        mcs_arr_eval = mcs_masks["mcs_arr_eval"]

        llrs_ = []
        for idx in range(self._num_mcss_supported):
            llrs__ = self._bdemapper[idx]([x_hat, no_eff])
            llrs_.append(llrs__)

        indices = mcs_arr_eval
        # only keep the LLRs of the MCS indices in mcs_arr_eval
        llrs = []
        num_tx = tf.shape(x_hat)[1]
        batch_size = tf.shape(x_hat)[0]
        for idx in indices:
            llrs_[idx] = tf.cast(llrs_[idx], tf.float32)

            # [batch_size, 1, num_tx, num_ofdm_symbols, fft_size, num_bits_per_symbol]
            llrs_[idx] = tf.transpose(llrs_[idx], [0, 1, 3, 2, 4])
            llrs_[idx] = tf.expand_dims(llrs_[idx], axis=1)

            # [batch_size, num_tx, 1, num_data_symbols, num_bit_per_symbols]
            llrs_[idx] = self._rg_demapper(llrs_[idx])
            llrs_[idx] = llrs_[idx][:,:num_tx]

            llrs_[idx] = tf.reshape(llrs_[idx], [batch_size, num_tx, 1, self._num_data_symbols*self._num_bits_per_symbol[idx]])

            # [batch_size, num_tx, num_data_symbols*num_bit_per_symbols]
            if self._layer_demappers is None:
                llrs_[idx] = tf.squeeze(llrs_[idx], axis=-2)
            else:
                llrs_[idx] = self._layer_demappers[idx](llrs_[idx])
            llrs.append(llrs_[idx])

        if training:
            return llrs
        else:
            return llrs[0]


@register_block
class LLRLoss(layers.Layer):
    """Binary cross-entropy between LLRs and transmitted bits.

    The per-UE loss is restricted to the MCS each UE uses and, with
    ``snr_weighting``, weighted by ``log2(1 + 1/no)``. Returns a dict
    with ``"BCE"`` and ``"Total"`` (``lambda_tot`` times BCE), each [B*T];
    zeros when not training.
    """

    def __init__(   self,
                    lambda_tot=1.0,
                    snr_weighting=True,
                    name="LLRLoss",
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._real_dtype = tf.as_dtype(dtype).real_dtype

        self.snr_weighting = snr_weighting
        self.lambda_tot = tf.cast(lambda_tot, dtype=self.dtype)

        self._bce = tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE)

    def call(self, llrs_, bits, mcs_masks, noise, shapes, training=True):
        if not training:
            return {"BCE":0.,
                "Total": 0.}

        s=shapes
        no = noise["no"]
        mcs_arr_eval = mcs_masks["mcs_arr_eval"]
        mcs_ue_mask = mcs_masks["mcs_ue_mask"]

        batch_size = tf.shape(llrs_[0])[0]
        num_tx = tf.shape(llrs_[0])[1]

        L_b = []
        indices = mcs_arr_eval
        for idx in range(len(indices)):
            L_b_ = self._bce(bits[idx], llrs_[idx]) # [batch_size, num_tx]
            mcs_ue_mask_ = expand_to_rank(
                tf.gather(mcs_ue_mask, indices=indices[idx], axis=2),
                tf.rank(L_b_), axis=-1)

            # select data loss only for associated MCSs
            L_b_ = tf.multiply(L_b_, mcs_ue_mask_)

            if self.snr_weighting:
                snr_mul = tf.math.log(1 + 1/no) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
                snr_mul = expand_to_rank(snr_mul,2,-1)
                L_b_ = tf.multiply(L_b_, snr_mul)

            L_b.append(L_b_)
        L_b = tf.reduce_sum(tf.stack(L_b, axis=0), axis=0)

        L_b = tf.reshape(L_b, [s.B*s.T])

        total = L_b * self.lambda_tot
        return {"BCE":L_b,
                "Total": total}


@register_block
class ChLoss(layers.Layer):
    """Channel estimation loss (MSE or Huber) with optional mask and SNR weighting.

    ``y_pred`` is ``h_ft14``, ``(h_ft14, mask)`` or ``(h_ft, h_ft14, mask)``,
    with channels of shape [B*T*RA, F, S, 2] and a mask [B*T*RA, F, S, 1]
    (possibly the last element of a tuple). With ``time_steps='pilots'`` and a
    pilot-symbol estimate ``h_ft``, the loss is computed on the pilot symbols
    (2, 11) only. Returns a dict with ``"MSE"`` and ``"Total"``, each [B*T];
    zeros when not training.

    Args:
        lambda_tot: Weight of the total loss.
        snr_weighting: Weight each UE by ``log2(1 + 1/no)``.
        loss_type: ``'mse'`` or ``'huber'``.
        delta: Huber threshold.
        normalize_target: Normalize the target with ``HFreqNormalizer``.
        time_steps: ``'pilots'`` or ``'all'``.
    """

    def __init__(   self,
                    lambda_tot=1.0,
                    snr_weighting=True,
                    loss_type="mse",
                    delta=1.,
                    normalize_target=False,
                    time_steps="pilots",
                    name="ChLoss",
                    dtype=tf.complex64,
                    **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._real_dtype = tf.as_dtype(dtype).real_dtype

        self._snr_weighting = snr_weighting
        self._lambda_tot = tf.cast(lambda_tot, dtype=self._real_dtype)
        self._delta = tf.cast(delta, dtype=self._real_dtype)
        self._loss_type = loss_type
        self._time_steps = time_steps
        self._normalize_target = normalize_target

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            self._multiple_ins = True
            self._num_ins = len(input_shape)
        else:
            self._multiple_ins = False
            self._num_ins = 1
        if self._num_ins == 2:
            self._mask_is_tuple = isinstance(input_shape[1], (list, tuple))
        elif self._num_ins >= 3:
            self._mask_is_tuple = isinstance(input_shape[2], (list, tuple))
        else:
            self._mask_is_tuple = False
        if self._normalize_target:
            self.norm = HFreqNormalizer(dtype=self.dtype)

        super().build(input_shape)

    def _masked_mean(self, x, mask=None):
        """Per-sample (masked) mean of x [N, H, W, C]; mask broadcastable to x. Returns [N]."""
        if mask is None:
            return tf.reduce_mean(x, axis=(1, 2, 3))

        mask = tf.cast(mask, x.dtype)
        mask = tf.broadcast_to(mask, tf.shape(x))

        num = tf.reduce_sum(x * mask, axis=(1, 2, 3))
        den = tf.reduce_sum(mask, axis=(1, 2, 3))

        den = tf.maximum(den, tf.cast(1e-8, x.dtype))
        return num / den

    def call(self, y_pred, y_true, noise, shapes, training=True):
        if not training:
            return {
                "MSE": 0.,
                "Total": 0.,
            }

        h_ft = None

        if self._num_ins == 1:
            h_ft14 = y_pred
            mask = None
        elif self._num_ins == 2:
            h_ft14, mask = y_pred
        elif self._num_ins >= 3:
            h_ft, h_ft14, mask = y_pred[:3]

        if self._mask_is_tuple:
            mask = mask[-1]

        if self._normalize_target:
            y_true, msk = self.norm((y_true, mask), y_true=y_true, shapes=shapes)

        no = noise["no"]
        s = shapes
        h_true = y_true

        if self._time_steps == "pilots" and h_ft is not None:
            inds = tf.constant([2, 11], tf.int32)
            h_true = tf.gather(h_true, inds, axis=2)
            h_pred = h_ft
            if mask is not None:
                mask = tf.gather(mask, inds, axis=2)
        else:
            h_pred = h_ft14

        e_h = h_pred - h_true

        if self._loss_type == 'huber':
            L_h_elem = huber(e_h, delta=self._delta)
        elif self._loss_type == 'mse':
            L_h_elem = mse(e_h)
        else:
            raise NotImplementedError(
                f"Loss type '{self._loss_type}' is not implemented."
            )

        L_h_mse_elem = mse(e_h)

        # Per-sample masked reduction: [N]
        L_h = self._masked_mean(L_h_elem, mask=mask)
        L_h_mse = self._masked_mean(L_h_mse_elem, mask=mask)

        # [B*T*RA] -> [B*T, RA]
        L_h = tf.reshape(L_h, [s.B * s.T, s.RA])
        L_h_mse = tf.reshape(L_h_mse, [s.B * s.T, s.RA])

        L_h = tf.reduce_mean(L_h, axis=1)
        L_h_mse = tf.reduce_mean(L_h_mse, axis=1)

        if self._snr_weighting:
            if not isinstance(no, tf.Tensor):
                no = tf.constant(no, dtype=self._real_dtype)
            else:
                no = tf.cast(no, dtype=self._real_dtype)

            no = tf.reshape(no, [-1, 1])
            no = tf.broadcast_to(no, [s.B, s.T])
            no = tf.reshape(no, [s.B * s.T])

            snr_mul = tf.math.log(1 + 1 / no) / tf.math.log(
                tf.constant(2.0, dtype=self._real_dtype)
            )
            L_h = snr_mul * L_h

        L_h = L_h * self._lambda_tot

        return {
            "MSE": L_h_mse,
            "Total": L_h,
        }


@register_block
class LS(layers.Layer):
    """Least-squares channel estimation from known or estimated symbols.

    Computes ``y * conj(x)`` for every transmitter, where ``x`` holds pilot or
    (estimated) data symbols, and multiplies the result by ``pilot_mask``.
    With ``res=True`` the contributions of the other transmitters, rebuilt
    from the current channel estimate, are first subtracted from ``y``
    (data-aided LS with interference cancellation).
    """

    def __init__(   self,
                    num_tx=1,
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._dtype = dtype
        self._num_tx = num_tx

    def ls_res(self,y,x,h,batch_size):
        """Per-transmitter LS estimate after cancelling the other transmitters.

        Args:
            y: [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant], complex.
            x: [batch_size, num_tx, num_subcarriers, num_ofdm_symbols], complex.
            h: [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant], real.

        Returns:
            [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant].
        """
        num_tx = self._num_tx

        real_part, imag_part = tf.split(h, num_or_size_splits=2, axis=-1)
        h = tf.complex(real_part, imag_part)

        # [batch_size,num_subcarriers, num_ofdm_symbols, num_rx_ant, num_tx], tf.complex
        h = tf.transpose(h, perm=[0,2,3,4,1])

        # [batch_size, num_subcarriers, num_ofdm_symbols, num_tx], tf.complex
        x = tf.transpose(x,perm=[0,2,3,1])

        batch_size_, num_subcarriers, num_ofdm_symbols, num_rx_ant = h.shape[:-1]
        Hx = [
            tf.zeros((batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant), dtype=h.dtype)
            for _ in range(num_tx)
        ]
        for i in range(num_tx):
            xi = tf.expand_dims(x[...,i],axis=-1)
            hi = h[...,i]
            Hx_i = tf.multiply(hi, xi)
            Hx[i]=Hx_i

        # [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant]
        Hx_sum = tf.add_n(Hx)

        # [batch_size,num_subcarriers, num_ofdm_symbols, num_rx_ant,num_tx]
        Hxs = tf.stack(Hx,axis=-1)

        Hx_sum = tf.expand_dims(Hx_sum,axis=-1)

        # interference from the other transmitters, per tx
        res_tx = Hx_sum - Hxs

        y = tf.expand_dims(y,axis=-1)

        # y with the other transmitters removed, per tx
        # [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant, num_tx]
        hx_tx = y - res_tx

        x = tf.expand_dims(x,axis=3)

        h_ls_d = tf.multiply(hx_tx,tf.math.conj(x))

        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        h_ls_d = tf.transpose(h_ls_d,perm=[0,4,1,2,3])
        h_ls_d = tf.concat([tf.math.real(h_ls_d), tf.math.imag(h_ls_d)], axis=-1)

        return h_ls_d

    def check_is_finit(self, ls):
        tf.debugging.assert_all_finite(ls, "x_real contains non-finite values")

    def call(self, inputs, rx_signal, pilot_mask, shapes, res=True):
        """Compute LS channel estimates.

        Args:
            inputs: ``(h, (x, err))`` with the current channel estimate ``h``
                [B*T*RA, F, 14, 2] and symbol estimates ``x`` [B, T, F, 14]
                (complex); ``err`` is unused.
            rx_signal: dict with ``"y"`` [B, 1, RA, 14, F] (complex).
            pilot_mask: [1, T, F, 14, 1] mask applied to the estimate.
            shapes: ``Shapes`` object with B, T, RA, F.
            res: Cancel the other transmitters before the LS estimate.

        Returns:
            [B*T*RA, F, 14, 2].
        """
        h, xe = inputs
        x, err = xe
        y = rx_signal["y"]
        S = shapes
        batch_size = S.B

        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
        h = tf.reshape(h, [S.B, S.T, S.RA, S.F, 14, 2])
        h = tf.transpose(h, perm=[0, 1, 3, 4, 2, 5])

        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        h = tf.concat([h[...,0], h[...,1]], axis=-1)

        y = tf.squeeze(y, axis=1)
        # [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant]
        y = tf.transpose(y, perm=[0, 3, 2, 1])

        if res:
            ls = self.ls_res(y,x,h, batch_size)
            ls = tf.multiply(pilot_mask,ls)
            self.check_is_finit(ls)

            real_part, imag_part = tf.split(ls, num_or_size_splits=2, axis=-1)
            ls = tf.stack([real_part, imag_part], axis=-1) # [B,T,F,14,RA,2]
            ls = tf.transpose(ls, perm=[0,1,4, 2,3,5]) # [B,T,RA, F,14,2]
            ls = tf.reshape(ls, [S.B*S.T*S.RA, S.F,14,2])
            return ls

        # [batch_size, 1, num_subcarriers, num_ofdm_symbols, num_rx_ant]
        y = tf.expand_dims(y,axis=-1)
        y = tf.transpose(y, perm=[0,4,1,2,3])

        x = tf.expand_dims(x,axis=-1)
        ls = tf.multiply(y, tf.math.conj(x))

        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        ls = tf.concat([tf.math.real(ls), tf.math.imag(ls)], axis=-1)

        ls = tf.multiply(pilot_mask,ls)
        self.check_is_finit(ls)

        real_part, imag_part = tf.split(ls, num_or_size_splits=2, axis=-1)
        ls = tf.stack([real_part, imag_part], axis=-1) # [B,T,F,14,RA,2]
        ls = tf.transpose(ls, perm=[0,1,4, 2,3,5]) # [B,T,RA, F,14,2]
        ls = tf.reshape(ls, [S.B*S.T*S.RA, S.F,14,2])

        return ls


__all__ = [
    "BDemapper",
    "BDemapper_",
    "ChLoss",
    "LLRLoss",
    "LMMSE",
    "LS",
]
