# Mahdi Abdollahpour (mahdi.abdollahpour@unibo.it)
# 2025


"""MDX: model-driven neural receiver for 5G NR PUSCH (arXiv:2508.12892).

"""

import core.runtime as _runtime

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, SeparableConv2D, Layer, LayerNormalization, BatchNormalization
from tensorflow.nn import relu
from sionna.utils import flatten_dims, split_dim, flatten_last_dims, insert_dims, expand_to_rank, matrix_inv
from sionna.ofdm import ResourceGridDemapper
from sionna.nr import TBDecoder, LayerDemapper, PUSCHLSChannelEstimator
from sionna.mapping import Demapper, SymbolDemapper, Constellation
from sionna.mimo import lmmse_equalizer
from sionna.ofdm import LinearDetector
from sionna.mimo.utils import whiten_channel
import sionna as sn


class BDemapper(Layer):
    """Soft-output QAM demapper applied per resource element.

    Input: ``(x, no)``, both [batch_size, num_tx, num_subcarriers,
    num_ofdm_symbols]; ``x`` is complex. Output: LLRs of shape
    [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_bits_per_symbol].
    """

    def __init__(   self,
                    num_bits_per_symbol,
                    demapping_type,
                    dtype=tf.float32,
                    **kwargs):

        super().__init__(dtype=dtype,**kwargs)

        self._dtype = dtype
        self._cdtype = tf.complex64
        if dtype==tf.float64:
            self._cdtype = tf.complex128

        constellation_ = Constellation.create_or_check_constellation(
                                    "qam",
                                    num_bits_per_symbol,
                                    constellation=None,
                                    dtype=self._cdtype)
        self._demapper = Demapper(demapping_type,
                constellation=constellation_,
                hard_out=False,
                dtype=self._cdtype)

    def call(self,inputs):
        x, no = inputs
        x_shape = tf.shape(x)
        x = flatten_last_dims(x,2)
        no = flatten_last_dims(no,2)
        llrs = self._demapper([x,no])

        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_bits_per_symbol]
        llrs = tf.reshape(llrs, tf.concat([x_shape, [-1]], axis=0) )

        tf.debugging.assert_all_finite(llrs, "llrs contains non-finite values")

        return llrs


class LMMSE(Layer):
    """LMMSE equalizer with learnable scaling of the error statistics.

    The noise-plus-channel-estimation-error covariance ``s`` is scaled by a
    learnable per-PRB matrix (12 x 14, shared across PRBs) or scalar before
    equalization. The post-equalization error variance passed to the demapper
    is scaled likewise, optionally with an extra learnable factor per
    modulation order, and a learnable scalar scales the equalized symbols.

    ``num_units_agg`` selects the variants: ``[1][0]`` (input covariance) and
    ``[2][0]`` (output error variance) with 0: none, 1: scalar, 2: per-PRB
    matrix; ``[5][0] == 0`` enables the per-MCS factors; ``[8][0] == 0``
    returns ``diag(GH)`` instead of the error variance.
    """

    def __init__(   self,
                    num_units_agg=[[2],[1],[1],[0],[1],[1],[1],[1],[1]],
                    num_mcs=1,
                    name_suffix="",
                    dtype=tf.float32,
                    **kwargs):
        name = f"LMMSE_{name_suffix}"

        self._name_suffix = name_suffix
        self._dtype = dtype
        self._cdtype = tf.complex64
        if dtype==tf.float64:
            self._cdtype = tf.complex128
        super().__init__(name=name, dtype=dtype,**kwargs)
        self._num_mcs = num_mcs
        self._num_units_agg = num_units_agg
        self._gamma = None
        self._theta = None
        self._zeta  = None
        self._mcs_mul = None

    def build(self, input_shape):
        shape = None
        if self._num_units_agg[1][0]==2:
            shape=(12,14)
            name = f"in_noise_multiplier_mat_{self._name_suffix}"
        if self._num_units_agg[1][0]==1:
            shape=()
            name = f"noise_multiplier_{self._name_suffix}"
        if shape is not None:
            self._gamma = self.add_weight(
                name=name,
                shape=shape,
                initializer=tf.keras.initializers.Constant(1.),
                trainable=True,
                dtype = self._dtype
            )

        shape = None
        if self._num_units_agg[2][0]==2:
            shape=(12,14)
            name = f"out_noise_multiplier_mat_{self._name_suffix}"
        if self._num_units_agg[2][0]==1:
            shape=()
            name = f"noise_multiplier_out_{self._name_suffix}"
        if shape is not None:
            self._theta = self.add_weight(
                name=name,
                shape=shape,
                initializer=tf.keras.initializers.Constant(1.),
                trainable=True,
                dtype = self._dtype
            )

        name = f"x_multiplier_out_{self._name_suffix}"
        self._zeta = self.add_weight(
            name=name,
            shape=(),
            initializer=tf.keras.initializers.Constant(1.),
            trainable=True,
            dtype = self._dtype
        )

        if self._num_units_agg[5][0] == 0:
            self._mcs_mul = []
            if self._num_mcs>1:
                for i in range(self._num_mcs):
                    name = f"x_mcs_mul_{i}"
                    mcs_mul_ = self.add_weight(
                        name=name,
                        shape=(),
                        initializer=tf.keras.initializers.Constant(1.),
                        trainable=True,
                        dtype = self._dtype
                    )
                    self._mcs_mul.append(mcs_mul_)

    def cholesky_inverse(self, matrix):
        """Inverse of a Hermitian positive-definite matrix via Cholesky."""
        matrix = (matrix + tf.linalg.adjoint(matrix)) / tf.constant(2, self._cdtype)

        L = tf.linalg.cholesky(matrix)
        identity = tf.eye(tf.shape(L)[-1], dtype=L.dtype)
        matrix_inv = tf.linalg.cholesky_solve(L, identity)
        tf.debugging.assert_all_finite(tf.math.real(matrix_inv), f"[cholesky_inverse] [{self._name_suffix}] inverted matrix, (real part) contains non-finite values")
        tf.debugging.assert_all_finite(tf.math.imag(matrix_inv), f"[cholesky_inverse] [{self._name_suffix}] inverted matrix, (imag part) contains non-finite values")

        return matrix_inv

    def matrix_inv(self, tensor):
        """Matrix inverse as in Sionna (eigendecomposition-based in XLA mode)."""
        if tensor.dtype in [tf.complex64, tf.complex128] \
                        and sn.config.xla_compat \
                        and not tf.executing_eagerly():
            s, u = tf.linalg.eigh(tensor)
            s = tf.abs(s)
            one = tf.constant(1.,dtype=s.dtype, shape=())
            s = one/s
            s = tf.cast(s, u.dtype)
            s = tf.expand_dims(s, -2)
            return tf.matmul(u*s, u, adjoint_b=True)
        else:
            return tf.linalg.inv(tensor)

    def lmmse(self, y, h, s, shape, whiten_interference=False):
        """Per-RE LMMSE: x_hat = diag(GH)^-1 G y, G = H^H (H H^H + S)^-1."""
        # y [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant]
        shape = tf.shape(y)
        num_prbs = shape[1]//12
        num_subcarriers = shape[1]
        num_ofdm_symbols = shape[2]
        num_rx_ant = shape[3]
        # s [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant, num_rx_ant]
        if self._num_units_agg[1][0]==1:
            s = tf.cast(self._gamma, self._cdtype) * s
        if self._num_units_agg[1][0]==2:
            gamma = tf.expand_dims(self._gamma,axis=0)
            gamma = tf.expand_dims(gamma,axis=0)
            gamma = tf.expand_dims(gamma,axis=-1)
            gamma = tf.expand_dims(gamma,axis=-1)
            gamma = tf.broadcast_to(gamma, [1, num_prbs, 12, 14, num_rx_ant, num_rx_ant])
            gamma = tf.reshape(gamma, [1, num_subcarriers, 14, num_rx_ant, num_rx_ant])
            s = tf.cast(gamma, self._cdtype) * s

        # Ensure positive noise variance
        e = tf.constant(0.00001,self._dtype)
        s =  tf.nn.relu( tf.math.real(s) ) + e
        s = tf.cast(s,self._cdtype)

        g = tf.matmul(h, h, adjoint_b=True) + s
        g_inv = self.cholesky_inverse(g)
        # [..., num_tx, num_rx_ant]
        g = tf.matmul(h, g_inv, adjoint_a=True)
        tf.debugging.assert_all_finite(tf.math.real(g_inv), "g_inv real contains non-finite values")
        tf.debugging.assert_all_finite(tf.math.imag(g_inv), "g_inv imag contains non-finite values")

        y = tf.expand_dims(y, -1)
        gy = tf.squeeze(tf.matmul(g, y), axis=-1)
        gh = tf.matmul(g, h)
        d = tf.linalg.diag_part(gh)

        x_hat = tf.math.divide_no_nan(gy,d)
        x_hat = tf.cast(self._zeta,self._cdtype) * x_hat

        # Residual error variance
        d = tf.math.real(d)
        one = tf.constant(1.,dtype=self._dtype, shape=())
        d = tf.cast(d, dtype=self._dtype)
        no_eff = tf.math.divide_no_nan(one,d) - one

        num_tx = tf.shape(x_hat)[-1]
        if self._num_units_agg[2][0]==1:
            no_eff = tf.cast(self._theta, self._dtype) * no_eff

        if self._num_units_agg[2][0]==2:
            theta = tf.expand_dims(self._theta,axis=0)
            theta = tf.expand_dims(theta,axis=0)
            theta = tf.expand_dims(theta,axis=-1)
            theta = tf.broadcast_to(theta, [1, num_prbs, 12, 14, num_tx])
            theta = tf.reshape(theta, [1, num_subcarriers, 14, num_tx])
            no_eff = tf.cast(theta, self._dtype) * no_eff

        # Ensure positive error variance
        no_eff =  tf.nn.relu( no_eff ) + e

        if self._num_units_agg[8][0]==0:
            no_eff = d

        tf.debugging.assert_all_finite(no_eff, "no_eff contains non-finite values")
        tf.debugging.assert_all_finite(tf.math.real(x_hat), "x_real contains non-finite values")
        tf.debugging.assert_all_finite(tf.math.imag(x_hat), "x_imag contains non-finite values")

        return x_hat, no_eff

    def check_is_finit(self, x_real,x_imag,no_eff):
        tf.debugging.assert_all_finite(x_real, "x_real contains non-finite values")
        tf.debugging.assert_all_finite(x_imag, "x_imag contains non-finite values")
        tf.debugging.assert_all_finite(no_eff, "no_eff contains non-finite values")

    def call(self, inputs, mcs_ue_mask):
        """
        Args:
            inputs: ``(y, h_hat, active_tx_x, s)`` with
                y: [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant], complex
                h_hat: [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
                active_tx_x: [batch_size, num_tx, 1, 1], complex
                s: [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant, num_rx_ant], complex
            mcs_ue_mask: [batch_size, num_tx, num_mcs], one-hot MCS of each user

        Returns:
            x_hat: [batch_size, num_tx, num_subcarriers, num_ofdm_symbols], complex
            no_eff: [batch_size, num_tx, num_subcarriers, num_ofdm_symbols]
        """
        y, h_hat, active_tx_x, s = inputs

        real_part, imag_part = tf.split(h_hat, num_or_size_splits=2, axis=-1)
        h_hat = tf.complex(real_part, imag_part)

        # [batch_size,num_subcarriers, num_ofdm_symbols,num_rx_ant, num_tx], tf.complex
        h_hat = tf.transpose(h_hat, perm=[0,2,3,4,1])

        shape = tf.shape(h_hat)
        x_hat, no_eff = self.lmmse(y, h_hat, s, shape, whiten_interference=False)

        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols]
        x_hat = tf.transpose(x_hat, perm=[0,3,1,2])
        no_eff = tf.transpose(no_eff, perm=[0,3,1,2])

        if self._num_mcs>1 and self._num_units_agg[5][0]==0:
            no_eff_ = tf.zeros_like(no_eff)
            for i in range(self._num_mcs):
                mask_i = tf.expand_dims(mcs_ue_mask[:, :, i], axis=-1)
                mask_i = tf.expand_dims(mask_i, axis=-1)
                no_eff_ = no_eff_ + self._mcs_mul[i]* mask_i * no_eff
            no_eff = no_eff_

        x_hat = tf.multiply(active_tx_x,x_hat)

        self.check_is_finit(tf.math.real(x_hat),tf.math.imag(x_hat),no_eff)

        return x_hat, no_eff


class LS(Layer):
    """Data-aided least-squares (DA-LS) channel estimation.

    The current symbol estimates act as pilots on the data REs (``y * conj(x)``).
    With ``res=True`` the contributions of the other users, computed from the
    current channel estimate ``h``, are removed from ``y`` first. Pilot REs are
    zeroed by ``pilot_mask``.

    Input: ``(y, x, pilot_mask, h, batch_size)`` with
    y [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant] (complex),
    x [batch_size, num_tx, num_subcarriers, num_ofdm_symbols] (complex),
    pilot_mask [1, num_tx, num_subcarriers, num_ofdm_symbols, 1] and
    h [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant].

    Output: [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
    """

    def __init__(   self,
                    num_tx,
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._dtype = dtype
        self._num_tx = num_tx

    def ls_res(self,y,x,h,batch_size):
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

        Hx_sum = tf.add_n(Hx)
        # [batch_size,num_subcarriers, num_ofdm_symbols, num_rx_ant,num_tx]
        Hxs = tf.stack(Hx,axis=-1)
        Hx_sum = tf.expand_dims(Hx_sum,axis=-1)

        # Remove the other users' contributions: y - sum_{j != i} h_j x_j
        res_tx = Hx_sum - Hxs
        y = tf.expand_dims(y,axis=-1)
        hx_tx = y - res_tx
        x = tf.expand_dims(x,axis=3)

        # Data-aided LS estimate per user
        h_ls_d = tf.multiply(hx_tx,tf.math.conj(x))

        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        h_ls_d = tf.transpose(h_ls_d,perm=[0,4,1,2,3])
        h_ls_d = tf.concat([tf.math.real(h_ls_d), tf.math.imag(h_ls_d)], axis=-1)

        return h_ls_d

    def check_is_finit(self, ls):
        tf.debugging.assert_all_finite(ls, "x_real contains non-finite values")

    def call(self, inputs, res=False):
        if res:
            y, x, pilot_mask, h, batch_size = inputs
            ls = self.ls_res(y,x,h, batch_size)
            ls = tf.multiply(pilot_mask,ls)
            self.check_is_finit(ls)
            return ls
        else:
            y, x, pilot_mask, h, batch_size = inputs

        # [batch_size, 1, num_subcarriers, num_ofdm_symbols, num_rx_ant]
        y = tf.expand_dims(y,axis=-1)
        y = tf.transpose(y, perm=[0,4,1,2,3])

        x = tf.expand_dims(x,axis=-1)
        ls = tf.multiply(y, tf.math.conj(x))

        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        ls = tf.concat([tf.math.real(ls), tf.math.imag(ls)], axis=-1)

        # Zero the pilot REs
        ls = tf.multiply(pilot_mask,ls)
        self.check_is_finit(ls)

        return ls


class ResBlock(Layer):
    """Residual block of the channel refinement network.

    Each receive antenna is processed separately with depthwise-separable
    convolutions (3x3 depthwise over subcarriers x OFDM symbols followed by a
    1x1 pointwise projection), and a learned embedding of the positional
    encoding is added to the hidden features. With ``num_skip_connections=2``
    a second output head produces an update for the DA-LS branch.

    Args:
        num_units: number of hidden channels
        num_rx_ant: number of receive antennas
        h_k0: number of input channels per antenna
        pe_k0: number of positional-encoding channels
        pe_type: 1 or 2 for a per-PRB [12, 14, pe_k0] encoding, 0 or 3 for an
            encoding covering the full resource grid
        input_relu, input_norm: apply ReLU / batch normalization to the inputs
        num_skip_connections: number of output heads (1 or 2)
    """

    def __init__(   self,
                    num_units,
                    num_rx_ant,
                    name_suffix="",
                    h_k0=6,
                    pe_k0=2,
                    pe_type=0,
                    input_relu=0,
                    input_norm=0,
                    num_skip_connections=1,
                    dtype=tf.float32,
                    **kwargs):
        self._name_suffix = name_suffix
        name = f"ResBlock_{name_suffix}"
        super().__init__(name=name,dtype=dtype,**kwargs)

        self._pe_type = pe_type
        self._num_rx_ant = num_rx_ant
        self._num_units = num_units
        self._input_relu = input_relu
        self._input_norm = input_norm
        self._name_suffix = name_suffix
        self._num_skip_connections = num_skip_connections
        if input_norm:
            self._BN = BatchNormalization(axis=-1, name=f"input_norm")
            self._BN1 = BatchNormalization(axis=-1, name=f"input_norm1")

        act_v = None
        act_h = None
        self._input_vert_h=[]
        self._input_horz_h=[]
        self._input_vert_pe=[]
        self._input_horz_pe=[]

        in_num_ant = 1
        k_last = num_units
        str = ""
        # Conv3D inputs are [batch, num_subcarriers, num_ofdm_symbols, 1, channels]:
        # a grouped 3x3x1 (depthwise) conv followed by a 1x1x1 (pointwise) conv.
        input_horz_h = Conv3D(filters=h_k0*in_num_ant, kernel_size=(3,3,1), activation=act_h,
                        data_format='channels_last', padding='same', strides=(1,1,1), dtype=dtype, name=f"input_horzH{str}", groups=h_k0*in_num_ant )

        input_vert_h = Conv3D(filters=k_last, kernel_size=(1,1,1), activation=act_v,
                        data_format='channels_last', padding='valid', strides=(1,1,1), dtype=dtype, name=f"input_vertH{str}")
        self._input_vert_h.append(input_vert_h)
        self._input_horz_h.append(input_horz_h)

        # Positional-encoding embedding
        input_horz_pe = Conv3D(filters=pe_k0, kernel_size=(3,3,1), activation=act_h,
                        data_format='channels_last', padding='same', strides=(1,1,1), dtype=dtype, name=f"input_horzP", groups=pe_k0)

        input_vert_pe = Conv3D(filters=k_last, kernel_size=(1,1,1), activation=act_v,
                        data_format='channels_last', padding='valid', strides=(1,1,1), dtype=dtype, name=f"input_vertP")

        self._input_vert_pe.append(input_vert_pe)
        self._input_horz_pe.append(input_horz_pe)

        # Output heads
        self._outpt_vert_h=[]
        self._outpt_horz_h=[]
        k0 = k_last
        k1 = 2

        horz_h = Conv3D(filters=k0, kernel_size=(3,3,1), activation=act_h,
                        data_format='channels_last', padding='same', strides=(1,1,1), dtype=dtype, name=f"outpt_horzH", groups=k0)

        vert_h = Conv3D(filters=k1, kernel_size=(1,1,1), activation=act_v,
                        data_format='channels_last', padding='same', strides=(1,1,1), dtype=dtype, name=f"outpt_vertH" )

        self._outpt_vert_h.append(vert_h)
        self._outpt_horz_h.append(horz_h)

        if num_skip_connections==2:
            horz_h = Conv3D(filters=k0, kernel_size=(3,3,1), activation=act_h,
                            data_format='channels_last', padding='same', strides=(1,1,1), dtype=dtype, name=f"outpt_horzH1", groups=k0)

            vert_h = Conv3D(filters=k1, kernel_size=(1,1,1), activation=act_v,
                            data_format='channels_last', padding='same', strides=(1,1,1), dtype=dtype, name=f"outpt_vertH1" )

            self._outpt_vert_h.append(vert_h)
            self._outpt_horz_h.append(horz_h)

    def call(self,h1, h2, pe):
        """
        Args:
            h1, h2: [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
            pe: [12, 14, pe_k0] for pe_type 1/2, otherwise
                [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, pe_k0]

        Returns:
            Updates of the two heads, each [batch_size*num_tx, num_subcarriers,
            num_ofdm_symbols, num_rx_ant, 2]; the second is an empty list if
            ``num_skip_connections == 1``.
        """
        shape = tf.shape(h1)
        num_prbs = shape[1]//12
        num_subcarriers = shape[1]
        num_ofdm_symbols = shape[2]

        if self._input_norm:
            h1 = self._BN(h1)
            h2 = self._BN1(h2)
        if self._input_relu:
            h1 =  tf.nn.relu(h1)
            h2 =  tf.nn.relu(h2)

        # h  [batch_size * num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, h_k0]
        h = tf.concat([h2, h1], axis=-1)

        # Positional-encoding embedding
        if self._pe_type==1 or self._pe_type==2:
            pe = tf.expand_dims(pe,axis=0)

        pe = tf.expand_dims(pe,axis=-1)
        pe = tf.transpose(pe,perm=[0,1,2,4,3])
        pe = self._input_horz_pe[0]( pe )
        pe = self._input_vert_pe[0](pe)

        if self._pe_type==1 or self._pe_type==2:
            # [1, 1, 12, 14, 1, num_units], broadcast over the PRBs
            pe = tf.expand_dims(pe,axis=0)

        h_=[]
        h_1=[]
        for i in range(self._num_rx_ant):
            # h_i [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, 1, h_k0]
            h_i = h  [:,:,:,i,:]
            h_i = tf.expand_dims(h_i,axis=3)
            h_i = self._input_horz_h[0](  h_i  )
            h_i = self._input_vert_h[0](  h_i  )

            # Add the positional-encoding embedding
            if self._pe_type==1 or self._pe_type==2:
                h_i = tf.reshape(h_i, [-1, num_prbs, 12, num_ofdm_symbols, 1, self._num_units])
                h_i = h_i + pe
                h_i = tf.reshape(h_i, [-1, num_subcarriers, num_ofdm_symbols, 1, self._num_units])
            else:
                h_i = h_i + pe

            h_i = tf.nn.relu(h_i)

            if self._num_skip_connections==2:
                h_i1 = self._outpt_horz_h[1](h_i)
                h_i1 = self._outpt_vert_h[1](h_i1)
                h_1.append(h_i1)

            h_i = self._outpt_horz_h[0](h_i)
            h_i = self._outpt_vert_h[0](h_i)
            h_.append(h_i)

        # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
        h_ = tf.concat(h_,axis=3)
        if self._num_skip_connections==2:
            h_1 = tf.concat(h_1,axis=3)

        return h_, h_1


class CHNN(Layer):
    """Channel refinement network of one receiver iteration.

    Refines the channel estimate from the current estimate, the DA-LS estimate
    and the positional encoding with a stack of :class:`ResBlock` layers
    (``len(num_units_init)`` blocks with ``num_units_init[i]`` channels). The
    output of each block is scaled by learnable per-PRB residual multipliers
    (12 x 14, initialized close to zero) before it is added to the estimate.

    ``arch`` must contain ``"res_blocks"``; ``"res_blocks2"`` also refines the
    DA-LS branch through a second skip connection. ``num_units_agg[0][0]``
    selects the residual multipliers (0: scalar, 1: one matrix, 2: separate
    matrices for the real and imaginary parts, 3: one matrix per block);
    ``num_units_agg[9][0] == 0`` adds a learnable multiplier per modulation
    order.
    """

    def __init__(   self,
                    num_units_state,
                    arch="2D",
                    layer_type="sepconv",
                    num_rx_ant=4,
                    norm="batch_norm",
                    name_suffix="",
                    h_k0=6,
                    pe_k0=2,
                    pe_type=0,
                    d_s=1,
                    num_units_init=[5],
                    num_units_agg=[2],
                    num_mcs=1,
                    dtype=tf.float32,
                    **kwargs):
        self._name_suffix = name_suffix
        name = f"CHNN_{name_suffix}"
        super().__init__(name=name,dtype=dtype,**kwargs)

        self._num_mcs = num_mcs
        self._mcs_mul = None
        self._gamma_real = None
        self._gamma_imag = None
        self._gamma_real_sclr = None
        self._h_k0 = h_k0
        self._arch = arch
        self._norm = norm
        self._num_rx_ant = num_rx_ant
        self._pe_k0 = pe_k0
        self._input_type = d_s
        self._pe_type = pe_type
        self._num_units = num_units_state
        self._num_units_init = num_units_init
        self._num_units_agg = num_units_agg
        if self._input_type == 1:
            self._h_k0 = 2
        if self._input_type == 2:
            self._h_k0 = 3
        if self._input_type == 3:
            self._h_k0 = 4
        if self._input_type == 4:
            self._h_k0 = 6
        if self._input_type == 5:
            self._h_k0 = 10
        if self._input_type == 6:
            self._h_k0 = 2
        if self._input_type == 7:
            self._h_k0 = 3
        if self._input_type == 8:
            self._h_k0 = 9
        h_k0 = self._h_k0
        if norm is not None:
            if norm=="batch_norm":
                norm_layer = BatchNormalization
            if norm=="layer_norm":
                norm_layer = LayerNormalization

        self._num_input_layers = 1

        if "res_blocks" in arch:
            in_num_ant = 1
            self._num_input_layers = 1

        self._in_num_ant = in_num_ant
        self._arch = arch

        self._norm_layer = []
        if norm is not None:
            axis = -1
            self._norm_layer.append(norm_layer(axis=axis, name=f"input_norm1"))
            self._norm_layer.append(norm_layer(axis=axis, name=f"input_norm2"))
            self._norm_layer.append(norm_layer(axis=axis, name=f"input_norm3"))

        act_v = None
        act_h = None
        self._input_vert_h=[]
        self._input_horz_h=[]
        self._input_vert_pe=[]
        self._input_horz_pe=[]

        if "res_blocks" in arch:
            self._num_res_blocks = len(num_units_init)
            if "2" in arch:
                self._num_skip_connections=2
            else:
                self._num_skip_connections=1
            self._res_blocks = []
            for i in range(len(num_units_init)):
                if i==0:
                    input_relu=False
                    input_norm=False
                else:
                    input_relu=True
                    input_norm=True

                if i<len(num_units_init)-1:
                    num_skip_connections = self._num_skip_connections
                else:
                    num_skip_connections = 1

                name_suffix = f"_{i}"
                res_block = ResBlock(num_units_init[i], num_rx_ant, name_suffix=name_suffix, h_k0=h_k0, pe_k0=pe_k0,
                                     pe_type=pe_type, input_relu=input_relu, input_norm=input_norm,
                                     num_skip_connections=num_skip_connections)
                self._res_blocks.append(res_block)

    def build(self, input_shape):
        if self._num_units_agg[0][0] == 0:
            name = f"skip_multiplier_real_scaler"
            self._gamma_real_sclr = self.add_weight(
                name=name,
                shape=(),
                initializer=tf.keras.initializers.Constant(0.0001),
                trainable=True
            )
            name = f"skip_multiplier_imag_scaler"
            self._gamma_imag_sclr = self.add_weight(
                name=name,
                shape=(),
                initializer=tf.keras.initializers.Constant(0.0001),
                trainable=True
            )
        if self._num_units_agg[0][0] == 1:
            name = f"skip_multiplier_real"
            self._gamma_real = self.add_weight(
                name=name,
                shape=(12,14),
                initializer=tf.keras.initializers.Constant(0.0001),
                trainable=True
            )
        if "res_blocks2" in self._arch and self._num_units_agg[3][0] > 0:
            name = f"skip_multiplier_tilde"
            self._gamma_tilde = self.add_weight(
                name=name,
                shape=(12,14),
                initializer=tf.keras.initializers.Constant(0.0001),
                trainable=True
            )

        if self._num_units_agg[0][0] == 2:
            name = f"skip_multiplier_real"
            self._gamma_real = self.add_weight(
                name=name,
                shape=(12,14),
                initializer=tf.keras.initializers.Constant(0.0001),
                trainable=True
            )
            name = f"skip_multiplier_imag"
            self._gamma_imag = self.add_weight(
                name=name,
                shape=(12,14),
                initializer=tf.keras.initializers.Constant(0.0001),
                trainable=True
            )

        if self._num_units_agg[0][0] == 3:
            self._gamma_real = []
            for i in range(self._num_res_blocks):
                name = f"skip_multiplier_{i}"
                gamma_ = self.add_weight(
                    name=name,
                    shape=(12,14),
                    initializer=tf.keras.initializers.Constant(0.0001),
                    trainable=True
                )
                self._gamma_real.append(gamma_)

        if self._num_units_agg[9][0] == 0:
            self._mcs_mul = []
            if self._num_mcs>1:
                for i in range(self._num_mcs):
                    name = f"mcs_mul_{i}"
                    mcs_mul_ = self.add_weight(
                        name=name,
                        shape=(),
                        initializer=tf.keras.initializers.Constant(1.),
                        trainable=True,
                        dtype = self._dtype
                    )
                    self._mcs_mul.append(mcs_mul_)

    def resNN(self, h_hat, hdres, pe, batch_size=None, num_tx=None, mcs_ue_mask=None):
        """Apply the residual blocks to ``h_hat`` and the DA-LS branch ``hdres``."""
        shape = tf.shape(h_hat)
        num_prbs = shape[1]//12
        num_subcarriers = shape[1]
        num_ofdm_symbols = shape[2]

        for i, block in enumerate(self._res_blocks):
            h_hat_new, hdres_new = block(h_hat, hdres,pe)

            # Per-PRB residual multipliers, [1, 1, 12, 14, 1, 1] (or scalars)
            if self._num_units_agg[0][0]==0:
                gamma_real = self._gamma_real_sclr
                gamma_imag = self._gamma_real_sclr
            if self._num_units_agg[0][0]==1:
                gamma_real = insert_dims(self._gamma_real, num_dims=2, axis=0)
                gamma_real = insert_dims(gamma_real, num_dims=2, axis=-1)
                gamma_imag = gamma_real
            if self._num_units_agg[0][0]==2:
                gamma_real = insert_dims(self._gamma_real, num_dims=2, axis=0)
                gamma_real = insert_dims(gamma_real, num_dims=2, axis=-1)
                gamma_imag = insert_dims(self._gamma_imag, num_dims=2, axis=0)
                gamma_imag = insert_dims(gamma_imag, num_dims=2, axis=-1)
            if self._num_units_agg[0][0]==3:
                gamma_real = insert_dims(self._gamma_real[i], num_dims=2, axis=0)
                gamma_real = insert_dims(gamma_real, num_dims=2, axis=-1)
                gamma_imag = gamma_real

            # [batch_size*num_tx, [num_prbs, 12], num_ofdm_symbols, num_rx_ant, 2]
            h_hat_new = split_dim(h_hat_new, [num_prbs, 12], 1)
            real, imag = tf.expand_dims(h_hat_new[...,0],axis=-1), tf.expand_dims(h_hat_new[...,1],axis=-1)
            real = tf.multiply(real, gamma_real)
            imag = tf.multiply(imag, gamma_imag)
            h_hat_new = tf.concat([real, imag], axis=-1)
            h_hat_new = tf.reshape(h_hat_new, [-1, num_subcarriers, num_ofdm_symbols, self._num_rx_ant, 2])

            # Per-modulation residual multipliers
            if self._num_units_agg[9][0]==0 and self._num_mcs>1:
                # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
                h_hat_new = split_dim(h_hat_new, [batch_size, num_tx], 0)

                h__ = tf.zeros_like(h_hat_new)
                for i_ in range(self._num_mcs):
                    mask_i = tf.expand_dims(mcs_ue_mask[:, :, i_], axis=-1)
                    mask_i = tf.expand_dims(mask_i, axis=-1)
                    mask_i = tf.expand_dims(mask_i, axis=-1)
                    mask_i = tf.expand_dims(mask_i, axis=-1)
                    h__ = h__ + self._mcs_mul[i_] * mask_i * h_hat_new
                h_hat_new = h__
                h_hat_new = flatten_dims(h_hat_new, num_dims=2, axis=0)

            h_hat = h_hat + h_hat_new

            # Second skip connection (DA-LS branch)
            if self._num_skip_connections==2:
                if i==len(self._res_blocks)-1 and self._num_units_agg[3][0] > 0:
                    gamma_tilde = insert_dims(self._gamma_tilde, num_dims=2, axis=0)
                    gamma_tilde = insert_dims(gamma_tilde, num_dims=2, axis=-1)
                    hdres_new = split_dim(hdres_new, [num_prbs, 12], 1)
                    real1, imag1 = tf.expand_dims(hdres_new[...,0],axis=-1), tf.expand_dims(hdres_new[...,1],axis=-1)
                    real1 = tf.multiply(real1, gamma_tilde)
                    imag1 = tf.multiply(imag1, gamma_tilde)
                    hdres_new = tf.concat([real1, imag1], axis=-1)
                    hdres_new = tf.reshape(hdres_new, [-1, num_subcarriers, num_ofdm_symbols, self._num_rx_ant, 2])
                if i < len(self._res_blocks)-1:
                    hdres = hdres + hdres_new

        if self._num_skip_connections==1:
            hdres = h_hat

        return h_hat, hdres

    def preprocess(self, h1, h2, h3):
        """Split real/imaginary parts per antenna, normalize and stack ``[h2, h1]``."""
        shape = tf.shape(h1)
        num_subcarriers = shape[1]
        num_ofdm_symbols = shape[2]
        num_rx_ant = shape[3]//2

        # h [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
        real, imag = tf.split(h1, num_or_size_splits=2, axis=-1)
        h1 = tf.stack([real, imag], axis=-1)

        real, imag = tf.split(h2, num_or_size_splits=2, axis=-1)
        h2 = tf.stack([real, imag], axis=-1)

        real, imag = tf.split(h3, num_or_size_splits=2, axis=-1)
        h3 = tf.stack([real, imag], axis=-1)

        if self._norm is not None:
            h1 = self._norm_layer[0](h1)
            h2 = self._norm_layer[1](h2)
            h3 = self._norm_layer[2](h3)

        # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 4]
        h = tf.concat([h2, h1], axis=-1)

        return h

    def check_is_finit(self,h1,h2,h1_old,h2_old,pe):
        tf.debugging.assert_all_finite(h1, "x_real contains non-finite values")
        tf.debugging.assert_all_finite(h2, "x_imag contains non-finite values")
        tf.debugging.assert_all_finite(h1_old, "no_eff contains non-finite values")
        tf.debugging.assert_all_finite(h2_old, "no_eff contains non-finite values")

    def call(self, inputs, mcs_ue_mask=None):
        """
        Args:
            inputs: ``(h_hat, h_d_res, h_d, pe, active_tx_h)``: current channel
                estimate and DA-LS estimates (with and without removal of the
                other users), each [batch_size, num_tx, num_subcarriers,
                num_ofdm_symbols, 2*num_rx_ant], the positional encoding and
                the active-user mask.
            mcs_ue_mask: [batch_size, num_tx, num_mcs], one-hot MCS of each user

        Returns:
            (h_hat, h_tilde): refined channel estimate and refined DA-LS
            branch, each [batch_size, num_tx, num_subcarriers,
            num_ofdm_symbols, 2*num_rx_ant].
        """
        h_hat, h2, h3, pe, active_tx_h = inputs

        h1 = tf.identity(h_hat)

        shape = tf.shape(h1)
        num_prbs = shape[2]//12
        num_subcarriers = shape[2]
        num_ofdm_symbols = shape[3]
        batch_size = shape[0]
        num_tx = shape[1]

        self.check_is_finit(h1,h2,h1,h2,pe)

        h1 = tf.multiply(h1,active_tx_h)
        h2 = tf.multiply(h2,active_tx_h)
        h3 = tf.multiply(h3,active_tx_h)

        # move num_tx to batch dimension
        # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols,  2*num_rx_ant]
        h1 = flatten_dims(h1, num_dims=2, axis=0)
        h2 = flatten_dims(h2, num_dims=2, axis=0)
        h3 = flatten_dims(h3, num_dims=2, axis=0)

        if self._pe_type==0 or self._pe_type==3:
            # [batch_size * num_tx, num_subcarriers, num_ofdm_symbols, 2]
            pe = tf.tile(tf.expand_dims(pe, axis=0), [batch_size, 1, 1, 1, 1])
            pe = flatten_dims(pe, 2, 0)

        hp = self.preprocess(h1,h2,h3)

        h_tilde = h1
        h_hat = h1

        # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant, 2]
        h_hat, h_tilde = self.resNN(hp[...,2:4],hp[...,0:2], pe, batch_size, num_tx, mcs_ue_mask)

        real, imag = tf.split(h_hat, num_or_size_splits=2, axis=-1)
        h_hat = tf.concat([tf.squeeze(real,axis=[-1]), tf.squeeze(imag,axis=[-1])], axis=-1)
        h_hat = split_dim(h_hat, [batch_size, num_tx], 0)

        real, imag = tf.split(h_tilde, num_or_size_splits=2, axis=-1)
        h_tilde = tf.concat([tf.squeeze(real,axis=[-1]), tf.squeeze(imag,axis=[-1])], axis=-1)
        h_tilde = split_dim(h_tilde, [batch_size, num_tx], 0)

        return h_hat, h_tilde


class CGNN(Model):
    """Iterative core of the MDX receiver.

    After an initial LMMSE equalization with the initial channel estimate,
    each of the ``num_it`` iterations (with its own weights) performs DA-LS
    estimation (:class:`LS`), channel refinement (:class:`CHNN`), learnable
    LMMSE equalization (:class:`LMMSE`) and demapping (:class:`BDemapper`).

    ``call`` returns lists over the tracked iterations (all of them in
    training, only the last one otherwise) of the LLRs (one entry per MCS),
    channel estimates and symbol estimates, plus the last error variance and
    the refined DA-LS estimates.
    """

    def __init__(   self,
                    num_bits_per_symbol,
                    num_rx_ant,
                    num_it,
                    arch,
                    d_s,
                    num_units_init,
                    num_units_agg,
                    num_units_state ,
                    num_units_readout,
                    layer_type_dense,
                    layer_type_conv,
                    layer_type_readout,
                    pilot_mask,
                    constellation,
                    demapping_type,
                    max_num_tx,
                    training=False,
                    apply_multiloss=False,
                    var_mcs_masking=False,
                    pe_d=2,
                    pe_type=False,
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(dtype=dtype,**kwargs)

        self._training = training

        self._apply_multiloss = apply_multiloss
        self._var_mcs_masking = var_mcs_masking
        self._pilot_mask = pilot_mask
        self._dtype = dtype
        self._cdtype = tf.complex64
        self._num_units_agg = num_units_agg
        if dtype==tf.float64:
            self._cdtype = tf.complex128
        self._pe_type = pe_type
        self._chnn = []
        self._num_mcs = len(num_bits_per_symbol)
        self._lmmse0 = LMMSE(name_suffix=f"dummy",dtype=dtype)

        if num_units_agg[4][0] > 0:
            self._lmmse_init = LMMSE(num_mcs=self._num_mcs,name_suffix=f"init",dtype=dtype)
        if num_units_agg[4][0] == 0:
            self._lmmse_init = LMMSE(num_mcs=self._num_mcs,num_units_agg=num_units_agg,name_suffix=f"init",dtype=dtype)

        self._lmmse = []

        for i in range(num_it):
            lmmse = LMMSE(num_mcs=self._num_mcs,num_units_agg=num_units_agg, name_suffix=f"{i}",dtype=dtype)
            self._lmmse.append(lmmse)
            chnn = CHNN( num_units_state[i],
                        arch=arch,
                        num_rx_ant=num_rx_ant,
                        name_suffix=f"{i}",
                        pe_k0=pe_d,
                        d_s = d_s,
                        num_units_init=num_units_init,
                        num_units_agg=num_units_agg,
                        pe_type=pe_type,
                        num_mcs=self._num_mcs,
                        dtype=dtype)
            self._chnn.append(chnn)

        self._bdemapper = []
        for num_bits_per_symbol_ in num_bits_per_symbol:
            bd = BDemapper(num_bits_per_symbol_, demapping_type)
            self._bdemapper.append(bd)

        self._ls = LS(num_tx=max_num_tx,dtype=dtype)
        self._num_it = num_it

        self._num_mcss_supported = len(num_bits_per_symbol)
        self._num_bits_per_symbol = num_bits_per_symbol

        self._inc_num_it = 1
        self._step = 0

    @property
    def apply_multiloss(self):
        """Average loss over all iterations or eval just the last iteration."""
        return self._apply_multiloss

    @apply_multiloss.setter
    def apply_multiloss(self, val):
        assert isinstance(val, bool), "apply_multiloss must be bool."
        self._apply_multiloss = val

    @property
    def num_it(self):
        """Number of receiver iterations."""
        return self._num_it

    @num_it.setter
    def num_it(self, val):
        assert (val >= 1) and (val <= len(self._chnn)),\
            "Invalid number of iterations"
        self._num_it = val

    def call(self, inputs):
        """
        Args:
            inputs: ``(y, pe, h_hat, active_tx, mcs_ue_mask, no, x, h, err_var, batch_size)``
                y: [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
                pe: positional encoding
                h_hat: initial channel estimate, [batch_size, num_tx,
                    num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
                active_tx: [batch_size, num_tx], active-user mask
                mcs_ue_mask: [batch_size, num_tx, num_mcs], one-hot MCS of each user
                no: [batch_size], noise variance
                err_var: [batch_size, num_rx, num_subcarriers, num_ofdm_symbols,
                    num_rx_ant], error variance of the initial estimate
                x, h, batch_size: not used
        """
        y, pe, h_hat, active_tx, mcs_ue_mask, no, x, h, err_var, batch_size = inputs

        x = tf.complex(x[...,0],x[...,1])

        batch_size = tf.shape(y)[0]

        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols], complex
        x_hat = tf.zeros(tf.shape(h_hat)[0:-1],dtype=self._cdtype)

        active_tx_h = expand_to_rank(active_tx, tf.rank(h_hat), axis=-1)
        h_ls = h_hat
        h_tilde = h_hat

        pilot_mask = self._pilot_mask

        active_tx_x = expand_to_rank(active_tx, 4, axis=-1)
        active_tx_x = tf.complex(active_tx_x, tf.zeros_like(active_tx_x) )
        active_tx_x = tf.cast(active_tx_x, self._cdtype)

        active_tx_llr = expand_to_rank(active_tx, 5, axis=-1)

        # [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant], complex
        real_part, imag_part = tf.split(y, num_or_size_splits=2, axis=-1)
        y = tf.complex(real_part, imag_part)

        llrs = []
        h_hats = []
        h_tildes = []
        x_hats = []

        batch_size = tf.shape(y)[0]
        num_rx_ant = tf.shape(y)[-1]

        # Noise plus channel estimation error covariance
        # [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant, num_rx_ant]
        err_var = tf.squeeze(err_var,axis=1)
        err_var_ = tf.linalg.diag(err_var)
        err_var_ = tf.cast(err_var_,self._cdtype)

        no_ = expand_to_rank(no,2,-1)
        no_ = tf.broadcast_to(no_,[batch_size, num_rx_ant])
        no_ = insert_dims(no_,num_dims=2,axis=1)
        no_ = tf.linalg.diag(no_)
        no_ = tf.cast(no_,self._cdtype)

        s = no_ + err_var_

        # Reference LMMSE on the initial estimate (not included in the loss)
        h_debug = h_ls
        x_hat_debug, no_eff = self._lmmse0([y, h_debug, active_tx_x, s], mcs_ue_mask)

        if self._num_units_agg[8][0]==1:
            llrs_ = []
            for idx in range(self._num_mcss_supported):
                llrs__ = self._bdemapper[idx]([x_hat_debug, no_eff])
                llrs_.append(llrs__)
            llrs.append(llrs_)
            h_hats.append(h_debug)
            h_tildes.append(h_debug)
            x_hats.append(x_hat_debug)

        # Initial equalization
        x_hat, no_eff = self._lmmse_init([y, h_hat, active_tx_x, s],mcs_ue_mask)
        if self._num_units_agg[8][0]==1:
            llrs_ = []
            for idx in range(self._num_mcss_supported):
                llrs__ = self._bdemapper[idx]([x_hat, no_eff])
                llrs_.append(llrs__)
            llrs.append(llrs_)

        for i in range(self._num_it):
            # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
            h_d_res = self._ls([y,x_hat,pilot_mask,h_hat, batch_size], res=True)
            h_d = self._ls([y,x_hat,pilot_mask,h_hat, batch_size], res=False)

            # Channel refinement
            h_hat, h_tilde = self._chnn[i]([h_hat, h_d_res, h_d, pe, active_tx_h], mcs_ue_mask=mcs_ue_mask)

            x_hat, no_eff = self._lmmse[i]([y, h_hat, active_tx_x, s],mcs_ue_mask)

            # only during training every intermediate iteration is tracked
            if self._training or i==self._num_it-1:
                h_hats.append(h_hat)
                h_tildes.append(h_tilde)
                x_hats.append(x_hat)
                llrs_ = []
                for idx in range(self._num_mcss_supported):
                    # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_bits_per_symbol]
                    llrs__ = self._bdemapper[idx]([x_hat, no_eff])
                    llrs_.append(llrs__)
                llrs.append(llrs_)

        return llrs, h_hats, x_hats, no_eff, h_tildes


class CGNNOFDM(Model):
    """OFDM wrapper of :class:`CGNN`.

    Precomputes the positional encoding and the pilot mask, reshapes the
    inputs, runs the iterative receiver, extracts the LLRs of the data REs and,
    in training mode, computes the losses.
    """

    def __init__(self,
                 sys_parameters,
                 max_num_tx,
                 training,
                 num_it=5,
                 d_s=1,
                 num_units_init=[64],
                 num_units_agg=[[64]],
                 num_units_state=[[64]],
                 num_units_readout=[64],
                 layer_demappers=None,
                 layer_type_dense="dense",
                 layer_type_conv="sepconv",
                 layer_type_readout="dense",
                 nrx_dtype=tf.float32,
                 **kwargs):
        super().__init__(**kwargs)

        self._num_units_agg = num_units_agg
        self._training = training
        self._max_num_tx = max_num_tx
        self._layer_demappers = layer_demappers
        self._sys_parameters = sys_parameters
        self._nrx_dtype = nrx_dtype
        self._nrx_cdtype = tf.complex64
        if self._nrx_dtype==tf.float64:
            self._nrx_cdtype=tf.complex128
        self._num_it = num_it

        self._num_mcss_supported = len(sys_parameters.mcs_index)

        self._rg = sys_parameters.transmitters[0]._resource_grid

        # all UEs in the same pusch config must use the same MCS
        self._num_bits_per_symbol = []
        self._constellation = []
        self._demapper = []
        _sys_demapper = []
        for mcs_list_idx in range(self._num_mcss_supported):
            num_bits_per_symbol_ = sys_parameters.pusch_configs[mcs_list_idx][0].tb.num_bits_per_symbol
            constellation_ = Constellation.create_or_check_constellation(
                                    "qam",
                                    num_bits_per_symbol_,
                                    constellation=None,
                                    dtype=self._nrx_cdtype)
            demapper_ = Demapper(self._sys_parameters.demapping_type,
                    constellation=constellation_,
                    hard_out=False,
                    dtype=self._nrx_cdtype)
            sym_demapper = SymbolDemapper(constellation=constellation_,
                                            hard_out=False,
                                            dtype=self._nrx_cdtype)
            self._num_bits_per_symbol.append(num_bits_per_symbol_)
            self._constellation.append( constellation_ )
            self._demapper.append( demapper_ )
            _sys_demapper.append( sym_demapper )

        self._rg_demapper = ResourceGridDemapper(self._rg,
                                                 sys_parameters.sm)
        self._num_data_symbols = self._rg.pilot_pattern.num_data_symbols
        if training:
            self._bce = tf.keras.losses.BinaryCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.NONE)
            self._mse = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE)

        # Positional encoding: distance to the nearest pilot in time and frequency
        rg_type = self._rg.build_type_grid()[:,0]
        pilot_ind = tf.where(rg_type==1)
        pilots = flatten_last_dims(self._rg.pilot_pattern.pilots, 3)
        # [max_num_tx, num_effective_subcarriers, num_ofdm_symbols]
        pilots_only = tf.scatter_nd(pilot_ind, pilots,
                                    rg_type.shape)
        pilot_ind = tf.where(tf.abs(pilots_only) > 1e-3)
        pilot_ind = np.array(pilot_ind)

        pilot_ind_sorted = [ [] for _ in range(max_num_tx) ]

        for p_ind in pilot_ind:
            tx_ind = p_ind[0]
            re_ind = p_ind[1:]
            pilot_ind_sorted[tx_ind].append(re_ind)
        pilot_ind_sorted = np.array(pilot_ind_sorted)

        pilots_dist_time = np.zeros([   max_num_tx,
                                        self._rg.num_ofdm_symbols,
                                        self._rg.fft_size,
                                        pilot_ind_sorted.shape[1]])
        pilots_dist_freq = np.zeros([   max_num_tx,
                                        self._rg.num_ofdm_symbols,
                                        self._rg.fft_size,
                                        pilot_ind_sorted.shape[1]])

        t_ind = np.arange(self._rg.num_ofdm_symbols)
        f_ind = np.arange(self._rg.fft_size)

        for tx_ind in range(max_num_tx):
            for i, p_ind in enumerate(pilot_ind_sorted[tx_ind]):
                pt = np.expand_dims(np.abs(p_ind[0] - t_ind), axis=1)
                pilots_dist_time[tx_ind, :, :, i] = pt

                pf = np.expand_dims(np.abs(p_ind[1] - f_ind), axis=0)
                pilots_dist_freq[tx_ind, :, :, i] = pf

        # Normalizing the tensors of distance to force zero-mean and
        # unit variance.
        nearest_pilot_dist_time = np.min(pilots_dist_time, axis=-1)
        nearest_pilot_dist_freq = np.min(pilots_dist_freq, axis=-1)
        nearest_pilot_dist_time -= np.mean(nearest_pilot_dist_time,
                                            axis=1, keepdims=True)
        std_ = np.std(nearest_pilot_dist_time, axis=1, keepdims=True)
        nearest_pilot_dist_time = np.where(std_ > 0.,
                                           nearest_pilot_dist_time / std_,
                                           nearest_pilot_dist_time)
        nearest_pilot_dist_freq -= np.mean(nearest_pilot_dist_freq,
                                            axis=2, keepdims=True)
        std_ = np.std(nearest_pilot_dist_freq, axis=2, keepdims=True)
        nearest_pilot_dist_freq = np.where(std_ > 0.,
                                           nearest_pilot_dist_freq / std_,
                                           nearest_pilot_dist_freq)

        nearest_pilot_dist = np.stack([ nearest_pilot_dist_time,
                                        nearest_pilot_dist_freq],
                                        axis=-1)
        nearest_pilot_dist = tf.constant(nearest_pilot_dist, tf.float32)
        # [max_num_tx, num_subcarriers, num_ofdm_symbols, 2]
        self._nearest_pilot_dist = tf.transpose(nearest_pilot_dist,
                                                [0, 2, 1, 3])

        num_rx_ant = sys_parameters.num_rx_antennas
        arch = sys_parameters.arch

        pilot_pattern = self._rg.pilot_pattern
        num_pilot_symbols = pilot_pattern.num_pilot_symbols
        # [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], 1 at pilots
        pilot_mask = pilot_pattern.mask

        # [num_tx, num_effective_subcarriers, num_ofdm_symbols], 0 at pilots
        pilot_mask = 1 - pilot_mask[:,0]
        pilot_mask = tf.transpose(pilot_mask, perm=[0,2,1])
        # [1,num_tx,num_effective_subcarriers, num_ofdm_symbols,1]
        pilot_mask = insert_dims(pilot_mask,num_dims=1,axis=0)
        pilot_mask = insert_dims(pilot_mask,num_dims=1,axis=-1)
        pilot_mask = tf.cast(pilot_mask,self._dtype)

        pe_d = self._sys_parameters.pe_d

        if num_units_agg[8][0]==-1:
            layer_type_readout="sepconv"

        self._cgnn = CGNN(self._num_bits_per_symbol,
                          num_rx_ant,
                          num_it,
                          arch,
                          d_s,
                          num_units_init,
                          num_units_agg,
                          num_units_state,
                          num_units_readout,
                          pilot_mask=pilot_mask,
                          constellation=self._constellation,
                          demapping_type=self._sys_parameters.demapping_type,
                          max_num_tx = self._sys_parameters.max_num_tx,
                          training=training,
                          layer_type_dense=layer_type_dense,
                          layer_type_conv=layer_type_conv,
                          layer_type_readout=layer_type_readout,
                          var_mcs_masking=None,
                          pe_d=self._sys_parameters.pe_d,
                          pe_type=self._sys_parameters.pe_type,
                          dtype=nrx_dtype)
        # [1, num_tx, num_subcarriers, num_ofdm_symbols]
        self._pilot_mask = pilot_mask
        self._pilot_mask = tf.squeeze(self._pilot_mask,axis=-1)

    @property
    def num_it(self):
        """Number of receiver iterations. No weight sharing is used."""
        return self._cgnn.num_it

    @num_it.setter
    def num_it(self, val):
        self._cgnn.num_it = val

    def call(self, inputs, mcs_arr_eval, mcs_ue_mask_eval=None, no=None):
        """
        Args:
            inputs: in training ``(y, h_hat_init, active_tx, bits, h,
                mcs_ue_mask, x, no, err_var, batch_size)``, otherwise
                ``(y, h_hat_init, active_tx, no, x, h, err_var, batch_size)``
                y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], complex
                h_hat_init, h: [batch_size, num_tx, num_subcarriers,
                    num_ofdm_symbols, 2*num_rx_ant]
                x: [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2]
                mcs_ue_mask: [batch_size, num_tx, num_mcs]
            mcs_arr_eval: indices of the evaluated MCSs
            mcs_ue_mask_eval: optional MCS mask for inference; by default all
                users use ``mcs_arr_eval[0]``

        Returns:
            In training a dict with ``"llr_loss"`` (BCE), ``"ch_loss"`` (MSE)
            and ``"Total"``; otherwise the LLRs and channel estimate of the
            last iteration.
        """
        if self._training:
            y, h_hat_init, active_tx, bits, h, mcs_ue_mask, x, no,err_var, batch_size = inputs
        else:
            y, h_hat_init, active_tx, no, x, h,err_var, batch_size = inputs
            if mcs_ue_mask_eval is None:
                mcs_ue_mask = tf.one_hot(mcs_arr_eval[0],
                                         depth=self._num_mcss_supported)
            else:
                mcs_ue_mask = mcs_ue_mask_eval
            mcs_ue_mask = expand_to_rank(mcs_ue_mask, 3, axis=0)

        # total number of possible streams; not all of them might be active.
        num_tx = tf.shape(active_tx)[1]

        num_prbs = tf.shape(y)[4]//12

        # mask pilots for pilotless communications
        if self._sys_parameters.mask_pilots:
            rg_type = self._rg.build_type_grid()
            rg_type = tf.expand_dims(rg_type, axis=0)
            rg_type = tf.broadcast_to(rg_type, tf.shape(y))
            y = tf.where(rg_type==1, tf.constant(0., y.dtype), y)

        # [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        y = y[:,0]
        y = tf.transpose(y, [0, 3, 2, 1])
        y = tf.concat([tf.math.real(y), tf.math.imag(y)], axis=-1)

        # pe_type 0: nearest-pilot distance, 1: sinusoidal, 2: per-PRB, 3: 0 and 2 combined
        if self._sys_parameters.pe_type==0:
            # [num_tx, num_subcarriers, num_ofdm_symbols, 2]
            pe = self._nearest_pilot_dist[:num_tx]
        if self._sys_parameters.pe_type==1 or self._sys_parameters.pe_type==2:
            # [12, 14, 2]
            pe = self._sys_parameters.PE
        if self._sys_parameters.pe_type==3:
            pe = tf.expand_dims(self._sys_parameters.PE,axis=0)
            pe = tf.expand_dims(pe,axis=0)
            pe = tf.broadcast_to(pe,[num_tx, num_prbs, 12,14,2])
            pe = tf.reshape(pe, [num_tx, num_prbs*12, 14, 2])

            # [num_tx, num_subcarriers, num_ofdm_symbols, 4]
            pe = tf.concat([pe, self._nearest_pilot_dist[:num_tx]], axis=-1)

        y = tf.cast(y, self._nrx_dtype)
        pe = tf.cast(pe, self._nrx_dtype)

        if h_hat_init is not None:
            h_hat_init = tf.cast(h_hat_init, self._nrx_dtype)
        active_tx = tf.cast(active_tx, self._nrx_dtype)

        h_temp=0
        llrs_, h_hats_, x_hats_, no_, h_tildes_ = self._cgnn([y, pe, h_hat_init, active_tx, mcs_ue_mask, no, x, h_temp,err_var, batch_size])

        indices = mcs_arr_eval

        # list of lists; outer list separates iterations, inner list the MCSs
        llrs = []
        h_hats = []
        h_tildes = []
        x_hats = []
        for llrs_ in llrs_:
            _llrs_ = []
            for idx in indices:
                llrs_[idx] = tf.cast(llrs_[idx], tf.float32)

                # [batch_size, 1, num_tx, num_ofdm_symbols, fft_size, num_bits_per_symbol]
                llrs_[idx] = tf.transpose(llrs_[idx], [0, 1, 3, 2, 4])
                llrs_[idx] = tf.expand_dims(llrs_[idx], axis=1)

                # [batch_size, num_tx, 1, num_data_symbols, num_bit_per_symbols]
                llrs_[idx] = self._rg_demapper(llrs_[idx])
                llrs_[idx] = llrs_[idx][:,:num_tx]

                # [batch_size, num_tx, 1, num_data_symbols*num_bit_per_symbols]
                llrs_[idx] = tf.reshape(llrs_[idx], [batch_size, self._max_num_tx, 1, self._num_data_symbols*self._num_bits_per_symbol[idx]])

                # Remove the stream dimension (one stream per user)
                if self._layer_demappers is None:
                    llrs_[idx] = tf.squeeze(llrs_[idx], axis=-2)
                else:
                    llrs_[idx] = self._layer_demappers[idx](llrs_[idx])
                _llrs_.append(llrs_[idx])

            llrs.append(_llrs_)

        for h_hat_, x_hat_, h_tilde_ in zip(h_hats_, x_hats_, h_tildes_):
            x_hat_ = tf.stack([tf.math.real(x_hat_), tf.math.imag(x_hat_)], axis=-1)
            x_hat_ = tf.cast(x_hat_, tf.float32)
            h_hat_ = tf.cast(h_hat_, tf.float32)
            h_tilde_ = tf.cast(h_tilde_, tf.float32)

            h_hats.append(h_hat_)
            h_tildes.append(h_tilde_)
            x_hats.append(x_hat_)

        if self._training:
            mcs_mul = tf.constant([10, 1, 1/5], dtype=tf.float32)

            # Skip the reference LMMSE output (only collected if num_units_agg[8][0] == 1)
            exclude_n = 1
            if self._num_units_agg[8][0]<1:
                exclude_n = 0

            # BCE loss on the LLRs
            loss_data = tf.constant(0.0, dtype=tf.float32)
            l0 = [tf.constant(0.0, dtype=tf.float32) for _ in range(self._sys_parameters.num_nrx_iter+2)]
            l0_rel = [tf.constant(0.0, dtype=tf.float32) for _ in range(self._sys_parameters.num_nrx_iter+2)]
            i=0
            for llrs_ in llrs:
                i=i+1
                for idx in range(len(indices)):
                    # [batch_size, max_num_tx]
                    loss_data_ = self._bce(bits[idx], llrs_[idx])

                    mcs_ue_mask_ = expand_to_rank(
                        tf.gather(mcs_ue_mask, indices=indices[idx], axis=2),
                        tf.rank(loss_data_), axis=-1)

                    # select data loss only for associated MCSs
                    loss_data_ = tf.multiply(loss_data_, mcs_ue_mask_)

                    # only focus on active users
                    active_tx_data = expand_to_rank(active_tx,
                                                    tf.rank(loss_data_),
                                                    axis=-1)

                    loss_data_ = tf.multiply(loss_data_, active_tx_data)

                    l0_ = loss_data_
                    # Optional SNR and per-MCS weighting
                    if self._num_units_agg[6][0]==1:
                        snr_mul = tf.math.log(1 + 1/no) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
                        snr_mul = expand_to_rank(snr_mul,2,-1)
                        snr_mul = tf.broadcast_to(snr_mul,[batch_size, num_tx])
                        snr_mul = tf.math.log(1 + snr_mul) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
                        loss_data_ = tf.multiply(loss_data_, snr_mul)

                    if  self._num_units_agg[7][0]==1 and len(indices)>1:
                        loss_data_ = mcs_mul[i-1]*loss_data_

                    l0_ = tf.reduce_mean(l0_)
                    loss_data_ = tf.reduce_mean(loss_data_)

                    if i==1:
                        ref = l0_
                    if i>exclude_n:
                        loss_data += loss_data_
                    l0[i-1] = l0_
                    l0_rel[i-1] = l0_-ref

            # MSE loss on the channel estimates
            loss_chest = tf.constant(0.0, dtype=tf.float32)
            l1=[tf.constant(0.0, dtype=tf.float32) for _ in range(self._sys_parameters.num_nrx_iter+1)]

            loss_tilde = tf.constant(0.0, dtype=tf.float32)
            l1t=[tf.constant(0.0, dtype=tf.float32) for _ in range(self._sys_parameters.num_nrx_iter+1)]

            if self._num_units_agg[6][0]==1:
                snr_mul = tf.math.log(1 + 1/no) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
                snr_mul = expand_to_rank(snr_mul,4,-1)
                snr_mul = tf.broadcast_to(snr_mul,[batch_size, tf.shape(h)[1], tf.shape(h)[2], tf.shape(h)[3]])
                snr_mul = tf.math.log(1 + snr_mul) / tf.math.log(tf.constant(2.0, dtype=tf.float32))

            i = 0
            if h_hats is not None:
                for h_hat_, h_tilde_ in zip(h_hats, h_tildes):
                    i = i + 1
                    if h is not None:
                        if i>exclude_n:
                            # [batch size, num_tx, num_subcarriers, num_ofdm_symbols]
                            loss_ = self._mse(h, h_hat_)

                            if self._num_units_agg[6][0]==1:
                                loss_ = tf.multiply(loss_, snr_mul)
                            if  self._num_units_agg[7][0]==1 and len(indices)>1:
                                mask_i = tf.expand_dims(mcs_ue_mask[:, :, 0], axis=-1)
                                mask_i = tf.expand_dims(mask_i,axis=-1)
                                loss__ = mcs_mul[0]*mask_i*loss_

                                mask_i = tf.expand_dims(mcs_ue_mask[:, :, 1], axis=-1)
                                mask_i = tf.expand_dims(mask_i,axis=-1)
                                loss__ = loss__ + mcs_mul[1]*mask_i*loss_

                                if len(indices) > 2:
                                    mask_i = tf.expand_dims(mcs_ue_mask[:, :, 2], axis=-1)
                                    mask_i = tf.expand_dims(mask_i,axis=-1)
                                    loss__ = loss__ + mcs_mul[2]*mask_i*loss_
                                loss_ = loss__

                            loss_chest += loss_

                            loss_t = self._mse(h, h_tilde_)
                            if self._num_units_agg[6][0]==1:
                                loss_t = tf.multiply(loss_t, snr_mul)
                            loss_tilde += loss_t

                        l1[i-1] = self._mse(h, h_hat_)
                        l1t[i-1] = self._mse(h, h_tilde_)

            # only focus on active users
            active_tx_chest = expand_to_rank(active_tx,
                                             tf.rank(loss_chest), axis=-1)
            loss_chest = tf.multiply(loss_chest, active_tx_chest)
            loss_chest = tf.reduce_mean(loss_chest)

            loss_tilde = tf.multiply(loss_tilde, active_tx_chest)
            loss_tilde = tf.reduce_mean(loss_tilde)

            for i in range(len(l1)):
                l1_ = tf.multiply(l1[i], active_tx_chest)
                l1_ = tf.reduce_mean(l1_)
                l1[i] = l1_

                l1t_ = tf.multiply(l1t[i], active_tx_chest)
                l1t_ = tf.reduce_mean(l1t_)
                l1t[i] = l1t_

            return {
                "llr_loss": loss_data,
                "ch_loss": loss_chest,
                "Total": loss_data+loss_chest,
            }
        else:
            # Only return the last iteration during inference
            return llrs[-1][0], h_hats[-1]


class MDNeuralPUSCHReceiver(Layer):
    """MDX 5G NR PUSCH receiver.

    Wraps :class:`CGNNOFDM` with the 5G NR processing: PA-LS channel
    estimation, transport block (TB) re-encoding of the labels during training
    and TB decoding during inference.

    Args:
        sys_parameters: system parameters (resource grid, PUSCH configurations,
            transmitters and MDX hyperparameters)
        training: if True, ``call`` returns the training losses

    Input:
        In training ``(y, active_tx, b, h, mcs_ue_mask, x, no, batch_size)``,
        otherwise ``(y, active_tx, no, mcs_ue_mask, x, h, batch_size)``.
            y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], complex
            active_tx: [batch_size, num_tx], active-user mask
            b: information bits, one entry per evaluated MCS
            h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant,
                num_ofdm_symbols, fft_size], complex ground-truth channel
            mcs_ue_mask: [batch_size, num_tx, num_mcs], one-hot MCS of each user
            x: [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size],
                complex transmitted symbols
            no: [batch_size], noise variance

    Output:
        In training a dict of losses, otherwise
        ``(b_hat, h_hat_refined, h_hat, tb_crc_status)`` with the decoded TBs
        [batch_size, num_tx, tb_size], the refined and the initial channel
        estimates [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
        2*num_rx_ant], and the TB CRC status [batch_size, num_tx].
    """

    def __init__(self,
                sys_parameters,
                training=False,
                **kwargs):
        super().__init__(**kwargs)

        self._sys_parameters = sys_parameters

        self._training = training

        self._tb_encoders = []
        self._tb_decoders= []

        self._num_mcss_supported = len(sys_parameters.mcs_index)
        for mcs_list_idx in range(self._num_mcss_supported):
                self._tb_encoders.append(
                    self._sys_parameters.transmitters[mcs_list_idx]._tb_encoder)

                self._tb_decoders.append(
                    TBDecoder(self._tb_encoders[mcs_list_idx],
                              num_bp_iter=sys_parameters.num_bp_iter,
                              cn_type=sys_parameters.cn_type))

        # Precoding matrix [num_tx, num_tx_ant, num_layers = 1]
        if hasattr(sys_parameters.transmitters[0], "_precoder"):
            self._precoding_mat = sys_parameters.transmitters[0]._precoder._w
        else:
            self._precoding_mat = tf.ones([sys_parameters.max_num_tx,
                                           sys_parameters.num_antenna_ports, 1], tf.complex64)

        rg = sys_parameters.transmitters[0]._resource_grid
        pc =  sys_parameters.pusch_configs[0][0]
        self._ls_est = PUSCHLSChannelEstimator(
                resource_grid=rg,
                dmrs_length=pc.dmrs.length,
                dmrs_additional_position=pc.dmrs.additional_position,
                num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
                interpolation_type="nn")

        rg_type = rg.build_type_grid()[:,0]
        pilot_ind = tf.where(rg_type==1)
        self._pilot_ind = np.array(pilot_ind)

        self._layer_demappers = []
        for mcs_list_idx in range(self._num_mcss_supported):
                self._layer_demappers.append(
                    LayerDemapper(
                            self._sys_parameters.transmitters[mcs_list_idx]._layer_mapper,
                            sys_parameters.transmitters[mcs_list_idx]._num_bits_per_symbol))

        self._neural_rx = CGNNOFDM(
                    sys_parameters,
                    max_num_tx=sys_parameters.max_num_tx,
                    training=training,
                    num_it=sys_parameters.num_nrx_iter,
                    d_s=sys_parameters.d_s,
                    num_units_init=sys_parameters.num_units_init,
                    num_units_agg=sys_parameters.num_units_agg,
                    num_units_state=sys_parameters.num_units_state,
                    num_units_readout=sys_parameters.num_units_readout,
                    layer_demappers=self._layer_demappers,
                    layer_type_dense=sys_parameters.layer_type_dense,
                    layer_type_conv=sys_parameters.layer_type_conv,
                    layer_type_readout=sys_parameters.layer_type_readout,
                    dtype=sys_parameters.nrx_dtype)

    def estimate_channel(self, y, num_tx,no):
        """PA-LS channel estimate and its error variance.

        Returns ``h_hat`` [batch_size, num_tx, num_subcarriers,
        num_ofdm_symbols, 2*num_rx_ant] and ``err_var`` [batch_size, num_rx,
        num_subcarriers, num_ofdm_symbols, num_rx_ant].
        """
        if self._sys_parameters.initial_chest == 'ls':
            if self._sys_parameters.mask_pilots:
                raise ValueError("Cannot use initial channel estimator if " \
                                "pilots are masked.")
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #    num_ofdm_symbols, num_effective_subcarriers]
            h_hat, err_var = self._ls_est([y, no])

            err_var_dt = tf.broadcast_to(err_var, tf.shape(h_hat))
            err_var_dt = tf.transpose(err_var_dt, [0, 1, 5, 6, 2, 3, 4])
            err_var_dt = flatten_last_dims(err_var_dt, 2)
            err_var = tf.reduce_sum(err_var_dt, -1)

            h_hat = h_hat[:,0,:,:num_tx,0]
            h_hat = tf.transpose(h_hat, [0, 2, 4, 3, 1])
            h_hat = tf.concat([tf.math.real(h_hat), tf.math.imag(h_hat)],
                              axis=-1)
            err_var = tf.transpose(err_var,perm=[0,1,3,2,4])

        elif self._sys_parameters.initial_chest == None:
            h_hat = None
        return h_hat, err_var

    def preprocess_channel_ground_truth(self, h):
        """Effective (precoded) ground-truth channel in real representation.

        [batch_size, num_rx=1, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols,
        fft_size] -> [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
        2*num_rx_ant]
        """
        h = tf.squeeze(h, axis=1)
        # [batch_size, num_tx, num_effective_subcarriers, num_ofdm_symbols, num_rx_ant, num_tx_ant]
        h = tf.transpose(h, perm=[0,2,5,4,1,3])

        # [1, num_tx, 1, 1, num_tx_ant, 1]
        w = insert_dims(tf.expand_dims(self._precoding_mat, axis=0), 2, 2)
        h = tf.squeeze(tf.matmul(h, w), axis=-1)
        h = tf.concat([tf.math.real(h), tf.math.imag(h)], axis=-1)
        return h

    def preprocess_x_ground_truth(self, x):
        """Map the per-antenna ground-truth symbols back to one symbol per user.

        [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size] ->
        [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2]
        """
        # [batch_size, max_num_tx, num_ofdm_symbols, fft_size, num_tx_ant, 1]
        x = tf.transpose(x,perm=[0,1,3,4,2])
        x = tf.expand_dims(x,axis=-1)

        w = self._precoding_mat
        # Inverse precoding weights
        w = 1/ (2 * self._precoding_mat)
        w = insert_dims(tf.expand_dims(w, axis=0), 2, 2)
        x = tf.matmul(w,x,transpose_a=True)
        x = tf.squeeze(x, axis=[4,5])

        x = tf.transpose(x,perm=[0,1,3,2])
        x = tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)

        return x

    def call(self, inputs, mcs_arr_eval=[0], mcs_ue_mask_eval=None):
        """Run the receiver; see the class docstring for inputs and outputs."""
        if self._training:
            y, active_tx, b, h, mcs_ue_mask, x, no, batch_size  = inputs
            # re-encode bits in training mode to generate labels
            # avoids the need for post-FEC bits as labels
            if len(mcs_arr_eval)==1 and not isinstance(b, list):
                b = [b]
            bits = []
            for idx in range(len(mcs_arr_eval)):
                bits.append(
                    self._sys_parameters.transmitters[mcs_arr_eval[idx]]._tb_encoder(b[idx]))

            # Initial channel estimation
            num_tx = tf.shape(active_tx)[1]
            h_hat, err_var = self.estimate_channel(y, num_tx, no)

            if h is not None:
                h = self.preprocess_channel_ground_truth(h)
            if x is not None:
                x = self.preprocess_x_ground_truth(x)

            losses = self._neural_rx((y, h_hat, active_tx,
                                      bits, h, mcs_ue_mask, x, no,err_var, batch_size),
                                      mcs_arr_eval)
            return losses

        else:
            y, active_tx, no, mcs_ue_mask, x, h, batch_size = inputs

            num_tx = tf.shape(active_tx)[1]
            h_hat, err_var = self.estimate_channel(y, num_tx, no)

            x = self.preprocess_x_ground_truth(x)
            h = self.preprocess_channel_ground_truth(h)

            llr, h_hat_refined = self._neural_rx(
                                            (y, h_hat, active_tx, no,x,h, err_var, batch_size),
                                            [mcs_arr_eval[0]],
                                            mcs_ue_mask_eval=mcs_ue_mask_eval, no=no)

            b_hat, tb_crc_status = self._tb_decoders[mcs_arr_eval[0]](llr)

            return b_hat, h_hat_refined, h_hat, tb_crc_status
