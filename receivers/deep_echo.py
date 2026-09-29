# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026

# The 5G NR wrapper (DeepEcho5G) follows the structure of

"""DeepEcho: 5G NR PUSCH receiver hosting a config-defined block graph.

"""

import core.runtime as _runtime
from core.runtime import DEBUGD

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from sionna.utils import flatten_last_dims, insert_dims, expand_to_rank
from sionna.nr import TBDecoder, LayerDemapper, PUSCHLSChannelEstimator

from blocks import Shapes

from utils import dbg

from graphs.block_graph import ModularGraph


class DeepEcho(Model):
    """Runs the block graph on the per-link LS channel estimates.

    The LS channel estimate is reshaped to one real-valued
    [num_subcarriers, num_ofdm_symbols, 2] grid per (batch, user, receive
    antenna) link. Together with the auxiliary inputs (received signal, noise
    statistics, MCS masks, active-user mask) and, in training, the targets, it
    is passed to the graph. In training ``call`` returns the graph losses
    averaged over the active users, otherwise the LLRs; the refined channel
    estimate [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
    2*num_rx_ant] is returned in both cases.
    """

    _DENSE_NOISE_S_BLOCK_TYPES = frozenset(("LMMSE",))

    @classmethod
    def _block_config_needs_dense_noise_s(cls, block_config):
        """True if a block needs the dense noise-plus-error covariance ``s``."""
        return any(
            isinstance(block, dict)
            and block.get("type") in cls._DENSE_NOISE_S_BLOCK_TYPES
            for block in (block_config or ())
        )

    def __init__(self, sys_parameters, training,
                        cplx_dtype=tf.complex64, **kwargs):

        super().__init__(**kwargs)
        self._dtype = cplx_dtype
        self._real_dtype = tf.as_dtype(cplx_dtype).real_dtype
        self._sys_parameters = sys_parameters
        self._training = training

        self._loss_dtype = tf.float32

        self._num_mcs_supported = len(sys_parameters.mcs_index)
        self._needs_dense_noise_s = self._block_config_needs_dense_noise_s(
            sys_parameters.block_config
        )

        # System description passed to the blocks
        sys = {}
        sys["rg"] = self._sys_parameters.transmitters[0]._resource_grid
        sys["sm"] = self._sys_parameters.sm
        sys["num_mcss_supported"] = self._num_mcs_supported
        sys["transmitters"]       = self._sys_parameters.transmitters
        sys["demapping_type"]     = sys_parameters.demapping_type

        num_bits_per_symbol=[]
        for mcs_list_idx in range(sys["num_mcss_supported"]):
            num_bits_per_symbol.append(sys_parameters.pusch_configs[mcs_list_idx][0].tb.num_bits_per_symbol)
        sys["num_bits_per_symbol"] = num_bits_per_symbol
        self._num_bits_per_symbol_by_mcs = tf.constant(num_bits_per_symbol, dtype=tf.int32)

        if hasattr(self._sys_parameters, "num_cols_per_panel"):
            sys["num_cols_per_panel"] = self._sys_parameters.num_cols_per_panel
        if hasattr(self._sys_parameters, "num_rows_per_panel"):
            sys["num_rows_per_panel"] = self._sys_parameters.num_rows_per_panel

        sys["n_size_bwp_training"] = self._sys_parameters.n_size_bwp_training
        sys["n_size_bwp_eval"]     = self._sys_parameters.n_size_bwp_eval

        # The graph diagram is only rendered in training, since its file name
        # depends only on the config label.
        self.nn = ModularGraph(config=sys_parameters.block_config, sys=sys,
                    shapes=None, rg_params=None,
                    graph_label=sys_parameters.label, name="mdx",
                    render_graph=training, training=training)
        self._graph_has_llr_stash = "llr" in self.nn.stash_keys
        self._graph_has_channel_stash = "channel" in self.nn.stash_keys

        self._rg = sys_parameters.transmitters[0]._resource_grid
        self._num_data_symbols = self._rg.pilot_pattern.num_data_symbols
        pilot_pattern = self._rg.pilot_pattern
        pilot_mask = pilot_pattern.mask
        pilot_mask = 1 - pilot_mask[:,0]
        pilot_mask = tf.transpose(pilot_mask, perm=[0,2,1])
        # [1,num_tx,num_effective_subcarriers, num_ofdm_symbols,1]
        pilot_mask = insert_dims(pilot_mask,num_dims=1,axis=0)
        pilot_mask = insert_dims(pilot_mask,num_dims=1,axis=-1)
        pilot_mask = tf.cast(pilot_mask,self._real_dtype)
        self._pilot_mask = pilot_mask

    def _make_extras(self, y, shapes, no, err_var, s,
                        mcs_ue_mask, mcs_arr_eval):
        """Collect the auxiliary inputs of the graph."""
        extras = {}
        extras["shapes"] = shapes
        extras["pilot_mask"] = self._pilot_mask

        rx_signal = {}
        rx_signal["y"] = y # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_subcarriers]
        extras["rx_signal"] = rx_signal

        mcs_masks = {}
        mcs_masks["mcs_ue_mask"] = mcs_ue_mask # [batch_size, max_num_tx, depth(num_mcss)]
        mcs_masks["mcs_arr_eval"] = mcs_arr_eval
        extras["mcs_masks"] = mcs_masks

        noise = {}
        noise["no"] = no # [batch_size]
        noise["err_var"] = err_var # [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant]
        noise["s"] = s  # None, or [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant, num_rx_ant] complex
        extras["noise"] = noise

        return extras

    def make_targets(self, h_ft, bits):
        """Training targets of the graph.

        ``h_ft``: ground-truth channel [batch_size*num_tx*num_rx_ant,
        num_subcarriers, num_ofdm_symbols, 2] or None; ``bits``: coded bits or None.
        """
        targets = {}
        targets["channel_ofdm"] = h_ft
        targets["bits"] = bits

        return targets

    def call(self, inputs, mcs_arr_eval, mcs_ue_mask_eval=None,  h_true=None, no=None):
        """
        Args:
            inputs: in training ``(y, h_hat, active_tx, bits, mcs_ue_mask, no, err_var)``,
                otherwise ``(y, h_hat, active_tx, no, err_var)``
                y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], complex
                h_hat: [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant], complex
                active_tx: [batch_size, num_tx]
                mcs_ue_mask: [batch_size, num_tx, num_mcs]
                no: [batch_size]
                err_var: [batch_size, num_rx, num_subcarriers, num_ofdm_symbols, num_rx_ant]
            mcs_arr_eval: indices of the evaluated MCSs
            mcs_ue_mask_eval: optional MCS mask for inference
            h_true: optional ground-truth channel, same shape as ``h_hat``
        """
        if self._training:
            (y, h_hat, active_tx, bits, mcs_ue_mask, no , err_var) = inputs
        else:
            (y, h_hat, active_tx, no, err_var) = inputs
            if mcs_ue_mask_eval is None:
                mcs_ue_mask = tf.one_hot(mcs_arr_eval[0],
                                         depth=self._num_mcs_supported)
            else:
                mcs_ue_mask = mcs_ue_mask_eval
            mcs_ue_mask = expand_to_rank(mcs_ue_mask, 3, axis=0)

            h_true = None
            bits = None

        batch_size = tf.shape(y)[0]
        num_tx = tf.shape(h_hat)[1]
        num_ant_rx = tf.shape(y)[2]
        fft_size = tf.shape(y)[4]

        S = Shapes(batch_size=batch_size, num_tx=num_tx, num_ant_rx=num_ant_rx,
                    num_ant_tx=1, num_rx=1, fft_size=fft_size)

        # [batch_size * num_tx * num_rx_ant, num_effective_subcarriers, num_ofdm_symbols,2]
        h_hat = tf.stack([tf.math.real(h_hat), tf.math.imag(h_hat)], axis=-1)
        h_hat = tf.transpose(h_hat,perm=[0,1,4,2,3,5])
        h_hat = tf.reshape(h_hat, [S.B*S.T*S.RA, S.F, 14, 2])
        h_hat = tf.stop_gradient(h_hat)

        if h_true is not None:
            h_true = tf.stack([tf.math.real(h_true), tf.math.imag(h_true)], axis=-1)
            h_true = tf.transpose(h_true,perm=[0,1,4,2,3,5])
            h_true = tf.reshape(h_true, [S.B*S.T*S.RA, S.F, 14, 2])
            h_true = tf.stop_gradient(h_true)

        err_var = tf.squeeze(err_var,axis=1)
        if self._needs_dense_noise_s:
            # s = N0*I + diag(err_var)
            # [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant, num_rx_ant]
            err_var_ = tf.linalg.diag(err_var)
            err_var_ = tf.cast(err_var_,self._dtype)

            no_ = expand_to_rank(no,2,-1)
            no_ = tf.broadcast_to(no_,[S.B, S.RA])
            no_ = insert_dims(no_,num_dims=2,axis=1)
            no_ = tf.linalg.diag(no_)
            no_ = tf.cast(no_,self._dtype)

            s = no_ + err_var_
        else:
            s = None

        targets =  self.make_targets(h_true, bits)

        extras = self._make_extras(y, S,
                        no=no, err_var=err_var, s=s,
                        mcs_ue_mask=mcs_ue_mask, mcs_arr_eval=mcs_arr_eval)

        # Active-user mask, [batch_size*num_tx, 1, 1, 1]
        mask_tx = tf.identity(active_tx)
        mask_tx = tf.reshape(mask_tx, [-1,1,1,1])
        mask_tx = tf.cast(mask_tx, tf.bool)
        extras["mask"] = mask_tx

        last_out, loss_dict, stash, stash_view = self.nn(h_hat, targets=targets, extras=extras, training=self._training)
        dbg("Run graph")

        if self._training:
            # Average the per-user losses over the active users
            def __process_loss(x, shapes, active_tx):
                s = shapes
                x = tf.reshape(x, [s.B, s.T])
                x = tf.multiply(x, active_tx)
                x = tf.reduce_sum(x) / (tf.reduce_sum(active_tx)+1)
                return x

            def _process_loss(x, shapes, active_tx):
                if tf.is_tensor(x):
                    return __process_loss(x, shapes, active_tx)
                if isinstance(x, (list, tuple)):
                    return type(x)(_process_loss(y, shapes, active_tx) for y in x)
                if isinstance(x, dict):
                    return {k: _process_loss(v, shapes, active_tx) for k, v in x.items()}
                return x

            loss_dict = _process_loss(loss_dict, S, active_tx)

            dbg("loss processed.")

            if DEBUGD['print']>0:
                def log_nonfinite_losses(loss_tree, prefix="loss"):
                    """Print the status ("ok", "nan", "inf" or "both") of every loss tensor."""
                    def _status(x: tf.Tensor) -> tf.Tensor:
                        any_nan = tf.reduce_any(tf.math.is_nan(x))
                        any_inf = tf.reduce_any(tf.math.is_inf(x))
                        both = tf.logical_and(any_nan, any_inf)

                        status = tf.where(both,
                                        tf.constant("both"),
                                        tf.where(any_nan,
                                                tf.constant("nan"),
                                                tf.where(any_inf, tf.constant("inf"), tf.constant("ok"))))
                        return status

                    def _walk(node, key_path):
                        if tf.is_tensor(node):
                            s = _status(node)
                            tf.print("[nonfinite]", key_path, "=>", s, "| dtype:", node.dtype, "| shape:", tf.shape(node), "| value:", node)
                            return
                        if isinstance(node, dict):
                            for k, v in node.items():
                                _walk(v, key_path + f".{k}")
                            return
                        if isinstance(node, (list, tuple)):
                            for i, v in enumerate(node):
                                _walk(v, key_path + f"[{i}]")
                            return
                        return

                    _walk(loss_tree, prefix)
                    print("-------------------------\n")

                log_nonfinite_losses(loss_dict, prefix="loss")

        if self._graph_has_llr_stash:
            llr = stash_view["llr"]
        else:
            dbg("Warning: 'llr' not found in stash_view; using zeros instead.")
            llrs = []
            for mcs_idx in mcs_arr_eval:
                num_llrs = tf.cast(
                    self._num_data_symbols * self._num_bits_per_symbol_by_mcs[mcs_idx],
                    S.B.dtype,
                )
                llrs.append(
                    tf.zeros(
                        tf.stack([S.B, S.T, num_llrs]),
                        dtype=self._real_dtype,
                    )
                )
            llr = llrs[0] if len(llrs) == 1 else llrs

        if self._graph_has_channel_stash:
            ch_refined = stash_view["channel"]
        else:
            ch_refined = tf.zeros([S.B* S.T* S.RA, S.F, 14, 2], dtype=self._real_dtype)
            dbg("Warning: 'channel' not found in stash_view; using zeros instead.")

        # [batch_size*num_tx*num_rx_ant, num_subcarriers, 14, 2]
        # -> [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        ch_refined = tf.reshape(ch_refined, [S.B, S.T, S.RA, S.F, 14, 2])
        ch_refined = tf.transpose(ch_refined, perm=[0,1,3,4,2,5])
        ch_refined = tf.concat([ch_refined[...,0], ch_refined[...,1]], axis=-1)

        if self._training:
            return loss_dict, ch_refined
        else:
            return llr, ch_refined


class DeepEcho5G(Layer):
    # pylint: disable=line-too-long
    r"""
    5G NR PUSCH receiver hosting a config-defined block graph.

    The block graph (:class:`graphs.block_graph.ModularGraph`) is built from
    the ``block_config`` of the system parameters and receives the LS channel
    estimate of every antenna-layer link. This layer adds the 5G NR parts
    around it: LS channel estimation, transport block (TB) re-encoding of the
    labels during training, and TB decoding during inference. It is used to
    run MDX, MDELAN, CHEA and the baseline channel estimators.

    Parameters
    ----------
    sys_parameters : Parameters
        The system parameters.

    training : boolean
        Set to `True` if instantiated for training. Set to `False` otherwise.

    Input
    ------
    (y, active_tx, bits, h, mcs_ue_mask, no) in training,
    (y, active_tx, no, mcs_ue_mask, h) otherwise :

        y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
            The received OFDM resource grid after cyclic prefix removal and FFT.

        active_tx: [batch_size, num_tx], tf.float
            Active user mask where each `0` indicates non-active users and `1`
            indicates an active user.

        bits : list of [[batch_size, num_tx, num_data_symbols*num_bits_per_symbol],
                        tf.int]
            Transmitted information (uncoded) bits for each evaluated MCS.
            Only required for training to compute the loss function.

        h : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            num_ofdm_symbols, fft_size], tf.complex, or None
            Ground-truth channel, used as target by the loss blocks.

        mcs_ue_mask: [batch_size, max_num_tx, len(mcs_index)], tf.int32
            One-hot mask that specifies the MCS index of each UE for each batch
            sample. Only used in training to enable UE-specific MCS
            association.

        no : [batch_size], tf.float
            Noise variance.

    mcs_arr_eval : list with int elements
        Selects the elements (indices) of the mcs_index array to process.
        Defaults to [0]

    mcs_ue_mask_eval : [batch_size, max_num_tx, len(mcs_index)], tf.int32, None
        Optional additional parameter to specify an mcs_ue_mask for evaluation
        (self._training=False).
        Defaults to None, which internally assumes that all UEs are scheduled
        with mcs_arr_eval[0]

    Output
    ------
    Depending on the value of `training`:

    If Training set to `False`
        Inference only implemented for one MCS (first element in mcs_arr_eval)

    (b_hat, h_hat_refined, h_hat, tb_crc_status) : tuple

        b_hat : [batch_size, num_tx, tb_size], tf.float
            Reconstructed transport block bits after decoding.

        h_hat_refined, [batch_size, num_tx, num_effective_subcarriers,
                    num_ofdm_symbols, 2*num_rx_ant]
            Refined channel estimate from the block graph.

        h_hat, [batch_size, num_tx, num_effective_subcarriers,
                   num_ofdm_symbols, 2*num_rx_ant]
            Initial (LS) channel estimate fed to the block graph.

        tb_crc_status: [batch_size, num_tx]
            Status of the TB CRC for each decoded TB.

    If Training set to `True`

    losses : dict
        Losses of the loss blocks of the graph, averaged over active UEs.
    """

    def __init__(self,
                sys_parameters,
                training=False,
                **kwargs):
        super().__init__(**kwargs)

        self._sys_parameters = sys_parameters
        self._training = training

        self._dtype = sys_parameters.dtype
        self._cplx_dtype = sys_parameters.dtype
        self._real_dtype = tf.as_dtype(sys_parameters.dtype).real_dtype

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

        # Precoding matrix to post-process the ground-truth channel
        # [num_tx, num_tx_ant, num_layers = 1]
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
                interpolation_type="lin")

        rg_type = rg.build_type_grid()[:,0]
        pilot_ind = tf.where(rg_type==1)
        self._pilot_ind = np.array(pilot_ind)

        self._layer_demappers = []
        for mcs_list_idx in range(self._num_mcss_supported):
                self._layer_demappers.append(
                    LayerDemapper(
                            self._sys_parameters.transmitters[mcs_list_idx]._layer_mapper,
                            sys_parameters.transmitters[mcs_list_idx]._num_bits_per_symbol))

        self._deep_echo = DeepEcho(self._sys_parameters,
                                    training,
                                    cplx_dtype=self._cplx_dtype,
                                    )

    def estimate_channel(self, y, num_tx,no):
        """LS channel estimate and its error variance.

        Returns ``h_hat`` [batch_size, num_tx, num_subcarriers,
        num_ofdm_symbols, num_rx_ant] (complex) and ``err_var`` [batch_size,
        num_rx, num_subcarriers, num_ofdm_symbols, num_rx_ant].
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
            err_var = tf.transpose(err_var,perm=[0,1,3,2,4])

        elif self._sys_parameters.initial_chest == None:
            h_hat = None
        return h_hat, err_var

    def apply_precoding_effect(self, h):
        """Effective (precoded) ground-truth channel.

        [batch_size, num_rx=1, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols,
        fft_size] -> [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
        num_rx_ant], complex
        """
        h = tf.squeeze(h, axis=1)
        # [batch_size, num_tx, num_effective_subcarriers, num_ofdm_symbols, num_rx_ant, num_tx_ant]
        h = tf.transpose(h, perm=[0,2,5,4,1,3])

        # [1, num_tx, 1, 1, num_tx_ant, 1]
        w = insert_dims(tf.expand_dims(self._precoding_mat, axis=0), 2, 2)
        h = tf.squeeze(tf.matmul(h, w), axis=-1)
        return h

    def preprocess_channel_ground_truth(self, h):
        """Like :meth:`apply_precoding_effect`, with real output [..., 2*num_rx_ant]."""
        h = tf.squeeze(h, axis=1)
        # [batch_size, num_tx, num_effective_subcarriers, num_ofdm_symbols, num_rx_ant, num_tx_ant]
        h = tf.transpose(h, perm=[0,2,5,4,1,3])

        # [1, num_tx, 1, 1, num_tx_ant, 1]
        w = insert_dims(tf.expand_dims(self._precoding_mat, axis=0), 2, 2)
        h = tf.squeeze(tf.matmul(h, w), axis=-1)

        h = tf.concat([tf.math.real(h), tf.math.imag(h)], axis=-1)
        return h

    def call(self, inputs, mcs_arr_eval=[0], mcs_ue_mask_eval=None):
        """Run the receiver; see the class docstring for inputs and outputs."""
        if self._training:
            y, active_tx, b, h, mcs_ue_mask, no  = inputs
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
                h = self.apply_precoding_effect(h)

            losses, ch_refined = self._deep_echo ((y, h_hat, active_tx,
                                      bits, mcs_ue_mask, no, err_var),
                                      mcs_arr_eval, h_true=h)

            return losses

        else:
            y, active_tx, no, mcs_ue_mask, h = inputs

            num_tx = tf.shape(active_tx)[1]
            h_hat, err_var = self.estimate_channel(y, num_tx, no)

            if h is not None:
                h = self.apply_precoding_effect(h)

            llr, h_hat_refined = self._deep_echo(
                                            (y, h_hat, active_tx, no, err_var),
                                            [mcs_arr_eval[0]],
                                            mcs_ue_mask_eval=mcs_ue_mask_eval, h_true=h, no=no)

            # [batch_size, num_tx, num_effective_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
            h_hat = tf.concat([tf.math.real(h_hat), tf.math.imag(h_hat)],
                  axis=-1)

            b_hat, tb_crc_status = self._tb_decoders[mcs_arr_eval[0]](llr)

            return b_hat, h_hat_refined, h_hat, tb_crc_status
