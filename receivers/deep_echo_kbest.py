
# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026


"""DeepEcho5G variant with K-best detection on the refined channel estimate."""

import core.runtime as _runtime

import tensorflow as tf

from sionna.ofdm import KBestDetector
from sionna.utils import flatten_last_dims

from .deep_echo import DeepEcho5G


class DeepEchoKBest5G(DeepEcho5G):
    """Run DeepEcho channel refinement followed by OFDM K-best detection.

    Shape shorthand: B=batch, R=num_rx, T=num_tx, RA=num_rx_ant,
    F=num_effective_subcarriers, S=num_ofdm_symbols, L=num_streams,
    N=num_coded_bits.
    """

    def __init__(self, sys_parameters, training=False, **kwargs):
        super().__init__(sys_parameters, training=training, **kwargs)

        self._kbest_detectors = []
        for mcs_list_idx in range(self._num_mcss_supported):
            tx = self._sys_parameters.transmitters[mcs_list_idx]
            self._kbest_detectors.append(
                KBestDetector(
                    "bit",
                    self._sys_parameters.max_num_tx,
                    64,
                    tx._resource_grid,
                    self._sys_parameters.sm,
                    constellation_type="qam",
                    num_bits_per_symbol=tx._num_bits_per_symbol,
                )
            )

    def _channel_to_complex(self, h_hat):
        # [B, T, F, S, 2*RA] -> [B, T, F, S, RA], complex
        real_part, imag_part = tf.split(h_hat, num_or_size_splits=2, axis=-1)
        return tf.complex(real_part, imag_part)

    def _channel_for_kbest(self, h_hat):
        # [B, T, F, S, RA] -> [B, R=1, RA, T, L=1, S, F]
        h_hat = tf.transpose(h_hat, perm=[0, 4, 1, 3, 2])
        h_hat = tf.expand_dims(h_hat, axis=1)
        h_hat = tf.expand_dims(h_hat, axis=4)
        return h_hat

    def call(self, inputs, mcs_arr_eval=[0], mcs_ue_mask_eval=None):
        if not isinstance(mcs_arr_eval, (list, tuple)):
            mcs_arr_eval = [mcs_arr_eval]

        if self._training:
            y, active_tx, b, h, mcs_ue_mask, no  = inputs
            losses = super().call(inputs, mcs_arr_eval=mcs_arr_eval,
                                mcs_ue_mask_eval=mcs_ue_mask_eval)
        else:
            # inputs -> y: [B, R=1, RA, S, F], active_tx: [B, T], no: [B]
            y, active_tx, no, mcs_ue_mask, h = inputs
            b_hat, h_hat_refined, h_hat, tb_crc_status = super().call(inputs, mcs_arr_eval=mcs_arr_eval,
                                mcs_ue_mask_eval=mcs_ue_mask_eval)

        mcs_idx = mcs_arr_eval[0]
        num_tx = tf.shape(active_tx)[1]

        # [B, R, RA, T, num_streams_per_tx, S, F]
        h_hat_init, err_var = self._ls_est([y, no])

        err_var_init = tf.broadcast_to(err_var, tf.shape(h_hat_init))
        err_var_init = tf.transpose(err_var_init, [0, 1, 5, 6, 2, 3, 4])
        err_var_init = flatten_last_dims(err_var_init, 2)
        err_var_init = tf.reduce_sum(err_var_init, -1)
        # [B, R, F, S, RA]
        err_var_init = tf.transpose(err_var_init,perm=[0,1,3,2,4])

        h_hat_init = h_hat_init[:,0,:,:num_tx,0] # [B, RA, T, S, F]
        h_hat_init = tf.transpose(h_hat_init, [0, 2, 4, 3, 1]) # [B, T, F, S, RA]

        # _deep_echo -> h_hat_refined: [B, T, F, S, 2*RA]
        _, h_hat_refined = self._deep_echo(
            (y, h_hat_init, active_tx, no, err_var_init),
            [mcs_arr_eval[0]],
            mcs_ue_mask_eval=mcs_ue_mask_eval,
            no=no,
        )

        # [B, T, F, S, 2*RA]
        h_hat_init_out = tf.concat(
            [tf.math.real(h_hat_init), tf.math.imag(h_hat_init)],
            axis=-1,
        )

        h_hat_refined_cplx = self._channel_to_complex(h_hat_refined)
        h_hat_refined_det = self._channel_for_kbest(h_hat_refined_cplx)

        # Detector inputs: y [B, R=1, RA, S, F], h_hat_refined_det [B, R=1, RA, T, L=1, S, F],
        # err_var broadcastable to [B, R=1, RA, T, L=1, S, F], no [B].
        llr = self._kbest_detectors[mcs_idx](
            [y, h_hat_refined_det, err_var/1, no/1]
        )
        # llr -> [B, T, N]
        llr = self._layer_demappers[mcs_idx](llr)
        # llr: [B, T, N] -> b_hat: [B, T, tb_size], tb_crc_status: [B, T]
        b_hat, tb_crc_status = self._tb_decoders[mcs_idx](llr)

        # return: b_hat [B, T, tb_size], h_hat_refined [B, T, F, S, 2*RA],
        # h_hat_init_out [B, T, F, S, 2*RA], tb_crc_status [B, T]
        return b_hat, h_hat_refined, h_hat_init_out, tb_crc_status
