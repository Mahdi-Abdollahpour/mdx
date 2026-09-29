# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

##### E2E model for system evaluations #####
import core.runtime as _runtime

import tensorflow as tf
from tensorflow.keras import Model
from sionna.channel import gen_single_sector_topology
from sionna.utils import BinarySource, ebnodb2no, expand_to_rank, log10, flatten_dims, split_dim, insert_dims
from .baseline_rx import BaselineReceiver
from receivers.deep_echo import DeepEcho5G
from receivers.deep_echo_kbest import DeepEchoKBest5G
from receivers.ch_saver import ChSaver5G
from receivers.md_rx import MDNeuralPUSCHReceiver
from .neural_rx import NeuralPUSCHReceiver
from receivers.neural_rx_kbest import NeuralKBestPUSCHReceiver


class E2E_Model(Model):
    r"""E2E_Model(sys_parameters, training=False, return_tb_status=False, **kwargs)
    End-to-end model for system evaluation.

    The model can be used with the neural receiver as well as with the baseline
    receivers.

    Parameters
    ----------
    sys_parameters : Parameters
        The system parameters.

    training : bool
        If True, the model returns the loss instead of b and b_hat.

    return_tb_status : bool
        If True, the model returns if the transport block CRC holds.

    mcs_arr_eval_idx : int
        Selects the element (index) of the mcs_index array to evaluate. Can be
        overwritten as call argument.
        Defaults to 0, which selects the very first element specified in the
        mcs_index array.

    output_channel_est_only : bool
        Currently unused. Defaults to False.

    Input
    -----
    batch_size : int
        Batch size of random transmit signals to be generated

    ebno_db: float
        SNR in dB. Interpreted as rate-adjusted SNR if ``sys_parameters.ebno``
        is `True`.

    num_tx : int | None
        Number of active DMRS ports. Remark: the model always simulates all
        max_num_tx UEs. However, they are masked at the tx output, i.e., no
        energy is transmitted. This allows training with varying number of
        users without re-building the TensorFlow graph.

    return_channel : bool
        If True, the ground-truth and estimated channels are returned as well.

    mcs_arr_eval_idx : int
        Selects the element (index) of the mcs_index array to evaluate.
        If not specified (defaults to None), the index set by the constructor
        is used. Has no effect if an mcs_ue_mask is specified.

    mcs_ue_mask : None, or [batch_size, max_num_tx, len(mcs_index)], tf.int32
        One-hot mask that specifies the MCS index of each UE for each batch
        sample. If not specified (defaults to None), the mcs_ue_mask will be
        inferred from msc_arr_eval_index.

    active_dmrs : None, or [batch_size, max_num_tx], tf.int32
        Optional one-hot mask that specifies the active DMRS for each batch
        sample. If active_dmrs is None (default), the E2E model generates the
        active_dmrs mask by itself.

    Output
    ------
    In inference mode (training=False)
        Depending on if return_tb_status and return_channel is True

        Tuple : b, b_hat, [tb_crc_status], [h, h_hat_refined, h_hat]

            b : [batch_size, num_tx, tb_size], tf.float
                    Transmitted information bits for selected MCS.

            b_hat : [batch_size, num_tx, tb_size], tf.float
                Decoded information bits for selected MCS.

            tb_crc_status: [batch_size, num_tx], tf.float
                Status of the TB CRC for each decoded TB.

            h: [batch_size, num_tx, num_effective_subcarriers,
                    num_ofdm_symbols, 2*num_rx_ant]
                Ground truth channel CFRs

            h_hat_refined, [batch_size, num_tx, num_effective_subcarriers,
                    num_ofdm_symbols, 2*num_rx_ant]
                Refined channel estimate from the NRX.

            h_hat, [batch_size, num_tx, num_effective_subcarriers,
                   num_ofdm_symbols, 2*num_rx_ant]
                Initial channel estimate used for the NRX.

    In training mode:
        losses: tuple with (loss_data, loss_chest) from NeuralPUSCHReceiver
            Only if ``training`` is `True`. The "mdx" and "deep_echo"
            receivers return a dict of named losses instead.

            loss_data: tf.float
                Binary cross-entropy loss on LLRs. Computed from active UEs and
                their selected MCSs.

            loss_chest: tf.float
                Mean-squared-error (MSE) loss between channel estimates and
                ground truth channel CFRs. Only relevant if double-readout is
                used.

    """

    def __init__(self, sys_parameters, training=False, return_tb_status=False,
                 mcs_arr_eval_idx=0, output_channel_est_only=False):

        super().__init__()

        assert isinstance(mcs_arr_eval_idx, int), "E2E Model can only evaluate one MCS at a time. For mixed MCS evaluation, use the E2E_Model_Mixed_MCS class."

        self._sys_parameters = sys_parameters
        self._training = training
        self._return_tb_status = return_tb_status
        self._mcs_arr_eval_idx = mcs_arr_eval_idx
        self._output_channel_est_only = output_channel_est_only
        ###################################
        # Transmitter
        ###################################

        self._source = BinarySource()
        self._transmitters = sys_parameters.transmitters
        if hasattr(sys_parameters.transmitters[0], "_precoder"):
            self._precoding_mat = sys_parameters.transmitters[0]._precoder._w
        ###################################
        # Channel
        ###################################
        self._channel = sys_parameters.channel

        ###################################
        # Receiver
        ###################################

        if self._sys_parameters.system == 'baseline_perf_csi_kbest':
            self._sys_name = "Baseline - Perf. CSI & K-Best"
            self._receiver = BaselineReceiver(
                                self._sys_parameters,
                                return_tb_status=return_tb_status,
                                mcs_arr_eval_idx=mcs_arr_eval_idx)

        elif self._sys_parameters.system == 'baseline_perf_csi_lmmse':
            self._sys_name = "Baseline - Perf. CSI & LMMSE"
            self._receiver = BaselineReceiver(
                                self._sys_parameters,
                                return_tb_status=return_tb_status,
                                mcs_arr_eval_idx=mcs_arr_eval_idx)

        elif self._sys_parameters.system == 'baseline_lmmse_kbest':
            self._sys_name = f"Baseline - LMMSE+K-Best"
            self._receiver = BaselineReceiver(
                                self._sys_parameters,
                                return_tb_status=return_tb_status,
                                mcs_arr_eval_idx=mcs_arr_eval_idx)

        elif self._sys_parameters.system == 'baseline_lmmse_lmmse':
            self._sys_name = f"Baseline - LMMSE+LMMSE"
            self._receiver = BaselineReceiver(
                            self._sys_parameters,
                            return_tb_status=return_tb_status,
                            mcs_arr_eval_idx=mcs_arr_eval_idx)

        elif self._sys_parameters.system == 'baseline_lsnn_lmmse':
            self._sys_name = f"Baseline - LS/nn+LMMSE"
            self._receiver = BaselineReceiver(
                            self._sys_parameters,
                            return_tb_status=return_tb_status,
                            mcs_arr_eval_idx=mcs_arr_eval_idx)

        elif self._sys_parameters.system == 'baseline_lslin_lmmse':
            self._sys_name = f"Baseline - LS/lin+LMMSE"
            self._receiver = BaselineReceiver(
                                self._sys_parameters,
                                return_tb_status=return_tb_status,
                                mcs_arr_eval_idx=mcs_arr_eval_idx)
        elif self._sys_parameters.system == 'baseline_lslin_kbest':
            self._sys_name = f"Baseline - LS/lin+K-Best"
            self._receiver = BaselineReceiver(
                                self._sys_parameters,
                                return_tb_status=return_tb_status,
                                mcs_arr_eval_idx=mcs_arr_eval_idx)
        elif self._sys_parameters.system == "nrx":
            self._sys_name = "Neural Receiver"
            self._receiver = NeuralPUSCHReceiver(
                                    self._sys_parameters,
                                    training)
        elif self._sys_parameters.system == "nrx_kbest":
            self._sys_name = "Neural Receiver + K-Best"
            self._receiver = NeuralKBestPUSCHReceiver(
                                    self._sys_parameters,
                                    training)
        elif self._sys_parameters.system == "mdx":
            self._sys_name = "Model Driven Receiver"
            self._receiver = MDNeuralPUSCHReceiver(
                                    self._sys_parameters,
                                    training)
        elif self._sys_parameters.system == "deep_echo":
            self._sys_name = "Deep Echo Multi-Path Component Estimatore"
            self._receiver = DeepEcho5G(
                                    self._sys_parameters,
                                    training)
        elif self._sys_parameters.system == "deep_echo_kbest":
            self._sys_name = "Deep Echo + K-Best"
            self._receiver = DeepEchoKBest5G(
                                    self._sys_parameters,
                                    training)
        elif self._sys_parameters.system == "ch_saver":
            self._sys_name = "Channel Saver"
            self._receiver = ChSaver5G(
                                    self._sys_parameters,
                                    save_path=getattr(self._sys_parameters,
                                                      'ch_save_path', 'ch_save.h5'))
        else:
            raise NotImplementedError("Unknown system selected!")

    def _apply_precoding_effect(self, h):
        """Apply the precoder to the ground-truth channel (single rx assumed).

        Input ``h``: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant,
        num_ofdm_symbols, num_effective_subcarriers], complex.

        Returns the effective channel as [batch_size, num_tx,
        num_effective_subcarriers, num_ofdm_symbols, 2*num_rx_ant], real.
        """
        h = tf.squeeze(h, axis=1)

        # [batch_size, num_tx, num_effective_subcarriers, num_ofdm_symbols, num_rx_ant, num_tx_ant]
        h = tf.transpose(h, perm=[0,2,5,4,1,3])

        # precoding_mat [num_tx, num_tx_ant, 1] -> [1, num_tx, 1, 1, num_tx_ant, 1]
        w = insert_dims(tf.expand_dims(self._precoding_mat, axis=0), 2, 2)

        # [batch_size, num_tx, num_effective_subcarriers, num_ofdm_symbols, num_rx_ant]
        h = tf.squeeze(tf.matmul(h, w), axis=-1)

        # Complex-to-real
        h = tf.concat([tf.math.real(h), tf.math.imag(h)], axis=-1)

        return h

    def _active_dmrs_mask(self, batch_size, num_tx, max_num_tx):
        """Sample mask of num_tx active DMRS ports (=users/streams).
        Draws different realization per batch sample.

        Input
        -----
        batch_size: int
            Batchsize of mask

        num_tx: int
            Number of active DMRS ports.

        max_num_tx: int
            Total number of DMRS ports.

        Output
        ------
        dmrs_mask: [batch_size, max_num_tx], tf.float32
            Mask of randomly activated DMRS ports.
        """

        max_num_tx = tf.cast(max_num_tx, tf.int32)
        num_tx = tf.cast(num_tx, tf.int32)
        r = tf.range(max_num_tx, dtype=tf.int32)
        r = tf.expand_dims(r, axis=0)
        r = tf.tile(r, (batch_size,1))
        x = tf.where(r<tf.cast(num_tx, tf.int32),
                     tf.ones_like(r),
                     tf.zeros_like(r))
        x = tf.expand_dims(x, axis=-1)
        x_p = tf.map_fn(tf.random.shuffle, x)
        x_p = tf.cast(x_p, tf.float32)
        return tf.squeeze(x_p, axis=-1)

    def _mask_active_dmrs(self, b, b_hat, num_tx, active_dmrs,
                          mcs_arr_eval_idx, tb_crc_status=None):
        """Remove inactive users/layers from b and b_hat.

        ``b``, ``b_hat``: [batch_size, max_num_tx, tb_size] ->
        [batch_size, num_tx, tb_size]; ``tb_crc_status`` (optional):
        [batch_size, max_num_tx] -> [batch_size, num_tx].
        """
        batch_size = tf.shape(b)[0]

        # only focus on active users
        # [batch_size, max_num_tx, tb_size]
        a_mask = expand_to_rank(active_dmrs, tf.rank(b_hat), axis=-1)
        a_mask = tf.broadcast_to(a_mask, tf.shape(b_hat))

        b_hat = tf.boolean_mask(b_hat, a_mask)
        b_hat = tf.reshape(b_hat,
                           (batch_size, num_tx, self._transmitters
                            [mcs_arr_eval_idx]._tb_size))

        b = tf.boolean_mask(b, a_mask)
        b = tf.reshape(b, (batch_size, num_tx, self._transmitters
                           [mcs_arr_eval_idx]._tb_size))

        if tb_crc_status is not None:
            a_mask = expand_to_rank(active_dmrs, tf.rank(tb_crc_status),
                                    axis=-1)
            a_mask = tf.broadcast_to(a_mask, tf.shape(tb_crc_status))
            tb_crc_status = tf.boolean_mask(tb_crc_status, a_mask)
            tb_crc_status = tf.reshape(tb_crc_status, (batch_size, num_tx))
            return b, b_hat, tb_crc_status

        return b, b_hat

    def _mask_active_dmrs_channels(self, h,
                                   batch_size, num_tx, active_dmrs):
        """Remove inactive users/layers from channel tensors.

        Input shapes:

            h: [batch_size, max_num_tx, fft, num_ofdm_symbols, 2*num_rx_ant]

            batch_size: []
            num_tx: []
            active_dmrs: [batch_size, max_num_tx]

        Output:
            h: [batch_size, num_tx, fft, num_ofdm_symbols, 2*num_rx_ant]
        """
        # [5] -> [batch_size, num_tx, fft, num_ofdm_symbols, 2*num_rx_ant]
        output_shape = tf.concat([tf.stack([batch_size, num_tx]),
                                  tf.shape(h)[2:]],
                                 axis=0)
        # [batch_size, max_num_tx, 1, 1, 1]
        a_mask = expand_to_rank(active_dmrs,
                                tf.rank(h), axis=-1)

        # [batch_size, max_num_tx, fft, num_ofdm_symbols, 2*num_rx_ant]
        a_mask = tf.broadcast_to(a_mask, tf.shape(h))

        h = tf.boolean_mask(h, a_mask)
        h = tf.reshape(h, output_shape)

        return h

    def _mask_active_dmrs_channels_(self, h,
                                   output_shape, active_dmrs):
        """Same as :meth:`_mask_active_dmrs_channels`, with a precomputed
        ``output_shape`` ([batch_size, num_tx, fft, num_ofdm_symbols,
        2*num_rx_ant]).
        """
        # [batch_size, max_num_tx, 1, 1, 1]
        a_mask = expand_to_rank(active_dmrs,
                                tf.rank(h), axis=-1)

        # [batch_size, max_num_tx, fft, num_ofdm_symbols, 2*num_rx_ant]
        a_mask = tf.broadcast_to(a_mask, tf.shape(h))

        h = tf.boolean_mask(h, a_mask)
        h = tf.reshape(h, output_shape)

        return h

    def _set_transmitter_random_pilots(self):
        """
        Sample a random slot number and assigns its pilots to the transmitter
        """
        pilot_set = self._sys_parameters.pilots
        num_pilots = tf.shape(pilot_set)[0]
        random_pilot_ind = tf.random.uniform((), 0, num_pilots, dtype=tf.int32)
        pilots = tf.gather(pilot_set, random_pilot_ind, axis=0)
        for mcs_list_idx in range(len(self._sys_parameters.mcs_index)):
            self._transmitters[mcs_list_idx].pilot_pattern.pilots = pilots

    def call(self, batch_size, ebno_db, num_tx=None, return_channel=False,
             mcs_arr_eval_idx=None, mcs_ue_mask=None, active_dmrs=None):
        """defines end-to-end system model."""

        batch_size = tf.cast(batch_size, tf.int32)

        # randomly sample num_tx active dmrs ports
        if num_tx is None:
            num_tx = self._sys_parameters.max_num_tx
        num_tx = tf.cast(num_tx, tf.int32)

        # if nothing is specified, select one pre-specified MCS
        if mcs_arr_eval_idx is None:
            mcs_arr_eval_idx = self._mcs_arr_eval_idx

        # generate active dmrs mask (if not specified)
        if active_dmrs is None:
            active_dmrs = self._active_dmrs_mask(
                                batch_size,
                                num_tx,
                                self._sys_parameters.max_num_tx)

        # mcs_ue_mask: [batch_size, max_num_tx, depth], depth being num supported MCSs
        if mcs_ue_mask is None:
            # No mcs-to-ue-mask specified --> evaluate pre-specified MCS only
            assert isinstance(mcs_arr_eval_idx, int), "Pre-defined MCS UE mask only works if mcs_arr_eval_idx is an integer"
            mcs_ue_mask = tf.one_hot(mcs_arr_eval_idx,
                                     depth=len(self._sys_parameters.mcs_index))
            mcs_ue_mask = expand_to_rank(mcs_ue_mask, 3, axis=0)
            mcs_ue_mask = tf.tile(mcs_ue_mask,
                                  multiples=[batch_size,
                                             self._sys_parameters.max_num_tx,
                                             1])
            mcs_arr_eval = [mcs_arr_eval_idx]
        else:
            # mcs_ue_mask is not none --> we now need to process all MCSs
            if isinstance(mcs_arr_eval_idx, (list, tuple)):
                # some different order specified. This is useful to evaluate
                # mixed MCS scenarios.
                # E2E model outputs data for mcs_arr_eval_idx[0], i.e., only
                # compute BER/BLER for UEs scheduled with mcs_arr_eval_idx[0].
                assert len(mcs_arr_eval_idx) == len(self._sys_parameters.mcs_index), "mcs_arr_eval_idx list not compatible with length of mcs_index array"
                mcs_arr_eval = mcs_arr_eval_idx
            else:
                # process in order of mcs_index array
                mcs_arr_eval = list(range(len(self._sys_parameters.mcs_index)))

        ###################################
        # Transmitters
        # one transmitter for each supported MCS
        ###################################

        b = []
        for idx in range(len(mcs_arr_eval)):
            b.append(
                self._source([batch_size,
                            self._sys_parameters.max_num_tx,
                            self._transmitters[mcs_arr_eval[idx]]._tb_size]))

        # Sample a random slot number and assigns its pilots to the transmitter
        if self._training:
            self._set_transmitter_random_pilots()

        # Combine transmit signals from all MCSs; UEs scheduled with a
        # different MCS are zeroed by the per-MCS mask.
        # _mcs_ue_mask: [batch_size, max_num_tx, 1, 1, 1]
        # x: [batch_size, max_num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        _mcs_ue_mask = tf.cast(expand_to_rank(
                                    tf.gather(mcs_ue_mask,
                                              indices=mcs_arr_eval[0],
                                              axis=2), 5, axis=-1),
                                              dtype=tf.complex64)
        x = _mcs_ue_mask * self._transmitters[mcs_arr_eval[0]](b[0])

        for idx in range(1, len(mcs_arr_eval)):
            _mcs_ue_mask = tf.cast(expand_to_rank(
                                        tf.gather(mcs_ue_mask,
                                                  indices=mcs_arr_eval[idx],
                                                  axis=2), 5, axis=-1),
                                                  dtype=tf.complex64)
            x = x + _mcs_ue_mask * self._transmitters[mcs_arr_eval[idx]](b[idx])

        # mask non-active DMRS ports by multiplying with 0
        a_tx = expand_to_rank(active_dmrs, tf.rank(x), axis=-1)
        x = tf.multiply(x, tf.cast(a_tx, tf.complex64))

        ###################################
        # Channel
        ###################################

        # Apply TX hardware impairments
        # CFO is applied per UE (i.e., must be done at TX side)
        if self._sys_parameters.frequency_offset is not None:
            x = self._sys_parameters.frequency_offset(x)

        # Rate adjusted SNR; for e2e learning non-rate adjusted is sometimes
        # preferred as pilotless communications changes the rate.
        if self._sys_parameters.ebno:

            # If pilot masking is used (for e2e), we account for the resulting
            # rate shift the assumption is that the empty REs are not
            # considered during transmission
            if self._sys_parameters.mask_pilots:
                tx = self._sys_parameters.transmitters[0]
                num_pilots = tf.cast(tx._resource_grid.num_pilot_symbols,
                                     tf.float32)
                num_res = tf.cast(tx._resource_grid.num_resource_elements,
                                  tf.float32)
                ebno_db -= 10.*log10(1. - num_pilots/num_res)
                # Remark: this also counts empty REs from other CDM groups
                # (e.g., used for other DMRS ports).
                # In the current e2e config; this case does not occur.

            # Translate Eb/No [dB] to N0 for first evaluated MCS
            no = ebnodb2no(
                    ebno_db,
                    self._transmitters[mcs_arr_eval[0]]._num_bits_per_symbol,
                    self._transmitters[mcs_arr_eval[0]]._target_coderate,
                    self._transmitters[mcs_arr_eval[0]]._resource_grid)

        else:
            # ebno_db is actually SNR when self._sys_parameters.ebno==False
            # [batch_size]
            no = 10**(-ebno_db/10)

        # Update topology only required for 3GPP UMi/UMa models
        if self._sys_parameters.channel_type in ("UMi", "UMa"):
            if self._sys_parameters.channel_type == "UMi":
                ch_type = 'umi'
            else:
                ch_type = 'uma'
            # Topology update only required for 3GPP pilot patterns
            topology = gen_single_sector_topology(
                        batch_size,
                        self._sys_parameters.max_num_tx,
                        ch_type,
                        min_ut_velocity=self._sys_parameters.min_ut_velocity,
                        max_ut_velocity=self._sys_parameters.max_ut_velocity,
                        indoor_probability=0.) # disable indoor users
            self._sys_parameters.channel_model.set_topology(*topology)

        # Apply channel
        if self._sys_parameters.channel_type == "AWGN":
            y = self._channel([x, no])
            h = tf.ones_like(y) # simple AWGN channel
        else:
            y, h = self._channel([x, no])

        rg = self._transmitters[0]._resource_grid
        fft = getattr(rg, "num_effective_subcarriers", rg.fft_size)
        num_ofdm_symbols = rg.num_ofdm_symbols
        num_rx_ant = self._sys_parameters.num_rx_antennas

        # h: [batch_size, num_rx, num_rx_ant, max_num_tx, num_tx_ant, num_ofdm_symbols, fft]
        # h_perf (precoded, single rx): [batch_size, max_num_tx, fft, num_ofdm_symbols, 2*num_rx_ant]
        h_perf = self._apply_precoding_effect(h)

        shape = tf.concat([tf.stack([batch_size, num_tx]),
                           tf.constant([fft, num_ofdm_symbols, 2*num_rx_ant],
                                       dtype=tf.int32)],
                          axis=0)

        ###################################
        # Receiver
        ###################################

        if self._sys_parameters.system in ('baseline_lmmse_kbest',
                                           'baseline_lmmse_lmmse',
                                           'baseline_lsnn_lmmse',
                                           'baseline_lslin_lmmse',
                                           'baseline_lslin_kbest'):

            # y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,
            #     num_effective_subcarriers]
            if self._return_tb_status:
                # b_hat: [batch_size, max_num_tx, tb_size]
                # tb_crc_status: [batch_size, max_num_tx]
                # h_hat: [batch_size, num_rx, num_rx_ant, num_tx,
                #         num_streams, num_ofdm_symbols, fft_size]
                b_hat, tb_crc_status, h_hat = self._receiver([y, no])
                b, b_hat, tb_crc_status = self._mask_active_dmrs(
                    b[0],
                    b_hat,
                    num_tx,
                    active_dmrs,
                    mcs_arr_eval[0],
                    tb_crc_status)
            else:
                # b_hat: [batch_size, max_num_tx, tb_size]
                # h_hat: [batch_size, num_rx, num_rx_ant, num_tx,
                #         num_streams, num_ofdm_symbols, fft_size]
                b_hat, h_hat = self._receiver([y, no])
                tb_crc_status = None
                # return b[0] and b_hat only for active DMRS ports
                # b only holds bits corresponding to MCS indices specified
                # in mcs_arr_eval --> evaluation for one MCS only --> b[0]
                b, b_hat = self._mask_active_dmrs(b[0], b_hat, num_tx,
                                                  active_dmrs,
                                                  mcs_arr_eval[0])

            if h_hat is not None: # single rx and single stream assumed
                # to [batch_size, max_num_tx, fft, num_ofdm_symbols, num_rx_ant]
                h_hat = tf.squeeze(h_hat, axis=(1,4,)) #[B,RA,T,14, F]
                h_hat = tf.transpose(h_hat, perm=[0,2,4,3,1]) #[B,T,F,14,RA]
                # Complex-to-real
                # [batch size, max_num_tx, fft, num_ofdm_symbols, 2*num_rx_ant]
                h_hat = tf.concat([tf.math.real(h_hat), tf.math.imag(h_hat)], axis=-1)

                # return channel only for active DMRS ports
                # [batch_size, num_tx, fft, num_ofdm_symbols, 2*num_rx_ant]
                h_hat = self._mask_active_dmrs_channels(h_hat, batch_size, num_tx, active_dmrs)

            if self._return_tb_status:
                if return_channel:
                    return b, b_hat, tb_crc_status, h_perf, h_hat
                else:
                    return b, b_hat, tb_crc_status
            else:
                if return_channel:
                    return b, b_hat, h_perf, h_hat
                else:
                    return b, b_hat

        elif self._sys_parameters.system in ('baseline_perf_csi_kbest',
                                             'baseline_perf_csi_lmmse'):
            # perfect CSI receiver needs ground truth channel
            if self._return_tb_status:
                # b_hat: [batch_size, max_num_tx, tb_size]
                # tb_crc_status: [batch_size, max_num_tx]
                # h_hat: [batch_size, num_rx, num_rx_ant, num_tx,
                #         num_streams, num_ofdm_symbols, fft_size]
                b_hat, tb_crc_status, h_hat = self._receiver([y, h, no])
                b, b_hat, tb_crc_status = self._mask_active_dmrs(
                    b[0],
                    b_hat,
                    num_tx,
                    active_dmrs,
                    mcs_arr_eval[0],
                    tb_crc_status)
            else:
                # b_hat: [batch_size, max_num_tx, tb_size]
                # h_hat: [batch_size, num_rx, num_rx_ant, num_tx,
                #         num_streams, num_ofdm_symbols, fft_size]
                b_hat, h_hat = self._receiver([y, h, no])
                tb_crc_status = None
                # return b[0] and b_hat only for active DMRS ports
                # b only holds bits corresponding to MCS indices specified
                # in mcs_arr_eval --> evaluation for one MCS only --> b[0]
                b, b_hat = self._mask_active_dmrs(b[0], b_hat, num_tx,
                                                  active_dmrs,
                                                  mcs_arr_eval[0])

            if h_hat is not None: # single rx and single stream assumed
                # to [batch_size, max_num_tx, fft, num_ofdm_symbols, num_rx_ant]
                h_hat = tf.squeeze(h_hat, axis=(1,4,)) #[B,RA,T,14, F]
                h_hat = tf.transpose(h_hat, perm=[0,2,4,3,1]) #[B,T,F,14,RA]
                # Complex-to-real
                # [batch size, max_num_tx, fft, num_ofdm_symbols, 2*num_rx_ant]
                h_hat = tf.concat([tf.math.real(h_hat), tf.math.imag(h_hat)], axis=-1)

                # return channel only for active DMRS ports
                # [batch_size, num_tx, fft, num_ofdm_symbols, 2*num_rx_ant]
                h_hat = self._mask_active_dmrs_channels(h_hat, batch_size, num_tx, active_dmrs)

            if self._return_tb_status:
                if return_channel:
                    return b, b_hat, tb_crc_status, h_perf, h_hat
                else:
                    return b, b_hat, tb_crc_status
            else:
                if return_channel:
                    return b, b_hat, h_perf, h_hat
                else:
                    return b, b_hat

        elif self._sys_parameters.system in ("nrx", "nrx_kbest"):
            # NRX branch inputs:
            # y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,
            #     num_subcarriers]
            # active_dmrs: [batch_size, max_num_tx]
            # b[0]: [batch_size, max_num_tx, tb_size]
            # h: [batch_size, num_rx, num_rx_ant, max_num_tx, num_tx_ant,
            #     num_ofdm_symbols, num_effective_subcarriers]
            # mcs_ue_mask: [batch_size, max_num_tx, len(mcs_index)]
            # no: [] or [batch_size]
            # in training mode, only the losses are required
            if self._training:
                losses = self._receiver([y, active_dmrs, b, h, mcs_ue_mask],
                                        mcs_arr_eval)  # tuple(loss_data: [], loss_chest: [])
                return losses
            else:
                # in inference mode, the neural receiver returns:
                # - reconstructed payload bits b_hat
                # - refined channel estimate h_hat_refined
                # - initial channel estimate h_hat
                # - [optional] transport block CRC status
                if self._sys_parameters.system == "nrx_kbest":
                    receiver_inputs = (y, active_dmrs, no)
                else:
                    receiver_inputs = (y, active_dmrs)
                # b_hat: [batch_size, max_num_tx, tb_size]
                # h_hat_refined, h_hat: [batch_size, max_num_tx,
                #     num_effective_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
                # tb_crc_status: [batch_size, max_num_tx]
                b_hat, h_hat_refined, h_hat, tb_crc_status = \
                                self._receiver(receiver_inputs,
                                                mcs_arr_eval,
                                                mcs_ue_mask_eval=mcs_ue_mask)

                #################################
                # Only focus on active DMRS ports
                #################################
                # Data
                # b only holds bits corresponding to MCS indices specified
                # in mcs_arr_eval --> evaluation for one MCS only --> b[0]
                b, b_hat, tb_crc_status = self._mask_active_dmrs(b[0], b_hat,
                                                                num_tx,
                                                                active_dmrs,
                                                                mcs_arr_eval[0],
                                                                tb_crc_status)
                # Channel estimates
                # [5] -> [batch_size, num_tx, num_effective_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
                h_hat_output_shape = tf.concat([tf.stack([batch_size, num_tx]),
                                                tf.shape(h_hat_refined)[2:]],
                                               axis=0)
                # [batch_size, max_num_tx, num_effective_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
                a_mask = expand_to_rank(active_dmrs,
                                        tf.rank(h_hat_refined), axis=-1)
                a_mask = tf.broadcast_to(a_mask, tf.shape(h_hat_refined))
                if h_hat is not None:
                    h_hat = tf.boolean_mask(h_hat, a_mask)
                    h_hat = tf.reshape(h_hat, h_hat_output_shape)
                h_hat_refined = tf.boolean_mask(h_hat_refined, a_mask)
                h_hat_refined = tf.reshape(h_hat_refined, h_hat_output_shape)
                # Channel ground truth
                h = self._receiver.preprocess_channel_ground_truth(h)
                h = tf.boolean_mask(h, a_mask)
                h = tf.reshape(h, h_hat_output_shape)

                # if activated, return channel estimates (and ground truth CFRs)
                if self._return_tb_status:
                    if return_channel:
                        return b, b_hat, tb_crc_status, h, h_hat_refined, h_hat
                    else:
                        return b, b_hat, tb_crc_status
                else:
                    if return_channel:
                        return b, b_hat, h, h_hat_refined
                    else:
                        return b, b_hat

        elif self._sys_parameters.system == "mdx":

            # in training mode, only the losses are required
            if self._training:
                losses = self._receiver([y, active_dmrs, b, h, mcs_ue_mask, x, no, batch_size],
                                        mcs_arr_eval)

                return losses
            else:
                # in inference mode, the neural receiver returns:
                # - reconstructed payload bits b_hat
                # - refined channel estimate h_hat_refined
                # - initial channel estimate h_hat
                # - [optional] transport block CRC status
                b_hat, h_hat_refined, h_hat, tb_crc_status = \
                                self._receiver((y, active_dmrs, no, mcs_ue_mask,x,h, batch_size),
                                                mcs_arr_eval=mcs_arr_eval,
                                                mcs_ue_mask_eval=mcs_ue_mask)

                #################################
                # Only focus on active DMRS ports
                #################################
                # Data
                # b only holds bits corresponding to MCS indices specified
                # in mcs_arr_eval --> evaluation for one MCS only --> b[0]
                b, b_hat, tb_crc_status = self._mask_active_dmrs(b[0], b_hat,
                                                                num_tx,
                                                                active_dmrs,
                                                                mcs_arr_eval[0],
                                                                tb_crc_status)

                # Channel estimates
                h_hat_refined = self._mask_active_dmrs_channels(
                    h_hat_refined, batch_size, num_tx, active_dmrs)
                if h_hat is not None:
                    h_hat = self._mask_active_dmrs_channels(
                        h_hat, batch_size, num_tx, active_dmrs)
                # Channel ground truth
                h = self._receiver.preprocess_channel_ground_truth(h)
                h = self._mask_active_dmrs_channels(
                    h, batch_size, num_tx, active_dmrs)

                # if activated, return channel estimates (and ground truth CFRs)
                if self._return_tb_status:
                    if return_channel:
                        return b, b_hat, tb_crc_status, h, h_hat_refined
                    else:
                        return b, b_hat, tb_crc_status
                else:
                    if return_channel:
                        return b, b_hat, h, h_hat_refined, h_hat
                    else:
                        return b, b_hat

        elif self._sys_parameters.system in ("deep_echo", "deep_echo_kbest"):

            # in training mode, only the losses are required
            if self._training:

                losses = self._receiver([y, active_dmrs, b, h, mcs_ue_mask, no], mcs_arr_eval)

                return losses
            else:
                # b_hat: [batch_size, max_num_tx, tb_size]
                # h_hat_refined, h_hat: [batch_size, max_num_tx, fft, num_ofdm_symbols, 2*num_rx_ant]
                # tb_crc_status: [batch_size, max_num_tx]
                b_hat, h_hat_refined, h_hat, tb_crc_status = \
                                self._receiver((y, active_dmrs, no, mcs_ue_mask, h),
                                                mcs_arr_eval=mcs_arr_eval,
                                                mcs_ue_mask_eval=mcs_ue_mask)  

                #################################
                # Only focus on active DMRS ports
                #################################
                # Data
                # b only holds bits corresponding to MCS indices specified
                # in mcs_arr_eval --> evaluation for one MCS only --> b[0]
                b, b_hat, tb_crc_status = self._mask_active_dmrs(b[0], b_hat,
                                                                num_tx,
                                                                active_dmrs,
                                                                mcs_arr_eval[0],
                                                                tb_crc_status)

                # [batch_size, num_tx, fft, num_ofdm_symbols, 2*num_rx_ant]
                h_hat_refined = self._mask_active_dmrs_channels(h_hat_refined, batch_size, num_tx, active_dmrs)

                # Channel estimates
                # [5] -> [batch_size, num_tx, num_effective_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
                h_hat_output_shape = tf.concat([tf.stack([batch_size, num_tx]),
                                                tf.shape(h_hat_refined)[2:]],
                                               axis=0)
                a_mask = expand_to_rank(active_dmrs,
                                        tf.rank(h_hat_refined), axis=-1)
                a_mask = tf.broadcast_to(a_mask, tf.shape(h_hat_refined))
                if h_hat is not None:
                    h_hat = tf.boolean_mask(h_hat, a_mask)
                    h_hat = tf.reshape(h_hat, h_hat_output_shape)
                h_hat_refined = tf.boolean_mask(h_hat_refined, a_mask)
                h_hat_refined = tf.reshape(h_hat_refined, h_hat_output_shape)
                # Channel ground truth
                h = self._receiver.preprocess_channel_ground_truth(h)
                h = tf.boolean_mask(h, a_mask)
                h = tf.reshape(h, h_hat_output_shape)

                # if activated, return channel estimates (and ground truth CFRs)
                if self._return_tb_status:
                    if return_channel:
                        return b, b_hat, tb_crc_status, h_perf, h_hat_refined
                    else:
                        return b, b_hat, tb_crc_status
                else:
                    if return_channel:
                        return b, b_hat, h_perf, h_hat_refined
                    else:
                        return b, b_hat

        elif self._sys_parameters.system == "ch_saver":
            # Save (h_freq, h_ls, no) to HDF5; no decoding performed.
            h_freq, h_ls, no_vec = self._receiver(
                (y, active_dmrs, no, mcs_ue_mask, h), mcs_arr_eval)
            return h_freq, h_ls, no_vec

        else:
            raise ValueError("Unknown system selected!")


class E2E_Model_Mixed_MCS(E2E_Model):
    r"""E2E_Model_Mixed_MCS(sys_parameters, training=False, return_tb_status=False, **kwargs)
    Wrapper for end-to-end model for system evaluation in mixed MCS scenarios.
    This class allows to return b and b_hat and tb_crc_status only of one user.

    The model can be used with the neural receiver as well as with the baseline
    receivers.

    For mixed MCS scenarios, you must provide mcs_arr_eval_idx as a list and
    also specify an mcs_ue_mask. The receiver will only process and return bits
    and bit estimates for mcs_arr_eval_idx[0].
    Ensure that you select the right user index in ue_return that is scheduled
    with mcs_arr_eval_idx[0].

    Parameters
    ----------
    sys_parameters : Parameters
        The system parameters.

    training : bool
        If True, the model returns the loss instead of b and b_hat.

    return_tb_status : bool
        If True, the model returns if the transport block CRC holds.

    mcs_arr_eval_idx : int, list
        Selects the element (index) of the mcs_index array to evaluate. Can be
        overwritten as call argument.
        Necessary to provide as a list for mixed MCS simulations. When
        mcs_arr_eval_idx is a list,
        it must have the same length as sys_parameters.mcs_index. Even when
        provided as a list, the receiver
        will only process mcs_arr_eval_idx[0]. Defaults to 0, which selects the
        very first element specified in the mcs_index array.

    ue_return : int
        UE index to return the ground-truth bits and bit estimates. Defaults to
        0.

    mcs_ue_mask : None, or [batch_size, max_num_tx, len(mcs_index)], tf.int32
        One-hot mask that specifies the MCS index of each UE for each batch
        sample.
        Must be specified when mcs_arr_eval_idx is provided as a list.
        If not specified (defaults to None), the mcs_ue_mask will be inferred
        from mcs_arr_eval_index.

    Input
    -----
    batch_size : int
        Batch size of random transmit signals to be generated

    ebno_db: float
        SNR in dB. Interpreted as rate-adjusted SNR if ``sys_parameters.ebno``
        is `True`.

    num_tx : int | None
        Number of active DMRS ports. Remark: the model always simulates all
        max_num_tx UEs. However, they are masked at the tx output, i.e., no
        energy is transmitted. This allows training with varying number of
        users without re-building the TensorFlow graph.

    output_nrx_h_hat : bool
        If True, the NRX internal channel estimations are returned.
        This is required for training with double read-outs.

    Output
    ------
    Depending on if return_tb_status and output_nrx_h_hat is True

        Tuple : b, b_hat, [tb_crc_status], [h, h_hat_refined, h_hat]

            b : [batch_size, num_tx, tb_size], tf.float
                    Transmitted information bits for selected MCS.

            b_hat : [batch_size, num_tx, tb_size], tf.float
                Decoded information bits for selected MCS.

            tb_crc_status: [batch_size, num_tx], tf.float
                Status of the TB CRC for each decoded TB.

            h: [batch_size, num_tx, num_effective_subcarriers,
                    num_ofdm_symbols, 2*num_rx_ant]
                Ground truth channel CFRs

            h_hat_refined, [batch_size, num_tx, num_effective_subcarriers,
                    num_ofdm_symbols, 2*num_rx_ant]
                Refined channel estimate from the NRX.

            h_hat, [batch_size, num_tx, num_effective_subcarriers,
                   num_ofdm_symbols, 2*num_rx_ant]
                Initial channel estimate used for the NRX.
    """
    def __init__(self, sys_parameters, training=False, return_tb_status=False,
                 mcs_arr_eval_idx=0, ue_return=0, mcs_ue_mask=None):
        if isinstance(mcs_arr_eval_idx, (list, tuple)):
            assert len(mcs_arr_eval_idx)==len(sys_parameters.mcs_index), "If mcs_arr_eval_idx is a list, it must have the same length as sys_parameters.mcs_index"
            assert mcs_ue_mask is not None, "Must specify mcs_ue_mask if mcs_arr_eval_idx is given as list"
            super().__init__(sys_parameters=sys_parameters,
                             training=training,
                             return_tb_status=return_tb_status,
                             mcs_arr_eval_idx=mcs_arr_eval_idx[0])
            self._mcs_arr_eval = mcs_arr_eval_idx
        else:
            super().__init__(sys_parameters=sys_parameters,
                             training=training,
                             return_tb_status=return_tb_status,
                             mcs_arr_eval_idx=mcs_arr_eval_idx)
            self._mcs_arr_eval = mcs_arr_eval_idx
        self._ue_return = ue_return

        self._mcs_ue_mask = mcs_ue_mask

    def call(self, batch_size, ebno_db, num_tx=None, output_nrx_h_hat=False):
        if self._return_tb_status:
            if output_nrx_h_hat:
                b, b_hat, tb_crc_status, h, h_hat_refined, h_hat = \
                        super().call(batch_size, ebno_db, num_tx,
                                     output_nrx_h_hat,
                                     mcs_arr_eval_idx=self._mcs_arr_eval,
                                     mcs_ue_mask=self._mcs_ue_mask)
            else:
                b, b_hat, tb_crc_status = \
                            super().call(batch_size,
                                        ebno_db, num_tx,
                                        output_nrx_h_hat,
                                        mcs_arr_eval_idx=self._mcs_arr_eval,
                                        mcs_ue_mask=self._mcs_ue_mask)
        else:
            if output_nrx_h_hat:
                b, b_hat, h, h_hat_refined, h_hat = \
                            super().call(batch_size, ebno_db, num_tx,
                                         output_nrx_h_hat,
                                         mcs_arr_eval_idx=self._mcs_arr_eval,
                                         mcs_ue_mask=self._mcs_ue_mask)
            else:
                b, b_hat = super().call(batch_size, ebno_db, num_tx,
                                        output_nrx_h_hat,
                                        mcs_arr_eval_idx=self._mcs_arr_eval,
                                        mcs_ue_mask=self._mcs_ue_mask)

        b = tf.gather(b, indices=[self._ue_return], axis=1)
        b_hat = tf.gather(b_hat, indices=[self._ue_return], axis=1)

        if self._return_tb_status:
            tb_crc_status = tf.gather(tb_crc_status,
                                      indices=[self._ue_return], axis=1)
            if output_nrx_h_hat:
                return b, b_hat, tb_crc_status, h, h_hat_refined, h_hat
            else:
                return b, b_hat, tb_crc_status
        else:
            if output_nrx_h_hat:
                return b, b_hat, h, h_hat_refined, h_hat
            else:
                return b, b_hat
