# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026

"""Receiver stub that stores channel realizations and LS estimates to HDF5."""

import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from sionna.utils import expand_to_rank, flatten_last_dims
from sionna.nr import PUSCHLSChannelEstimator


class ChSaver5G(Model):
    """Receiver stub that saves channel data to HDF5 instead of decoding.

    Mirrors the DeepEcho5G interface so it plugs into E2E_Model unchanged.
    On each call it runs a LS channel estimate from the received signal and
    appends (h_freq, h_ls, no, mcs) to the output HDF5 file.

    Call finalize() after the generation loop to flush and close the file.
    """

    def __init__(self, sys_parameters, save_path, **kwargs):
        super().__init__(**kwargs)

        self._sys_parameters = sys_parameters
        self._save_path = save_path
        self._h5 = None
        self._n_saved = 0
        self._mcs_index_list = list(sys_parameters.mcs_index)

        # LS channel estimator, configured as in DeepEcho5G
        rg = sys_parameters.transmitters[0]._resource_grid
        pc = sys_parameters.pusch_configs[0][0]

        self._ls_est = PUSCHLSChannelEstimator(
            resource_grid=rg,
            dmrs_length=pc.dmrs.length,
            dmrs_additional_position=pc.dmrs.additional_position,
            num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
            interpolation_type="lin")

    def _open_h5(self, h_shape, hls_shape):
        os.makedirs(os.path.dirname(os.path.abspath(self._save_path)), exist_ok=True)
        self._h5 = h5py.File(self._save_path, 'w')
        self._h5.attrs['codex_layout'] = 'sampler_direct'
        self._h5.attrs['mcs_index_list'] = self._mcs_index_list
        self._h5.create_dataset(
            'h_freq',
            shape=(0,) + h_shape[1:],
            maxshape=(None,) + h_shape[1:],
            chunks=True, dtype='complex64')
        self._h5.create_dataset(
            'h_ls',
            shape=(0,) + hls_shape[1:],
            maxshape=(None,) + hls_shape[1:],
            chunks=True, dtype='complex64')
        self._h5.create_dataset(
            'no',
            shape=(0,),
            maxshape=(None,),
            chunks=True, dtype='float32')
        self._h5.create_dataset(
            'mcs',
            shape=(0,),
            maxshape=(None,),
            chunks=True, dtype='int32')

        print("\nDataset created:")
        print(f"  h_freq : {(None,) + h_shape[1:]}  complex64"
              f"  →  [N, num_rx, rx_ant, num_tx, tx_ant, ofdm_sym, fft]")
        print(f"  h_ls   : {(None,) + hls_shape[1:]}  complex64"
              f"  →  [N, num_tx, fft, ofdm_sym, rx_ant]")
        print(f"  no     : (N,)  float32  →  noise variance per example")
        print(f"  mcs    : (N,)  int32    →  MCS index value (from {self._mcs_index_list})")

    def _append(self, h_np, hls_np, no_np, mcs_np):
        first_batch = self._h5 is None
        if first_batch:
            self._open_h5(h_np.shape, hls_np.shape)

        n = h_np.shape[0]
        for key, arr in [('h_freq', h_np), ('h_ls', hls_np)]:
            ds = self._h5[key]
            ds.resize(ds.shape[0] + n, axis=0)
            ds[-n:] = arr
        for key, arr in [('no', no_np), ('mcs', mcs_np)]:
            ds = self._h5[key]
            ds.resize(ds.shape[0] + n, axis=0)
            ds[-n:] = arr
        self._h5.flush()
        self._n_saved += n

        if first_batch:
            print(f"\nFirst batch shapes (actual):")
            print(f"  h_freq : {h_np.shape}")
            print(f"  h_ls   : {hls_np.shape}")
            print(f"  no     : {no_np.shape}  (value: {no_np[0]:.4e})")
            print(f"  mcs    : {mcs_np.shape}  (value: {int(mcs_np[0])})")
            print()

    def finalize(self):
        """Close the HDF5 file. Call once after the generation loop."""
        if self._h5 is not None:
            print(f"\nFinalizing: {self._n_saved} examples in {self._save_path}")
            self._h5.close()
            self._h5 = None

    def _estimate_channel(self, y, num_tx, no):
        """LS channel estimate as in ``DeepEcho5G.estimate_channel``."""
        # h_hat: [batch, num_rx, num_rx_ant, num_tx, num_streams, ofdm_sym, fft]
        h_hat, err_var = self._ls_est([y, no])

        err_var_dt = tf.broadcast_to(err_var, tf.shape(h_hat))
        err_var_dt = tf.transpose(err_var_dt, [0, 1, 5, 6, 2, 3, 4])
        err_var_dt = flatten_last_dims(err_var_dt, 2)
        err_var = tf.reduce_sum(err_var_dt, -1)

        # [batch, num_tx, fft, ofdm_sym, num_rx_ant]
        h_hat = h_hat[:, 0, :, :num_tx, 0]
        h_hat = tf.transpose(h_hat, [0, 2, 4, 3, 1])
        err_var = tf.transpose(err_var, perm=[0, 1, 3, 2, 4])
        return h_hat, err_var

    def call(self, inputs, mcs_arr_eval=None, mcs_ue_mask_eval=None):
        """Estimate the channel, append the batch to the HDF5 file and return it.

        ``inputs = (y, active_tx, no, mcs_ue_mask, h)``. Stored per example:
          h_freq : [batch, 1, rx_ant, num_tx, tx_ant, ofdm_sym, fft]  complex64
          h_ls   : [batch, num_tx, fft, ofdm_sym, rx_ant]             complex64
          no     : [batch]                                             float32
          mcs    : [batch]  int32, MCS index value (e.g. 9, 14, 19)

        Returns ``(h, h_ls, no_vec)``.
        """
        if mcs_arr_eval is None:
            mcs_arr_eval = [0]

        y, active_tx, no, mcs_ue_mask, h = inputs
        num_tx = tf.shape(active_tx)[1]
        batch_size = tf.shape(y)[0]

        h_ls, _ = self._estimate_channel(y, num_tx, no)

        # broadcast scalar no to [batch] for consistent storage
        no_scalar = no if no.shape.rank == 0 else no[0]
        no_vec = tf.cast(tf.fill([batch_size], no_scalar), tf.float32)

        # actual MCS index value (e.g. 9, 14, 19) broadcast to [batch]
        mcs_val = self._mcs_index_list[int(mcs_arr_eval[0])]
        mcs_vec = np.full(int(batch_size.numpy()), mcs_val, dtype=np.int32)

        self._append(h.numpy(), h_ls.numpy(), no_vec.numpy(), mcs_vec)

        return h, h_ls, no_vec
