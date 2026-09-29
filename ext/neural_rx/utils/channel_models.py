# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Implements different channel models for performance evaluation
import core.runtime as _runtime
from core.runtime import DEBUGD

from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
import sionna
from sionna.channel import GenerateOFDMChannel, ApplyOFDMChannel, ChannelModel
from sionna.channel.tr38901 import CDL, TDL
from sionna.channel.utils import exp_corr_mat
import random

def gnb_correlation_matrix(num_ant, alpha):
    assert num_ant in [1,2,4,8]
    if num_ant==1:
        exponents = np.array([0])
    elif num_ant==2:
        exponents =  np.array([0, 1])
    elif num_ant==4:
        exponents = np.array([0, 1/9, 4/9, 1])
    elif num_ant==8:
        exponents = np.array([0, 1/49, 4/49, 9/49, 16/49, 25/49, 36/49, 1])
    row = alpha**exponents
    col = np.conj(row)
    r = tf.linalg.LinearOperatorToeplitz(col, row)
    return tf.cast(r.to_dense(), tf.complex64)

def ue_correlation_matrix(num_ant, beta):
    assert num_ant in [1,2,4]
    return gnb_correlation_matrix(num_ant, beta)


class NTDLChannel(tf.keras.layers.Layer):
    """
    Channel model that stacks N independent 3GPP TDL links, one per user.

    Sionna's TDL model is defined for a single transmitter, so an N user
    system is built by instantiating one TDL per user and concatenating the
    frequency responses along the num_tx axis.

    Parameters
    ---------
    carrier_frequency: float
        Carrier frequency of the simulation.

    resource_grid: ResourceGrid
        Resource grid used for the simulation.

    num_rx_ant: int
        Number of receiver antennas.

    num_tx_ant: int
        Number of transmit antennas for each user.

    max_num_tx: int
        Maximum number of users. ``tdl_models`` is tiled to at least this
        length.

    norm_channel: bool
        If True, the channel is normalized.

    correlation: "low" | "medium" | "high"
        Antenna correlation according to 38.901.

    tdl_models: list of str
        TDL profiles to draw from, each one of "A", "B", "C", "D", "E".

    delay_spread_min / delay_spread_max: float
        Delay spread range in nano seconds. Resampled for every call.

    doppler_shift_max: float
        Maximum Doppler shift in Hz. Defines the maximum UT speed.

    Input
    -----

    (x, no):
        Tuple:

    x :  [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size],
         tf.complex
        Channel inputs

    no : Scalar or Tensor, tf.float
        Scalar or tensor whose shape can be broadcast to the shape of the
        channel outputs

    Output
    -------
    y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        Channel outputs
    h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant,
              num_ofdm_symbols, fft_size], tf.complex
        Channel frequency responses.
    """
    def __init__(self,
                 carrier_frequency,
                 resource_grid,
                 num_rx_ant=16,
                 num_tx_ant=2,
                 max_num_tx=4,
                 norm_channel=False,
                 correlation="low",
                 tdl_models=["A"], # A, B, C, D, E
                 delay_spread_min=10,   # in nano seconds
                 delay_spread_max=300,  # in nano seconds
                 doppler_shift_max=325  # Hz
                 ):
        super().__init__()

        assert correlation in ["low", "medium", "high"]

        print(f"\n\tLoading NTDL {tdl_models} with {correlation} correlation.\n")

        if correlation=="low":
            alpha = beta = 0
        elif correlation=="medium":
            alpha = 0.9
            beta = 0.3
        else:
            alpha = 0.9
            beta = 0.9

        self._tx_corr_mat = exp_corr_mat(beta, num_tx_ant)
        self._rx_corr_mat = exp_corr_mat(alpha, num_rx_ant)

        self._max_num_tx = max_num_tx
        self._num_rx_ant = num_rx_ant
        self._num_tx_ant = num_tx_ant
        self._delay_spread_min = delay_spread_min
        self._delay_spread_max = delay_spread_max
        self._doppler_shift_max = doppler_shift_max
        self._carrier_frequency = carrier_frequency
        self._tdl_models = tdl_models
        self._resource_grid = resource_grid
        self._norm_channel = norm_channel
        self._apply_channel = ApplyOFDMChannel()
        self._speed = self._doppler_shift_max * sionna.SPEED_OF_LIGHT / self._carrier_frequency
        self._num_tdls = len(tdl_models)

        if len(tdl_models) < max_num_tx:
            # Repeat the list so that its length covers max_num_tx
            repeat_times = (max_num_tx // len(tdl_models)) + 1
            tdl_models = tdl_models * repeat_times

        self._gens = []
        self._tdls = []
        for model in tdl_models:

            delay_spread = delay_spread_max*1e-9 # seconds

            tdl = TDL(model,
                      delay_spread,
                      self._carrier_frequency,
                      max_speed=self._speed,
                      num_tx_ant=self._num_tx_ant,
                      num_rx_ant=self._num_rx_ant,
                      rx_corr_mat=self._rx_corr_mat,
                      tx_corr_mat=self._tx_corr_mat)

            gen_channel = GenerateOFDMChannel_(
                tdl,
                self._resource_grid,
                normalize_channel=self._norm_channel
            )
            self._gens.append(gen_channel)

    def call(self, inputs):

        x, no = inputs
        size = tf.shape(x)
        batch_size = size[0]
        num_tx = size[1]

        k = len(self._gens) 

        # Compute all links, each with its own random delay spread
        outputs = [
            gen(
                batch_size,
                delay_spread=tf.random.uniform(
                    shape=(),
                    minval=self._delay_spread_min,
                    maxval=self._delay_spread_max,
                    dtype=tf.float32
                ) * 1e-9  # nano Seconds
            )
            for gen in self._gens
        ]

        all_outputs = tf.stack(outputs, axis=0)

        # Randomly assign links to the active users
        indices = tf.range(k)
        shuffled_indices = tf.random.shuffle(indices)
        selected_indices = shuffled_indices[:num_tx]
        selected_outputs = tf.gather(all_outputs, selected_indices, axis=0)
        h_list = tf.unstack(selected_outputs, axis=0)

        # Concatenate along the num_tx axis
        h = tf.concat(h_list, axis=3)
        y = self._apply_channel([x, h, no])
        return y, h


class NCDLChannel(tf.keras.layers.Layer):
    """
    Channel model that stacks N independent 3GPP CDL links. Sionna's CDL is
    defined for a single transmitter and a single receiver, so an N user
    system is built by instantiating one CDL per user and concatenating the
    frequency responses along the num_tx axis (same approach as
    :class:`NTDLChannel`).

    Unlike the TDL-based models, the spatial correlation is not injected via
    Kronecker correlation matrices but follows from the antenna array
    geometry, hence ``bs_array``/``ut_array`` are required instead of
    ``num_rx_ant``/``num_tx_ant``.

    Parameters
    ---------
    carrier_frequency: float
        Carrier frequency of the simulation.

    resource_grid: ResourceGrid
        Resource grid used for the simulation.

    bs_array: PanelArray
        Base station antenna array. Defines num_rx_ant.

    ut_array: PanelArray
        User terminal antenna array. Defines num_tx_ant.

    max_num_tx: int
        Maximum number of users. ``cdl_models`` is tiled to at least this
        length.

    norm_channel: bool
        If True, the channel is normalized.

    cdl_models: list of str
        CDL profiles to draw from, each one of "A", "B", "C", "D", "E".

    min_speed / max_speed: float
        UT velocity range in m/s.

    delay_spread_min / delay_spread_max: float
        Delay spread range in nano seconds. Resampled for every call.

    spread_ut_orientation: bool
        CDL cluster angles are fixed per profile, so several users sharing a
        profile would also share their angular signature and be hard to
        separate. If True (default), each user gets a distinct UT azimuth
        spread over [0, 2*pi). Set to False for textbook CDL. For a single
        user this is a no-op (the sionna default orientation is used).

    Input
    -----

    (x, no):
        Tuple:

    x :  [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size],
         tf.complex
        Channel inputs

    no : Scalar or Tensor, tf.float
        Scalar or tensor whose shape can be broadcast to the shape of the
        channel outputs

    Output
    -------
    y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        Channel outputs
    h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant,
              num_ofdm_symbols, fft_size], tf.complex
        Channel frequency responses.
    """
    def __init__(self,
                 carrier_frequency,
                 resource_grid,
                 bs_array,
                 ut_array,
                 max_num_tx=4,
                 norm_channel=False,
                 cdl_models=["A"],  # A, B, C, D, E
                 min_speed=0.,
                 max_speed=34.,
                 delay_spread_min=10,   # in nano seconds
                 delay_spread_max=300,  # in nano seconds
                 spread_ut_orientation=True,
                 ):
        super().__init__()

        for model in cdl_models:
            assert model in ("A", "B", "C", "D", "E"), \
                f"Invalid CDL model '{model}', expected one of A, B, C, D, E."

        print(f"\n\tLoading NCDL {cdl_models}.\n")

        self._max_num_tx = max_num_tx
        self._delay_spread_min = delay_spread_min
        self._delay_spread_max = delay_spread_max
        self._carrier_frequency = carrier_frequency
        self._cdl_models = cdl_models
        self._resource_grid = resource_grid
        self._norm_channel = norm_channel
        self._apply_channel = ApplyOFDMChannel()

        if len(cdl_models) < max_num_tx:
            # Repeat the list so that its length covers max_num_tx
            repeat_times = (max_num_tx // len(cdl_models)) + 1
            cdl_models = cdl_models * repeat_times

        num_links = len(cdl_models)

        self._gens = []
        for idx, model in enumerate(cdl_models):

            delay_spread = delay_spread_max * 1e-9  # seconds

            # Sionna's default is [PI, 0, 0] (UT facing the BS); index 0 keeps
            # it so that a single-user setup reproduces textbook CDL exactly.
            if spread_ut_orientation and num_links > 1:
                azimuth = np.pi + 2. * np.pi * idx / num_links
                ut_orientation = tf.constant([azimuth, 0., 0.], tf.float32)
            else:
                ut_orientation = None

            cdl = CDL(model=model,
                      delay_spread=delay_spread,
                      carrier_frequency=self._carrier_frequency,
                      ut_array=ut_array,
                      bs_array=bs_array,
                      direction='uplink',
                      ut_orientation=ut_orientation,
                      min_speed=min_speed,
                      max_speed=max_speed)

            gen_channel = GenerateOFDMChannel_(
                cdl,
                self._resource_grid,
                normalize_channel=self._norm_channel
            )
            self._gens.append(gen_channel)

    def call(self, inputs):

        x, no = inputs
        size = tf.shape(x)
        batch_size = size[0]
        num_tx = size[1]

        k = len(self._gens)

        # Compute all links, each with its own random delay spread
        outputs = [
            gen(
                batch_size,
                delay_spread=tf.random.uniform(
                    shape=(),
                    minval=self._delay_spread_min,
                    maxval=self._delay_spread_max,
                    dtype=tf.float32
                ) * 1e-9  # nano seconds
            )
            for gen in self._gens
        ]

        all_outputs = tf.stack(outputs, axis=0)

        # Randomly assign links to the active users
        indices = tf.range(k)
        shuffled_indices = tf.random.shuffle(indices)
        selected_indices = shuffled_indices[:num_tx]
        selected_outputs = tf.gather(all_outputs, selected_indices, axis=0)

        h_list = tf.unstack(selected_outputs, axis=0)

        # Concatenate along the num_tx axis
        h = tf.concat(h_list, axis=3)
        y = self._apply_channel([x, h, no])
        return y, h


#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class for generating channel frequency responses"""


from sionna.channel.utils import subcarrier_frequencies, cir_to_ofdm_channel


class GenerateOFDMChannel_:
    # pylint: disable=line-too-long
    r"""GenerateOFDMChannel_(channel_model, resource_grid, normalize_channel=False, dtype=tf.complex64)

    Variant of Sionna's ``GenerateOFDMChannel`` that forwards an optional
    per-call ``delay_spread`` to the channel model.

    Generate channel frequency responses.
    The channel impulse response is constant over the duration of an OFDM symbol.

    Given a channel impulse response
    :math:`(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1`, generated by the ``channel_model``,
    the channel frequency response for the :math:`s^{th}` OFDM symbol and
    :math:`n^{th}` subcarrier is computed as follows:

    .. math::
        \widehat{h}_{s, n} = \sum_{m=0}^{M-1} a_{m}(s) e^{-j2\pi n \Delta_f \tau_{m}}

    where :math:`\Delta_f` is the subcarrier spacing, and :math:`s` is used as time
    step to indicate that the channel impulse response can change from one OFDM symbol to the
    next in the event of mobility, even if it is assumed static over the duration
    of an OFDM symbol.

    Parameters
    ----------
    channel_model : :class:`~sionna.channel.ChannelModel` object
        An instance of a :class:`~sionna.channel.ChannelModel` object, such as
        :class:`~sionna.channel.RayleighBlockFading` or
        :class:`~sionna.channel.tr38901.UMi`.

    resource_grid : :class:`~sionna.ofdm.ResourceGrid`
        Resource grid

    normalize_channel : bool
        If set to `True`, the channel is normalized over the resource grid
        to ensure unit average energy per resource element. Defaults to `False`.

    dtype : tf.DType
        Complex datatype to use for internal processing and output.
        Defaults to `tf.complex64`.

    Input
    -----

    batch_size : int
        Batch size. Defaults to `None` for channel models that do not require this parameter.

    delay_spread : float or None
        Delay spread [s] passed to the channel model. Defaults to `None`.

    Output
    -------
    h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex
        Channel frequency responses
    """

    def __init__(self, channel_model, resource_grid, normalize_channel=False,
                 dtype=tf.complex64):

        # Callable used to sample channel input responses
        self._cir_sampler = channel_model

        # We need those in call()
        self._num_ofdm_symbols = resource_grid.num_ofdm_symbols
        self._subcarrier_spacing = resource_grid.subcarrier_spacing
        self._num_subcarriers = resource_grid.fft_size
        self._normalize_channel = normalize_channel
        self._sampling_frequency = 1./resource_grid.ofdm_symbol_duration

        # Frequencies of the subcarriers
        self._frequencies = subcarrier_frequencies(self._num_subcarriers,
                                                   self._subcarrier_spacing,
                                                   dtype)

    def __call__(self, batch_size=None, delay_spread=None):

        # Sample channel impulse responses
        h, tau = self._cir_sampler( batch_size,
                                    self._num_ofdm_symbols,
                                    self._sampling_frequency,
                                    delay_spread)

        h_freq = cir_to_ofdm_channel(self._frequencies, h, tau,
                                     self._normalize_channel)

        return h_freq


class DoubleTDLChannel(tf.keras.layers.Layer):
    """
    Channel model that stacks a 3GPP TDL-B100-400 and TDL-C-300-100 channel
    model. This allows to benchmark a two user system in a 3GPP compliant
    scenario.

    Parameters
    ---------
    carrier_frequency: float
        Carrier frequency of the simulation.

    resource_grid: ResourceGrid
        Resource grid used for the simulation.

    num_rx_ant: int
        Number of receiver antennas.

    num_tx_ant: int
        Number of transmit antennas for each user.

    norm_channel: bool
        If True, the channel is normalized.

    correlation: "low" | "medium" | "high"
        Antenna correlation according to 38.901.

    Input
    -----

    (x, no) or x:
        Tuple or Tensor:

    x :  [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size],
         tf.complex
        Channel inputs

    no : Scalar or Tensor, tf.float
        Scalar or tensor whose shape can be broadcast to the shape of the
        channel outputs

    Output
    -------
    y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        Channel outputs
    h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant,
              num_ofdm_symbols, fft_size], tf.complex
        Channel frequency responses.
    """
    def __init__(self,
                 carrier_frequency,
                 resource_grid,
                 num_rx_ant=4,
                 num_tx_ant=2,
                 norm_channel=False,
                 correlation="low"):
        super().__init__()

        assert correlation in ["low", "medium", "high"]

        print(f"Loading DoubleTDL with {correlation} correlation.")

        if correlation=="low":
            alpha = beta = 0
        elif correlation=="medium":
            alpha = 0.9
            beta = 0.3
        else:
            alpha = 0.9
            beta = 0.9

        tx_corr_mat = exp_corr_mat(beta, num_tx_ant)
        rx_corr_mat = exp_corr_mat(alpha, num_rx_ant)

        # TDL B100 model
        delay_spread_1 = 100e-9
        doppler_spread_1 = 400
        speed_1 = doppler_spread_1 * sionna.SPEED_OF_LIGHT / carrier_frequency
        tdl1 = TDL("B100",
           delay_spread_1,
           carrier_frequency,
           max_speed=speed_1,
           num_tx_ant=num_tx_ant,
           num_rx_ant=num_rx_ant,
           rx_corr_mat=rx_corr_mat,
           tx_corr_mat=tx_corr_mat)

        # TDL C300 model
        delay_spread_2 = 300e-9
        doppler_spread_2 = 100
        speed_2 = doppler_spread_2 * sionna.SPEED_OF_LIGHT / carrier_frequency
        tdl2 = TDL("C300",
           delay_spread_2,
           carrier_frequency,
           max_speed=speed_2,
           num_tx_ant=num_tx_ant,
           num_rx_ant=num_rx_ant,
           rx_corr_mat=rx_corr_mat,
           tx_corr_mat=tx_corr_mat)

        self._gen_channel_1 = GenerateOFDMChannel(
                                        tdl1,
                                        resource_grid,
                                        normalize_channel=norm_channel)
        self._gen_channel_2 = GenerateOFDMChannel(
                                        tdl2,
                                        resource_grid,
                                        normalize_channel=norm_channel)

        self._apply_channel = ApplyOFDMChannel()

    def call(self, inputs):

        x, no = inputs
        batch_size = tf.shape(x)[0]
        h1 = self._gen_channel_1(batch_size)
        h2 = self._gen_channel_2(batch_size)

        # stack the two models
        h = tf.concat([h1, h2], axis=3)

        y = self._apply_channel([x, h, no])
        return y, h

class DatasetChannel(ChannelModel):
    """Channel model from a TFRecords Dataset File
       The entire dataset is read in memory.

       This version supports XLA acceleration.


    Parameter
    ---------
    tfrecord_filename: str
        Filename of the pre-computed dataset.

    max_num_examples: int
        Max number of samples loaded from dataset. If equals to "-1"
        the entire dataset will be loaded. Defines memory occupation.

    Input
    -----
    batchsize: int
        How many samples shall be returned.

    Output
    ------
    a: [batch_size,...]
        batch_size samples from ``a``. Exact shape depends on dataset.

    tau: [batch_size,...]
        batch_size samples from ``tau``. Exact shape depends on dataset.

    """
    def __init__(self, tfrecord_filename, max_num_examples=-1, training=True,
                 num_tx=1, random_subsampling=True):

        self._training = training
        self._num_tx = num_tx
        self._random_subsampling = random_subsampling

        # Read raw dataset
        dataset = tf.data.TFRecordDataset([tfrecord_filename]) \
                  .map(self._parse_function,
                       num_parallel_calls=tf.data.AUTOTUNE) \
                  .take(max_num_examples) \
                  .batch(1024)

        # Load entire dataset into memory as large tensor
        a = None
        tau = None
        for example in dataset:
            # aggregate all channels in batch direction to multiple users.
            # i.e., move batch direction to num_tx direction.
            #
            # Evaluation data set already has two active users for each batch
            # sample.
            # Thus, every other sample after the aggregation belong to the same
            # user.
            a_ex, tau_ex = example
            a_ex = tf.split(a_ex, a_ex.shape[0], axis=0)
            a_ex = tf.concat(a_ex, axis=3)
            tau_ex = tf.split(tau_ex, tau_ex.shape[0], axis=0)
            tau_ex = tf.concat(tau_ex, axis=2)
            if a is None:
                a = a_ex
                tau = tau_ex
            else:
                a = tf.concat([a, a_ex], axis=3)
                tau = tf.concat([tau, tau_ex], axis=2)

        if training:
            # User positions are randomly sampled. In order to avoid sampling
            # the same positions multiple times within one batch sample, we
            # split the dataset into equal parts for each user to sample from
            # during simulations.
            num_examples = int(a.shape[3]/self._num_tx)
            self._num_examples = num_examples
            self._a = []
            self._tau = []
            for i in range(self._num_tx):
                self._a.append(a[:,:,:,i*num_examples:(i+1)*num_examples])
                self._tau.append(tau[:,:,i*num_examples:(i+1)*num_examples])
        else:
            self._num_examples = a.shape[3]
            self._a = [a,]
            self._tau = [tau,]

    @staticmethod
    def _parse_function(proto):
        description = {
                'a': tf.io.FixedLenFeature([], tf.string),
                'tau': tf.io.FixedLenFeature([], tf.string),
            }
        features = tf.io.parse_single_example(proto, description)
        a = tf.io.parse_tensor(features['a'], out_type=tf.complex64)
        tau = tf.io.parse_tensor(features['tau'], out_type=tf.float32)
        return a, tau


    def __call__(self, batch_size=None,
                       num_time_steps=None,
                       sampling_frequency=None):
        # default values are used for compatibility with other TF functions.

        # Remark: this is random subsampling
        # random sampling is also done in eval mode; keep in mind that even
        # though UE is on trajectory, we need many slot realizations for good
        # BLER curves (in any case we sample new AWGN noise)

        a = None
        tau = None

        if self._training:
            if not self._random_subsampling:
                ind = tf.random.uniform([batch_size],
                                     maxval=self._num_examples, dtype=tf.int32)
            # randomly subsample from different subsets
            for ue_idx in range(self._num_tx):
                if self._random_subsampling:
                    ind = tf.random.uniform(
                                        [batch_size],
                                        maxval=self._num_examples,
                                        dtype=tf.int32)

                # Gather reshape and combine
                a_ = tf.gather(self._a[ue_idx], ind, axis=3)
                a_ = tf.transpose(a_, perm=[3, 1, 2, 0, 4, 5, 6])
                tau_ = tf.gather(self._tau[ue_idx], ind, axis=2)
                tau_ = tf.transpose(tau_, perm=[2, 1, 0, 3])
                if a is not None:
                    a = tf.concat([a, a_], axis=3)
                    tau = tf.concat([tau, tau_], axis=2)
                else:
                    a = a_
                    tau = tau_
        else:
            # samples in self._a alternating between both trajectories
            if not self._random_subsampling:
                # no random sub-sampling: take subsequent two samples
                ind = tf.random.uniform([batch_size],
                                     maxval=self._num_examples//self._num_tx,
                                     dtype=tf.int32)
                ind = tf.repeat(tf.expand_dims(ind, axis=-1),
                                repeats=self._num_tx, axis=-1)
            else:
                ind = tf.random.uniform([batch_size, self._num_tx],
                                     maxval=self._num_examples//self._num_tx,
                                     dtype=tf.int32)
            # sample subsequent points from all ues
            ind = self._num_tx * ind + tf.expand_dims(
                                        tf.range(self._num_tx, dtype=tf.int32),
                                        axis=0)

            a = tf.transpose(
                    tf.squeeze(tf.gather(self._a[0], ind, axis=3), axis=0),
                    perm=[2,0,1,3,4,5,6])
            tau = tf.transpose(
                    tf.squeeze(tf.gather(self._tau[0], ind, axis=2), axis=0),
                    perm=[1,0,2,3])

        return a, tau

try:
    import h5py
except ImportError:
    h5py = None

try:
    from scipy import io as scipy_io
except ImportError:
    scipy_io = None

import numpy as np
import tensorflow as tf

class OFDMDatasetChannelSampler(ChannelModel):
    """Channel model from a MATLAB file containing precomputed OFDM channels.

    The entire dataset is read into memory once during initialization.

    Expected dataset format in the MAT-file
    ---------------------------------------
    h_freq : complex tensor with shape
        [num_examples, num_rx, num_rx_ant, num_tx, num_tx_ant,
         num_ofdm_symbols, fft_size]

    Parameters
    ----------
    mat_filename : str
        Path to the MATLAB ``.mat`` file containing the precomputed channel
        dataset. Both standard MAT files and MATLAB v7.3 / HDF5 files are
        supported.
    max_num_examples : int, optional
        Maximum number of channel examples to load from the file.
        ``-1`` (default) loads all available examples.
    training : bool, optional
        If ``True`` (default), the dataset is split into ``num_tx`` equal
        subsets and each subset is independently sampled per transmitter.
        If ``False``, examples are drawn in an interleaved pattern across
        transmitters (evaluation mode).
    num_tx : int, optional
        Number of transmit users / streams. Default is ``1``.
    random_subsampling : bool, optional
        If ``True`` (default), a fresh random batch of indices is drawn for
        every call. If ``False``, the same index vector is reused across all
        transmitter subsets within one call, producing correlated samples.
    dataset_name : str, optional
        Key used to look up the channel tensor inside the MAT file.
        Default is ``"h_freq"``.
    prbs : int or None, optional
        Number of Physical Resource Blocks to retain in the frequency
        dimension. One PRB spans 12 consecutive subcarriers (LTE/NR
        standard). When set, only the first ``prbs * 12`` subcarriers are
        kept; the slice is applied once at load time to reduce memory usage.
        ``None`` (default) keeps the full ``fft_size``.
    dtype : tf.DType, optional
        TensorFlow data type for the channel tensor. Default is
        ``tf.complex64``.

    Notes
    -----
    - Standard MATLAB ``.mat`` files are read with ``scipy.io.loadmat``.
    - MATLAB v7.3 / HDF5 files are read with ``h5py`` when available.
    - HDF5-backed ``.mat`` files carrying a ``codex_layout="sampler_direct"``
      attribute are read without the legacy MATLAB v7.3 axis reversal.
    - The runtime sampling path in ``__call__`` is TensorFlow graph compatible.
    - This class returns OFDM frequency-domain channels directly, not ``a`` and ``tau``.
    """

    def __init__(self,
                 mat_filename,
                 max_num_examples=-1,
                 training=True,
                 num_tx=1,
                 random_subsampling=True,
                 dataset_name="h_freq",
                 prbs=None,
                 dtype=tf.complex64):
        self._training = training
        self._num_tx = num_tx
        self._random_subsampling = random_subsampling
        self._dtype = dtype

        h_freq = self._load_mat_complex(dataset_name=dataset_name,
                                        mat_filename=mat_filename,
                                        max_num_examples=max_num_examples)

        # Ensure dtype expected by the simulator
        h_freq = tf.cast(h_freq, self._dtype)

        # PRB slicing: 1 PRB = 12 subcarriers (LTE/NR standard)
        if prbs is not None:
            fft_size = int(h_freq.shape[-1])
            if fft_size % 12 != 0:
                raise ValueError(
                    f"fft_size {fft_size} is not divisible by 12 (subcarriers per PRB)."
                )
            max_prbs = fft_size // 12
            if prbs > max_prbs:
                raise ValueError(
                    f"Requested {prbs} PRBs but dataset only has {max_prbs} PRBs "
                    f"({fft_size} subcarriers)."
                )
            h_freq = h_freq[..., :prbs * 12]

        # h_freq shape:
        # [N, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        if training:
            # split dataset into equal subsets, one subset per user stream.
            num_examples = int(h_freq.shape[0] // self._num_tx)
            self._num_examples = num_examples
            self._h = []
            for i in range(self._num_tx):
                self._h.append(h_freq[i*num_examples:(i+1)*num_examples])
        else:
            self._num_examples = int(h_freq.shape[0])
            self._h = [h_freq]

    @staticmethod
    def _decode_complex_array(arr, dataset_name="h_freq"):
        """Decode numeric or compound complex arrays into a complex numpy array."""
        if np.iscomplexobj(arr):
            return arr
        if arr.dtype.fields is not None:
            fields = set(arr.dtype.fields.keys())
            if {"real", "imag"}.issubset(fields):
                return arr["real"] + 1j * arr["imag"]
            if {"r", "i"}.issubset(fields):
                return arr["r"] + 1j * arr["i"]
            raise ValueError(
                f"Unsupported complex compound dtype fields for '{dataset_name}': {list(fields)}"
            )
        raise ValueError(
            f"Dataset '{dataset_name}' is not complex."
        )

    @staticmethod
    def _decode_hdf5_attr(value):
        """Normalize HDF5 attribute values to plain Python strings when possible."""
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value[()]
        if isinstance(value, np.bytes_):
            return value.decode()
        if isinstance(value, bytes):
            return value.decode()
        return value

    @staticmethod
    def _slice_max_examples(h, max_num_examples, mat_filename, dataset_name):
        if max_num_examples == -1:
            return h

        num_examples = h.shape[0]
        if max_num_examples > num_examples:
            print(
                f"Warning: requested max_num_examples={max_num_examples}, but "
                f"dataset '{dataset_name}' in '{mat_filename}' only contains "
                f"{num_examples} examples. Loading all available examples."
            )
            return h

        return h[:max_num_examples]

    @classmethod
    def _load_mat_complex(cls, mat_filename, dataset_name="h_freq", max_num_examples=-1):
        """Load a complex OFDM channel tensor from MATLAB MAT files.

        Returns
        -------
        tf.Tensor
            Tensor of shape
            [num_examples, num_rx, num_rx_ant, num_tx, num_tx_ant,
            num_ofdm_symbols, fft_size]
        """
        errors = []
        h = None

        if scipy_io is not None:
            try:
                mat = scipy_io.loadmat(mat_filename)
                if dataset_name not in mat:
                    raise KeyError(f"Dataset '{dataset_name}' not found in '{mat_filename}'")
                h = cls._decode_complex_array(mat[dataset_name], dataset_name=dataset_name)
                if h.ndim != 7:
                    raise ValueError(
                        f"Expected 7-D tensor for '{dataset_name}', got shape {h.shape}"
                    )
                h = cls._slice_max_examples(
                    h, max_num_examples, mat_filename, dataset_name
                )
            except Exception as exc:  # fallback to HDF5/v7.3 reader if needed
                errors.append(exc)
                h = None

        if h is None and h5py is not None:
            try:
                with h5py.File(mat_filename, "r") as f:
                    if dataset_name not in f:
                        raise KeyError(f"Dataset '{dataset_name}' not found in '{mat_filename}'")

                    ds = f[dataset_name]
                    layout = cls._decode_hdf5_attr(ds.attrs.get("codex_layout"))
                    arr = ds[()]

                h = cls._decode_complex_array(arr, dataset_name=dataset_name)

                if h.ndim != 7:
                    raise ValueError(
                        f"Expected 7-D tensor for '{dataset_name}', got shape {h.shape}"
                    )

                if layout != "sampler_direct":
                    # MATLAB v7.3/HDF5 dimension order appears reversed in h5py.
                    # Read shape example:
                    #   (fft_size, num_sym, num_tx_ant, num_tx, num_rx_ant, num_rx, batch)
                    # Convert to:
                    #   (batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_sym, fft_size)
                    h = np.transpose(h, (6, 5, 4, 3, 2, 1, 0))
                h = cls._slice_max_examples(
                    h, max_num_examples, mat_filename, dataset_name
                )
            except Exception as exc:
                errors.append(exc)
                h = None

        if h is None:
            err_lines = [f"Could not load MATLAB dataset '{dataset_name}' from '{mat_filename}'."]
            if scipy_io is None:
                err_lines.append("scipy.io is not available for standard MAT files.")
            if h5py is None:
                err_lines.append("h5py is not available for MATLAB v7.3 / HDF5 files.")
            if errors:
                err_lines.append("Loader errors:")
                err_lines.extend([f"- {type(err).__name__}: {err}" for err in errors[-2:]])
            raise ValueError("\n".join(err_lines))

        with tf.device("/CPU:0"):
            return tf.convert_to_tensor(h, dtype=tf.complex64)

    def __call__(self,
                 batch_size=None,
                 num_time_steps=None,
                 sampling_frequency=None):
        # Unused arguments kept for compatibility with existing interfaces.
        del num_time_steps
        del sampling_frequency

        if batch_size is None:
            raise ValueError("batch_size must be provided.")

        if self._training:
            h = None

            if not self._random_subsampling:
                ind = tf.random.uniform([batch_size],
                                        maxval=self._num_examples,
                                        dtype=tf.int32)

            # Randomly subsample from different subsets
            for ue_idx in range(self._num_tx):
                if self._random_subsampling:
                    ind = tf.random.uniform([batch_size],
                                            maxval=self._num_examples,
                                            dtype=tf.int32)

                # self._h[ue_idx]:
                # [Nsub, num_rx, num_rx_ant, num_tx, num_tx_ant, num_sym, fft]
                h_ = tf.gather(self._h[ue_idx], ind, axis=0)
                # h_:
                # [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_sym, fft]

                if h is not None:
                    # Aggregate sampled users along num_tx axis
                    h = tf.concat([h, h_], axis=3)
                else:
                    h = h_
        else:
            # Evaluation mode: samples alternate between users
            if not self._random_subsampling:
                ind = tf.random.uniform([batch_size],
                                        maxval=self._num_examples // self._num_tx,
                                        dtype=tf.int32)
                ind = tf.repeat(tf.expand_dims(ind, axis=-1),
                                repeats=self._num_tx,
                                axis=-1)
            else:
                ind = tf.random.uniform([batch_size, self._num_tx],
                                        maxval=self._num_examples // self._num_tx,
                                        dtype=tf.int32)

            ind = self._num_tx * ind + tf.expand_dims(
                tf.range(self._num_tx, dtype=tf.int32), axis=0
            )

            # self._h[0]:
            # [N, num_rx, num_rx_ant, num_tx, num_tx_ant, num_sym, fft]
            h = tf.gather(self._h[0], ind, axis=0)
            # h:
            # [batch, num_tx_users, num_rx, num_rx_ant, num_tx, num_tx_ant, num_sym, fft]

            # Merge gathered user dimension into num_tx
            b = tf.shape(h)[0]
            n_users = tf.shape(h)[1]
            num_rx = tf.shape(h)[2]
            num_rx_ant = tf.shape(h)[3]
            num_tx = tf.shape(h)[4]
            num_tx_ant = tf.shape(h)[5]
            num_sym = tf.shape(h)[6]
            fft_size = tf.shape(h)[7]

            h = tf.transpose(h, [0, 2, 3, 1, 4, 5, 6, 7])
            h = tf.reshape(h, [b, num_rx, num_rx_ant,
                               n_users * num_tx, num_tx_ant,
                               num_sym, fft_size])

        return h


class OFDMDatasetChannel(Layer):
    """
    Applies precomputed OFDM channels to the channel input.

    Channel frequency responses are drawn from ``channel_model`` and applied
    with :class:`~sionna.channel.ApplyOFDMChannel`.

    Parameters
    ----------
    channel_model : callable
        Sampler that returns channel frequency responses for a given batch
        size, e.g. :class:`OFDMDatasetChannelSampler`.

    add_awgn : bool
        If set to `False`, no white Gaussian noise is added.
        Defaults to `True`.

    normalize_channel : bool
        If set to `True`, the sampled channel is normalized to unit average
        energy per RX/TX link. Defaults to `True`.

    return_channel : bool
        If set to `True`, the channel response is returned in addition to the
        channel output. Defaults to `False`.

    dtype : tf.DType
        Complex datatype to use for internal processing and output.
        Defaults to tf.complex64.

    Input
    -----

    (x, no) or x:
        Tuple or Tensor:

    x :  [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
        Channel inputs

    no : Scalar or Tensor, tf.float
        Scalar or tensor whose shape can be broadcast to the shape of the
        channel outputs:
        [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size].
        Only required if ``add_awgn`` is set to `True`.

    Output
    -------
    y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        Channel outputs
    h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
        (Optional) Channel frequency responses. Returned only if
        ``return_channel`` is set to `True`.
    """

    def __init__(self, channel_model, add_awgn=True,
                normalize_channel=True, return_channel=False,
                dtype=tf.complex64, **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)

        self._channel_sampler = channel_model
        self._add_awgn = add_awgn
        self._normalize_channel = normalize_channel
        self._return_channel = return_channel

    def build(self, input_shape): #pylint: disable=unused-argument

        self._apply_channel = ApplyOFDMChannel( self._add_awgn,
                                                tf.as_dtype(self.dtype))

    def _normalize_h_freq(self, h_freq):
        """Match Sionna's cir_to_ofdm_channel normalization on loaded OFDM data.

        Input
        -----
        h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant,
                  num_ofdm_symbols, fft_size], tf.complex
            Channel frequency responses.

        Output
        ------
        h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant,
                  num_ofdm_symbols, fft_size], tf.complex
            Channel frequency responses normalized to unit average energy per
            RX/TX link.
        """
        # Shapes: B=batch, R=num_rx, RA=num_rx_ant, T=num_tx, TA=num_tx_ant,
        # S=num_ofdm_symbols, F=fft_size
        # h_freq: [B, R, RA, T, TA, S, F]
        rank = h_freq.shape.rank
        if rank is None:
            # Average over RA, TA, S, and F for dynamic-rank tensors.
            reduce_axes = (2, 4, 5, 6)
        else:
            # Keep batch, RX, and TX link axes and average over every other
            # dimension so each sampled link has unit average energy.
            reduce_axes = tuple(axis for axis in range(rank) if axis not in (0, 1, 3))

        # scale: [B, R, 1, T, 1, 1, 1]
        scale = tf.reduce_mean(
            tf.square(tf.abs(h_freq)),
            axis=reduce_axes,
            keepdims=True,
        )
        scale = tf.cast(tf.sqrt(scale), h_freq.dtype)
        # return: [B, R, RA, T, TA, S, F]
        return tf.math.divide_no_nan(h_freq, scale)

    def call(self, inputs):
        # Shapes: B=batch, R=num_rx, RA=num_rx_ant, T=num_tx, TA=num_tx_ant, S=num_ofdm_symbols, F=fft_size
        if self._add_awgn:
            x, no = inputs # x:[B, T, TA, S, F], no:broadcastable to [B, R, RA, S, F]
        else:
            x = inputs # x:[B, T, TA, S, F]

        h_freq = self._channel_sampler(tf.shape(x)[0]) # [B, R, RA, T, TA, S, F]

        if self._normalize_channel:
            h_freq = self._normalize_h_freq(h_freq) # [B, R, RA, T, TA, S, F]

        if self._add_awgn:
            y = self._apply_channel([x, h_freq, no]) # [B, R, RA, S, F]
        else:
            y = self._apply_channel([x, h_freq]) # [B, R, RA, S, F]

        if self._return_channel:
            return y, h_freq # y:[B, R, RA, S, F], h_freq:[B, R, RA, T, TA, S, F]
        else:
            return y # [B, R, RA, S, F]
