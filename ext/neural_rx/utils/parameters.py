# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# reads in the configuration file and initialized all relevant system components
# This allows to train and evaluate different system configurations on the same
# server and simplifies logging of the training process.
import core.runtime as _runtime
import os
import sys
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import configparser
import tensorflow as tf
from os.path import exists
from sionna.nr import PUSCHConfig, PUSCHDMRSConfig, TBConfig, CarrierConfig, PUSCHTransmitter, PUSCHPilotPattern
from sionna.channel.tr38901 import PanelArray, UMi, TDL, UMa
from sionna.mimo import StreamManagement
from sionna.channel import OFDMChannel, AWGN
from .channel_models import DoubleTDLChannel, DatasetChannel, NTDLChannel, NCDLChannel, OFDMDatasetChannel, OFDMDatasetChannelSampler
from .impairments import FrequencyOffset

def resolve_channel_models(*candidates):
    """Pick the first non-empty model list among ``candidates`` and normalize it.

    The TDL and CDL families share the same profile letters (A..E), so a single
    list selects the profiles for whichever family ``channel_type`` picks.
    Entries may carry a redundant "TDL-"/"CDL-" prefix (e.g. "CDL-B"), which is
    stripped. Returns None if every candidate is empty.
    """
    for models in candidates:
        if models:
            return [m.split("-", 1)[1]
                    if m.upper().startswith(("TDL-", "CDL-")) else m
                    for m in models]
    return None


# ---------------------------------------------------------------------------
# LMMSE covariance matrices: naming, provenance and validation
# ---------------------------------------------------------------------------
#
# The matrices are sized by the resource grid and the BS array, which may be
# overridden at runtime, so one .cfg can be evaluated on several geometries.
# The file key therefore encodes every overridable quantity, and a JSON
# sidecar records the rest so that mismatching matrices are rejected.

COV_META_SCHEMA_VERSION = 1

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))

WEIGHTS_DIR = os.path.join(_REPO_ROOT, "weights")


def cov_mat_key(params):
    """Filename stem shared by the covariance triple and its metadata sidecar.

    Used by both compute_cov_mat.py (writer) and Parameters._load_cov_mats()
    (reader).
    """
    rows = getattr(params, "num_rows_per_panel", None)
    cols = getattr(params, "num_cols_per_panel", None)
    # "ra<r>x<c>" and "ant<N>" differ on purpose: e.g. a (4,4) dual-pol panel
    # and the (16,1) ULA fallback both have 32 antennas but different spatial
    # covariance.
    if rows is not None and cols is not None:
        geom = f"ra{rows}x{cols}"
    else:
        geom = f"ant{params.num_rx_antennas}"
    return f"{params.label}_prb{params.n_size_bwp}_{geom}_{params.max_num_tx}u"


def cov_mat_paths(params):
    """Absolute paths of the covariance triple and its metadata sidecar."""
    key = cov_mat_key(params)
    return {
        "freq": os.path.join(WEIGHTS_DIR, f"{key}_freq_cov_mat.npy"),
        "time": os.path.join(WEIGHTS_DIR, f"{key}_time_cov_mat.npy"),
        "space": os.path.join(WEIGHTS_DIR, f"{key}_space_cov_mat.npy"),
        "meta": os.path.join(WEIGHTS_DIR, f"{key}_cov_meta.json"),
    }


def cov_mat_expected_shapes(params):
    """Expected covariance shapes per axis, read from the resource grid in use."""
    rg = params.transmitters[0]._resource_grid
    return {
        "freq": (int(rg.fft_size), int(rg.fft_size)),
        "time": (int(rg.num_ofdm_symbols), int(rg.num_ofdm_symbols)),
        "space": (int(params.num_rx_antennas), int(params.num_rx_antennas)),
    }


def _cov_identity_fields(params):
    """Run-defining fields compared between the sidecar and live Parameters.

    The channel type and UT velocity range are recorded in the sidecar but not
    compared: covariance computation always uses a 3GPP UMi/UMa channel and
    the config's velocity range, independent of the evaluation overrides.
    """
    rg = params.transmitters[0]._resource_grid
    return {
        "label": params.label,
        "n_size_bwp": int(params.n_size_bwp),
        "fft_size": int(rg.fft_size),
        "num_ofdm_symbols": int(rg.num_ofdm_symbols),
        "num_rx_antennas": int(params.num_rx_antennas),
        "num_rows_per_panel": getattr(params, "num_rows_per_panel", None),
        "num_cols_per_panel": getattr(params, "num_cols_per_panel", None),
        "max_num_tx": int(params.max_num_tx),
        "carrier_frequency": float(params.carrier_frequency),
    }


def _config_digest(params):
    """Hash of the parsed config, so a .cfg edit after generation is visible."""
    return hashlib.sha256(
        getattr(params, "config_str", "").encode("utf-8")).hexdigest()[:16]


def cov_mat_provenance(params, **extra):
    """Record written next to the matrices and re-checked when they are read."""
    shapes = cov_mat_expected_shapes(params)
    meta = {
        "schema_version": COV_META_SCHEMA_VERSION,
        "config_path": getattr(params, "config_path", None),
        "config_digest": _config_digest(params),
        # Recorded but not enforced, see _cov_identity_fields().
        "cov_channel_type": str(params.channel_type),
        "min_ut_velocity": float(params.min_ut_velocity),
        "max_ut_velocity": float(params.max_ut_velocity),
        "freq_cov_mat_shape": list(shapes["freq"]),
        "time_cov_mat_shape": list(shapes["time"]),
        "space_cov_mat_shape": list(shapes["space"]),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv),
    }
    meta.update(_cov_identity_fields(params))
    meta.update(extra)
    return meta


def _regenerate_hint(params):
    """The exact command that produces the matrices this run needs."""
    cmd = ["python scripts/compute_cov_mat.py",
           f"-config_name {params.config_name}",
           f"-n_size_bwp_eval {params.n_size_bwp}",
           f"-num_tx_eval {params.max_num_tx}"]
    rows = getattr(params, "num_rows_per_panel", None)
    cols = getattr(params, "num_cols_per_panel", None)
    if rows is not None and cols is not None:
        cmd.append(f"-num_rx_antennas {params.num_rx_antennas}")
        cmd.append(f"-num_rows_per_panel {rows} -num_cols_per_panel {cols}")
    return ("Re-run without -skip_cov_compute, or generate them with:\n"
            "    " + " \\\n        ".join(cmd))


# Order in which sionna's LMMSEInterpolator applies its 1D passes. "s" (spatial
# smoothing across receive antennas) is optional; "f" and "t" are not, since
# every resource element has to end up with an estimate.
LMMSE_ORDER_DEFAULT = "s-f-t"


def validate_lmmse_order(order):
    """Check an LMMSEInterpolator ``order`` string and return it unchanged.

    Fails early with a readable error instead of an assert inside
    LMMSEInterpolator at build time.
    """
    steps = str(order).split("-")
    unknown = [s for s in steps if s not in ("s", "f", "t")]
    if unknown:
        raise ValueError(
            f"Invalid lmmse_order '{order}': unknown step(s) "
            f"{', '.join(repr(u) for u in unknown)}. Steps are 's' (space), "
            f"'f' (frequency) and 't' (time), joined by '-', e.g. 's-f-t'.")
    repeated = sorted({s for s in steps if steps.count(s) > 1})
    if repeated:
        raise ValueError(
            f"Invalid lmmse_order '{order}': step(s) "
            f"{', '.join(repr(r) for r in repeated)} given more than once.")
    missing = [s for s in ("f", "t") if s not in steps]
    if missing:
        raise ValueError(
            f"Invalid lmmse_order '{order}': {' and '.join(missing)} "
            f"interpolation is mandatory -- without it some resource elements "
            f"get no channel estimate. Valid orders are permutations of "
            f"'f-t' with an optional 's', e.g. 's-f-t' or 'f-t'.")
    return order


class Parameters:
    r"""
    Simulation parameters

    Parameters
    ----------
    config_name : str
        name of the config file.

    system : str
        Receiver algorithm to be used. Must be one of:
        * "deep_echo" : DeepEcho receiver
        * "deep_echo_kbest" : DeepEcho channel refinement with K-Best detection
        * "mdx" : Model driven
        * "nrx" : Neural receiver
        * "nrx_kbest" : Neural receiver channel refinement with K-Best detection
        * "baseline_lmmse_kbest" : LMMSE estimation and K-Best detection
        * "baseline_perf_csi_kbest" : perfect CSI and K-Best detection
        * "baseline_lmmse_lmmse" : LMMSE estimation and LMMSE equalization
        * "baseline_lsnn_lmmse" : LS estimation/nn interpolation and LMMSE equalization
        * "baseline_lsnn_kbest" : LS estimation/nn interpolation and K-Best equalization
        * "ch_saver" : saves (h_freq, h_ls, no) to HDF5 (``ch_save_path``,
          default 'ch_save.h5'); no decoding performed.
        * "dummy" : stops after parameter import. Can be used only to parse the
          config.

    training: bool, False,
        If True, training parameters are loaded. Otherwise, the evaluation
        parameters are used.

    verbose: bool, False
        If True, additional information is printed during init.

    compute_cov: bool, False
        If True, the UMi channel model is loaded automatically to avoid
        overfitting to TDL models.

    num_tx_eval: int or None
        If provided, the max number of users is limited to ``num_tx_eval``.
        For this, the first DMRS ports are selected.

    Evaluation-time overrides (bandwidth, antenna geometry, channel model,
    LMMSE order, ...) are applied by calling :meth:`re_init` again.
    """
    def __init__(self, config_name, system="dummy", training=False, verbose=False, compute_cov=False, num_tx_eval=None):

        # check input for consistency
        assert isinstance(verbose, bool), "verbose must be bool."
        assert isinstance(training, bool), "training must be bool."
        assert isinstance(config_name, str), "config_name must be str."
        assert isinstance(system, str), "system must be str."
        assert isinstance(compute_cov, bool), "compute_cov must be bool."

        self.system = system
        self.config_name = config_name

        ###################################
        ##### Load configuration file #####
        ###################################

        # create parser object and read config file
        config_names = [config_name]
        if not config_name.endswith(".cfg"):
            config_names.append(f"{config_name}.cfg")

        repo_config_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "config")
        )

        candidate_paths = []
        for cfg_name in config_names:
            if os.path.isabs(cfg_name):
                candidate_paths.append(cfg_name)
            else:
                candidate_paths.append(cfg_name)
                candidate_paths.append(os.path.join("..", "config", cfg_name))
                candidate_paths.append(os.path.join(repo_config_dir, cfg_name))

        fn = next((path for path in candidate_paths if exists(path)), None)
        if fn is None:
            raise FileNotFoundError(
                f"Unknown config file '{config_name}'. Looked in '{repo_config_dir}'."
            )

        config = configparser.RawConfigParser()
        config.read(fn)
        self.config_path = os.path.abspath(fn)

        # and import all parameters as attributes
        self.config_str = ""
        for section in config.sections():
            s = f"\n---- {section} ----- "
            self.config_str += s + "<br />" # add linebreak for Tensorboard
            if verbose:
                print(s)
            for option in config.options(section):
                setattr(self, f"{option}", eval(config.get(section,option)))
                s = f"{option}: {eval(config.get(section,option))}"
                self.config_str += s + "<br />" # add linebreak for Tensorboard
                if verbose:
                    print(s)

        # Training bandwidth, stored before re_init() replaces n_size_bwp by
        # the eval value.
        self.n_size_bwp_training = self.n_size_bwp

        self._training = training
        self._verbose = verbose
        self._compute_cov = compute_cov
        self._num_tx_eval = num_tx_eval

        if hasattr(self, "dtype"):
            if not self.dtype.is_complex:
                raise TypeError(f"Expected a complex dtype, but got {self.dtype.name}")
            self.real_dtype = self.dtype.real_dtype

        self.re_init(training, verbose, compute_cov, num_tx_eval)

    def re_init(self,training=None, verbose=None, compute_cov=None, num_tx_eval=None,
                n_size_bwp_eval=None, batch_size_eval=None, batch_size_eval_small=None,
                max_ut_velocity_eval=None, min_ut_velocity_eval=None,
                channel_norm_eval=None, mat_filename_eval=None,
                channel_type_eval=None, channel_models=None,
                tdl_models=None, cdl_models=None,
                num_rx_antennas=None, num_rows_per_panel=None,
                num_cols_per_panel=None,
                lmmse_order=None):
        """(Re-)initialize the system components.

        Called once from ``__init__`` and again with evaluation overrides
        (``*_eval``, antenna geometry, channel type/models, LMMSE order).
        Arguments left as None keep their current or config value.
        """

        # Drop memoized covariance matrices; the geometry may have changed.
        self._cov_cache = None

        if training is None:
            training = self._training
        if verbose is None:
            verbose = self._verbose
        if compute_cov is None:
            compute_cov = self._compute_cov
        if num_tx_eval is None:
            num_tx_eval = self._num_tx_eval

        if n_size_bwp_eval is not None:
            self.n_size_bwp_eval = n_size_bwp_eval
        if batch_size_eval is not None:
            self.batch_size_eval = batch_size_eval
        if batch_size_eval_small is not None:
            self.batch_size_eval_small = batch_size_eval_small            
        if max_ut_velocity_eval is not None:
            self.max_ut_velocity_eval = max_ut_velocity_eval
        if min_ut_velocity_eval is not None:
            self.min_ut_velocity_eval = min_ut_velocity_eval
        if channel_norm_eval is not None:
            self.channel_norm_eval = channel_norm_eval

        # BS array override, so that a single config can be evaluated on
        # several antenna geometries. Applied before the "dummy" early return
        # so both re_init passes see the same geometry.
        if (num_rows_per_panel is None) != (num_cols_per_panel is None):
            raise ValueError("num_rows_per_panel and num_cols_per_panel must "
                             "be provided together.")
        if num_rx_antennas is not None:
            self.num_rx_antennas = num_rx_antennas
        if num_rows_per_panel is not None:
            self.num_rows_per_panel = num_rows_per_panel
            self.num_cols_per_panel = num_cols_per_panel

        # LMMSE interpolation order used by BaselineReceiver (default "s-f-t").
        if lmmse_order is not None:
            self.lmmse_order = lmmse_order
        elif not hasattr(self, "lmmse_order"):
            self.lmmse_order = LMMSE_ORDER_DEFAULT
        validate_lmmse_order(self.lmmse_order)

        # Overwrite channel and PRBs in inference mode with "eval" parameters
        # This allows to configure different parameters during training and
        # evaluation.
        if not training:
            if hasattr(self, 'mat_filename') and mat_filename_eval is not None:
                print(f"mat_filename:{self.mat_filename}, \n changed to eval mat_filename:{mat_filename_eval}")
                self.mat_filename = mat_filename_eval
            self.n_size_bwp = self.n_size_bwp_eval
            self.max_ut_velocity = self.max_ut_velocity_eval
            self.min_ut_velocity = self.min_ut_velocity_eval
            self.channel_norm = self.channel_norm_eval
            self.cfo_offset_ppm = self.cfo_offset_ppm_eval
            self.tfrecord_filename = self.tfrecord_filename_eval
            if self.channel_type == "Dataset":
                self.random_subsampling = self.random_subsampling_eval

        # load only config parameters and return without initializing the rest
        # of the system
        if self.system == "dummy":
            return

        #####################################
        ##### Init PUSCH configurations #####
        #####################################

        # init PUSCHConfig
        carrier_config = CarrierConfig(
                n_cell_id=self.n_cell_id,
                cyclic_prefix=self.cyclic_prefix,
                subcarrier_spacing=int(self.subcarrier_spacing/1e3), # in kHz
                n_size_grid=self.n_size_bwp,
                n_start_grid=self.n_start_grid,
                slot_number=self.slot_number,
                frame_number=self.frame_number)

        # init DMRSConfig
        pusch_dmrs_config=PUSCHDMRSConfig(
                config_type=self.dmrs_config_type,
                type_a_position=self.dmrs_type_a_position,
                additional_position=self.dmrs_additional_position,
                length=self.dmrs_length,
                dmrs_port_set=self.dmrs_port_sets[0], # first user
                n_scid=self.n_scid,
            num_cdm_groups_without_data=self.num_cdm_groups_without_data)

        mcs_list = self.mcs_index
        # generate pusch configs for all MCSs
        self.pusch_configs = []   # self.pusch_configs[MCS_CONFIG][N_UE]
        for mcs_list_idx in range(len(mcs_list)):
            self.pusch_configs.append([])
            mcs_index = mcs_list[mcs_list_idx]
            # init TBConfig
            tb_config = TBConfig(
                    mcs_index=mcs_index,
                    mcs_table=self.mcs_table,
                    channel_type="PUSCH")
            if self.num_antenna_ports >1:
                # first user PUSCH config
                pc = PUSCHConfig(
                        carrier_config=carrier_config,
                        pusch_dmrs_config=pusch_dmrs_config,
                        tb_config=tb_config,
                        num_antenna_ports = self.num_antenna_ports,
                        precoding = self.precoding,
                        symbol_allocation = self.symbol_allocation,
                        tpmi = self.tpmi,
                        mapping_type=self.dmrs_mapping_type,)
            else:
                # first user PUSCH config
                pc = PUSCHConfig(
                        carrier_config=carrier_config,
                        pusch_dmrs_config=pusch_dmrs_config,
                        tb_config=tb_config,
                        num_antenna_ports = self.num_antenna_ports,
                        symbol_allocation = self.symbol_allocation,
                        mapping_type=self.dmrs_mapping_type,)

            # clone new PUSCHConfig for each additional user
            for idx,_ in enumerate(self.dmrs_port_sets):
                p = pc.clone() # generate new PUSCHConfig
                # set user specific parts
                p.dmrs.dmrs_port_set = self.dmrs_port_sets[idx]
                # n_id, DMRS n_id and n_rnti keep their default values
                self.pusch_configs[mcs_list_idx].append(p)

        ##############################
        ##### Consistency checks #####
        ##############################

        # after training we can only reduce the number of iterations
        assert self.num_nrx_iter_eval<=self.num_nrx_iter, \
            "num_nrx_iter_eval must be smaller or equal num_nrx_iter."

        # for the evaluation, only activate num_tx_eval configs
        if not training:
            # overwrite num_tx_eval if explicitly provided
            if num_tx_eval is not None:
                num_tx_eval = num_tx_eval
            else: # if not provided use all available port sets
                num_tx_eval = len(self.dmrs_port_sets)
            self.max_num_tx = num_tx_eval # non-varying users for evaluation
            self.min_num_tx = num_tx_eval # non-varying users for evaluation

        for mcs_list_idx in range(len(mcs_list)):
            self.pusch_configs[mcs_list_idx] = self.pusch_configs[mcs_list_idx][:self.max_num_tx]
        print(f"Evaluating the first {self.max_num_tx} port sets.")

        ##################################
        ##### Configure Transmitters #####
        ##################################

        # Generate and store DMRS for all slot numbers
        self.pilots = []
        for slot_num in range(carrier_config.num_slots_per_frame):
            for pcs in self.pusch_configs:
                for pc in pcs:
                    pc.carrier.slot_number = slot_num
            # only generate pilot pattern for first MCS's PUSCH config, as
            # pilots are independent from MCS index
            pilot_pattern = PUSCHPilotPattern(self.pusch_configs[0])
            self.pilots.append(pilot_pattern.pilots)
        self.pilots = tf.stack(self.pilots, axis=0)
        self.pilots = tf.constant(self.pilots)
        for pcs in self.pusch_configs:
            for pc in pcs:
                pc.carrier.slot_number = self.slot_number

        # transmitter is a list of PUSCHTransmitters, one for each MCS
        self.transmitters = []
        for mcs_list_idx in range(len(mcs_list)):
            # and init transmitter
            self.transmitters.append(
                PUSCHTransmitter(
                            self.pusch_configs[mcs_list_idx],
                            return_bits=False,
                            output_domain="freq",
                            verbose=self.verbose))

            # support end-to-end learning / custom constellations
            # see https://arxiv.org/pdf/2009.05261 for details
            if self.custom_constellation: # trainable constellations
                print("Activating trainable custom constellations.")
                self.transmitters[mcs_list_idx]._mapper.constellation.trainable = True
            # Center constellations. This could be also deactivated for more
            # degrees of freedom.
            self.transmitters[mcs_list_idx]._mapper.constellation.center = True

        # chest will fail if we use explicit masking of pilots.
        if self.mask_pilots and self.initial_chest in ("ls", "nn"):
            print("Warning: initial_chest will fail with masked pilots.")

        # StreamManagement required for KBestDetector
        self.sm = StreamManagement(np.ones([1, self.max_num_tx], int), 1)

        num_mcs = len(mcs_list)
        self.points_list = [self.transmitters[i]._mapper.constellation.points for i in range(num_mcs)]  # List of tensors for each transmitter's points

        ##############################
        ##### Initialize Channel #####
        ##############################
        self.channel_models = ["A"]
        self._pc = pc

        # One list of profile letters serves both the TDL and the CDL family;
        # -tdl_models / -cdl_models are kept as aliases of -channel_models.
        channel_models = resolve_channel_models(channel_models, cdl_models,
                                                tdl_models)
        if channel_models is not None:
            self.channel_models = channel_models
        # Legacy attribute names
        self.tdl_models = self.channel_models
        self.cdl_models = self.channel_models

        if channel_type_eval is not None:
            self.initialize_channel(compute_cov=compute_cov, channel_type_eval=channel_type_eval,
                                    channel_models=channel_models)
        else:
            self.initialize_channel(compute_cov=compute_cov)

        # Hardware impairments
        if self.cfo_offset_ppm>0:
            offset = self.carrier_frequency / 1e6 * self.cfo_offset_ppm
            max_rel_offset = offset/self.transmitters[0].resource_grid.bandwidth    # resource grid and bandwidth is independent of MCS
            self.frequency_offset = FrequencyOffset(
                                    max_rel_offset,
                                    "freq",
                                    self.transmitters[0].resource_grid,             # resource grid is independent of MCS
                                    constant_offset=(not training)) # fix offset for evaluation
        else:
            self.frequency_offset = None

        ##############################
        ##### Positional Encoding #####
        ##############################

        if hasattr(self, 'pe_type'):
            if self.system=="mdx":
                if self.pe_type == 0: # NRX default
                    self.pe_d = 2
                if self.pe_type == 1: # sin coding
                    seq_len = 12*14
                    d = self.pe_d
                    n = self.pe_n

                    PE = np.zeros((seq_len, d))
                    for k in range(seq_len):
                        for i in np.arange(int(d/2)):
                            denominator = np.power(n, 2*i/d)
                            PE[k, 2*i] = np.sin(k/denominator)
                            PE[k, 2*i+1] = np.cos(k/denominator)

                    PE = tf.convert_to_tensor(PE, dtype=tf.float32)
                    self.PE = tf.reshape(PE,[12,14,d])
                if self.pe_type==2 or self.pe_type==3: # PRB coding or NRX default + PRB coding
                    rows = tf.cast(tf.range(12, dtype=tf.float32) / 11.0, dtype=tf.float32)  # 0 to 11, normalized to [0, 1]
                    cols = tf.cast(tf.range(14, dtype=tf.float32) / 13.0, dtype=tf.float32)  # 0 to 13, normalized to [0, 1]

                    A_0 = tf.tile(tf.expand_dims(rows, axis=1), [1, 14])  
                    A_1 = tf.tile(tf.expand_dims(cols, axis=0), [12, 1]) 

                    # pe [12, 14, 2]
                    self.PE = tf.stack([A_0, A_1], axis=-1)
                    self.pe_d = 2
                if self.pe_type==3: # NRX default + PRB coding
                    self.pe_d = self.pe_d + 2

    # ------------------------------------------------------------------
    # LMMSE covariance matrices
    # ------------------------------------------------------------------
    #
    # Loaded lazily on first access (from BaselineReceiver), i.e. after all
    # evaluation overrides have been applied by re_init, so that the file key
    # and the shape check use the geometry actually in use.

    def _cov_mats(self):
        """Load, validate and memoize the covariance triple."""
        if getattr(self, "_cov_cache", None) is None:
            self._cov_cache = self._load_cov_mats()
        return self._cov_cache

    @property
    def freq_cov_mat(self):
        return self._cov_mats()["freq"]

    @property
    def time_cov_mat(self):
        return self._cov_mats()["time"]

    @property
    def space_cov_mat(self):
        return self._cov_mats()["space"]

    def _load_cov_mats(self):
        """Read the covariance triple, refusing anything that is not this run's.

        Missing files, a schema mismatch, differing run-defining fields or
        wrong shapes are all fatal errors.
        """
        if self.system not in ("baseline_lmmse_kbest", "baseline_lmmse_lmmse"):
            raise RuntimeError(
                f"Covariance matrices are only used by the LMMSE baselines, "
                f"but system is '{self.system}'.")

        paths = cov_mat_paths(self)
        expected = cov_mat_expected_shapes(self)
        want = _cov_identity_fields(self)

        missing = [p for p in paths.values() if not exists(p)]
        if missing:
            raise FileNotFoundError(
                f"No covariance matrices for this run "
                f"('{cov_mat_key(self)}').\nMissing:\n  "
                + "\n  ".join(missing) + "\n" + _regenerate_hint(self))

        with open(paths["meta"], encoding="utf-8") as f:
            meta = json.load(f)

        version = meta.get("schema_version")
        if version != COV_META_SCHEMA_VERSION:
            raise RuntimeError(
                f"{paths['meta']} has schema_version {version!r}, expected "
                f"{COV_META_SCHEMA_VERSION}.\n" + _regenerate_hint(self))

        # The filename already pins PRBs, geometry and users, so a mismatch
        # here means the sidecar and the matrices were written by different
        # runs, or the .cfg changed something the key does not encode.
        diffs = [f"{k}: recorded {meta.get(k)!r}, this run {v!r}"
                 for k, v in want.items() if meta.get(k) != v]

        arrays = {name: np.load(paths[name])
                  for name in ("freq", "time", "space")}
        diffs += [f"{os.path.basename(paths[name])}: found {tuple(arr.shape)}, "
                  f"expected {expected[name]}"
                  for name, arr in arrays.items()
                  if tuple(arr.shape) != expected[name]]

        if diffs:
            raise RuntimeError(
                f"Stale covariance matrices for '{self.label}' in "
                f"{WEIGHTS_DIR}:\n  " + "\n  ".join(diffs)
                + f"\n  written {meta.get('created_utc')} on channel "
                  f"{meta.get('cov_channel_type')} by: "
                  f"{' '.join(meta.get('argv') or ['?'])}\n"
                + _regenerate_hint(self))

        mats = {name: tf.cast(arr, tf.complex64)
                for name, arr in arrays.items()}

        # A warning rather than an error: the digest covers the whole .cfg,
        # including training-only keys that cannot affect these matrices.
        if meta.get("config_digest") != _config_digest(self):
            print(f"WARNING: {os.path.basename(self.config_path)} has changed "
                  f"since {os.path.basename(paths['meta'])} was written. The "
                  f"shape-relevant fields still match, but the covariance "
                  f"matrices may be out of date.")

        return mats

    def _build_panel_arrays(self):
        """Build the BS and UT antenna arrays for the array-based 3GPP models
        (UMi, UMa, CDL). Returns (bs_array, ut_array)."""
        if self.num_rx_antennas==1: # ignore polarization for single antenna
            print("Using vertical polarization for single antenna setup.")
            num_cols_per_panel = 1
            num_rows_per_panel = 1
            polarization = "single"
            polarization_type = 'V'
        else:

            if hasattr(self, 'num_cols_per_panel') and hasattr(self, 'num_rows_per_panel'):
                num_cols_per_panel = self.num_cols_per_panel
                num_rows_per_panel = self.num_rows_per_panel
                assert num_cols_per_panel * num_rows_per_panel * 2 == self.num_rx_antennas, \
    f"Invalid antenna configuration: {num_cols_per_panel} columns * {num_rows_per_panel} rows * 2 " \
    f"does not equal {self.num_rx_antennas} receive antennas"
            else:
                # we use a ULA array to be aligned with TDL models
                num_cols_per_panel = self.num_rx_antennas//2
                num_rows_per_panel = 1

            polarization = "dual"
            polarization_type = 'cross'

        bs_array = PanelArray(num_rows_per_panel = num_rows_per_panel,
                              num_cols_per_panel = num_cols_per_panel,
                              polarization = polarization,
                              polarization_type  = polarization_type,
                              antenna_pattern = '38.901',
                              carrier_frequency = self.carrier_frequency)

        ut_array = PanelArray(num_rows_per_panel = 1,
                              num_cols_per_panel = self._pc.num_antenna_ports,
                              polarization = 'single',
                              polarization_type = 'V',
                              antenna_pattern = 'omni',
                              carrier_frequency = self.carrier_frequency)

        return bs_array, ut_array

    def initialize_channel(self,channel_type_eval=None, channel_models=None,
                                tdl_models=None, cdl_models=None, compute_cov=False,
                                delay_spread_min=10,   # in nano seconds
                                delay_spread_max=300,  # in nano seconds
                                doppler_shift_max=325  # Hz
                                ):
        """Instantiate ``self.channel`` according to ``self.channel_type``.

        ``channel_type_eval`` and ``channel_models`` (or its aliases
        ``tdl_models``/``cdl_models``) override the config values. With
        ``compute_cov``, non-UMi/UMa channels are replaced by UMi.
        """

        # Enable/Disable random number of clusters for geometric channel models (implemented only for UMi)
        if hasattr(self, 'random_num_clusters'):
            random_num_clusters = self.random_num_clusters
        else:
            random_num_clusters = False

        # Enable/Disable random number of rays for geometric channel models (implemented only for UMi)
        if hasattr(self, 'random_num_rays'):
            random_num_rays = self.random_num_rays
        else:
            random_num_rays = False

        if hasattr(self, 'mask_doa'):
            mask_doa = self.mask_doa
        else:
            mask_doa = False        

        if hasattr(self, 'num_rays'):
            num_rays = self.num_rays
        else:
            num_rays = None       

        if channel_type_eval is not None:
            self.channel_type = channel_type_eval
        channel_models = resolve_channel_models(channel_models, cdl_models,
                                                tdl_models)
        if channel_models is not None:
            self.channel_models = channel_models
            # Legacy attribute names
            self.tdl_models = self.channel_models
            self.cdl_models = self.channel_models
        # use UMi (or UMa, if selected) to calculate covariance matrix
        if compute_cov:
            if not self.channel_type in ("UMi", "UMa"):
                print("Setting channel type to UMi for covariance computation.")
                self.channel_type = "UMi"

        # Sanity check
        if self.channel_type in ("DoubleTDLlow","DoubleTDLmedium",
                                 "DoubleTDLhigh") and self.max_num_tx==1:
                print("Warning: SelectedDoubleTDL model only defined for 2 "\
                      "users. Selecting TDL-B100 instead.")
                self.channel_type = "TDL-B100"

        # Initialize channel
        # Remark: new channel models can be added here

        if self.channel_type in ("UMi", "UMa"):
            bs_array, ut_array = self._build_panel_arrays()

            if self.channel_type == "UMi":
                self.channel_model = UMi(
                                carrier_frequency=self.carrier_frequency,
                                o2i_model = 'low',
                                bs_array = bs_array,
                                ut_array = ut_array,
                                direction = 'uplink',
                                enable_pathloss = False,
                                enable_shadow_fading = False,
                                random_num_clusters = random_num_clusters,
                                random_num_rays = random_num_rays,
                                mask_doa = mask_doa,
                                num_rays = num_rays)
            else: # UMa
                ignored = [n for n, v in (
                                ("random_num_clusters", random_num_clusters),
                                ("random_num_rays", random_num_rays),
                                ("mask_doa", mask_doa),
                                ("num_rays", num_rays)) if v]
                if ignored:
                    print("Warning: " + ", ".join(ignored) + " are "
                          "implemented for UMi only and are ignored for "
                          "channel_type='UMa'.")
                self.channel_model = UMa(
                                carrier_frequency=self.carrier_frequency,
                                o2i_model = 'low',
                                bs_array = bs_array,
                                ut_array = ut_array,
                                direction = 'uplink',
                                enable_pathloss = False,
                                enable_shadow_fading = False)

            self.channel = OFDMChannel(
                    channel_model=self.channel_model,
                    resource_grid=self.transmitters[0]._resource_grid,          # resource grid is independent of MCS
                    add_awgn=True,
                    normalize_channel=self.channel_norm,
                    return_channel=True)

        elif self.channel_type == "TDL-B100":
            tdl = TDL(model="B100",
                      delay_spread=100e-9,
                      carrier_frequency=self.carrier_frequency,
                      min_speed=self.min_ut_velocity,
                      max_speed=self.max_ut_velocity,
                      num_tx_ant=self._pc.num_antenna_ports,
                      num_rx_ant=self.num_rx_antennas)
            self.channel = OFDMChannel(tdl,
                                       self.transmitters[0].resource_grid,      # resource grid is independent of MCS
                                       add_awgn=True,
                                       normalize_channel=self.channel_norm,
                                       return_channel=True)
        elif self.channel_type == "TDL-C300":
            tdl = TDL(model="C300",
                      delay_spread=300e-9,
                      carrier_frequency=self.carrier_frequency,
                      min_speed=self.min_ut_velocity,
                      max_speed=self.max_ut_velocity,
                      num_tx_ant=self._pc.num_antenna_ports,
                      num_rx_ant=self.num_rx_antennas)
            self.channel = OFDMChannel(tdl,
                                       self.transmitters[0].resource_grid,      # resource grid is independent of MCS
                                       add_awgn=True,
                                       normalize_channel=self.channel_norm,
                                       return_channel=True)
        # DoubleTDL for evaluation
        elif self.channel_type == "DoubleTDLlow":
            self.channel = DoubleTDLChannel(self.carrier_frequency,
                                    self.transmitters[0].resource_grid,         # resource grid is independent of MCS
                                    correlation="low",
                                    num_tx_ant=self._pc.num_antenna_ports,
                                    num_rx_ant = self.num_rx_antennas,
                                    norm_channel=self.channel_norm)
        # DoubleTDL for evaluation
        elif self.channel_type == "DoubleTDLmedium":
            self.channel= DoubleTDLChannel(self.carrier_frequency,
                                    self.transmitters[0].resource_grid,         # resource grid is independent of MCS
                                    correlation="medium",
                                    num_tx_ant=self._pc.num_antenna_ports,
                                    norm_channel=self.channel_norm)
        # DoubleTDL for evaluation
        elif self.channel_type == "DoubleTDLhigh":
            self.channel = DoubleTDLChannel(self.carrier_frequency,
                                    self.transmitters[0].resource_grid,         # resource grid is independent of MCS
                                    correlation="high",
                                    num_tx_ant=self._pc.num_antenna_ports,
                                    norm_channel=self.channel_norm)
        # NTDL for evaluation
        elif self.channel_type == "NTDLlow":
            self.channel = NTDLChannel(carrier_frequency=self.carrier_frequency,
                                    resource_grid=self.transmitters[0].resource_grid,        # resource grid is independent of MCS
                                    correlation="low",
                                    num_tx_ant=self._pc.num_antenna_ports,
                                    num_rx_ant = self.num_rx_antennas,
                                    max_num_tx = self.max_num_tx,
                                    norm_channel=self.channel_norm,
                                    tdl_models=self.channel_models,
                                    delay_spread_min=delay_spread_min,   # in nano seconds
                                    delay_spread_max=delay_spread_max,  # in nano seconds
                                    doppler_shift_max=doppler_shift_max  # Hz
                                    )

        # Single TDL profile "TDL-A".."TDL-E" via NTDLChannel (supports
        # max_num_tx > 1, unlike the "TDL-B100"/"TDL-C300" presets above).
        elif self.channel_type.startswith("TDL-"):
            tdl_models_ = [self.channel_type.split("-", 1)[1]]
            self.channel = NTDLChannel(carrier_frequency=self.carrier_frequency,
                                    resource_grid=self.transmitters[0].resource_grid,
                                    correlation="low",
                                    num_tx_ant=self._pc.num_antenna_ports,
                                    num_rx_ant = self.num_rx_antennas,
                                    max_num_tx = self.max_num_tx,
                                    norm_channel=self.channel_norm,
                                    tdl_models=tdl_models_,
                                    delay_spread_min=delay_spread_min,   # in nano seconds
                                    delay_spread_max=delay_spread_max,  # in nano seconds
                                    doppler_shift_max=doppler_shift_max  # Hz
                                    )

        # CDL for evaluation. "CDL-A" .. "CDL-E" select a single profile
        # (matching the "TDL-B100"/"TDL-C300" naming), plain "CDL" draws the
        # per-user profiles from self.channel_models.
        elif self.channel_type == "CDL" or self.channel_type.startswith("CDL-"):
            if self.channel_type.startswith("CDL-"):
                cdl_models_ = [self.channel_type.split("-", 1)[1]]
            else:
                cdl_models_ = self.channel_models

            bs_array, ut_array = self._build_panel_arrays()

            self.channel = NCDLChannel(
                            carrier_frequency=self.carrier_frequency,
                            resource_grid=self.transmitters[0].resource_grid,    # resource grid is independent of MCS
                            bs_array=bs_array,
                            ut_array=ut_array,
                            max_num_tx=self.max_num_tx,
                            norm_channel=self.channel_norm,
                            cdl_models=cdl_models_,
                            min_speed=self.min_ut_velocity,
                            max_speed=self.max_ut_velocity,
                            delay_spread_min=delay_spread_min,   # in nano seconds
                            delay_spread_max=delay_spread_max,   # in nano seconds
                            )

        elif self.channel_type == "AWGN":
            self.channel = AWGN()

        elif self.channel_type == "Dataset":
            channel_model = DatasetChannel("../data/" + self.tfrecord_filename,
                                    max_num_examples=-1, # loads entire dataset
                                    training=self._training,
                                    num_tx=self.max_num_tx,
                                    random_subsampling=self.random_subsampling,
                                    )
            self.channel = OFDMChannel(channel_model,
                                       self.transmitters[0].resource_grid,      # resource grid is independent of MCS
                                       add_awgn=True,
                                       normalize_channel=self.channel_norm,
                                       return_channel=True)

        elif self.channel_type == "OFDMDataset":
            if os.path.isabs(self.mat_filename):
                file_path = self.mat_filename
            else:
                file_path = os.path.join("..", "data", self.mat_filename)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            self.channel_model = OFDMDatasetChannelSampler(file_path,
                                    max_num_examples=getattr(self, "max_num_examples", -1),
                                    training=self._training,
                                    num_tx=self.max_num_tx,
                                    random_subsampling=False,
                                    prbs=self.n_size_bwp,
                                    )

            self.channel = OFDMDatasetChannel(self.channel_model,
                                       add_awgn=True,
                                       normalize_channel=self.channel_norm,
                                       return_channel=True)

        else:
            raise ValueError(f"{self.name} Unknown Channel type \'{self.channel_type}\'!")
