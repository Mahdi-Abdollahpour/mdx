#!/usr/bin/python3



"""Evaluate receivers (BER/BLER and/or channel-estimation MSE/NMSE) over SNR.

Results are stored per (system, num_tx, MCS index) in
``<dir><label><name_suffix>_results``. Supported ``-methods``:

- ``deep_echo``: DeepEcho receiver (config-defined block graph)
- ``deep_echo_kbest``: DeepEcho channel refinement with K-Best detection
- ``mdx``: model-driven neural receiver
- ``nrx``, ``nrx_kbest``: neural receiver, optionally with K-Best detection
- ``baseline_lslin_lmmse``, ``baseline_lslin_kbest``: LS estimation with linear
  interpolation, LMMSE or K-Best detection
- ``baseline_lmmse_lmmse``, ``baseline_lmmse_kbest``: LMMSE channel estimation
  (requires covariance matrices, see ``compute_cov_mat.py``)
- ``baseline_perf_csi_lmmse``, ``baseline_perf_csi_kbest``: perfect CSI
"""

import argparse
import os
import pickle
import subprocess
import sys
import time
from os.path import exists

# Keep TensorFlow/XLA startup logs from breaking progress tables.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.append("../")
import core.runtime as _runtime  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("-config_name", help="config filename", type=str)
parser.add_argument(
    "-num_target_block_errors",
    help="Number of target block errors for BER/BLER evaluation",
    type=int,
    default=500,
)
parser.add_argument(
    "-max_mc_iter",
    help="Maximum Monte Carlo iterations",
    type=int,
    default=500,
)
parser.add_argument(
    "-target_bler",
    help="Early stop threshold for BLER",
    type=float,
    default=0.001,
)
parser.add_argument(
    "-target_mse",
    help="Early stop threshold for MSE",
    type=float,
    default=None,
)
parser.add_argument(
    "-num_cov_samples",
    help="Number of samples for covariance generation",
    type=int,
    default=1000000,
)
parser.add_argument("-gpu", help="GPU to use", type=int, default=0)
parser.add_argument(
    "-num_tx_eval",
    help="Number of active users",
    type=int,
    nargs="+",
    default=-1,
)
parser.add_argument(
    "-mcs_arr_eval_idx",
    help="Select the MCS array index for evaluation. Use -1 to evaluate all MCSs.",
    type=int,
    default=-1,
)
parser.add_argument(
    "-debug",
    help="Set debugging configuration",
    action="store_true",
    default=False,
)
parser.add_argument(
    "-all_gpus",
    help="Distribute on all GPUs",
    action="store_true",
    default=False,
)
parser.add_argument(
    "-dont_load",
    help="Do not load previous evaluation",
    action="store_true",
    default=False,
)
parser.add_argument(
    "-forced_transfer",
    help="Transfer weights specified in cfg file",
    action="store_true",
    default=False,
)
parser.add_argument(
    "-skip_cov_compute",
    help="Skip covariance matrix generation",
    action="store_true",
    default=False,
)
parser.add_argument(
    "-start_token_model",
    help="Override transfer-learning start token for model weights",
    type=str,
    default=None,
)
parser.add_argument(
    "-start_token_file",
    help="Override transfer-learning start token for checkpoint weights",
    type=str,
    default=None,
)
parser.add_argument(
    "-transfer_weights_path",
    help="Override transfer-learning checkpoint path",
    type=str,
    default=None,
)

parser.add_argument("-snr_db_eval_min", type=float, default=-5.0)
parser.add_argument("-snr_db_eval_max", type=float, default=5.0)
parser.add_argument("-snr_db_eval_stepsize", type=float, default=1.0)
parser.add_argument("-max_ut_velocity_eval", type=float, default=34.0)
parser.add_argument("-channel_type_eval", type=str, default="NTDLlow")
parser.add_argument(
    "-channel_models",
    help=(
        "Profile letters for the selected channel family, one per user: "
        "-channel_models A B C ... The TDL and CDL families share the same "
        "letters (A..E); a redundant 'TDL-'/'CDL-' prefix is accepted and "
        "stripped. Only used by the multi-user 'NTDLlow' and 'CDL' models."
    ),
    type=str,
    nargs="+",
    default=None,
)
# Deprecated aliases of -channel_models
parser.add_argument("-tdl_models", type=str, nargs="+", default=None,
                    help=argparse.SUPPRESS)
parser.add_argument("-cdl_models", type=str, nargs="+", default=None,
                    help=argparse.SUPPRESS)
parser.add_argument("-n_size_bwp_eval", type=int, default=132)
parser.add_argument("-batch_size_eval", type=int, default=30)
parser.add_argument("-batch_size_eval_small", type=int, default=3)
parser.add_argument(
    "-num_rx_antennas",
    help=(
        "Override the config's BS antenna count for evaluation. Must be given "
        "together with -num_rows_per_panel/-num_cols_per_panel, and "
        "rows*cols*2 == num_rx_antennas (dual-pol panel). "
        "Default: keep the config value."
    ),
    type=int,
    default=None,
)
parser.add_argument(
    "-num_rows_per_panel",
    help="Override the BS panel rows for evaluation. See -num_rx_antennas.",
    type=int,
    default=None,
)
parser.add_argument(
    "-num_cols_per_panel",
    help="Override the BS panel columns for evaluation. See -num_rx_antennas.",
    type=int,
    default=None,
)
parser.add_argument(
    "-lmmse_order",
    help=(
        "Order of the LMMSE interpolation passes for the baseline_lmmse_* "
        "receivers: 's' (spatial smoothing across RX antennas), 'f' "
        "(frequency) and 't' (time), joined by '-'. 'f' and 't' are "
        "mandatory, 's' is optional, e.g. 's-f-t' or 'f-t'. "
        "Default: keep the config value ('s-f-t' when unset)."
    ),
    type=str,
    default=None,
)
parser.add_argument(
    "-dir",
    type=str,
    help="Directory to save results",
    default="../results/",
)
parser.add_argument(
    "-name_suffix",
    type=str,
    help="Suffix added to results filename",
    default="",
)
parser.add_argument(
    "-mcs_index",
    help="-mcs_index 9 14 19",
    type=int,
    nargs="+",
    default=[-1],
)

parser.add_argument(
    "-methods",
    nargs="+",
    help=(
        "methods list: deep_echo, deep_echo_kbest, mdx, nrx, baseline_lslin_lmmse, "
        "nrx_kbest, "
        "baseline_lslin_kbest, baseline_lmmse_kbest, "
        "baseline_perf_csi_lmmse, baseline_lmmse_lmmse, "
        "baseline_perf_csi_kbest"
    ),
    type=str,
    default=["baseline_lslin_lmmse"],
)

parser.add_argument(
    "-snr_dbs",
    nargs="+",
    help="List of SNR values in dB (e.g., -5 -4 0 10)",
    type=float,
    default=[],
)
parser.add_argument(
    "-mat_filename_eval",
    help="Evaluation dataset mat filename",
    type=str,
    default=None,
)

parser.add_argument(
    "-eval_mode",
    help="Metric mode to evaluate: ber, mse, or both",
    type=str,
    choices=["ber", "mse", "both"],
    default="ber",
)

parser.add_argument(
    "-target_squared_error",
    help="Target accumulated squared error for MSE evaluation",
    type=float,
    default=5.0e5,
)

parser.add_argument(
    "-channel_norm",
    help="Override channel normalization (true/false). Default: use config value.",
    type=lambda x: x.lower() in ('1', 'true', 'yes'),
    default=None,
)

args = parser.parse_args()

config_name = args.config_name
max_mc_iter = args.max_mc_iter
num_target_block_errors = args.num_target_block_errors
num_cov_samples = args.num_cov_samples
gpu = args.gpu
target_bler = args.target_bler
target_mse = args.target_mse
num_tx_eval = args.num_tx_eval
mcs_arr_eval_idx = args.mcs_arr_eval_idx
dont_load = args.dont_load
methods = args.methods
res_dir = args.dir
name_suffix = args.name_suffix
mcs_index = args.mcs_index
snr_dbs = args.snr_dbs
forced_transfer = args.forced_transfer
skip_cov_compute = args.skip_cov_compute
mat_filename_eval = args.mat_filename_eval
eval_mode = args.eval_mode

distribute = "all" if args.all_gpus else None

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

gpus = tf.config.list_physical_devices("GPU")

if distribute != "all":
    try:
        tf.config.set_visible_devices(gpus[args.gpu], "GPU")
        print("Only GPU number", args.gpu, "used.")
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
    except RuntimeError as e:
        print(f"error\n:{e}")

sys.path.append("../")

import numpy as np
import sionna as sn

from utils.sim_metrics import sim_metrics
from ext.neural_rx.utils.e2e_model import E2E_Model
from utils.model_weights import load_or_transfer_weights
from ext.neural_rx.utils.parameters import Parameters, resolve_channel_models
from ext.neural_rx.utils.utils import load_weights

if args.debug:
    tf.config.run_functions_eagerly(True)

REQUIRED_RESULT_FIELDS = ("snr", "ber", "bler", "mse", "nmse")
OPTIONAL_RESULT_FIELDS = ("avg_runtime_per_sample",)
RESULT_FIELDS = REQUIRED_RESULT_FIELDS + OPTIONAL_RESULT_FIELDS


def empty_results():
    """Create an empty results container."""
    return {k: {} for k in RESULT_FIELDS}


def save_results(results_filename, results, max_retries=3, wait_time=5):
    """Save results using a fixed schema."""
    for attempt in range(max_retries):
        try:
            with open(results_filename, "wb") as f:
                pickle.dump(results, f)
            time.sleep(2)
            print(f"File saved successfully: {results_filename}")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("All attempts failed.")
                return False


def load_results(results_filename, dont_load=False):
    """Load existing results or create an empty structure."""
    if not dont_load and exists(results_filename):
        print(f"### File '{results_filename}' found. It will be updated.")
        with open(results_filename, "rb") as f:
            data = pickle.load(f)

        for field in REQUIRED_RESULT_FIELDS:
            if field not in data:
                raise ValueError(
                    f"Invalid results file '{results_filename}'. Missing key: '{field}'"
                )
        for field in OPTIONAL_RESULT_FIELDS:
            data.setdefault(field, {})
        return data

    print(
        f"### No existing results file at '{results_filename}', or loading is disabled. "
        f"Initializing empty results."
    )
    return empty_results()


def result_key(sys_name, num_tx_eval, mcs_arr_eval_idx):
    """Build a unique storage key for one evaluated setup."""
    return (sys_name, num_tx_eval, mcs_arr_eval_idx)


def store_metric_results(results, key, snr, metrics):
    """Store whichever metrics were produced by sim_metrics."""
    results["snr"][key] = snr

    if "ber" in metrics:
        results["ber"][key] = metrics["ber"]
    if "bler" in metrics:
        results["bler"][key] = metrics["bler"]
    if "mse" in metrics:
        results["mse"][key] = metrics["mse"]
    if "nmse" in metrics:
        results["nmse"][key] = metrics["nmse"]
    if "avg_runtime_per_sample" in metrics:
        results["avg_runtime_per_sample"][key] = metrics["avg_runtime_per_sample"]


def antenna_override_args(args):
    """CLI flags forwarded to compute_cov_mat.py, empty when nothing is set."""
    if args.num_rx_antennas is None:
        return []
    return [
        "-num_rx_antennas", str(args.num_rx_antennas),
        "-num_rows_per_panel", str(args.num_rows_per_panel),
        "-num_cols_per_panel", str(args.num_cols_per_panel),
    ]


def validate_antenna_overrides(args):
    """Check that the BS array overrides are given together and are consistent.

    Parameters only reads the panel geometry when both rows and cols are set,
    so a partial override would silently change the antenna array.
    """
    overrides = (args.num_rx_antennas,
                 args.num_rows_per_panel,
                 args.num_cols_per_panel)
    if all(v is None for v in overrides):
        return
    if any(v is None for v in overrides):
        raise ValueError(
            "-num_rx_antennas, -num_rows_per_panel and -num_cols_per_panel "
            "must be given together."
        )
    expected = args.num_rows_per_panel * args.num_cols_per_panel * 2
    if expected != args.num_rx_antennas:
        raise ValueError(
            f"Invalid antenna override: {args.num_rows_per_panel} rows * "
            f"{args.num_cols_per_panel} cols * 2 = {expected} "
            f"!= {args.num_rx_antennas} receive antennas"
        )


def set_eval_params(sys_parameters, args):
    """Apply evaluation-specific overrides to Parameters."""
    channel_models = resolve_channel_models(
        args.channel_models, args.cdl_models, args.tdl_models) or ["A"]

    if args.channel_type_eval in ["OFDMDataset", "Dataset"]:
        print(
            f"setting the evaluation channel model to: {args.channel_type_eval}\n"
            f"Data: {args.mat_filename_eval}"
        )
    else:
        print(
            f"setting the evaluation channel model to: "
            f"{args.channel_type_eval} {channel_models}"
        )

    print(f"setting n_size_bwp_eval to {args.n_size_bwp_eval}")

    validate_antenna_overrides(args)
    if args.num_rx_antennas is not None:
        print(
            f"overriding BS array: {args.num_rx_antennas} antennas, "
            f"{args.num_rows_per_panel}x{args.num_cols_per_panel} dual-pol panel"
        )
    if args.lmmse_order is not None:
        print(f"overriding LMMSE interpolation order: {args.lmmse_order}")

    sys_parameters.re_init(
        n_size_bwp_eval=args.n_size_bwp_eval,
        batch_size_eval=args.batch_size_eval,
        batch_size_eval_small=args.batch_size_eval_small,
        max_ut_velocity_eval=args.max_ut_velocity_eval,
        channel_type_eval=args.channel_type_eval,
        channel_models=channel_models,
        mat_filename_eval=args.mat_filename_eval,
        channel_norm_eval=args.channel_norm,
        num_rx_antennas=args.num_rx_antennas,
        num_rows_per_panel=args.num_rows_per_panel,
        num_cols_per_panel=args.num_cols_per_panel,
        lmmse_order=args.lmmse_order,
    )

    if args.start_token_model is not None:
        sys_parameters.start_token_model = args.start_token_model
    if args.start_token_file is not None:
        sys_parameters.start_token_file = args.start_token_file
    if args.transfer_weights_path is not None:
        sys_parameters.transfer_weights_path = args.transfer_weights_path

    return sys_parameters


def validate_channel_compatibility(sys_parameters, num_tx_eval):
    """Check that the selected channel model matches the requested number of users."""
    if sys_parameters.channel_type == "TDL-B100":
        assert num_tx_eval == 1, (
            "Channel model 'TDL-B100' only works with one transmitter"
        )
    elif sys_parameters.channel_type in (
        "DoubleTDLlow",
        "DoubleTDLmedium",
        "DoubleTDLhigh",
    ):
        assert num_tx_eval == 2, (
            "Channel model 'DoubleTDL' only works with exactly two transmitters"
        )
    elif sys_parameters.channel_type.startswith("CDL-"):
        assert num_tx_eval == 1, (
            f"Channel model '{sys_parameters.channel_type}' pins a single CDL "
            "profile and only works with one transmitter. Use "
            "-channel_type_eval=CDL with -cdl_models for multi-user setups."
        )


def _make_load_result(model, *, transferred_all):
    """Synthetic result dict for load_weights (which returns no diagnostic info)."""
    n = len(model.weights)
    ok = n if transferred_all else 0
    return {"summary": {"transferred": ok, "total_model_weights": n}}


def maybe_load_weights_for_deep_echo(e2e_nn, sys_parameters, forced_transfer):
    """Load or transfer weights for deep_echo."""
    filename = f"../weights/{sys_parameters.label}_weights.h5"

    start_token_model = getattr(sys_parameters, "start_token_model", None)
    start_token_file = getattr(sys_parameters, "start_token_file", None)
    verbose=3
    if forced_transfer:
        if exists(sys_parameters.transfer_weights_path):
            transfer_result = load_or_transfer_weights(
                e2e_nn,
                sys_parameters.transfer_weights_path,
                start_token_model=start_token_model,
                start_token_file=start_token_file,
                verbose=verbose,
            )
            print(f"weights transferred from:\n{sys_parameters.transfer_weights_path}")
            return transfer_result
        else:
            print("Transfer weights do not exist.")
            return _make_load_result(e2e_nn, transferred_all=False)

    if exists(filename):
        load_weights(e2e_nn, filename)
        print(f"weights loaded from:\n{filename}")
        return _make_load_result(e2e_nn, transferred_all=True)
    elif exists(sys_parameters.transfer_weights_path):
        transfer_result = load_or_transfer_weights(
            e2e_nn,
            sys_parameters.transfer_weights_path,
            start_token_model=start_token_model,
            start_token_file=start_token_file,
            verbose=verbose,
        )
        print(f"weights transferred from:\n{sys_parameters.transfer_weights_path}")
        return transfer_result
    else:
        print("weights do not exist.")
        return _make_load_result(e2e_nn, transferred_all=False)


def maybe_load_weights_for_mdx(e2e_nn, sys_parameters, forced_transfer):
    """Load or transfer weights for mdx."""
    filename = f"../weights/{sys_parameters.label}_weights.h5"

    if forced_transfer:
        if exists(sys_parameters.transfer_weights_path):
            transfer_result = load_or_transfer_weights(
                e2e_nn,
                sys_parameters.transfer_weights_path,
                start_token="neural_pusch_receiver/cgnnofdm",
                verbose=False,
            )
            print(f"weights transferred from:\n{sys_parameters.transfer_weights_path}")
            return transfer_result
        else:
            print("Transfer weights do not exist.")
            return _make_load_result(e2e_nn, transferred_all=False)

    if exists(filename):
        load_weights(e2e_nn, filename)
        print(f"weights loaded from:\n{filename}")
        return _make_load_result(e2e_nn, transferred_all=True)
    elif exists(sys_parameters.transfer_weights_path):
        transfer_result = load_or_transfer_weights(
            e2e_nn,
            sys_parameters.transfer_weights_path,
            start_token="neural_pusch_receiver/cgnnofdm",
            verbose=False,
        )
        print(f"weights transferred from:\n{sys_parameters.transfer_weights_path}")
        return transfer_result
    else:
        print("weights do not exist.")
        return _make_load_result(e2e_nn, transferred_all=False)


def maybe_load_weights_for_nrx(e2e_nn, sys_parameters, forced_transfer):
    """Load or transfer weights for nrx."""
    filename = f"../weights/{sys_parameters.label}_weights"

    start_token_model = getattr(sys_parameters, "start_token_model", None)
    start_token_file = getattr(sys_parameters, "start_token_file", None)

    if forced_transfer:
        if exists(sys_parameters.transfer_weights_path):
            transfer_result = load_or_transfer_weights(
                e2e_nn,
                sys_parameters.transfer_weights_path,
                start_token_model=start_token_model,
                start_token_file=start_token_file,
                verbose=1,
            )
            print(f"weights transferred from:\n{sys_parameters.transfer_weights_path}")
            return transfer_result
        else:
            print("Transfer weights do not exist.")
            return _make_load_result(e2e_nn, transferred_all=False)

    if exists(filename):
        load_weights(e2e_nn, filename)
        print(f"weights loaded from:\n{filename}")
        return _make_load_result(e2e_nn, transferred_all=True)
    elif exists(sys_parameters.transfer_weights_path):
        transfer_result = load_or_transfer_weights(
            e2e_nn,
            sys_parameters.transfer_weights_path,
            start_token_model=start_token_model,
            start_token_file=start_token_file,
            verbose=1,
        )
        print(f"weights transferred from:\n{sys_parameters.transfer_weights_path}")
        return transfer_result
    else:
        print("weights do not exist.")
        return _make_load_result(e2e_nn, transferred_all=False)


def should_run_eval_after_transfer(sys_parameters, transfer_result):
    """Skip evaluation when weights were not fully assigned (100%)."""
    summary = transfer_result.get("summary", {})
    transferred = summary.get("transferred")
    total_model_weights = summary.get("total_model_weights")
    if transferred is not None and transferred == total_model_weights:
        return True

    weights_path = getattr(sys_parameters, "transfer_weights_path", "unknown")
    print(
        "Skipping evaluation for {}: only assigned {}/{} model weights from\n{}".format(
            sys_parameters.system,
            transferred,
            total_model_weights,
            weights_path,
        )
    )
    return False


def run_model_eval(
    model,
    snr_values,
    batch_size,
    graph_mode,
    results,
    num_tx_eval,
    mcs_arr_eval_idx,
    eval_mode,
    distribute=None,
):
    """Run ``sim_metrics`` for ``eval_mode`` ("ber", "mse" or "both") and store the results."""
    key = result_key(model._sys_name, num_tx_eval, mcs_arr_eval_idx)
    system = getattr(getattr(model, "_sys_parameters", None), "system", "")
    perfect_csi_mode = system in ("baseline_perf_csi_lmmse", "baseline_perf_csi_kbest")
    disable_mse_targets = perfect_csi_mode and eval_mode in ("mse", "both")
    mse_target = target_mse
    num_target_samples = None
    if disable_mse_targets:
        mse_target = None if eval_mode == "mse" else float("inf")
        if eval_mode == "mse":
            num_target_samples = 1
    metrics = sim_metrics(
        model,
        graph_mode=graph_mode,
        ebno_dbs=snr_values,
        max_mc_iter=max_mc_iter,
        mode=eval_mode,
        batch_size=batch_size,
        distribute=distribute,
        early_stop=True,
        num_target_block_errors=num_target_block_errors if eval_mode in ("ber", "both") else None,
        target_bler=target_bler if eval_mode in ("ber", "both") else None,
        target_mse=mse_target if eval_mode in ("mse", "both") else None,
        num_target_samples=num_target_samples,
        target_squared_error=None if disable_mse_targets else (
            args.target_squared_error if eval_mode in ("mse", "both") else None
        ),
        forward_keyboard_interrupt=True,
    )

    store_metric_results(results, key, snr_values, metrics)


if os.path.isdir(res_dir):
    print(f"Directory '{res_dir}' exists")
else:
    print(f"Directory '{res_dir}' does not exist")

# Dummy system, used only for labels and config-dependent defaults
sys_parameters = Parameters(
    config_name,
    training=True,
    system="dummy",
)

sys_parameters.snr_db_eval_min = args.snr_db_eval_min
sys_parameters.snr_db_eval_max = args.snr_db_eval_max
sys_parameters.snr_db_eval_stepsize = args.snr_db_eval_stepsize
sys_parameters = set_eval_params(sys_parameters, args)

batch_size = sys_parameters.batch_size_eval
batch_size_small = sys_parameters.batch_size_eval_small

results_filename = f"{res_dir}{sys_parameters.label}{name_suffix}_results"
results = load_results(results_filename, dont_load=dont_load)

if len(snr_dbs) > 0:
    ebno_db = np.array(snr_dbs, dtype=float)
    print(f"ebno_db set to: {ebno_db}")
else:
    ebno_db = np.arange(
        sys_parameters.snr_db_eval_min,
        sys_parameters.snr_db_eval_max,
        sys_parameters.snr_db_eval_stepsize,
    )

if num_tx_eval == -1:
    num_tx_evals = np.arange(
        sys_parameters.min_num_tx,
        sys_parameters.max_num_tx + 1,
        1,
    )
else:
    if isinstance(num_tx_eval, int):
        num_tx_evals = [num_tx_eval]
    elif isinstance(num_tx_eval, (list, tuple)):
        num_tx_evals = num_tx_eval
    else:
        raise ValueError("num_tx_eval must be int or list of ints.")

if mcs_arr_eval_idx == -1:
    mcs_arr_eval_idxs = list(range(len(sys_parameters.mcs_index)))
else:
    if isinstance(mcs_arr_eval_idx, int):
        mcs_arr_eval_idxs = [mcs_arr_eval_idx]
    elif isinstance(mcs_arr_eval_idx, (list, tuple)):
        mcs_arr_eval_idxs = mcs_arr_eval_idx
    else:
        raise ValueError("mcs_arr_eval_idx must be int or list of ints.")

print(
    f"Evaluating for {num_tx_evals} active users and "
    f"mcs_index elements {mcs_arr_eval_idxs}."
)

for num_tx_eval in num_tx_evals:

    # LMMSE channel estimation needs precomputed covariance matrices
    if (
        not skip_cov_compute
        and (
            "baseline_lmmse_kbest" in methods
            or "baseline_lmmse_lmmse" in methods
        )
    ):
        cov_cmd = [
            sys.executable, "compute_cov_mat.py",
            "-config_name", str(config_name),
            "-gpu", str(gpu),
            "-num_samples", str(num_cov_samples),
            "-num_tx_eval", str(num_tx_eval),
            "-n_size_bwp_eval", str(args.n_size_bwp_eval),
        ] + antenna_override_args(args)
        print("Generating cov matrix:", " ".join(cov_cmd))
        returncode = subprocess.run(cov_cmd).returncode
        if returncode != 0:
            raise RuntimeError(
                f"Covariance generation failed with exit code {returncode}: "
                f"{' '.join(cov_cmd)}"
            )

    for mcs_arr_eval_idx in mcs_arr_eval_idxs:

        if "deep_echo" in methods:
            print("running deep_echo ...")
            sn.config.xla_compat = True

            sys_parameters = Parameters(
                config_name,
                training=False,
                num_tx_eval=num_tx_eval,
                system="deep_echo",
            )
            sys_parameters = set_eval_params(sys_parameters, args)
            validate_channel_compatibility(sys_parameters, num_tx_eval)

            e2e_nn = E2E_Model(
                sys_parameters,
                training=False,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

            print("\nRunning:", sys_parameters.system)
            e2e_nn(1, 1.0)
            transfer_result = maybe_load_weights_for_deep_echo(
                e2e_nn,
                sys_parameters,
                forced_transfer,
            )
            if should_run_eval_after_transfer(sys_parameters, transfer_result):
                run_model_eval(
                    model=e2e_nn,
                    snr_values=ebno_db,
                    batch_size=batch_size,
                    graph_mode="xla",
                    results=results,
                    num_tx_eval=num_tx_eval,
                    mcs_arr_eval_idx=mcs_arr_eval_idx,
                    eval_mode=eval_mode,
                    distribute=distribute,
                )
                save_results(results_filename, results)

            tf.keras.backend.clear_session()
            del e2e_nn
            sn.config.xla_compat = False
        else:
            print("skipping DeepEcho")

        if "deep_echo_kbest" in methods:
            print("running deep_echo_kbest ...")
            sn.config.xla_compat = True

            sys_parameters = Parameters(
                config_name,
                training=False,
                num_tx_eval=num_tx_eval,
                system="deep_echo_kbest",
            )
            sys_parameters = set_eval_params(sys_parameters, args)
            validate_channel_compatibility(sys_parameters, num_tx_eval)

            e2e_nn = E2E_Model(
                sys_parameters,
                training=False,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

            print("\nRunning:", sys_parameters.system)
            e2e_nn(1, 1.0)
            transfer_result = maybe_load_weights_for_deep_echo(
                e2e_nn,
                sys_parameters,
                forced_transfer,
            )
            if should_run_eval_after_transfer(sys_parameters, transfer_result):
                run_model_eval(
                    model=e2e_nn,
                    snr_values=ebno_db,
                    batch_size=batch_size,
                    graph_mode="xla",
                    results=results,
                    num_tx_eval=num_tx_eval,
                    mcs_arr_eval_idx=mcs_arr_eval_idx,
                    eval_mode=eval_mode,
                    distribute=distribute,
                )
                save_results(results_filename, results)

            tf.keras.backend.clear_session()
            del e2e_nn
            sn.config.xla_compat = False
        else:
            print("skipping DeepEcho + K-Best")

        if "mdx" in methods:
            print("running mdx ...")
            sn.config.xla_compat = True

            sys_parameters = Parameters(
                config_name,
                training=False,
                num_tx_eval=num_tx_eval,
                system="mdx",
            )
            sys_parameters = set_eval_params(sys_parameters, args)
            validate_channel_compatibility(sys_parameters, num_tx_eval)

            e2e_nn = E2E_Model(
                sys_parameters,
                training=False,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

            print("\nRunning:", sys_parameters.system)
            e2e_nn(1, 1.0)
            transfer_result = maybe_load_weights_for_mdx(
                e2e_nn,
                sys_parameters,
                forced_transfer,
            )
            if should_run_eval_after_transfer(sys_parameters, transfer_result):
                run_model_eval(
                    model=e2e_nn,
                    snr_values=ebno_db,
                    batch_size=batch_size,
                    graph_mode="xla",
                    results=results,
                    num_tx_eval=num_tx_eval,
                    mcs_arr_eval_idx=mcs_arr_eval_idx,
                    eval_mode=eval_mode,
                    distribute=distribute,
                )
                save_results(results_filename, results)

            tf.keras.backend.clear_session()
            del e2e_nn
            sn.config.xla_compat = False
        else:
            print("skipping MDX")

        if "nrx" in methods:
            print("running nrx ...")
            sn.config.xla_compat = True

            sys_parameters = Parameters(
                config_name,
                training=False,
                num_tx_eval=num_tx_eval,
                system="nrx",
            )
            sys_parameters = set_eval_params(sys_parameters, args)
            validate_channel_compatibility(sys_parameters, num_tx_eval)

            e2e_nn = E2E_Model(
                sys_parameters,
                training=False,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

            print("\nRunning:", sys_parameters.system)
            e2e_nn(1, 1.0)
            transfer_result = maybe_load_weights_for_nrx(
                e2e_nn,
                sys_parameters,
                forced_transfer,
            )

            if should_run_eval_after_transfer(sys_parameters, transfer_result):
                e2e_nn._receiver._neural_rx.num_it = sys_parameters.num_nrx_iter_eval

                run_model_eval(
                    model=e2e_nn,
                    snr_values=ebno_db,
                    batch_size=batch_size,
                    graph_mode="xla",
                    results=results,
                    num_tx_eval=num_tx_eval,
                    mcs_arr_eval_idx=mcs_arr_eval_idx,
                    eval_mode=eval_mode,
                    distribute=distribute,
                )
                save_results(results_filename, results)

            tf.keras.backend.clear_session()
            del e2e_nn
            sn.config.xla_compat = False
        else:
            print("skipping NRX")

        if "nrx_kbest" in methods:
            print("running nrx_kbest ...")
            sn.config.xla_compat = True

            sys_parameters = Parameters(
                config_name,
                training=False,
                num_tx_eval=num_tx_eval,
                system="nrx_kbest",
            )
            sys_parameters = set_eval_params(sys_parameters, args)
            validate_channel_compatibility(sys_parameters, num_tx_eval)

            e2e_nn = E2E_Model(
                sys_parameters,
                training=False,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

            print("\nRunning:", sys_parameters.system)
            e2e_nn(1, 1.0)
            transfer_result = maybe_load_weights_for_nrx(
                e2e_nn,
                sys_parameters,
                forced_transfer,
            )

            if should_run_eval_after_transfer(sys_parameters, transfer_result):
                e2e_nn._receiver._neural_rx.num_it = sys_parameters.num_nrx_iter_eval

                run_model_eval(
                    model=e2e_nn,
                    snr_values=ebno_db,
                    batch_size=batch_size,
                    graph_mode="xla",
                    results=results,
                    num_tx_eval=num_tx_eval,
                    mcs_arr_eval_idx=mcs_arr_eval_idx,
                    eval_mode=eval_mode,
                    distribute=distribute,
                )
                save_results(results_filename, results)

            tf.keras.backend.clear_session()
            del e2e_nn
            sn.config.xla_compat = False
        else:
            print("skipping NRX + K-Best")

        if "baseline_lslin_lmmse" in methods:
            print("running baseline_lslin_lmmse ...")
            sn.config.xla_compat = True

            sys_parameters = Parameters(
                config_name,
                training=False,
                num_tx_eval=num_tx_eval,
                system="baseline_lslin_lmmse",
            )
            sys_parameters = set_eval_params(sys_parameters, args)

            e2e_baseline = E2E_Model(
                sys_parameters,
                training=False,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

            print("\nRunning:", sys_parameters.system)

            run_model_eval(
                model=e2e_baseline,
                snr_values=ebno_db,
                batch_size=batch_size,
                graph_mode="xla",
                results=results,
                num_tx_eval=num_tx_eval,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
                eval_mode=eval_mode,
                distribute=distribute,
            )
            save_results(results_filename, results)
            sn.config.xla_compat = False
        else:
            print("skipping LSlin & LMMSE")

        if "baseline_lslin_kbest" in methods:
            print("running baseline_lslin_kbest ...")
            sn.config.xla_compat = True

            sys_parameters = Parameters(
                config_name,
                training=False,
                num_tx_eval=num_tx_eval,
                system="baseline_lslin_kbest",
            )
            sys_parameters = set_eval_params(sys_parameters, args)

            e2e_baseline = E2E_Model(
                sys_parameters,
                training=False,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

            print("\nRunning:", sys_parameters.system)

            run_model_eval(
                model=e2e_baseline,
                snr_values=ebno_db,
                batch_size=batch_size,
                graph_mode="xla",
                results=results,
                num_tx_eval=num_tx_eval,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
                eval_mode=eval_mode,
                distribute=distribute,
            )
            save_results(results_filename, results)
            sn.config.xla_compat = False
        else:
            print("skipping LSlin & K-Best")

        if "baseline_lmmse_kbest" in methods:
            print("running baseline_lmmse_kbest ...")
            sn.config.xla_compat = False

            sys_parameters = Parameters(
                config_name,
                training=False,
                num_tx_eval=num_tx_eval,
                system="baseline_lmmse_kbest",
            )
            sys_parameters = set_eval_params(sys_parameters, args)

            e2e_baseline = E2E_Model(
                sys_parameters,
                training=False,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

            print("\nRunning:", sys_parameters.system)

            use_small_batch = eval_mode == "ber"
            run_model_eval(
                model=e2e_baseline,
                snr_values=ebno_db,
                batch_size=batch_size_small if use_small_batch else batch_size,
                graph_mode="graph",
                results=results,
                num_tx_eval=num_tx_eval,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
                eval_mode=eval_mode,
                distribute=None if eval_mode == "ber" else distribute,
            )
            save_results(results_filename, results)
            sn.config.xla_compat = False
        else:
            print("skipping LMMSE & KBest")

        if "baseline_perf_csi_lmmse" in methods:
            print("running baseline_perf_csi_lmmse ...")

            sys_parameters = Parameters(
                config_name,
                training=False,
                num_tx_eval=num_tx_eval,
                system="baseline_perf_csi_lmmse",
            )
            sys_parameters = set_eval_params(sys_parameters, args)

            e2e_baseline = E2E_Model(
                sys_parameters,
                training=False,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

            print("\nRunning:", sys_parameters.system)

            run_model_eval(
                model=e2e_baseline,
                snr_values=ebno_db,
                batch_size=batch_size,
                graph_mode="graph",
                results=results,
                num_tx_eval=num_tx_eval,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
                eval_mode=eval_mode,
                distribute=None,
            )
            save_results(results_filename, results)
        else:
            print("skipping Perfect CSI & LMMSE")

        if "baseline_lmmse_lmmse" in methods:
            print("running baseline_lmmse_lmmse ...")
            sn.config.xla_compat = False

            sys_parameters = Parameters(
                config_name,
                training=False,
                num_tx_eval=num_tx_eval,
                system="baseline_lmmse_lmmse",
            )
            sys_parameters = set_eval_params(sys_parameters, args)

            e2e_baseline = E2E_Model(
                sys_parameters,
                training=False,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

            print("Running:", sys_parameters.system)

            use_small_batch = eval_mode == "ber"
            run_model_eval(
                model=e2e_baseline,
                snr_values=ebno_db,
                batch_size=batch_size_small if use_small_batch else batch_size,
                graph_mode="graph",
                results=results,
                num_tx_eval=num_tx_eval,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
                eval_mode=eval_mode,
                distribute=None if eval_mode == "ber" else distribute,
            )
            save_results(results_filename, results)
            sn.config.xla_compat = False
        else:
            print("skipping LMMSE")

        if "baseline_perf_csi_kbest" in methods:
            print("running baseline_perf_csi_kbest ...")
            sn.config.xla_compat = False

            sys_parameters = Parameters(
                config_name,
                training=False,
                num_tx_eval=num_tx_eval,
                system="baseline_perf_csi_kbest",
            )
            sys_parameters = set_eval_params(sys_parameters, args)

            e2e_baseline = E2E_Model(
                sys_parameters,
                training=False,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

            print("\nRunning:", sys_parameters.system)

            use_small_batch = eval_mode == "ber"
            run_model_eval(
                model=e2e_baseline,
                snr_values=ebno_db,
                batch_size=batch_size_small if use_small_batch else batch_size,
                graph_mode="graph",
                results=results,
                num_tx_eval=num_tx_eval,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
                eval_mode=eval_mode,
                distribute=distribute,
            )
            save_results(results_filename, results)
            sn.config.xla_compat = False
        else:
            print("skipping Perfect CSI & K-Best")
