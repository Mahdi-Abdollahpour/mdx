#!/usr/bin/python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# training of the neural receiver for a given configuration file
# the training loop can be found in utils.training_loop

####################################################################
# Parse args
####################################################################

import argparse
from os.path import exists

parser = argparse.ArgumentParser()
# the config defines the sys parameters
parser.add_argument("-config_name", help="config filename", type=str)
# GPU to use
parser.add_argument("-gpu", help="GPU to use", type=int, default=0)
# Easier debugging with breakpoints when running the code eagerly
parser.add_argument("-debug", help="Set debugging configuration", action="store_true", default=False)
# seed
parser.add_argument("-seed", help="Set seed of training", type=int, default=43)
parser.add_argument("-system", help="Set system of training", type=str, default="nrx")
parser.add_argument("-log_grads", help="log gradients", action="store_true", default=False)
parser.add_argument("-log_grads_key", help="log grads including 'key'", type=str, default=None)

parser.add_argument("-v", help="verbose level", type=int, default=1)

# Dump TF debug info (graph/eager) to trace NaN/Inf sources
parser.add_argument(
    "-dump_debug",
    help="Enable tf.debugging.experimental.enable_dump_debug_info",
    action="store_true",
    default=False,
)
parser.add_argument(
    "-dump_dir",
    help="Directory for TF dump_debug_info",
    type=str,
    default="../logs/debug_trace",
)

# Parse all arguments
args = parser.parse_args()

####################################################################
# Imports and GPU configuration
####################################################################

import os
import sys

script_dir = os.path.abspath(os.path.dirname(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
repo_scripts_dir = os.path.join(repo_root, "scripts")

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Preserve the path assumptions used throughout the repo utilities.
os.chdir(repo_scripts_dir)

import core.runtime as _runtime  # noqa: F401

# Select the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

import tensorflow as tf

if args.dump_debug:
    tf.get_logger().setLevel("ERROR")

    dump_dir = args.dump_dir
    print(f"[train_nrx] TF debug dump enabled -> {dump_dir}")
    tf.debugging.experimental.enable_dump_debug_info(
        dump_dir,
        tensor_debug_mode="FULL_HEALTH",
        circular_buffer_size=1000,
    )

gpus = tf.config.list_physical_devices("GPU")
try:
    print("Only GPU number", args.gpu, "used.")
    tf.config.experimental.set_memory_growth(gpus[0], True)
except RuntimeError as e:
    print(e)

if args.debug:
    tf.config.run_functions_eagerly(True)

from ext.neural_rx.utils.e2e_model import E2E_Model
from utils.model_weights import compute_lr_multipliers, load_or_transfer_weights
from ext.neural_rx.utils.parameters import Parameters
from ext.neural_rx.utils.utils import load_weights, save_weights, training_loop

##################################################################
# Training parameters
##################################################################

# all relevant parameters are defined in the config_file
config_name = args.config_name

# initialize system parameters
sys_parameters = Parameters(config_name,
                            system=args.system,
                            training=True)

label = f"{sys_parameters.label}"
filename = "../weights/" + label + "_weights"
save_format = "pkl"
if args.system == "mdx" or args.system == "deep_echo":
    filename = "../weights/" + label + "_weights.h5"
    save_format = "h5"

training_logdir = "../logs"  # use TensorBoard to visualize

import numpy as np
random_seed = np.random.randint(0, 2**32)
training_seed = random_seed
print(f"training_seed:{training_seed}")
if args.debug:
    training_logdir = training_logdir + "/debug"


#################################################################
# Start training
#################################################################

sys_training = E2E_Model(sys_parameters, training=True)
sys_training(1, 1.)  # run once to init weights in TensorFlow
sys_training.summary()

# Load existing weights or, if specified, transfer weights
transfer_loaded = False
if hasattr(sys_parameters, "transfer_weights_path"):
    paths = sys_parameters.transfer_weights_path

    # `transfer_weights_path` may be a single path or a list of paths
    if isinstance(paths, (list, tuple)):
        missing = [p for p in paths if not exists(p)]
        if missing:
            print(
                "\nTransfer weights path(s) specified but the following do not exist:\n"
                + "\n".join(f"  - {p}" for p in missing)
            )
        else:
            print(
                "\nTransfer weights exist - loading transferred weights from:\n"
                + "\n".join(f"  - {p}" for p in paths)
            )
            start_token_model = getattr(sys_parameters, "start_token_model", None)
            start_token_file = getattr(sys_parameters, "start_token_file", None)

            load_or_transfer_weights(
                sys_training,
                paths,  # list of paths
                start_token_model=start_token_model,
                start_token_file=start_token_file,
                verbose=args.v,
            )
            transfer_loaded = True

    else:
        # Single path
        if exists(paths):
            print(
                "\nTransfer Weights exist - loading transferred weights from:\n"
                f"{paths}"
            )
            start_token_model = getattr(sys_parameters, "start_token_model", None)
            start_token_file = getattr(sys_parameters, "start_token_file", None)

            load_or_transfer_weights(
                sys_training,
                paths,  # single path
                start_token_model=start_token_model,
                start_token_file=start_token_file,
                verbose=args.v,
            )
            transfer_loaded = True
        else:
            print(
                "\nTransfer weights path specified but does not exist. "
                f"The specified filename:\n{paths}"
            )

if not transfer_loaded:
    if exists(filename):
        print("\nWeights exist already - loading stored weights...", end="..")
        load_weights(sys_training, filename)
        print(f"\b loaded from:\n{filename}")
    elif exists(f"{filename}.h5"):
        file_name_ = f"{filename}.h5"
        print("\nWeights exist with h5 format - loading stored weights...", end="..")
        load_weights(sys_training, file_name_, skip_mismatch=False, by_name=False)
        print(f"\b loaded from:\n{file_name_}")
    else:
        print(f"weights do not exist! specified filename:\n{filename}\n", flush=True)

if hasattr(sys_parameters, "mcs_training_snr_db_offset"):
    mcs_training_snr_db_offset = sys_parameters.mcs_training_snr_db_offset
else:
    mcs_training_snr_db_offset = None

if hasattr(sys_parameters, "mcs_training_probs"):
    mcs_training_probs = sys_parameters.mcs_training_probs
else:
    mcs_training_probs = None

if args.v > 2:
    save_weights(sys_training, filename, save_format, args.v)

# run the training / weights are automatically saved
# UEs' MCSs will be drawn randomly
training_loop(sys_training,
              label=label,
              filename=filename,
              training_logdir=training_logdir,
              training_seed=training_seed,
              training_schedule=sys_parameters.training_schedule,
              eval_ebno_db_arr=sys_parameters.eval_ebno_db_arr,
              min_num_tx=sys_parameters.min_num_tx,
              max_num_tx=sys_parameters.max_num_tx,
              sys_parameters=sys_parameters,
              mcs_arr_training_idx=list(range(len(sys_parameters.mcs_index))),  # train with all supported MCSs
              mcs_training_snr_db_offset=mcs_training_snr_db_offset,
              mcs_training_probs=mcs_training_probs,
              transfer_loaded=transfer_loaded,
              xla=sys_parameters.xla,
              save_format=save_format,
              log_grads=args.log_grads,
              grad_log_include_name=args.log_grads_key)
