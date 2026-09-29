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

# This script computes the covariance matrix for LMMSE channel estimation
# the matrices are stored in the weights/ folder

####################################################################
# Parse args
####################################################################

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-config_name", help="config filename", type=str)
parser.add_argument("-num_samples", help="Number of samples",
                    type=int, default=1000000)
parser.add_argument("-gpu", help="GPU to use", type=int, default=0)
parser.add_argument("-num_tx_eval", help="Number of active users",
                    type=int, default=1)
parser.add_argument("-n_size_bwp_eval", type=int, default=132)
# Optional BS array override (as in evaluate_metrics.py); the covariance
# matrices must match the array used for evaluation. Give all three together.
parser.add_argument("-num_rx_antennas", type=int, default=None)
parser.add_argument("-num_rows_per_panel", type=int, default=None)
parser.add_argument("-num_cols_per_panel", type=int, default=None)

# Parse all arguments
args = parser.parse_args()
config_name = args.config_name
num_tx_eval = args.num_tx_eval

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

# Avoid warnings from TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

import sionna as sn

sn.config.xla_compat = True
from sionna.channel import GenerateOFDMChannel, gen_single_sector_topology

from ext.neural_rx.utils.parameters import (
    Parameters, cov_mat_paths, cov_mat_provenance)
import json
import numpy as np

##################################################################
# Setup link
##################################################################
parameters = Parameters(config_name,
                        training=False,
                        num_tx_eval=num_tx_eval,
                        system="nrx",
                        compute_cov=True)  # load UMi channel in any case
parameters.re_init(n_size_bwp_eval=args.n_size_bwp_eval,
                   num_rx_antennas=args.num_rx_antennas,
                   num_rows_per_panel=args.num_rows_per_panel,
                   num_cols_per_panel=args.num_cols_per_panel)

batch_size = parameters.batch_size_eval
if hasattr(parameters, "batch_size_cov"):
    batch_size = parameters.batch_size_cov

NUM_SAMPLES = args.num_samples
# run multiple iterations to limit the batchsize/memory requirements
NUM_IT = int((NUM_SAMPLES // batch_size) + 1)

channel_type = parameters.channel_type
if parameters.channel_type == "OFDMDataset":
    gen_ofdm_channel = parameters.channel_model
else:
    channel_model = parameters.channel_model

    # OFDM channel in frequency domain
    gen_ofdm_channel = GenerateOFDMChannel(
        channel_model,
        parameters.transmitters[0]._resource_grid,
        normalize_channel=True,
    )

#################################################################
# Evaluate covariance matrices
#################################################################


def sample_channel(batch_size):
    # Sample a random network topology for each batch example
    if parameters.channel_type in ("UMi", "UMa"):
        topology = gen_single_sector_topology(
            batch_size,
            1,
            "umi",
            min_ut_velocity=parameters.min_ut_velocity,
            max_ut_velocity=parameters.max_ut_velocity,
        )
        channel_model.set_topology(*topology)

    # Sample channel frequency response
    # [batch size, 1, num_rx_ant, 1, 1, num_ofdm_symbols, fft_size]
    h_freq = gen_ofdm_channel(batch_size)
    # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
    h_freq = h_freq[:, 0, :, 0, 0]

    return h_freq


@tf.function(jit_compile=True)
def estimate_cov_mats(batch_size, num_it):
    rg = parameters.transmitters[0]._resource_grid
    freq_cov_mat = tf.zeros([rg.fft_size, rg.fft_size], tf.complex64)
    time_cov_mat = tf.zeros([rg.num_ofdm_symbols, rg.num_ofdm_symbols],
                            tf.complex64)
    space_cov_mat = tf.zeros([parameters.num_rx_antennas,
                              parameters.num_rx_antennas], tf.complex64)

    for _ in tf.range(num_it):
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        h_samples = sample_channel(batch_size)

        # Frequency covariance matrix estimation
        # [batch size, num_rx_ant, fft_size, num_ofdm_symbols]
        h_samples_ = tf.transpose(h_samples, [0, 1, 3, 2])
        # [batch size, num_rx_ant, fft_size, fft_size]
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [fft_size, fft_size]
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0, 1))
        freq_cov_mat += freq_cov_mat_

        # Time covariance matrix estimation
        # [batch size, num_rx_ant, num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0, 1))
        time_cov_mat += time_cov_mat_

        # Spatial covariance matrix estimation
        # [batch size, num_ofdm_symbols, num_rx_ant, fft_size]
        h_samples_ = tf.transpose(h_samples, [0, 2, 1, 3])
        # [batch size, num_ofdm_symbols, num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.reduce_mean(space_cov_mat_, axis=(0, 1))
        space_cov_mat += space_cov_mat_

    freq_cov_mat /= tf.complex(tf.cast(rg.num_ofdm_symbols * num_it, tf.float32), 0.0)
    time_cov_mat /= tf.complex(tf.cast(rg.fft_size * num_it, tf.float32), 0.0)
    space_cov_mat /= tf.complex(tf.cast(rg.fft_size * num_it, tf.float32), 0.0)
    return freq_cov_mat, time_cov_mat, space_cov_mat


freq_cov_mat, time_cov_mat, space_cov_mat = estimate_cov_mats(batch_size, NUM_IT)
freq_cov_mat = freq_cov_mat.numpy()
time_cov_mat = time_cov_mat.numpy()
space_cov_mat = space_cov_mat.numpy()

# Save covariance matrices to the weights directory. Filenames encode the PRB
# count, BS array and number of users (see cov_mat_key() in utils/parameters.py).
# Arrays are written atomically and the metadata sidecar last, so an
# interrupted run never leaves a set that looks complete.
paths = cov_mat_paths(parameters)
os.makedirs(os.path.dirname(paths["freq"]), exist_ok=True)

arrays = {"freq": freq_cov_mat, "time": time_cov_mat, "space": space_cov_mat}
for name, arr in arrays.items():
    tmp = paths[name] + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, paths[name])
    print(f"wrote {paths[name]}  {arr.shape}")

meta = cov_mat_provenance(parameters,
                          num_samples=NUM_SAMPLES,
                          batch_size=int(batch_size),
                          num_it=int(NUM_IT))
tmp = paths["meta"] + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, sort_keys=True)
os.replace(tmp, paths["meta"])
print(f"wrote {paths['meta']}")
