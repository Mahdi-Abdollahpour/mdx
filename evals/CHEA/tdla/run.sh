#!/bin/bash

# CHEA (arXiv:2607.16462, PIMRC 2026 Workshops), Fig. 3:
# channel-estimation MSE vs. SNR, 4x2 MU-MIMO with 2 users on TDL-A,
# (a) 10 PRBs and (b) 22 PRBs.
#
# The *_results files are written next to this script; nb_chea.ipynb plots them.
# Trained weights are loaded from <repo>/weights/<label>_weights.h5.
#
# Usage: ./run.sh [-gpu <id>]

# =======================================================================
# Setup
# =======================================================================
gpu=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -gpu)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for -gpu" >&2
                exit 1
            fi
            gpu="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# --- Path Discovery ---

dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)/
find_repo_root() {
    local current="${dir%/}"
    while [[ "${current}" != "/" ]]; do
        if [[ -f "${current}/scripts/evaluate_metrics.py" ]]; then
            printf '%s\n' "${current}"
            return 0
        fi
        current=$(dirname "${current}")
    done

    echo "Could not find repo root containing scripts/evaluate_metrics.py" >&2
    return 1
}
repo_root=$(find_repo_root) || exit 1

# --- Run Setup ---

run_eval() {
    (cd "${repo_root}/scripts" && python3 evaluate_metrics.py "$@")
}
echo "dir: ${dir}"

# Shared by all runs: TDL-A, 2 active users, SNR from -10 to 30 dB in 1 dB steps.
common=(
    -gpu="${gpu}" -dir="${dir}" -num_tx_eval 2
    -channel_type_eval=NTDLlow -channel_models A -max_ut_velocity_eval=34
    -snr_db_eval_min=-10 -snr_db_eval_max=31 -snr_db_eval_stepsize=1
    -num_target_block_errors=500 -target_bler=1e-4
)

# =======================================================================
# LS and LMMSE baselines
# =======================================================================
# baselines_4x2_chea.cfg evaluates MCS indices [9, 14, 19]; the paper uses
# MCS 14, i.e. -mcs_arr_eval_idx 1.
baselines=(
    -config_name=baselines_4x2_chea.cfg -mcs_arr_eval_idx=1
    -max_mc_iter=500 -target_mse=1e-4 -target_squared_error=2.5e6
)

# --- 10 PRBs ---
run_eval "${common[@]}" "${baselines[@]}" -n_size_bwp_eval=10 -batch_size_eval=30 \
    -methods baseline_lslin_lmmse -eval_mode=both -name_suffix=_10_ls
run_eval "${common[@]}" "${baselines[@]}" -n_size_bwp_eval=10 -batch_size_eval=3 \
    -methods baseline_lmmse_lmmse -eval_mode=mse -name_suffix=_10_lmmse

# --- 22 PRBs ---
run_eval "${common[@]}" "${baselines[@]}" -n_size_bwp_eval=22 -batch_size_eval=30 \
    -methods baseline_lslin_kbest -eval_mode=both -name_suffix=_22_ls
run_eval "${common[@]}" "${baselines[@]}" -n_size_bwp_eval=22 -batch_size_eval=3 \
    -methods baseline_lmmse_kbest -eval_mode=both -name_suffix=_22_lmmse

# =======================================================================
# Learned estimators (block graphs hosted by DeepEcho)
# =======================================================================
# These configs are trained for MCS 14 only, i.e. -mcs_arr_eval_idx 0.
models=(
    -methods deep_echo -mcs_arr_eval_idx=0 -eval_mode=mse
    -batch_size_eval=30 -target_mse=1e-7
)

for prb in 10 22; do
    # --- InterpolateNet ---
    run_eval "${common[@]}" "${models[@]}" -n_size_bwp_eval="${prb}" \
        -config_name="interpnet${prb}.cfg" -max_mc_iter=500 -target_squared_error=5.5e6

    # --- HA02, Channelformer, CEViT ---
    for config in "ha02_${prb}" "channelformer${prb}" "cevit${prb}"; do
        run_eval "${common[@]}" "${models[@]}" -n_size_bwp_eval="${prb}" \
            -config_name="${config}.cfg" -max_mc_iter=1000 -target_squared_error=5e6
    done

    # --- CHEA (chea16d) and CHEA-XL (chea64) ---
    for config in chea16d chea64; do
        run_eval "${common[@]}" "${models[@]}" -n_size_bwp_eval="${prb}" \
            -config_name="${config}.cfg" -name_suffix="_${prb}" \
            -max_mc_iter=1000 -target_squared_error=5.5e6
    done
done
