#!/bin/bash

# MDELAN (arXiv:2607.15127, SPAWC 2026), MSE_MCS9_suburban.png:
# channel-estimation MSE vs. SNR on the QuaDRiGa LEO NTN suburban dataset,
# 2x1 SIMO with 1 user, 11 PRBs, MCS 9.
#
# The evaluation dataset is read from <repo>/data/leo_ntn_channels_v73_suburban.mat.
# Trained weights are loaded from <repo>/weights/<label>_weights.h5.
# The *_results files are written next to this script; eval_2x1_ntn_suburban.ipynb
# plots them.
#
# Usage: ./run_suburban.sh [-gpu <id>]

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

# Shared by all runs: suburban dataset, 1 user, MCS 9 (-mcs_arr_eval_idx 0),
# 11 PRBs, SNR from -10 dB in 1 dB steps. Each SNR point runs the full
# -max_mc_iter budget: the squared-error stop is set out of reach, as for the
# released results.
#
# The released *_results files use an older list layout that
# evaluate_metrics.py cannot extend, so the first run writing each file
# passes -dont_load and starts it afresh.
common=(
    -gpu="${gpu}" -dir="${dir}" -num_tx_eval 1 -mcs_arr_eval_idx=0
    -channel_type_eval=OFDMDataset -n_size_bwp_eval=11 -batch_size_eval=30
    -max_ut_velocity_eval=34 -snr_db_eval_min=-10 -snr_db_eval_stepsize=1
    -eval_mode=mse -target_mse=1e-4 -target_squared_error=1e12
    -mat_filename_eval=leo_ntn_channels_v73_suburban.mat
)

# =======================================================================
# Runs
# =======================================================================
# --- LS and LMMSE -> baselines_2x1_suburban_results ---
run_eval "${common[@]}" -config_name=baselines_2x1.cfg -name_suffix=_suburban \
    -methods baseline_lslin_lmmse -snr_db_eval_max=21 -max_mc_iter=500 -dont_load
run_eval "${common[@]}" -config_name=baselines_2x1.cfg -name_suffix=_suburban \
    -methods baseline_lmmse_kbest -snr_db_eval_max=21 -max_mc_iter=200 \
    -num_cov_samples=100000

# --- MDX and MDX fine-tuned on suburban ---
run_eval "${common[@]}" -config_name=mdx_ntn.cfg -name_suffix=_original_suburban_new \
    -methods deep_echo -snr_db_eval_max=21 -max_mc_iter=1000 -dont_load
run_eval "${common[@]}" -config_name=mdx_ntn_fsuburban.cfg -name_suffix=_original_suburban_new \
    -methods deep_echo -snr_db_eval_max=12 -max_mc_iter=1000 -dont_load

# --- MDX:MDELAN and MDX:MDELAN fine-tuned on suburban ---
run_eval "${common[@]}" -config_name=mdx_ntn_mdelan2f.cfg -name_suffix=_suburban_new \
    -methods deep_echo -snr_db_eval_max=18 -max_mc_iter=2000 -dont_load
run_eval "${common[@]}" -config_name=mdx_ntn_mdelan2f_fsuburban.cfg -name_suffix=_suburban_new \
    -methods deep_echo -snr_db_eval_max=12 -max_mc_iter=1000 -dont_load
