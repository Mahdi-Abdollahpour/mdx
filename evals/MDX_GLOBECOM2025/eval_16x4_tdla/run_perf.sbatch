#!/bin/bash
#SBATCH --job-name=perf
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --environment=tf
#SBATCH --account=***
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mahdi.abdollahpour@unibo.it


# Mahdi Abdollahpour (mahdi.abdollahpour@unibo.it)
# 2025

cd "${SLURM_SUBMIT_DIR}" # submit from evals/MDX_GLOBECOM2025/eval_16x4_tdla

set -x

./run_perf_gpu0.sh &

./run_perf_gpu1.sh &

./run_perf_gpu2.sh &

./run_perf_gpu3.sh


wait
echo "[sbatch] LMMSE: All GPUs Finished!"