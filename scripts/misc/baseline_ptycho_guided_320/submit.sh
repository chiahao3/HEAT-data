#!/bin/bash
#SBATCH --job-name=wse2-guided-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/wse2-guided-mpl
cd "${SLURM_SUBMIT_DIR:-$(dirname -- "$0")}"
python3 -u run_baseline.py
