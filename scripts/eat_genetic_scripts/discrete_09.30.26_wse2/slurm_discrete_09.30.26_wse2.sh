#!/usr/bin/env bash
#SBATCH --job-name=wse2-discrete
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=wse2-discrete-%j.log
set -euo pipefail
# Submit from this directory. Supply your cluster partition/account with sbatch.
script_dir="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
exec bash "$script_dir/run_discrete_09.30.26_wse2.sh" "$@"
