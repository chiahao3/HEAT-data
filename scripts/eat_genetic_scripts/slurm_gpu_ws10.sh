#!/bin/bash
# Activate your environment before submission. Submit from this script directory.
# Supply your cluster partition/account through sbatch options.
#SBATCH -J H_sublatt -o slurm_out.o%j
#SBATCH --time=999:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2

# Activate virtual environment

# Run python3 script: python3 /path/to/main.py /path/to/infile /path/to/outdir
python3 eat_genetic_08.27.26_H_sublattice.py --config eat_genetic_08.27.26_H_sublattice.yaml
