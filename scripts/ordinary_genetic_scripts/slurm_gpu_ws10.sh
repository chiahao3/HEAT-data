#!/bin/bash
# Activate your environment before submission. Submit from this script directory.
# Supply your cluster partition/account through sbatch options.
#SBATCH -J H_ordinary -o slurm_out.o%j
#SBATCH --time=999:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2

# Activate virtual environment

# Run python3 script: python3 /path/to/main.py /path/to/infile /path/to/outdir
python3 ordinary_genetic_H_sublattice.py --config ordinary_genetic_H_sublattice.yaml
