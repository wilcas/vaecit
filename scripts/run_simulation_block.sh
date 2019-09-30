#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=42:00:00
#SBATCH --output=%N-%j.out

module load miniconda3
source activate vae_torch

cd /home/wcasazza/scratch/vaecit/scripts/
python vary_block_structure.py
