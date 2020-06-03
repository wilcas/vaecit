#!/bin/bash
#SBATCH -p p100
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=7-00:00:00
#SBATCH --output=%N-%j.out

source /h/wcasazza/miniconda3/bin/activate 
conda activate torch_stat

cd /h/wcasazza/vaecit/scripts/
python vary_num_genotypes.py fix_effects
