#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-saram
#SBATCH --mem=64000M
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

source $HOME/scratch/tensorflow/bin/activate
python $HOME/projects/def-saram/wcasazza/vaecit/scripts/pca_sim.py
