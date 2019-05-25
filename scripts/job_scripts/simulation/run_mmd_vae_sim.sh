#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-saram
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=127000M
#SBATCH --ntasks-per-node=16

source $HOME/scratch/tensorflow/bin/activate
python $HOME/projects/def-saram/wcasazza/vaecit/scripts/mmd_vae_sim.py
