#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-saram
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64000M
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

source $HOME/projects/def-saram/wcasazza/tensorflow/bin/activate
python $HOME/projects/def-saram/wcasazza/vaecit/scripts/mmd_vae_sim.py

