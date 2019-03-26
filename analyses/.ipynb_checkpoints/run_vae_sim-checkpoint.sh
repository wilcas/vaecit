#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-saram
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=127000M
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

source $HOME/projects/def-saram/wcasazza/tensorflow/bin/activate
python $HOME/projects/def-saram/wcasazza/vaecit/scripts/vae_sim.py
