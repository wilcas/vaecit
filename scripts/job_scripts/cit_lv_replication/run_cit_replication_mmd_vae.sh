#!/bin/bash
#PBS -l walltime=20:00:00
#PBS -l mem=22gb
#PBS -l select=2:ncpus=4
#PBS -q small
#PBS -m be
#PBS -M william.casazza@stat.ubc.ca

source activate tf_vae
cd /zfs3/users/william.casazza/william.casazza/vaecit/scripts
mpiexec -npernode 4 --oversubscribe python rosmap_cit_replication.py \
  --m-file="$HOME/methylationSNMnormpy.mat" \
  --ac-file="/zfs3/scratch/saram_lab/ROSMAP/data/acetylationNorm.mat" \
  --exp-file="/zfs3/scratch/saram_lab/ROSMAP/data/expressionAndPhenotype.mat" \
  --genotype-dir="/zfs3/scratch/saram_lab/ROSMAP/data/genotypeImputed/1kg/snpMatrix/" \
  --snp-coords="/zfs3/scratch/saram_lab/ROSMAP/data/genotypeImputed/1kg/snpPos/" \
  --cit-tests="$HOME/vaecit/CIT.txt" \
  --lv-method="mmdvae" \
  --num-latent=1 \
  --vae-depth=10 \
  --out-name="mmdvae_1_latent_depth_10_cit.csv"
