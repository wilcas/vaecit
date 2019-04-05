#!/bin/bash
#PBS -l walltime=20:00:00
#PBS -l nodes=1:ppn=6
#PBS -q medium
#PBS -m be
#PBS -M william.casazza@stat.ubc.ca

source activate tf_vae

cd /zfs3/users/william.casazza/william.casazza/vaecit/scripts
python rosmap_cit_replication.py \
  --m-file="$HOME/methylationSNMnormpy.mat" \
  --ac-file="/zfs3/scratch/saram_lab/ROSMAP/data/acetylationNorm.mat" \
  --exp-file="/zfs3/scratch/saram_lab/ROSMAP/data/expressionAndPhenotype.mat" \
  --genotype-dir="/zfs3/scratch/saram_lab/ROSMAP/data/genotypeImputed/1kg/snpMatrix/" \
  --cit-tests="$HOME/vaecit/CIT.txt" \
  --lv-method="pca" \
  --num-latent=3 \
  --out-name="pca_3_latent_cit.csv"
