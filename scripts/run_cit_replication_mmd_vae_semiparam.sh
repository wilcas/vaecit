#!/bin/bash
python rosmap_cit_replication.py \
  --m-file="$HOME/methylationSNMnormpy.mat" \
  --ac-file="$HOME/acetylationNorm.mat" \
  --exp-file="$HOME/expressionAndPhenotype.mat" \
  --genotype-dir="/zfs3/scratch/saram_lab/ROSMAP/data/genotypeImputed/1kg/snpMatrix/" \
  --genotype-file="/home/wcasazza/cit_genotypes.csv" \
  --snp-coords="" \
  --cit-tests="$HOME/vaecit/CIT.txt" \
  --lv-method="mmdvae" \
  --num-latent=1 \
  --vae-depth=10 \
  --num-bootstrap=0 \
  --out-name="mmdvae_1_latent_depth_10_cit_50_epochs.csv"
