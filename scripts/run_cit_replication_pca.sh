#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l mem=22GB
#PBS -l nodes=1:ppn=4
#PBS -q small
#PBS -m be
#PBS -M william.casazza@stat.ubc.ca

source activate tf_vae

cd /zfs3/users/william.casazza/william.casazza/vaecit/scripts
python rosmap_cit_replication.py \
  --m-file="$HOME/methylationSNMnormpy.mat" \
  --ac-file="/zfs3/scratch/saram_lab/ROSMAP/data/acetylationNorm.mat" \
  --exp-file="/zfs3/scratch/saram_lab/ROSMAP/data/expressionAndPhenotype.mat" \
  --genotype-dir="/zfs3/scratch/saram_lab/ROSMAP/data/genotypeImputed/1kg/snpMatrix/" \
  --genotype-file="/zfs3/users/william.casazza/william.casazza/vaecit/scripts/cit_genotypes.csv" \
  --snp-coords="/zfs3/scratch/saram_lab/ROSMAP/data/genotypeImputed/1kg/snpPos/" \
  --cit-tests="$HOME/vaecit/CIT.txt" \
  --lv-method="pca" \
  --num-latent=1 \
  --out-name="pca_1_latent_fix_3_cit.csv"
