#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l mem=22GB
#PBS -l nodes=1:ppn=4
#PBS -q small
#PBS -m be
#PBS -M william.casazza@stat.ubc.ca
source activate tf_vae

cd /zfs3/users/william.casazza/william.casazza/vaecit/scripts

python get_latent_embeddings.py \
  --genotype-file="/zfs3/users/william.casazza/william.casazza/vaecit/scripts/cit_genotypes.csv" \
  --cit-tests="$HOME/vaecit/CIT.txt" \
  --lv-method="pca" \
  --vae-depth=10 \
  --num-latent=1 \
  --out-name="lvs_1kg_cit_replication.csv"
