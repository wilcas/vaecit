#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l mem=22GB
#PBS -l nodes=1:ppn=4
#PBS -q small
#PBS -m be
#PBS -M william.casazza@stat.ubc.ca
LV_METHODS=( 'pca'  'mmdvae' 'lfa'  'kernelpca'  'fastica'  'mmdvae_warmup'  'mmdvae_batch'  'mmdvae_batch_warmup'  'ae' 'ae_batch' )
cd ~/vaecit/scripts

python get_latent_embeddings.py \
  --m-file="$HOME/methylationSNMnormpy.mat" \
  --ac-file="$HOME/acetylationNorm.mat" \
  --exp-file="$HOME/expressionAndPhenotype.mat" \
  --genotype-file="$HOME/cit_genotypes.csv" \
  --cit-tests="/home/wcasazza/eQTX_manifest.txt" \
  --lv-method="${LV_METHODS[$1]}" \
  --vae-depth=5 \
  --num-latent=1 \
  --model-dir="/media/wcasazza/DATA2/wcasazza/saved_models_eQTX/" \
  --out-name="lvs_1kg_cit_replication.csv"
