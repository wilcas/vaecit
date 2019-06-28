#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l mem=22GB
#PBS -l nodes=1:ppn=4
#PBS -q small
#PBS -m be
#PBS -M william.casazza@stat.ubc.ca
set -e -x
METHODS=( 'mmdvae_warmup' 'mmdvae_batch' 'mmdvae_batch_warmup' 'ae' 'ae_batch' )
METHOD="${METHODS[$1]}"
LATENT=1
DEPTH=5
if [ "$2" == "rev" ]; then
  rev_str="--run-reverse"
else
  rev_str=""
fi  


python rosmap_cit_replication.py \
  --m-file="$HOME/methylationSNMnormpy.mat" \
  --ac-file="$HOME/acetylationNorm.mat" \
  --exp-file="$HOME/expressionAndPhenotype.mat" \
  --genotype-dir="/zfs3/scratch/saram_lab/ROSMAP/data/genotypeImputed/1kg/snpMatrix/" \
  --genotype-file="$HOME/cit_genotypes.csv" \
  --snp-coords="/zfs3/scratch/saram_lab/ROSMAP/data/genotypeImputed/1kg/snpPos/" \
  --cit-tests="$HOME/vaecit/CIT.txt" \
  --lv-method=${METHOD} \
  --num-latent=${LATENT} \
  --vae-depth=${DEPTH} \
  --num-bootstrap=0 \
  --model-dir="/media/wcasazza/DATA2/wcasazza/saved_models_test_training/" \
  ${rev_str} \
  --out-name="${METHOD}_${LATENT}_latent_depth_${DEPTH}_cit.csv"

