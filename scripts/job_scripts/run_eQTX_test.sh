#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l mem=22GB
#PBS -l nodes=1:ppn=4
#PBS -q small
#PBS -m be
#PBS -M william.casazza@stat.ubc.ca
#PBS -t 1-21
cd $HOME/vaecit/scripts/
source activate tf_vae
python eQTX_window_test.py \
  --m-file="$HOME/methylationSNMnormpy.mat" \
  --ac-file="/zfs3/scratch/saram_lab/ROSMAP/data/acetylationNorm.mat" \
  --exp-file="/zfs3/scratch/saram_lab/ROSMAP/data/expressionAndPhenotype.mat" \
  --probe-map-file="/zfs3/scratch/saram_lab/ROSMAP/data/mapping/methyToGeneByChr/methyToGeneChr${PBS_ARRAYID}_1MB.mat" \
  --peak-map-file="/zfs3/scratch/saram_lab/ROSMAP/data/mapping/acetyToGeneByChr/acetyToGeneChr${PBS_ARRAYID}_1MB.mat" \
  --out-name="chr${PBS_ARRAYID}_eQTX.csv"
