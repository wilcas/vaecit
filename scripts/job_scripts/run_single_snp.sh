#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l mem=22GB
#PBS -l nodes=1:ppn=4
#PBS -q small
#PBS -m be
#PBS -M william.casazza@stat.ubc.ca
set -e -x
SINGLE="$1"
if [ "$2" == "rev" ]; then
  rev_str="--run-reverse"
else
  rev_str=""
fi  


python rosmap_cit_complete_rep.py \
  --m-file="/media/wcasazza/DATA2/wcasazza/ROSMAP/methylationSNMnormpy.mat" \
  --ac-file="/media/wcasazza/DATA2/wcasazza/ROSMAP/acetylationNorm.mat" \
  --exp-file="/media/wcasazza/DATA2/wcasazza/ROSMAP/expressionAndPhenotype.mat" \
  --genotype-dir="/zfs3/scratch/saram_lab/ROSMAP/data/genotypeImputed/1kg/snpMatrix/" \
  --genotype-file="/media/wcasazza/DATA2/wcasazza/ROSMAP/cit_genotypes.csv" \
  --snp-coords="/zfs3/scratch/saram_lab/ROSMAP/data/genotypeImputed/1kg/snpPos/" \
  --cit-tests="$HOME/vaecit/CIT.txt" \
  --num-bootstrap=0 \
  ${rev_str} \
  --out-name="replication_single_snp_${SINGLE}_cit.csv" \
  --separate-epigenetic=${SINGLE}
