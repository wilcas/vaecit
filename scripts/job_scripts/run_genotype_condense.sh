#!/bin/bash
#PBS -l walltime=20:00:00
#PBS -l mem=11gb
#PBS -l nodes=1:ppn=2
#PBS -q small
#PBS -m be
#PBS -M william.casazza@stat.ubc.ca

source activate tf_vae

cd /zfs3/users/william.casazza/william.casazza/vaecit/scripts
python save_cit_genotypes.py
