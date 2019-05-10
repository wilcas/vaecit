import pandas as pd
import os
import re
import numpy as np

def select_genotypes(path, snps, fname):
    lines = []
    for f in os.listdir(path):
        with open(os.path.join(path,f), 'r') as reader:
            samples = reader.readline()
            for line in reader:
                parsed = line.split(",")
                if parsed[0] in snps:
                    lines.append(line)
    with open(fname,'w+') as writer:
        writer.write(samples)
        writer.writelines(lines)

def select_genotypes_plink(path, snps, fname):
    dfs = []
    for f in os.listdir(path):
        with open(os.path.join(path,f), 'r') as reader:
            # get columns
            rsid_base = reader.readline().split(" ")
            rsids = [re.sub("_.*","",rsid) for rsid in rsid_base]
            rsid_idx = [i for (i,rsid) in enumerate(rsids) if rsid in snps]
        tmp_df = pd.read_csv(os.path.join(path,f),sep=" ", usecols=rsid_idx)
        dfs.append(tmp_df)
    pd.concat(dfs,axis=1).T.to_csv(fname)

if __name__ == "__main__":
    cit_table = pd.read_csv("/zfs3/users/william.casazza/william.casazza/vaecit/CIT.txt", sep = '\t')
    # df = select_genotypes("/zfs3/scratch/saram_lab/ROSMAP/data/genotypeImputed/1kg/snpMatrix/",cit_table.snp.unique(), "cit_genotypes.csv")
    select_genotypes_plink("/zfs3/scratch/saram_lab/ROSMAP/data/genotypeImputed/hrc/snpMatrix/",cit_table.snp.unique(), "cit_genotypes_hrc.csv")
