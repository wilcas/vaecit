import pandas as pd
import os
import re
import numpy as np
import joblib

def select_genotypes(path, snps, fname):
    def closure(i,f):
        lines = []
        with open(os.path.join(path,f), 'r') as reader:
            samples = reader.readline()
            for line in reader:
                parsed = line.split(",")
                if parsed[0] in snps:
                    lines.append(line)
        with open("{}_{}".format(i,fname),'w+') as writer:
            if i  == 0:
                writer.write(samples)
            writer.writelines(lines)
        
    with joblib.parallel_backend("loky"):
        all_lines = joblib.Parallel(n_jobs=-1, verbose=10)(joblib.delayed(closure)(i, f) for (i, f) in enumerate(os.listdir(path)))


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
    # get samples
    with open(os.path.join(path,f),"r") as reader:
        reader.readline()
        sample_names = []
        rosmap_samples = []
        for (i,row) in enumerate(reader):
            sample = row.split(" ")[0]
            if re.match("ROS|MAP|11AD",sample):
                rosmap_samples.append(i)
                sample = re.sub("ROS|MAP|11AD","",sample)
            sample_names.append(sample)
    df = pd.concat(dfs,axis=1).T
    df.columns = sample_names
    df.index =df.index.str.replace("_[A-Z]","")
    df.iloc[:,rosmap_samples].to_csv(fname)

if __name__ == "__main__":
    #cit_table = pd.read_csv("/zfs3/users/william.casazza/william.casazza/vaecit/CIT.txt", sep = '\t')
    cit_f = "/zfs3/users/william.casazza/william.casazza/eQTX_manifest.txt"
    snps = []
    with open(cit_f,'r') as f:
        f.readline()
        for line in f:
            snps.extend(line.split()[1].split(","))
    df = select_genotypes("/zfs3/scratch/saram_lab/ROSMAP/data/genotypeImputed/1kg/snpMatrix/",pd.Series(snps).unique(), "cit_genotypes.csv")
    #select_genotypes_plink("/zfs3/scratch/saram_lab/ROSMAP/data/genotypeImputed/hrc/snpMatrix/",cit_table.snp.unique(), "cit_genotypes_hrc.csv")
