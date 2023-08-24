"""
The goal is to generate a series of regression results under different causal scenarios, using
simulated genotype PCs as surrogates for genotype.
"""
import cit_sm as cit
import csv
import joblib

import data_model as dm
import numpy as np

from scipy.stats import zscore
from itertools import product

def compute_genotype_pcs(genotype):
    """
    Run PCA on genotype and return PCs and their percent variance explained
    """
    # A = zscore(genotype)
    (U, D, vh) = np.linalg.svd(genotype, full_matrices=False, compute_uv=True)
    return genotype@vh.T, D / np.sum(D)


def write_csv(results, vars_explained, filename):
    for (i,res) in enumerate(results):
        res['var_explained_pcs'] = vars_explained[i]
    cit.write_csv(results,filename)


def main():
    num_sim = 100
    num_subjects = 500
    num_genotypes = [10]
    pcs_to_test = [1]
    bootstraps = None
    block_structure = {
        "100": [1.0],
        "80-20": [0.8,0.2],
        "50-50": [0.5,0.5],
        "33-33-34": [0.33,0.33,0.34],
        "all-25": [0.25,0.25,0.25,0.25],
        "all-20": [0.2,0.2,0.2,0.2,0.2],
        "all": None
    }

    null_datasets = {
        (i, num_genotype, key):
        [dm.generate_null(
            n=num_subjects,
            p=num_genotype,
            genotype = dm.block_genotype(
                n = num_subjects,
                p = num_genotype,
                perc = block_structure[key])) for j in range(num_sim)]
        for (i, num_genotype, key) in product(pcs_to_test,num_genotypes, block_structure.keys())}
    null_PCs = {
        k: [compute_genotype_pcs(null_datasets[k][j][2]) for j in range(num_sim)]
        for k in null_datasets.keys()}
    caus1_datasets ={
        (i, num_genotype, key):
        [dm.generate_caus1(
            n=num_subjects,
            p=num_genotype,
            genotype = dm.block_genotype(
                n = num_subjects,
                p = num_genotype,
                perc = block_structure[key]))  for j in range(num_sim)]
        for (i, num_genotype, key) in product(pcs_to_test, num_genotypes, block_structure.keys())}
    caus1_PCs = {
        k: [compute_genotype_pcs(caus1_datasets[k][j][2]) for j in range(num_sim)]
        for k in caus1_datasets.keys()}
    ind1_datasets = {
        (i, num_genotype, key):
        [dm.generate_ind1(
            n=num_subjects,
            p=num_genotype,
            genotype = dm.block_genotype(
                n = num_subjects,
                p = num_genotype,
                perc = block_structure[key])) for j in range(num_sim)]
        for (i,num_genotype, key) in product(pcs_to_test,num_genotypes, block_structure.keys())}
    ind1_PCs = {
        k: [compute_genotype_pcs(ind1_datasets[k][j][2]) for j in range(num_sim)]
        for k in ind1_datasets.keys()}


    with joblib.parallel_backend('loky'):
        for (i, num_genotype, key) in product(pcs_to_test, num_genotypes, block_structure.keys()):
            data_key = (i, num_genotype, key)
            null_results = joblib.Parallel(n_jobs=-1, verbose=10)(
                    joblib.delayed(cit.cit)(trait, gene_exp, Z[:,0:i],bootstraps)
                    for ((trait, gene_exp, _), (Z,_)) in zip(null_datasets[data_key],null_PCs[data_key])
            )
            null_explained = [np.sum(D[0:i]) for (_,D) in null_PCs[data_key]]
            write_csv(null_results, null_explained, "cit_null_{}_PCs_{}_gen_{}_split.csv".format(i,num_genotype,key))

            caus1_results = joblib.Parallel(n_jobs=-1, verbose=10)(
                    joblib.delayed(cit.cit)(trait, gene_exp, Z[:,0:i], bootstraps)
                    for ((trait, gene_exp, _), (Z,_)) in zip(caus1_datasets[data_key],caus1_PCs[data_key])
            )
            caus1_explained = [np.sum(D[0:i]) for (_,D) in caus1_PCs[data_key]]
            write_csv(caus1_results, caus1_explained, "cit_caus1_{}_PCs_{}_gen_{}_split.csv".format(i,num_genotype,key))
            ind1_results = joblib.Parallel(n_jobs=-1, verbose=10)(
                    joblib.delayed(cit.cit)(trait, gene_exp, Z[:,0:i], bootstraps)
                    for ((trait, gene_exp, _), (Z,_)) in zip(ind1_datasets[data_key], ind1_PCs[data_key])
            )
            ind1_explained = [np.sum(D[0:i]) for (_,D) in ind1_PCs[data_key]]
            write_csv(ind1_results, ind1_explained, "cit_ind1_{}_PCs_{}_gen_{}_split.csv".format(i,num_genotype,key))
    return 0



if __name__ == '__main__':
    main()