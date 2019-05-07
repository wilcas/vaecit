"""
The goal is to generate a series of regression results under different causal scenarios, using
simulated genotype PCs as surrogates for genotype.
"""
import cit
import csv
import joblib

import data_model as dm
import numpy as np


def compute_genotype_pcs(genotype):
    """
    Run PCA on genotype and return PCs and their percent variance explained
    """
    (U, D, vh) = np.linalg.svd(genotype, full_matrices=False, compute_uv=True)
    return U@np.diag(D), D / np.sum(D)


def write_csv(results, vars_explained, filename):
    out_rows = []
    for (res,var_explained) in zip(results,vars_explained):
        cur_row = {}
        for j in range(1,5):
            cur_test = 'test{}'.format(j)
            cur_p = 'p{}'.format(j)
            for key in res[cur_test]:
                if key in ['rss', 'r2']: #single value for test
                    cur_key = '{}_{}'.format(cur_test,key)
                    cur_row[cur_key] = res[cur_test][key]
                else:
                    for k in range(len(res[cur_test][key])):
                        cur_key = '{}_{}{}'.format(cur_test,key,k)
                        cur_row[cur_key] = res[cur_test][key][k]
            cur_row[cur_p] = res[cur_p]
        cur_row['var_explained_pcs'] = var_explained
        cur_row['omni_p'] = res['omni_p']
        out_rows.append(cur_row)
    with open(filename, 'w') as f:
        names = out_rows[0].keys()
        writer = csv.DictWriter(f, names)
        writer.writeheader()
        writer.writerows(out_rows)


def main():
    num_sim = 100
    num_subjects = 500
    num_genotypes = [1,10,50,100]
    pcs_to_test = [1]
    block_structure = {
        "100": [1.0],
        "80-20": [0.8,0.2],
        "50-50": [0.5,0.5],
        "33-33-34": [0.33,0.33,0.34],
        "all-25": [0.25,0.25,0.25,0.25],
        "all-20": [0.2,0.2,0.2,0.2,0.2],
        "all": None
    }

    null_datasets = [
        dm.generate_null(
            n=num_subjects,
            p=num_genotypes,
            genotype = dm.block_genotype(
                n = num_sim,
                p = num_genotype,
                perc = block_strucure[key]))
        ) for (i, key) in zip(range(num_sim), num_genotypes, block_structure.keys())]
    null_PCs = [
        compute_genotype_pcs(genotype)
        for (_, _, genotype) in null_datasets]
    ind1_datasets = [
        dm.generate_null(
            n=num_subjects,
            p=num_genotypes,
            genotype = dm.block_genotype(
                n = num_sim,
                p = num_genotype,
                perc = block_strucure[key]))
        ) for (i, key) in zip(range(num_sim), num_genotypes, block_structure.keys())]
    ind1_PCs = [
        compute_genotype_pcs(genotype)
        for (_, _, genotype) in ind1_datasets]
    ind1_datasets = [
        dm.generate_null(
            n=num_subjects,
            p=num_genotype,
            genotype = dm.block_genotype(
                n = num_sim,
                p = num_genotypes,
                perc = block_strucure[key]))
        ) for (i, key) in zip(range(num_sim), num_genotypes, block_structure.keys())]
    ind1_PCs = [
        compute_genotype_pcs(genotype)
        for (_, _, genotype) in ind1_datasets]


    with joblib.parallel_backend('multiprocessing'):
        for (i,num_genotype,key) in zip(pcs_to_test,num_genotype,block_structure.keys()):
            null_results = joblib.Parallel(n_jobs=-1, verbose=10)(
                    joblib.delayed(cit.cit)(trait, gene_exp, Z[:,0:i],10000)
                    for ((trait, gene_exp, _), (Z,_) in zip(null_datasets,null_PCs)
            )
            null_explained = [np.sum(D[0:i]) for (_,D) in null_PCs]
            write_csv(null_results, null_explained, "cit_null_{}_PCs_{}_gen_{}_split.csv".format(i,num_genotype,key))

            caus1_results = joblib.Parallel(n_jobs=-1, verbose=10)(
                    joblib.delayed(cit.cit)(trait, gene_exp, Z[:,0:i], 10000)
                    for ((trait, gene_exp, _), (Z,_) in zip(caus1_datasets,caus1_PCs)
            )
            caus1_explained = [np.sum(D[0:i]) for (_,D) in caus1_PCs]
            write_csv(caus1_results, caus1_explained, "cit_caus1_{}_PCs_{}_gen_{}_split.csv".format(i,num_genotype,key))
            ind1_results = joblib.Parallel(n_jobs=-1, verbose=10)(
                    joblib.delayed(cit.cit)(trait, gene_exp, Z[:,0:i], 10000)
                    for ((trait, gene_exp, _), (Z,_) in zip(ind1_datasets, ind1_PCs)
            )
            ind1_explained = [np.sum(D[0:i]) for (_,D) in ind1_PCs]
            write_csv(ind1_results, ind1_explained, "cit_ind1_{}_PCs_{}_gen_{}_split.csv".format(i,num_genotype,key))
    return 0



if __name__ == '__main__':
    main()
