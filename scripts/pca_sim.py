"""
The goal is to generate a series of regression results under different causal scenarios, using
simulated genotype PCs as surrogates for genotype.
"""
import cit
import csv
import joblib

import data_model as dm
import numpy as np

from scipy.stats import zscore

def compute_genotype_pcs(genotype):
    (U, D, vh) = np.linalg.svd(zscore(genotype), full_matrices=False, compute_uv=True)
    return tmp@vh.T


def write_csv(results, filename):
    out_rows = []
    for res in results:
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
    num_genotypes = 50
    num_bootstrap = None
    pcs_to_test = [1]

    null_datasets = [dm.generate_null(n=num_subjects, p=num_genotypes) for i in range(num_sim)]
    null_PCs = [compute_genotype_pcs(genotype) for (_, _, genotype) in null_datasets]
    caus1_datasets = [dm.generate_caus1(n=num_subjects, p=num_genotypes) for i in range(num_sim)]
    caus1_PCs = [compute_genotype_pcs(genotype) for (_, _, genotype) in caus1_datasets]
    ind1_datasets = [dm.generate_ind1(n=num_subjects, p=num_genotypes) for i in range(num_sim)]
    ind1_PCs = [compute_genotype_pcs(genotype) for (_, _, genotype) in ind1_datasets]

    with joblib.parallel_backend('multiprocessing'):
        for i in pcs_to_test:
            null_results = joblib.Parallel(n_jobs=-1, verbose=10)(
                    joblib.delayed(cit.cit)(trait, gene_exp, Z[:,0:i],None)
                    for ((trait, gene_exp, _), Z) in zip(null_datasets,null_PCs)
                )
            write_csv(null_results, "cit_null_{}_PCs_{}_gen.csv".format(i,num_genotypes))

            caus1_results = joblib.Parallel(n_jobs=-1, verbose=10)(
                    joblib.delayed(cit.cit)(trait, gene_exp, Z[:,0:i], None)
                    for ((trait, gene_exp, _), Z) in zip(caus1_datasets,caus1_PCs)
                )
            write_csv(caus1_results, "cit_caus1_{}_PCs_{}_gen.csv".format(i, num_genotypes))

            ind1_results = joblib.Parallel(n_jobs=-1, verbose=10)(
                    joblib.delayed(cit.cit)(trait, gene_exp, Z[:,0:i], None)
                    for ((trait, gene_exp, _), Z) in zip(ind1_datasets, ind1_PCs)
                )
            write_csv(ind1_results, "cit_ind1_{}_PCs_{}_gen.csv".format(i, num_genotypes))

    return 0



if __name__ == '__main__':
    main()
