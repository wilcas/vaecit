"""
The goal is to generate a series of regression results under different causal scenarios, using
simulated genotype PCs as surrogates for genotype.
"""
import cit
import csv 
import joblib
import vae

import data_model as dm
import numpy as np
import tensorflow as tf

from itertools import product

tf.enable_eager_execution()


def train_vae(genotype, *params):
    model = vae.VAE(params)
    model.compile(loss=model.total, optimizer=tf.train.AdamOptimizer(1e-4))
    model.fit(genotype, genotype, epochs = 100, batch_size = 10, verbose=0)
    return model


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
    # Simulation parameters
    num_sim = 100
    num_subjects = 100
    num_genotypes = 200
    depths = [1,2,3] # number of hidden layers
    latent = [1,2,3,4,5] # number of latent variables
    
    # Generate datasets
    null_datasets = [dm.generate_null(n=num_subjects, p=num_genotypes) for i in range(num_sim)]
    caus1_datasets = [dm.generate_caus1(n=num_subjects, p=num_genotypes) for i in range(num_sim)]
    ind1_datasets = [dm.generate_ind1(n=num_subjects, p=num_genotypes) for i in range(num_sim)]

    # Train VAEs
    null_models = {}
    caus1_models = {}
    ind1_models = {}
    for param_set in product([num_genotype], latent, hidden):
        null_models[param_set] = [train_vae(genotype, param_set) for (_, _, genotype) in null_datasets]
        caus1_models[param_set] = [train_vae(genotype, param_set) for (_, _, genotype) in caus1_datasets]
        ind1_models[param_set] = [train_vae(genotype, param_set) for (_, _, genotype) in ind1_datasets]
    
    # Run causal inference test
    with joblib.parallel_backend('loky'):
        for param_set in product([num_genotype], latent, hidden):
            for trial in range(5):
                cur_null_model = null_models[param_set]
                Z_list = [cur_null_model.reparameterize(*cur_null_model.encode(genotype)) for (_,_,genotype) in null_datasets]
                null_results = joblib.Parallel(n_jobs=-1, verbose=10)(
                    joblib.delayed(cit.cit)(trait, gene_exp, Z, 1000)
                    for ((trait, gene_exp, _), Z) in zip(null_datasets, Z_list)
                )
                write_csv(null_results, "cit_null_vae_{}_depth_{}_latent_trial_{}.csv".format(param_set[1], param_set[2], trial)
            
                cur_caus1_model = caus1_models[param_set]
                Z_list = [cur_caus1_model.reparameterize(*cur_caus1_model.encode(genotype)) for (_,_,genotype) in caus1_datasets]
                caus1_results = joblib.Parallel(n_jobs=-1, verbose=10)(
                    joblib.delayed(cit.cit)(trait, gene_exp, Z, 1000)
                    for ((trait, gene_exp, _), Z) in zip(caus1_datasets, Z_list)
                )
                write_csv(caus1_results, "cit_caus1_vae_{}_depth_{}_latent_trial_{}.csv".format(param_set[1], param_set[2], trial)
            
                cur_ind1_model = ind1_models[param_set]
                Z_list = [cur_ind1_model.reparameterize(*cur_ind1_model.encode(genotype)) for (_,_,genotype) in ind1_datasets]
                ind1_results = joblib.Parallel(n_jobs=-1, verbose=10)(
                    joblib.delayed(cit.cit)(trait, gene_exp, Z, 1000)
                    for ((trait, gene_exp, _), Z) in zip(ind1_datasets, Z_list)
                )
                write_csv(ind1_results, "cit_ind1_vae_{}_depth_{}_latent_trial_{}.csv".format(param_set[1], param_set[2], trial)
            
    return 0
    
    
    
if __name__ == '__main__':
    main()