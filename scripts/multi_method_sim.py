"""
The goal is to generate a series of regression results under different causal scenarios, using
simulated genotype PCs as surrogates for genotype.
"""
import cit_sm as cit
import click
import csv
import joblib
import os

import vae_torch as vt
import torch
import data_model as dm
import numpy as np

from itertools import product

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


@click.command()
@click.option('--lv-method', type=click.Choice(['pca', 'mmdvae','lfa', 'kernelpca', 'fastica', 'mmdvae_warmup', 'mmdvae_batch', 'mmdvae_batch_warmup', 'ae','ae_batch']))
@click.option('--num-latent', type=int, required=True)
@click.option('--out-name', type=str, required=True,
    help="Suffix for output files, no path")
@click.option('--vae-depth', type=int, default=None)
@click.option('--model-dir')
def main(opts):
    # Simulation parameters
    num_sim = 100
    num_subjects = 500
    num_genotypes = 50
    num_bootstrap = None
    depths = [5] # number of hidden layers
    latent = [1] # number of latent variables

    # Generate datasets
    null_datasets = [
        dm.generate_null(n=num_subjects, p=num_genotypes)
        for i in range(num_sim)]
    caus1_datasets = [
        dm.generate_caus1(n=num_subjects, p=num_genotypes)
        for i in range(num_sim)]
    ind1_datasets = [
        dm.generate_ind1(n=num_subjects, p=num_genotypes)
        for i in range(num_sim)]

    # Train VAEs
    num_cpu = 16 #int(os.cpu_count() / 2)
    dm.reduce_genotype(cur_genotype, opts['lv_method'], opts['num_latent'], gene, opts['vae_depth'], opts['model_dir'])
    null_z = joblib.Parallel(n_jobs=num_cpu, verbose=10)(
        joblib.delayed(dm.reduce_genotype)((genotype, opts['lv_method'], opts['num_latent'], "null_" + str(i), opts['vae_depth'], opts['model_dir']))
        for (i,(_, _, genotype)) in enumerate(null_datasets))

    caus1_z = joblib.Parallel(n_jobs=num_cpu, verbose=10)(
        joblib.delayed(dm.reduce_genotype)((genotype, opts['lv_method'], opts['num_latent'], "caus1_" + str(i), opts['vae_depth'], opts['model_dir']))
        for (i, (_, _, genotype)) in enumerate(caus1_datasets))

    ind1_z = joblib.Parallel(n_jobs=num_cpu, verbose=10)(
        joblib.delayed(dm.reduce_genotype)((genotype, opts['lv_method'], opts['num_latent'], "ind1_" + str(i), opts['vae_depth'], opts['model_dir']))
        for (i, (_, _, genotype)) in enumerate(ind1_datasets))

    # Run causal inference test
    with joblib.parallel_backend('loky'):
        for param_set in product([num_genotypes], latent, depths):
            null_results = joblib.Parallel(n_jobs=-1, verbose=10)(
                joblib.delayed(cit.cit)(trait, gene_exp, Z, None)
                for ((trait, gene_exp, _), Z) in zip(null_datasets, null_z)
            )
            write_csv(
                null_results,
                "cit_null_mmdvae_{}_depth_{}_latent_{}_gen.csv".format(
                    param_set[2],
                    param_set[1],
                    num_genotypes))

            caus1_results = joblib.Parallel(n_jobs=-1, verbose=10)(
                joblib.delayed(cit.cit)(trait, gene_exp, Z, None)
                for ((trait, gene_exp, _), Z) in zip(caus1_datasets, caus1_z)
            )
            write_csv(
                caus1_results,
                "cit_caus1_mmdvae_{}_depth_{}_latent_{}_gen.csv".format(
                    param_set[2],
                    param_set[1],
                    num_genotypes))

            ind1_results = joblib.Parallel(n_jobs=-1, verbose=10)(
                joblib.delayed(cit.cit)(trait, gene_exp, Z, None)
                for ((trait, gene_exp, _), Z) in zip(ind1_datasets, ind1_z)
            )
            write_csv(
                ind1_results,
                "cit_ind1_{}_{}_gen.csv".format(
                    opts['lv_method'],
                    num_genotypes))

    return 0



if __name__ == '__main__':
    main()
