import data_model as dm

import click
import csv
import gc
import joblib
import logging
import os
import time
import torch

import numpy as np
import pandas as pd


def retrieve_latent(df, gene, opts, geno=None):
    if geno is None:
        raise ValueError('require genotype manifest')
    else:
        (g_samples, g_ids, genotype) = geno
        cur_genotype = genotype[:,np.isin(g_ids, df.snp.to_numpy())]
    latent_genotype = dm.reduce_genotype(cur_genotype, opts['lv_method'], opts['num_latent'], gene, opts['vae_depth'])
    if type(latent_genotype) != np.ndarray:
        latent_genotype = latent_genotype.numpy().astype(np.float64)
    return (gene, latent_genotype)


@click.command()
@click.option('--genotype-file', type=str, required=True)
@click.option('--cit-tests', type=str, required=True,
    help="Filename of manifest containing rsids, probe/peak ids and genes to test for causal mediation.")
@click.option('--lv-method', required=True, type=click.Choice(['pca', 'mmdvae']))
@click.option('--num-latent', type=int, required=True)
@click.option('--out-name', type=str, required=True,
    help="Suffix for output files, no path")
@click.option('--vae-depth', type=int, default=None)
def main(**opts):
    logging.basicConfig(
        filename="{}_run.log{}".format(
            opts['out_name'].split(".")[0],
            int(time.time())),
        level=logging.WARNING)

    # run tests by qtl Gene
    tests_df = pd.read_csv(opts['cit_tests'], sep='\t')
    if opts['genotype_file'] is not None:
        genotype_df = pd.read_csv(opts['genotype_file'], index_col=0)
        genotype = genotype_df.to_numpy().T
        g_samples = genotype_df.columns.to_numpy()
        g_ids = genotype_df.index.to_numpy()
        geno = (g_samples, g_ids, genotype)
    else:
        raise ValueError("require genotype manifest file")
    with joblib.parallel_backend("loky"):
       mediation_results = joblib.Parallel(n_jobs=-1, verbose=10)(
           joblib.delayed(retrieve_latent)(df, gene, opts, geno)
           for (gene, df) in tests_df.groupby('gene')
       )
    # mediation_results = [cit_on_qtl_set(df,gene,coord_df,methyl,acetyl,express,opts,geno) for (gene,df) in tests_df.groupby('gene')] # SEQUENTIAL VERSION
    genes = { gene: [elem[0] for elem in lv.tolist()] for (gene, lv) in mediation_results}
    out_df = pd.DataFrame(genes)
    out_df.index = g_samples
    # generate output
    out_df.to_csv("{}_{}".format(opts['lv_method'], opts['out_name']))


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    main()
