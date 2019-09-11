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


def retrieve_latent(df, gene, methyl, acetyl, express, opts, geno):
    if geno is None:
        raise ValueError('require genotype manifest')
    else:
        (g_samples, g_ids, genotype) = geno
        cur_genotype = genotype[:,np.isin(g_ids, df.snp.to_numpy())]
    latent_genotype = dm.reduce_genotype(cur_genotype, opts['lv_method'], opts['num_latent'], gene, opts['vae_depth'], opts['model_dir'])
    if type(latent_genotype) != np.ndarray:
        latent_genotype = latent_genotype.numpy().astype(np.float64)
    return (gene, latent_genotype)


@click.command()
@click.option('--m-file', type=str,required=True,
    help="Methylation MATLAB data filename")
@click.option('--ac-file', type=str, required=True,
    help="Acetylation MATLAB data filename")
@click.option('--exp-file', type=str,
    help="Expression MATLAB data filename")
@click.option('--genotype-file', type=str, required=True)
@click.option('--cit-tests', type=str, required=True,
    help="Filename of manifest containing rsids, probe/peak ids and genes to test for causal mediation.")
@click.option('--lv-method', required=True, type=click.Choice(['pca', 'mmdvae','lfa', 'kernelpca', 'fastica', 'mmdvae_warmup', 'mmdvae_batch', 'mmdvae_batch_warmup', 'ae','ae_batch']))
@click.option('--num-latent', type=int, required=True)
@click.option('--out-name', type=str, required=True,
    help="Suffix for output files, no path")
@click.option('--vae-depth', type=int, default=None)
@click.option('--model-dir')
def main(**opts):
    logging.basicConfig(
        filename="{}_run.log{}".format(
            opts['out_name'].split(".")[0],
            int(time.time())),
        level=logging.WARNING)
    pcs_to_remove = 10
    (m_samples, m_ids, methylation) = dm.load_methylation(opts['m_file'])
    (ac_samples, ac_ids, acetylation) = dm.load_acetylation(opts['ac_file'])
    (e_samples, e_ids, expression) = dm.load_expression(opts['exp_file'])
    # remove 'hidden' cell-type specific effects
    methylation = dm.standardize_remove_pcs(methylation, pcs_to_remove)
    acetylation = dm.standardize_remove_pcs(acetylation, pcs_to_remove)
    mask = ~np.all(np.isnan(acetylation),axis=0)
    acetylation = acetylation[:, mask]
    ac_ids = ac_ids[mask]
    expression = dm.standardize_remove_pcs(expression, pcs_to_remove)

    # run tests by qtl Gene
    tests_df = pd.read_csv(opts['cit_tests'], sep='\t')
    if pd.isna(tests_df).any().any():
        tests_df[pd.isna(tests_df)] = ""
    if opts['genotype_file'] is not None:
        genotype_df = pd.read_csv(opts['genotype_file'], index_col=0)
        genotype = genotype_df.to_numpy().T
        g_samples = genotype_df.columns.to_numpy()
        g_ids = genotype_df.index.to_numpy()
        (m_idx, ac_idx, e_idx, g_idx) =  dm.match_samples(m_samples, ac_samples, e_samples, g_samples)
        (g_samples, genotype) = (g_samples[g_idx], genotype[g_idx,:])
        (m_samples, methylation) = (m_samples[m_idx], methylation[m_idx,:])
        (ac_samples, acetylation) = (ac_samples[ac_idx], acetylation[ac_idx,:])
        (e_samples, expression) = (e_samples[e_idx], expression[e_idx,:])
        geno = (g_samples, g_ids, genotype)
        methyl = (m_samples, m_ids, methylation)
        acetyl = (ac_samples, ac_ids, acetylation)
        express = (e_samples, e_ids, expression)
    else:
        raise ValueError("require genotype manifest file")
    with joblib.parallel_backend("loky"):
       mediation_results = joblib.Parallel(n_jobs=6, verbose=10)(
           joblib.delayed(retrieve_latent)(df, gene, methyl, acetyl, express, opts, geno)
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
