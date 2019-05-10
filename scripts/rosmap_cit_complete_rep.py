import cit
import click
import csv
import gc
import joblib
import logging
import os
import time
import torch

import data_model as dm
import numpy as np
import pandas as pd


def cit_on_qtl_set(df, gene, coord_df, methyl, acetyl, express, opts, geno=None):
    (m_samples, m_ids, methylation) = methyl
    (ac_samples, ac_ids, acetylation) = acetyl
    (e_samples, e_ids, expression) = express
    gc.collect()
    if geno is None:
        # get load in genotypes associated with gene
        snp_files = dm.get_snp_groups(df.snp.values, coord_df, opts['genotype_dir'])
        df['fname'] = snp_files
        groups = iter(df.groupby('fname'))
        # grab from first file
        (snp_file, snp_df) = next(groups)
        g_samples, g_ids, genotype = dm.load_genotype(snp_file,snp_df.snp.values)
        # grab from remaining files
        for (snp_file, snp_df) in groups:
            _, g_ids_cur, genotype_cur = dm.load_genotype(snp_file,snp_df.snp.values)
            genotype = np.concatenate((genotype,genotype_cur),axis = 1)
            g_ids = np.concatenate((g_ids, g_ids_cur), axis = 0)
        # match samples
        (m_idx, ac_idx, e_idx, g_idx) =  dm.match_samples(m_samples, ac_samples, e_samples, g_samples)
        (m_samples, methylation) = (m_samples[m_idx], methylation[m_idx,:])
        (ac_samples, acetylation) = (ac_samples[ac_idx], acetylation[ac_idx,:])
        (e_samples, expression) = (e_samples[e_idx], expression[e_idx,:])
        (g_samples, genotype) = (g_samples[g_idx], genotype[g_idx,:])
    else:
        (g_samples, g_ids, genotype) = geno

    n = expression.shape[0]
    cur_exp = expression[:, e_ids == gene]
    mediation_results = []
    for (_, row) in df.iterrows():
        cur_epigenetic = dm.get_mediator(
            methylation,
            m_ids,
            row.probes.split(","),
            data2=acetylation,
            ids2=ac_ids,
            which_ids2=row.peaks.split(",")
        )
        # run CIT
        if opts['run_reverse']:
            mediation_results.append(
                cit.cit(
                    cur_epigenetic.reshape(n,1),
                    cur_exp.reshape(n,1),
                    genotype[:, g_ids == row.snp],
                    num_bootstrap=opts['num_bootstrap']))
        else:
            mediation_results.append(
                cit.cit(
                    cur_exp.reshape(n,1),
                    cur_epigenetic.reshape(n,1),
                    genotype[:, g_ids == row.snp],
                    num_bootstrap=opts['num_bootstrap']))
    return mediation_results


@click.command()
@click.option('--m-file', type=str,required=True,
    help="Methylation MATLAB data filename")
@click.option('--ac-file', type=str, required=True,
    help="Acetylation MATLAB data filename")
@click.option('--exp-file', type=str,
    help="Expression MATLAB data filename")
@click.option('--genotype-dir', type=str, required=True,
    help="Directory containing genotype CSVs")
@click.option('--genotype-file', type=str, default=None)
@click.option('--cit-tests', type=str, required=True,
    help="Filename of manifest containing rsids, probe/peak ids and genes to test for causal mediation.")
@click.option('--snp-coords', type=str, required=True,
    help="Directory of csv files containing snp coordinates")
@click.option('--out-name', type=str, required=True,
    help="Suffix for output files, no path")
@click.option('--num-bootstrap', type=int, default = 100000)
@click.option('--run-reverse', default=False, is_flag=True)
def main(**opts):
    logging.basicConfig(
        filename="{}_run.log{}".format(
            opts['out_name'].split(".")[0],
            int(time.time())),
        level=logging.WARNING)
    if opts['num_bootstrap'] < 1:
        opts['num_bootstrap'] = None
    pcs_to_remove = 10
    # load known data
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
    # get snp coordinates
    coord_files = [os.path.join(opts['snp_coords'],f) for f in os.listdir(opts['snp_coords']) if f.endswith('.csv')]
    coord_df = pd.concat([pd.read_csv(f, header=None, names=["snp", "chr", "pos"]) for f in  coord_files], axis=0, ignore_index = True)

    # run tests by qtl Gene
    tests_df = pd.read_csv(opts['cit_tests'], sep='\t')
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
        methyl = (m_samples, m_ids, methylation)
        acetyl = (ac_samples, ac_ids, acetylation)
        express = (e_samples, e_ids, expression)
        geno = None
    with joblib.parallel_backend("loky"):
       mediation_results = joblib.Parallel(n_jobs=-1, verbose=10)(
           joblib.delayed(cit_on_qtl_set)(df, gene, coord_df, methyl, acetyl, express, opts, geno)
           for (gene, df) in tests_df.groupby('gene')
        )
    # mediation_results = [cit_on_qtl_set(df,gene,coord_df,methyl,acetyl,express,opts, geno) for (gene,df) in tests_df.groupby('gene')] # SEQUENTIAL VERSION
    merged_results = [item for sublist in mediation_results for item in sublist]
    # generate output
    if opts['run_reverse']:
        opts['out_name'] = "rev_" + opts['out_name']
    if opts['num_bootstrap'] is None:
        opts['out_name'] = "perm_test_" + opts['out_name']
    cit.write_csv(merged_results, opts['out_name'])
    return 0


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    main()