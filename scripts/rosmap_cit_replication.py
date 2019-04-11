import cit
import click
import csv
import joblib
import os
import vae

import data_model as dm
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.enable_eager_execution()
tf.set_random_seed(hash("William Casazza"))


@click.command()
@click.option(
    '--m-file',
    type=str,
    required=True,
    help="Methylation MATLAB data filename")
@click.option(
    '--ac-file',
    type=str,
    required=True,
    help="Acetylation MATLAB data filename")
@click.option(
    '--exp-file',
    type=str,
    help="Expression MATLAB data filename")
@click.option(
    '--genotype-dir',
    type=str,
    required=True,
    help="Directory containing genotype CSVs")
@click.option(
    '--cit-tests',
    type=str,
    required=True,
    help="Filename of manifest containing rsids, probe/peak ids and genes to test for causal mediation.")
@click.option(
    '--snp-coords',
    type=str,
    required=True,
    help="Directory of csv files containing snp coordinates")
@click.option(
    '--lv-method',
    required=True,
    type=click.Choice(['pca', 'mmdvae']))
@click.option(
    '--num-latent',
    type=int,
    required=True)
@click.option(
    '--out-name',
    type=str,
    required=True,
    help="Suffix for output files, no path")
@click.option(
    '--vae-depth',
    type=int,
    default=None)
def main(**opts):
    pcs_to_remove = 10
    # load known data
    (m_samples, m_ids, methylation) = dm.load_methylation(opts['m_file'])
    (ac_samples, ac_ids, acetylation) = dm.load_acetylation(opts['ac_file'])
    (e_samples, e_ids, expression) = dm.load_expression(opts['exp_file'])
    # remove 'hidden' cell-type specific effects
    methylation = dm.standardize_remove_pcs(methylation, pcs_to_remove)
    acetylation = dm.standardize_remove_pcs(acetylation, pcs_to_remove)
    expression = dm.standardize_remove_pcs(expression, pcs_to_remove)
    # get snp coordinates
    coord_files = [os.path.join(opts['snp_coords'],f) for f in os.listdir(opts['snp_coords']) if f.endswith('.csv')]
    coord_df = pd.concat([pd.read_csv(f, header=None, names=["snp", "chr", "pos"]) for f in  coord_files], axis=0, ignore_index = True)
    methyl_results = []
    acetyl_results = []
    # parse out tests by qtl Gene
    tests_df = pd.read_csv(opts['cit_tests'], sep='\t')
    for (gene, df) in tests_df.groupby('gene'):
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
        (m_samples, cur_methylation) = (m_samples[m_idx], methylation[m_idx,:])
        (ac_samples, cur_acetylation) = (ac_samples[ac_idx], acetylation[ac_idx,:])
        (e_samples, cur_expression) = (e_samples[e_idx], expression[e_idx,:])
        (g_samples, cur_genotype) = (g_samples[g_idx], genotype[g_idx,:])
        # reduce genotype
        latent_genotype = dm.reduce_genotype(cur_genotype, opts['lv_method'], opts['num_latent'], opts['vae_depth'])
        # get probes and peaks
        cur_exp = expression[:, e_ids == gene]
        for (_, row) in df.iterrows():
            cur_epigenetic = dm.get_mediator(
                cur_methylation,
                m_ids,
                row.probes.split(","),
                data2=cur_acetylation,
                ids2=ac_ids,
                which_ids2=row.peaks.split(",")
            )

            # run CIT
            mediation_results += [cit.cit(cur_exp, cur_epigenetic, latent_genotype)]
    # generate output
    write_csv(opts['out_name'], mediation_results)
    return 0


if __name__ == '__main__':
    main()
