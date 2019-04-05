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
    help="Methylation MATLAB data filename")
@click.option(
    '--ac-file',
    type=str,
    help="Acetylation MATLAB data filename")
@click.option(
    '--exp-file',
    type=str,
    help="Expression MATLAB data filename")
@click.option(
    '--genotype-dir',
    type=str,
    help="Directory containing genotype CSVs")
@click.option(
    '--cit-tests',
    type=str,
    help="Filename of manifest containing rsids, probe/peak ids and genes to test for causal mediation.")
@click.option(
    '--snp-coords',
    type=str,
    help="Directory of csv files containing snp coordinates")
@click.option(
    '--lv-method',
    type=click.Choice(['pca', 'mmdvae']))
@click.option(
    '--num-latent',
    type=int)
@click.option(
    '--out-name',
    type=str,
    help="Suffix for output files, no path")
@click.option(
    '--vae-depth',
    type=int,
    default=None)
def main(**opts):
    # load known data
    (m_samples, m_ids, methylation) = dm.load_methylation(opts['m_file'])
    (ac_samples, ac_ids, acetylation) = dm.load_acetylation(opts['ac_file'])
    (e_samples, e_ids, expression) = dm.load_expression(opts['exp_file'])
    coord_files = [os.path.join(opts['snp_coords'],f) for f in os.listdir(opts['snp_coords']) if f.endswith('.csv')]
    coord_df = pd.concat([pd.read_csv(f, header=None, names=["snp", "chr", "pos"]) for f in  coord_files], axis=0, ignore_index = True)
    methyl_results = []
    acetyl_results = []
    # parse out tests by qtl Gene
    tests_df = pd.read_csv(opts['cit_tests'], sep='\t')
    for (gene, df) in tests_df.groupby('gene'):
        # get load in genotypes associated with gene
        snp_files = dm.get_snp_groups(df.snp.as_matrix(), coord_df, opts['genotype_dir'])
        df['fname'] = snp_files
        g_ids = np.array()
        genotype = np.array()
        for (snp_file, snp_df) in df.groupby('fname'):
            g_samples, g_ids_cur, genotype_cur = dm.load_genotype(snp_file,snp_df.snp.as_matrix())
            genotype = np.concatenate((genotype,genotype_cur),axis = 1)
            g_ids = np.concatenate((g_ids, g_ids_cur), axis = 0)
        # match samples
        (m_idx, ac_idx, e_idx, g_idx) =  dm.match_samples(m_samples, ac_samples, e_samples, g_samples)
        (m_samples, methylation) = (m_samples[m_idx], methylation[m_idx,:])
        (ac_samples, acetylation) = (ac_samples[ac_idx], acetylation[ac_idx,:])
        (e_samples, expression) = (e_samples[e_idx], expression[e_idx,:])
        (g_samples, genotype) = (g_samples[g_idx], genotype[g_idx,:])
        # reduce genotype
        latent_genotype = dm.reduce_genotype(genotype, opts['lv_method'], opts['num_latent'], opts['vae_depth'])

        cur_exp = expression[:, e_ids == row.gene]
        for (_, row) in df.iterrows():
            cur_methyl = dm.get_mediator(
                methylation,
                m_ids,
                row.probes.split(","))
            cur_acetyl = dm.get_mediator(
                acetylation,
                ac_ids,
                row.peaks.split(","))
            # run CIT
            methyl_results += [cit.cit(cur_exp, cur_methyl, latent_genotype)]
            acetyl_results += [cit.cit(cur_exp, cur_acetyl, latent_genotype)]
    # generate output
    write_csv("methyl_" + opts['out_name'], methyl_results)
    write_csv("acetyl_" + opts['out_name'], acetyl_results)
    return 0


if __name__ == '__main__':
    main()
