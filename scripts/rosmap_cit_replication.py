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
@click.add_option(
    '--m-file',
    type=str,
    help="Methylation MATLAB data filename")
@click.add_option(
    '--ac-file',
    type=str,
    help="Acetylation MATLAB data filename")
@click.add_option(
    '--exp-file',
    type=str,
    help="Expression MATLAB data filename")
@click.add_option(
    '--genotype-dir',
    type=str,
    help="Directory containing genotype CSVs")
@click.add_option(
    '--cit-tests',
    type=str,
    help="Filename of manifest containing rsids, probe/peak ids and genes to test for causal mediation.")
@click.add_option(
    '--snp-coords',
    type=str,
    help="Filename of file containing snp coordinates")
@click.add_option(
    '--lv-method',
    type=click.Choice(['pca', 'mmdvae']))
@click.add_option(
    '--num-latent',
    type=int)
@click.add_option(
    '--out-name',
    type=str,
    help="Suffix for output files, no path")
@click.add_option(
    '--vae-depth',
    type=int,
    default=None)
def main():
    # load known data
    (m_samples, m_ids, methylation) = dm.load_methylation(m_file)
    (ac_samples, ac_ids, acetylation) = dm.load_acetylation(ac_file)
    (e_samples, e_ids, expression) = dm.load_expression(exp_file)
    methyl_results = []
    acetyl_results = []
    # parse out tests by qtl Gene
    tests_df = pd.read_csv(cit_tests)
    for (gene, df) in tests_df.group_by('gene'):
        # get load in genotypes associated with gene
        snp_files = dm.get_snp_groups(df.snp, snp_coords, genotype_dir)
        df['fname'] = snp_files
        g_ids = np.array()
        genotype = np.array()
        for (snp_file, snp_df) in df.group_by('fname'):
            g_samples, g_ids_cur, genotype_cur = dm.load_genotype(snp_file,snp_df.snp)
            genotype = np.concatenate((genotype,genotype_cur),axis = 1)
            g_ids = np.concatenate((g_ids, g_ids_cur), axis = 0)
        # match samples
        (m_idx, ac_idx, e_idx, g_idx) =  dm.match_samples(m_samples, ac_samples, e_samples, g_samples)
        (m_samples, methylation) = (m_samples[m_idx], methylation[m_idx,:])
        (ac_samples, acetylation) = (ac_samples[ac_idx], acetylation[ac_idx,:])
        (e_samples, expression) = (e_samples[e_idx], expression[e_idx,:])
        (g_samples, genotype) = (g_samples[g_idx], genotype[g_idx,:])
        # reduce genotype
        latent_genotype = dm.reduce_genotype(genotype, lv_method, num_latent, vae_depth)

        cur_exp = expression[:, e_ids == row.gene]
        for (_, row) in df.iterrows():
            cur_methyl = dm.get_mediator(
                methylation,
                m_ids,
                row.probes.split(","),
                row.nProbes)
            cur_acetyl = dm.get_mediator(
                acetylation,
                ac_ids,
                row.peaks.split(","),
                row.nPeaks)
            # run CIT
            methyl_results += [cit.cit(cur_exp, cur_methyl, latent_genotype)]
            acetyl_results += [cit.cit(cur_exp, cur_acetyl, latent_genotype)]
    # generate output
    write_csv("methyl_" + out_name, methyl_results)
    write_csv("acetyl_" + out_name, acetyl_results)
    return 0


if __name__ == '__main__':
    main()
