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


def train_vae(genotype, params):
    model = vae.VAE(*params)
    model.compile(loss=model.total_loss, optimizer=tf.train.AdamOptimizer(1e-4))
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
    '--vae-depth',
    type=int)
def main():
    # load known data
    (m_samples, m_ids, methylation) = dm.load_methylation(m_file)
    (ac_samples, ac_ids, acetylation) = dm.load_acetylation(ac_file)
    (e_samples, e_ids, expression) = dm.load_expression(exp_file)
    # parse out tests by qtl Gene
    pd.read_csv()
    # load in genotypes

    sample_ids = dm.match_samples(m_samples, ac_samples, e_samples, g_samples)
    # match ids and process data
    # run CIT

    # generate output

    return 0



if __name__ == '__main__':
    main()
