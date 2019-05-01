"""
This Module serves to generate datasets under different causal structures. It mimics the simulation file in the R
version of this project.
"""
import csv
import h5py
import logging
import numpy as np
import pandas as pd
import os
import re
import random
import torch
import vae_torch as vt


from scipy import io,stats
from scipy.sparse.linalg import svds
from functools import reduce


def generate_null(n=100, p=200):
    trait = np.random.normal(size=(n,))
    genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    gene_exp = np.random.normal(size=(n,))
    return trait.reshape(n,1), gene_exp.reshape(n,1), genotype.astype(np.float64)


def generate_caus1(n=100, p=200):
    genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    exp_coeffs= np.array([(random.choice([-1,1])*np.random.uniform())for i in range(p)])
    gene_exp = random.choice([-1,1])*np.random.uniform() + (genotype@exp_coeffs) + np.random.normal(size=(n,))
    trait = random.choice([-1,1])*np.random.uniform() + random.choice([-1,1])*np.random.uniform() * gene_exp + np.random.normal(size=(n,))
    return trait.reshape(n,1), gene_exp.reshape(n,1), genotype.astype(np.float64)


def generate_ind1(n=100, p=200):
    genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    exp_coeffs= np.array([random.choice([-1,1])*np.random.uniform() for i in range(p)])
    gene_exp = random.choice([-1,1])*np.random.uniform()+(genotype@exp_coeffs)  + np.random.normal(size=(n,))
    trait_coeffs= np.array([random.choice([-1,1])*np.random.uniform() for i in range(p)])
    trait = random.choice([-1,1])*np.random.uniform()+(genotype@trait_coeffs)+ np.random.normal(size=(n,))
    return trait.reshape(n,1), gene_exp.reshape(n,1), genotype.astype(np.float64)


def generate_caus1_scale_kernel(n=100, p=200):
    """Make full mediation data where gene expression is generated from multivariate normal with  kernel covariance

        Returns:
            The return value: A tuple of numpy.ndarray for trait, gene expression, and genotype data

    """
    genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    K = genotype@genotype.T
    gene_exp = np.array([[1] for i in range(n)]) +\
        np.random.multivariate_normal(mean=np.zeros(n), cov=10*K).reshape(n,1) +\
        np.random.normal(size=(n, 1))
    trait = np.array([[2] for i in range(n)]) + 10 * gene_exp + np.random.normal(size=(n, 1))
    return trait, gene_exp, genotype.astype(np.float64)


def load_expression(fname):
    exp_and_phen = io.loadmat(fname)
    expression = exp_and_phen['data'][0][0][0]
    samples = exp_and_phen['data'][0][0][4]
    samples = np.array([re.sub(":.*","",sample[0][0]) for sample in samples])
    genes = exp_and_phen['data'][0][0][3]
    genes = np.array([re.sub(":.*","",gene[0][0]) for gene in genes])
    return samples.flatten(), genes.flatten(), expression


def load_genotype(fname,rsids):
    rsids = np.unique(rsids) #unique rsids
    if re.match(".*\.raw$", fname): #plink format
        sep = ' '
        with open(fname, 'r') as f:
            rsids_base = f.readline().split(sep=sep)
        rsids_file = [re.sub("_.*","",rsid) for rsid in rsids_base]
        rsids_file = np.array(rsids_file)
        rsid_idx = [np.nonzero(rsids_file == rsid)[0][0] for rsid in rsids]
        snps = np.array(rsids_file)[rsid_idx]
        not_found = [rsid for rsid in rsids if rsid not in rsids_file]
        samples = pd.read_csv(fname,sep=sep, skiprows=1, usecols=[0],header=None).to_numpy().flatten()
        samples_idx = [(("ROS" or "MAP") in sample) for sample in samples]
        samples = np.array([re.sub("[A-Z]*","",item) for item in samples[samples_idx]])
        genotype = pd.read_csv(fname, skiprows=1, usecols=rsid_idx, sep=sep,header=None).to_numpy()[samples_idx,:]
    elif re.match(".*\.csv$", fname): #csv
        sep = ','
        df = pd.read_csv(fname)
        rsids_file = df.index.to_numpy().flatten()
        samples = df.columns.to_numpy().flatten()
        not_found = [rsid for rsid in rsids if rsid not in rsids_file]
        rsids = np.setdiff1d(rsids, np.array(not_found))
        genotype = df.loc[rsids.tolist(),].T
        samples = genotype.index.to_numpy()
        snps = genotype.columns.to_numpy()
        genotype = genotype.to_numpy()
    else:
        raise NotImplementedError("Format of {} not recognized".format(fname))
    if len(not_found) > 0:
        logging.warning("rsids {} not found in {}".format(not_found, fname))
    return samples.flatten(), snps.flatten(), genotype


def load_methylation(fname):
    with h5py.File(fname, 'r') as h5:
        samples = bytes(h5['methyList'][()]).decode().replace("\x00","").split(",")
        probe_ids = bytes(h5['probeNames'][()]).decode().replace("\x00","").split(",")
        methylation = np.zeros(h5['methy'].shape,dtype = 'float32')
        h5['methy'].read_direct(methylation)
    return np.array(samples), np.array(probe_ids), methylation.T


def load_acetylation(fname):
    acetyl_obj = io.loadmat(fname)
    acetylation = acetyl_obj['acety']
    samples = np.array([tmp[0][0] for tmp in acetyl_obj["acetyList"]])
    peak_ids = np.array([tmp[0][0] for tmp in acetyl_obj["peakNames"]])
    return samples.flatten(), peak_ids.flatten(), acetylation


def standardize_remove_pcs(data, k):
    data_norm = stats.zscore(data)
    u,d,vt = svds(data_norm,k)
    return data_norm - (u @ np.diag(d) @ vt)


def match_samples(*samples):
    """Sort data by sample ids"""
    shared = reduce(np.intersect1d, samples)
    shared_idx = []
    for sample_vec in samples:
        indices = sample_vec.argsort()
        to_keep = np.array([elem in shared for elem in  sample_vec[indices]])
        shared_idx.append(np.extract(to_keep, indices))
    return shared_idx


def get_snp_groups(rsids, coord_df, genotype_dir, sep='\t'):
    """Get genotype file containing each rsid"""
    snp_files = []
    if re.match(".*hrc.*",genotype_dir): #plink raw, sample by snp
        for rsid in rsids:
            chrom = coord_df[coord_df['snp']== rsid]['chr'].values[0]
            snp_files.append(os.path.join(genotype_dir, "chr{}.raw".format(chrom)))
    elif re.match(".*1kg.*",genotype_dir): #csv file, snp by sample
        prev_chrom = ""
        for rsid in rsids:
            chrom = coord_df[coord_df['snp']== rsid]['chr'].values[0]
            if chrom != prev_chrom:
                fstring = os.path.join(genotype_dir, "snpMatrixChr{}{}.csv")
                fA = fstring.format(chrom,"a")
                fB = fstring.format(chrom,"b")
                file_rsids = pd.read_csv(fA, sep=',', skiprows=1, usecols=[0], header=None)
            if rsid in file_rsids.to_numpy().flatten():
                snp_files.append(fA)
            else:
                snp_files.append(fB)
            prev_chrom = chrom
    else:
        raise ValueError("Invalid Genotype group: {}".format(genotype_group))
    return snp_files


def compute_pcs(A):
    n = A.shape[0]
    A_std = stats.zscore(A)
    try:
        (U, D, vh) = np.linalg.svd(A_std, full_matrices=False, compute_uv=True)
    except np.linalg.LinAlgError:
        A2 = np.conj(A_std.T)@A_std
        (U,D,vh) = np.linalg.svd(A_std, full_matrices=False, compute_uv=True)
    return U@np.diag(D)


def get_mediator(data, ids, which_ids, data2= None, ids2 = None, which_ids2 = None):
    feature_idx = np.isin(ids, which_ids)
    tmp_data = data[:,feature_idx]
    if not np.isscalar(data2) and data2 != None:
        feature_idx2 = np.isin(ids2, which_ids2)
        tmp_data2 = data2[:, feature_idx2]
        cur_data = np.concatenate((tmp_data, tmp_data2), axis=1)
        cur_data = compute_pcs(cur_data)[:, 0]
    else:
        cur_data = compute_pcs(tmp_data)[:, 0]
    return cur_data


def reduce_genotype(genotype, lv_method, num_latent, vae_depth=None):
    if genotype.shape[1] == 1:
        return genotype
    num_latent = min(genotype.shape[1],num_latent)
    if lv_method == 'mmdvae':
        params = {
            "size": genotype.shape[1],
            "num_latent": num_latent,
            "depth": vae_depth}
        model = vt.train_mmd_vae(torch.Tensor(stats.zscore(genotype)), params)
        latent_genotype = model.encode(torch.Tensor(stats.zscore(genotype))).detach()
    elif lv_method == "pca":
        latent_genotype = compute_pcs(genotype)[:, 0:num_latent]
    else:
        raise NotImplemented
    return latent_genotype
