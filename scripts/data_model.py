"""
This Module serves to generate datasets under different causal structures. It mimics the simulation file in the R
version of this project.
"""
import csv
import h5py
import numpy as np
import os
import re
import vae

from scipy import io,stats
from functools import reduce

def generate_null(n=100, p=200):
    trait = np.random.normal(size=(n, 1))
    genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    gene_exp = np.random.normal(size=(n, 1))
    return trait, gene_exp, genotype.astype(np.float64)


def generate_caus1(n=100, p=200):
    genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    gene_exp = np.array([[1] for i in range(n)])+(genotype@np.array([[2] for i in range(p)])) + np.random.normal(size=(n, 1))
    trait = np.array([[1] for i in range(n)]) + 1 * gene_exp + np.random.normal(size=(n,1))
    return trait, gene_exp, genotype.astype(np.float64)


def generate_ind1(n=100, p=200):
    genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    gene_exp = np.array([[1] for i in range(n)])+(genotype@np.array([[-1] for i in range(p)])) + np.random.normal(size=(n, 1))
    trait = np.array([[2] for i in range(n)])+(genotype@np.array([[20] for i in range(p)])) + np.random.normal(size=(n, 1))
    return trait, gene_exp, genotype.astype(np.float64)


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
    genes = exp_and_phen['data'][0][0][3]
    return samples.flatten(), genes.flatten(), expression


def load_genotype(fname,rsids):
    rsids = list(set(rsids)) #unique rsids
    snps = []
    genotype = []
    if re.match(".*\.raw$", fname): #plink format
        sep = ' '
        with open(fname, 'r') as f:
            rsids_base = f.readline().split(sep=sep)
        rsids_file = [re.sub("_.*","",rsid) for rsid in rsids_base]
        rsids_file = np.array(rsids_file)
        rsid_idx = [np.argwhere(rsid in rsids_file) for rsid in rsids]
        rsids = [rsid for rsid in rsids if rsid not in rsids_file]
        genotype = np.loadtxt(fname, skiprows=1, usecols=rsid_idx, delimiter=sep)
        samples = np.loadtxt(fname, skiprows=1, usecols=0, delimiter=sep, dtype=str)

    elif re.match(".*\.csv$", fname): #csv
        sep = ','
        with open(fname, 'r') as f:
            samples  = f.readline().split(sep=sep)
            for line in f:
                items = line.split(sep=sep)
                if items[0] in rsids:
                    genotype.append(items[1:])
                    rsids.pop(items[0])
                    snps += [items[0]]
        genotype = np.ndarray(genotype).T
    else:
        raise NotImplementedError("Format of {} not recognized".format(fname))
    if len(rsids) > 0:
        raise LookupError("{} not found in {}".format(",".join(rsids), fname))
    else:
        return np.array(samples), np.array(snps),genotype


def load_methylation(fname):
    with h5py.File(fname, 'r') as h5:
        samples = []
        _, cols = h5['m']['id'].shape
        for i in range(cols):
            ref = h5['m']['id'][0,i]
            cur_id = ""
            for item in h5[ref]:
                digit = item.tostring().decode('utf-8').rstrip('\x00')
                cur_id += digit
            samples += [cur_id]
        probe_ids = []
        _, cols = h5['m']['rowlabels'].shape
        for i in range(cols):
            ref = h5['m']['rowlabels'][0,i]
            cur_id = ""
            for item in h5[ref]:
                c = item.tostring().decode('utf-8').rstrip('\x00')
                cur_id += c
            probe_ids += [cur_id]
        methylation = h5['m']['data'][:]
    return np.array(samples), np.array(probe_ids), methylation


def load_acetylation(fname):
    acetyl_obj = io.loadmat(fname)
    acetylation = acetyl_obj['aceR2'][0][0][0]
    samples = acetyl_obj['aceR2'][0][0][2]
    peak_id = acetyl_obj['aceR2'][0][0][1]
    return samples.flatten(), peak_id.flatten(), acetylation


def match_samples(*samples):
    """Sort data by sample ids"""
    shared = reduce(np.intersect1d, samples)
    shared_idx = []
    for sample_vec in samples:
        indices = sample_vec.argsort()
        to_keep = np.array([elem in shared for elem in  sample_vec[indices]])
        shared_idx += [np.extract(to_keep, indices)]
    return shared_idx

def get_snp_groups(rsids, coord_file, genotype_dir, sep='\t'):
    """Get genotype file containing each rsid"""
    coords = np.loadtxt(coord_file, delimiter=sep,dtype=str)
    coords_rsids = [re.sub("_.*","",rsid) for rsid in coords[:,0].flatten()]
    snp_files = []
    if re.match(".*hrc.*",genotype_dir): #plink raw, sample by snp
        for rsid in rsids:
            chrom_idx = np.argwhere(rsid in coords_rsids)
            chrom = coords[chrom_idx,1]
            snp_files += [os.path.join(genotype_dir, "chr{}.raw".format(chrom))]
    elif re.match(".*1kg.*",genotype_dir): #csv file, snp by sample
        for rsid in rsids:
            chrom_idx = np.argwhere(rsid in coords_rsids)
            chrom = coords[chrom_idx1]
            fstring = os.path.join(genotype_dir, "snpMatrixChr{}{}.csv")
            fA = fstring.format(chrom,"a")
            fB = fstring.format(chrom,"b")
            file_rsids = np.loadtxt(fA, delimiter=',', skiprows=1, usecols=0, dtype=str)
            if rsid in file_rsids:
                snp_files += [fA]
            else:
                snp_files += [fB]
    else:
        raise ValueError("Invalid Genotype group: {}".format(genotype_group))
    return snp_files


def compute_pcs(A):
    (U, D, vh) = np.linalg.svd(stats.zscore(A), full_matrices=False, compute_uv=True)
    return U@np.diag(D)


def get_mediator(data, ids, which_ids, nFeatures):
    if nFeatures > 1:
        feature_idx = ids[ids in which_ids]
        tmp_data = data[:,feature_idx]
        cur_data = compute_pcs(tmp_data)[:,0]
    else:
        cur_data = ids[:, ids == which_ids[0]]
    return cur_methyl


def reduce_genotype(genotype, lv_method, num_latent, vae_depth=None):
    if genotype.shape[1] == 1:
        return genotype
    if genotype.shape[1] < num_latent:
        num_latent = genotype.shape[1] / 2
    if lv_method == 'mmdvae':
        params = {
            "output_size": genotype.shape[0],
            "n_latent": num_latent,
            "n_hidden": vae_depth}
        model = vae.train_mmd_vae(stats.zscore(genotype), params)
        latent_genotype = model.encode(stats.zscore(genotype))
    elif lv_method == "pca":
        latent_genotype = compute_pcs(genotype)[:, 0:num_latent]
    else:
        raise NotImplemented
    return latent_genotype
