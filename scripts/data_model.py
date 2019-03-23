"""
This Module serves to generate datasets under different causal structures. It mimics the simulation file in the R
version of this project.
"""
import csv
import numpy as np
import h5py

from scipy import io


def generate_null(n=100, p=200):
    trait = np.random.normal(size=(n, 1))
    genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    gene_exp = np.random.normal(size=(n, 1))
    return trait, gene_exp, genotype.astype(np.float64)


def generate_caus1(n=100, p=200):
    genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    gene_exp = np.array([[1] for i in range(n)])+(genotype@np.array([[10] for i in range(p)])) + np.random.normal(size=(n, 1))
    trait = np.array([[2] for i in range(n)]) + 10 * gene_exp + np.random.normal(size=(n,1))
    return trait, gene_exp, genotype.astype(np.float64)


def generate_ind1(n=100, p=200):
    genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    gene_exp = np.array([[1] for i in range(n)])+(genotype@np.array([[10] for i in range(p)])) + np.random.normal(size=(n, 1))
    trait = np.array([[2] for i in range(n)])+(genotype@np.array([[10] for i in range(p)])) + np.random.normal(size=(n, 1))
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
    return samples, genes, expression


def load_genotype(fname,rsids, sep=','):
    rsids = list(set(rsids)) #unique rsids
    snps = []
    with open(fname, 'r') as f:
        genotype = []
        samples  = f.readline().split(sep=sep)
        for line in f:
            items = line.split(sep=sep)
            if items[0] in rsids:
                genotype.append(items[1:])
                rsids.pop(items[0])
                snps += [items[0]]
    if len(rsids) > 0:
        raise LookupError("{} not found in {}".format(",".join(rsids), fname))
    else:
        genotype = np.ndarray(genotype).T
        return samples, snps, genotype


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
    return samples, probe_ids, methylation


def load_acetylation(fname):
    raise NotImplemented
