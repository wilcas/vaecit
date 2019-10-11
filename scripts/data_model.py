"""
This Module serves to generate datasets under different causal structures. It mimics the simulation file in the R
version of this project.
"""
import csv
import h5py
import importlib
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
from sklearn.decomposition import FactorAnalysis, FastICA, KernelPCA
from functools import reduce


def write_csv(results, filename):
    with open(filename, 'w') as f:
        names = results[0].keys()
        writer = csv.DictWriter(f, names)
        writer.writeheader()
        writer.writerows(results)


def block_genotype(n=100,p=200, perc=[0.5,0.5]):
    """Generate 'blocky' correlation structured genotype"""
    if perc is None:
        return None
    grps = len(perc)
    base = np.random.binomial(n=2, p=0.25, size=(n,grps))
    result = np.zeros((n,p))
    j = 0
    if sum(perc) != 1:
        raise ValueError("perc must sum to 1!")
    for (col,frac) in zip(base.T,perc):
        for i in range(int(p * frac)):
            result[:,j] = col + np.random.normal(size=(n,))
            j += 1
    return result

def generate_null(n=100, p=200, genotype = None, fix_effects=False):
    if genotype is None:
        genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    trait = np.random.normal(size=(n,))
    gene_exp = np.random.normal(size=(n,))
    return trait.reshape(n,1), gene_exp.reshape(n,1), genotype.astype(np.float64)


def generate_caus1(n=100, p=200, genotype = None, fix_effects = False):
    if genotype is None:
        genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    if fix_effects:
        exp_coeffs = np.array([1 for i in range(p)])
    else:
        exp_coeffs= np.array([(random.choice([-1,1])*np.random.uniform())for i in range(p)])
    gene_exp =  (genotype@exp_coeffs) + np.random.normal(size=(n,))
    if fix_effects:
        trait_coeff = 1
    else:
        trait_coeff = random.choice([-1,1])*np.random.uniform()
    trait =  trait_coeff* gene_exp + np.random.normal(size=(n,))
    return trait.reshape(n,1), gene_exp.reshape(n,1), genotype.astype(np.float64)


def generate_ind1(n=100, p=200, genotype = None, fix_effects=False):
    if genotype is None:
        genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    if fix_effects:
        exp_coeffs = np.array([1 for i in range(p)])
    else:
        exp_coeffs= np.array([(random.choice([-1,1])*np.random.uniform())for i in range(p)])
    gene_exp = (genotype@exp_coeffs)  + np.random.normal(size=(n,))
    if fix_effects:
        trait_coeffs = np.array([1 for i in range(p)])
    else:
        trait_coeffs= np.array([(random.choice([-1,1])*np.random.uniform())for i in range(p)])
    trait = (genotype@trait_coeffs)+ np.random.normal(size=(n,))
    return trait.reshape(n,1), gene_exp.reshape(n,1), genotype.astype(np.float64)


def generate_caus_ind(n=100, p=200, genotype=None, fix_effects=False):
    if genotype is None:
        genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    if fix_effects:
        exp_coeffs= np.array([1 for i in range(p)])
    else:
        exp_coeffs= np.array([(random.choice([-1,1])*np.random.uniform())for i in range(p)])
    gene_exp = (genotype@exp_coeffs) + np.random.normal(size=(n,))
    if fix_effects:
        small_coeffs = np.array([0.01 for i in range(p)])
    else:
        small_coeffs = np.array([random.choice([-1,1])*np.random.uniform()* 1e-2 for i in range(p)])
    trait = exp_coeffs@gene_exp + genotype@small_coeffs + np.random.normal(size=(n,))
    return trait.reshape(n,1), gene_exp.reshape(n,1), genotype.astype(np.float64)


def generate_ind_hidden(n=100, p=200, genotype=None, fix_effects=False):
    if genotype is None:
        genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    if fix_effects:
        exp_coeffs = np.array([1 for i in range(p)])
    else:
        exp_coeffs= np.array([(random.choice([-1,1])*np.random.uniform())for i in range(p)])
    hidden_cov = np.sum(np.random.normal(size=(n,10)),1) * (p / 10.0)
    gene_exp = (genotype@exp_coeffs)  + hidden_cov + np.random.normal(size=(n,))
    if fix_effects:
        trait_coeffs = np.array([1 for i in range(p)])
    else:
        trait_coeffs= np.array([(random.choice([-1,1])*np.random.uniform())for i in range(p)])
    trait = (genotype@trait_coeffs)+ hidden_cov + np.random.normal(size=(n,))
    return trait.reshape(n,1), gene_exp.reshape(n,1), genotype.astype(np.float64)


def generate_caus_hidden(n=100, p=200, genotype=None, fix_effects = False):
    if genotype is None:
        genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    hidden_cov = np.sum(np.random.normal(size=(n,10)),1) * (p /10.0)
    if fix_effects:
        exp_coeffs = np.array([1 for i in range(p)])
        trait_coeffs = 1
    else:
        exp_coeffs = np.array([(random.choice([-1,1])*np.random.uniform())for i in range(p)])
        trait_coeffs = random.choice([-1,1])*np.random.uniform()
    gene_exp =  (genotype@exp_coeffs) + np.random.normal(size=(n,)) + hidden_cov
    trait = trait_coeffs*gene_exp + np.random.normal(size=(n,)) + hidden_cov
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
    if re.match("phen",fname):
        exp_and_phen = io.loadmat(fname)
        expression = exp_and_phen['data'][0][0][0]
        samples = exp_and_phen['data'][0][0][4]
        samples = np.array([re.sub(":.*","",sample[0][0]) for sample in samples])
        genes = exp_and_phen['data'][0][0][3]
        genes = np.array([re.sub(":.*","",gene[0][0]) for gene in genes])

    elif re.match("Normalize", fname):
        struct = io.loadmat(fname, squeeze_me=True)
        expression = struct['expr']
        samples = struct['exprList']
        genes = struct['geneSymbol']
    else:
        struct = io.loadmat(fname, squeeze_me=True)['Res']
        expression = struct['data'].item()
        samples = struct['rowlabels'].item()
        genes = struct['collabels'].item()
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


def load_mapping(f):
    data = io.loadmat(f)
    features = np.array([elem[0][0] for elem in data['xSet']])
    genes = np.array([elem[0][0] for elem in data['ySet']])
    mapping = data['XtoY']
    return (features,genes,mapping.tolil())


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
    return A_std@vh.T


def get_mediator(data, ids, which_ids, data2= None, ids2 = None, which_ids2 = None, lv_method="pca", vae_depth=None,num_latent=1, state_name="", model_dir=""):
    n = data.shape[0]
    feature_idx = np.isin(ids, which_ids)
    tmp_data = data[:,feature_idx]

    if not np.isscalar(data2) and data2 is not None:
        feature_idx2 = np.isin(ids2, which_ids2)
        tmp_data2 = data2[:, feature_idx2]
        cur_data = np.concatenate((tmp_data, tmp_data2), axis=1)
    else:
        cur_data = tmp_data
    if len(cur_data.shape) == 1:
        cur_data = cur_data.reshape((n,1))
    if re.search("ae", lv_method):
        params = {
            "size": cur_data.shape[1],
            "num_latent": num_latent,
            "depth": vae_depth}
        if re.search("batch", lv_method):
            params['batch_norm'] = True
        fname = os.path.join(model_dir, "{}_{}model_{}_depth.pt".format(state_name, lv_method, vae_depth))
        plot_name = os.path.join(model_dir, "{}_{}loss_{}_depth.png".format(state_name, lv_method, vae_depth))
        if os.path.isfile(fname):
            model = vt.MMD_VAE(**params)
            model.load_state_dict(torch.load(fname))
            model.eval()
        else:
            if lv_method == "ae" or lv_method == "ae_batch":
                model = vt.train_ae(torch.Tensor(stats.zscore(cur_data)), params, save_loss=plot_name)
            elif re.search("warmup", lv_method):
                model = vt.train_mmd_vae(torch.Tensor(stats.zscore(cur_data)), params, save_loss=plot_name, warmup=True)
            else:
                model = vt.train_mmd_vae(torch.Tensor(stats.zscore(cur_data)), params, save_loss=plot_name)
            torch.save(model.state_dict(), fname)
        cur_data = model.encode(torch.Tensor(stats.zscore(cur_data))).detach()
    elif lv_method == "pca":
        cur_data = compute_pcs(cur_data)[:, 0:num_latent]
    elif lv_method == "lfa":
        lfa = FactorAnalysis(n_components=1)
        cur_data = lfa.fit_transform(stats.zscore(cur_data))
    elif lv_method == "kernelpca":
        kpca = KernelPCA(n_components=1, kernel="rbf")
        cur_data = kpca.fit_transform(stats.zscore(cur_data))
    elif lv_method == "fastica":
        ica = FastICA(n_components=1)
        cur_data = ica.fit_transform(stats.zscore(cur_data))
    else:
        raise NotImplementedError("No such lv method implemented!")
    if len(cur_data.shape) == 1:
        cur_data = cur_data.reshape((n,1))
    else:
        cur_data = cur_data[:,0]
    return cur_data


def reduce_genotype(genotype, lv_method, num_latent, state_name, vae_depth=None, model_dir=""):
    if genotype.shape[1] == 1:
        return genotype
    num_latent = min(genotype.shape[1],num_latent)
    if re.search("ae", lv_method):
        params = {
            "size": genotype.shape[1],
            "num_latent": num_latent,
            "depth": vae_depth}
        if re.search("batch", lv_method):
            params['batch_norm'] = True
        if model_dir:
            fname = os.path.join(model_dir, "{}_{}model_{}_depth.pt".format(state_name, lv_method, vae_depth))
            plot_name = os.path.join(model_dir, "{}_{}loss_{}_depth.png".format(state_name, lv_method, vae_depth))
        else:
            fname = ""
            plot_name = ""
        if os.path.isfile(fname):
            model = vt.MMD_VAE(**params)
            model.load_state_dict(torch.load(fname))
            model.eval()
        else:
            if lv_method == "ae" or lv_method == "ae_batch":
                model = vt.train_ae(torch.Tensor(stats.zscore(genotype)), params, save_loss=plot_name)
            elif re.search("warmup", lv_method):
                model = vt.train_mmd_vae(torch.Tensor(stats.zscore(genotype)), params, save_loss=plot_name, warmup=True)
            else:
                model = vt.train_mmd_vae(torch.Tensor(stats.zscore(genotype)), params, save_loss=plot_name)
            if model_dir:
                torch.save(model.state_dict(), fname)
        latent_genotype = model.encode(torch.Tensor(stats.zscore(genotype))).detach()
    elif lv_method == "pca":
        latent_genotype = compute_pcs(genotype)[:, 0:num_latent]
    elif lv_method == "lfa":
        lfa = FactorAnalysis(n_components=1)
        latent_genotype = lfa.fit_transform(stats.zscore(genotype))
    elif lv_method == "kernelpca":
        kpca = KernelPCA(n_components=1, kernel="rbf")
        latent_genotype = kpca.fit_transform(stats.zscore(genotype))
    elif lv_method == "fastica":
        ica = FastICA(n_components=1)
        latent_genotype = ica.fit_transform(stats.zscore(genotype))
    else:
        raise NotImplementedError("LV method not defined!")
    return latent_genotype
