"""Code to partition data by age quantile and then print out mediation results for each method"""
import pandas as pd
import os
os.chdir("/home/wcasazza/vaecit/scripts")
import data_model as dm
import cit_sm as cit
import statsmodels.api as sm
import numpy as np
import joblib
def remove_covar(X, effect):
    result = []
    print("here")
    for col in X.T:
        fit = sm.OLS(col,effect).fit()
        result.append(fit.resid)
    return np.array(result).T
def cit_on_qtl_set(df, gene, methyl, acetyl, express, opts, geno=None):
    (m_samples, m_ids, methylation) = methyl
    (ac_samples, ac_ids, acetylation) = acetyl
    (e_samples, e_ids, expression) = express
    if geno is None:
        # get load in genotypes associated with gene
        snp_files = dm.get_snp_groups(df.snp.values, opts['genotype_dir'])
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
        if opts['lv_mediator']:
            cur_genotype = genotype[:, np.isin(g_ids, np.array([rsid for sublist in df.snp for rsid in sublist.split(",")]))]
        else:
            cur_genotype = genotype[:,np.isin(g_ids, df.snp.to_numpy())]

    # reduce genotype
    # latent_genotype = dm.reduce_genotype(cur_genotype, opts['lv_method'], opts['num_latent'], gene, opts['vae_depth'], opts['model_dir'])

    # if type(latent_genotype) != np.ndarray:
    #     latent_genotype = latent_genotype.numpy().astype(np.float64)
    # get probes and peaks
    cur_exp = expression[:, e_ids == gene]

    n = expression.shape[0]
    mediation_results = []
    for (_, row) in df.iterrows():
        if opts['lv_mediator']:
            lv_method = opts['lv_method']
            state_name = gene+"_mediator"
            model_dir = opts['model_dir']
            depth = opts['vae_depth']
        else:
            lv_method = "pca"
            state_name = ""
            model_dir = ""
            depth = None
        cur_epigenetic = dm.get_mediator(
            methylation,
            m_ids,
            row.probes.split(","),
            data2=acetylation,
            ids2=ac_ids,
            which_ids2=row.peaks.split(","),
            lv_method=lv_method,
            state_name=state_name,
            model_dir = model_dir,
            vae_depth = depth
        )
        
        if ("ae" in lv_method) and opts['lv_mediator']:
            cur_epigenetic = cur_epigenetic.numpy().astype(np.float64)
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
    
    
def main():
    methy_df = pd.read_csv('/home/wcasazza/ROSMAPmethylationWAgeSex.tar.gz', compression='gzip', header=0, sep=',', quotechar='"', error_bad_lines=False, index_col=0)
    phen_df = pd.read_csv("/home/wcasazza/ROSMAP_PHEN.csv", dtype={'Unnamed: 0':np.str_}).set_index('Unnamed: 0')
    acety_df = pd.read_csv("/home/wcasazza/acetylationNonAgeGenderAdjNoqNaN.csv", sep = "\t")
    (ac_samples,ac_ids,acetylation) = (acety_df.columns.to_numpy(), acety_df.index.to_numpy(), acety_df.to_numpy().T)
    (p_samples, p_ids, phenotype) = (phen_df.index.to_numpy(), phen_df.columns.to_numpy(), phen_df.to_numpy())
    (m_samples,m_ids,methylation) = (methy_df.columns.to_numpy(), methy_df.index.to_numpy(), methy_df.to_numpy().T)
    (e_samples, e_ids, expression) = dm.load_expression('/home/wcasazza/expressionAndPhenotype.mat')
    pcs_to_remove = 10
    
    
    
    # remove 'hidden' cell-type specific effects
    
    acetylation = dm.standardize_remove_pcs(acetylation, pcs_to_remove)
    mask = ~np.all(np.isnan(acetylation),axis=0)
    acetylation = acetylation[:, mask]
    ac_ids = ac_ids[mask]
    methylation = dm.standardize_remove_pcs(methylation, pcs_to_remove)
    mask = ~np.all(np.isnan(methylation),axis=0)
    methylation = methylation[:, mask]
    m_ids = m_ids[mask]
    expression = dm.standardize_remove_pcs(expression, pcs_to_remove)
    
    genotype_df = pd.read_csv("/home/wcasazza/cit_genotypes.csv", index_col=0)
    genotype = genotype_df.to_numpy().T
    g_samples = genotype_df.columns.to_numpy()
    g_ids = genotype_df.index.to_numpy()
    (m_idx, ac_idx, e_idx, g_idx, p_idx) =  dm.match_samples(m_samples, ac_samples, e_samples, g_samples, p_samples)
    (g_samples, genotype) = (g_samples[g_idx], genotype[g_idx,:])
    (m_samples, methylation) = (m_samples[m_idx], methylation[m_idx,:])
    (ac_samples, acetylation) = (ac_samples[ac_idx], acetylation[ac_idx,:])
    (e_samples, expression) = (e_samples[e_idx], expression[e_idx,:])
    (p_samples, phenotype) = (p_samples[p_idx], phenotype[p_idx,:])
    
    # remove sex effects from acetylation and methylation
    methylation = remove_covar(methylation, phenotype[:,p_ids == "msex"])
    acetylation = remove_covar(acetylation, phenotype[:,p_ids == "msex"])
    
    geno = (g_samples, g_ids, genotype)
    methyl = (m_samples, m_ids, methylation)
    acetyl = (ac_samples, ac_ids, acetylation)
    express = (e_samples, e_ids, expression) 
    pheno = (p_samples, p_ids, phenotype) 
    
    opts ={
      'lv_method' : None,
      'vae_depth': 5,
      'lv_mediator': False,
      'num_latent': 1,
      'model_dir': "/media/wcasazza/DATA2/wcasazza/saved_models_test_training/",
      'run_reverse': False,
      'num_bootstrap':0
    }
    quantiles = pd.qcut(phenotype[:, p_ids == 'age death'].flatten(),3, labels = [0,1])
    cit_df = pd.read_csv("/home/wcasazza/vaecit/CIT.txt", sep = "\t")
    lv_methods = ['pca', 'mmdvae','lfa', 'kernelpca', 'fastica', 'mmdvae_warmup', 'mmdvae_batch', 'mmdvae_batch_warmup', 'ae','ae_batch']
    for i in range(2):
        opts['run_reverse'] =  not opts['run_reverse']
        for lv_method in lv_methods:
            opts['lv_method'] = lv_method
            for cur_quant in range(5):
                geno = (g_samples, g_ids, genotype[quantiles == cur_quant,:])
                methyl = (m_samples, m_ids, methylation[quantiles == cur_quant,:])
                acetyl = (ac_samples, ac_ids, acetylation[quantiles == cur_quant,:])
                express = (e_samples, e_ids, expression[quantiles == cur_quant,:]) 
                with joblib.parallel_backend("loky"):
                    mediation_tmp = joblib.Parallel(n_jobs=5, verbose=10)(
                    joblib.delayed(cit_on_qtl_set)(df, gene, methyl, acetyl, express, opts, geno)
                        for (gene, df) in cit_df.groupby('gene')
                    )
                mediation_result = [item for sublist in mediation_tmp for item in sublist]
                cit.write_csv(mediation_result, "{}_{}_quantile_{}_5_cit_replication_perm_test.csv".format("rev" if opts['run_reverse'] else "",cur_quant,lv_method))


if __name__ == '__main__':
  main()
