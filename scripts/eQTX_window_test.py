'''
Test for the right window around TSS to select methylation and acetylation data
in ROSMAP
'''
import data_model as dm

import click
import joblib
import numpy as np
import statsmodels.api as sm

from scipy import io


def run_regression(exp_data, epi_data, mapping_data, gene):
    e_samples, e_ids, expression = exp_data
    epi_samples, epi_ids, epigenetic = epi_data
    features, genes, mapping = mapping_data
    window = mapping[:, genes == gene].nonzero()[0]
    n = e_samples.shape[0]
    results = []
    for feature in features[window]:
        cur_exp = expression[:,e_ids == gene]
        m = np.sum(e_ids == gene)
        if m == 1:
            fit = sm.OLS(cur_exp.reshape((n,1)), np.c_[np.ones((n,1)),epigenetic[:, epi_ids == feature].reshape((n,1))]).fit()
            pval = fit.pvalues[1]
            dist = mapping[features == feature, genes == gene]
            if dist.shape is ():
                results.append(
                    {
                        'feature': feature,
                        'gene': gene,
                        'pval': pval,
                        'dist': dist
                    }
                )
            else:
                for j,k in zip(*dist.nonzero()):
                    results.append(
                        {
                            'feature': feature,
                            'gene': gene,
                            'pval': pval,
                            'dist': dist[j,k]
                        }
                    )
        else:
            for i in range(m):
                fit = sm.OLS(cur_exp[:,i].reshape((n,1)), np.c_[np.ones((n,1)),epigenetic[:, epi_ids == feature].reshape((n,1))]).fit()
                pval = fit.pvalues[1]
                dist = mapping[features == feature, genes == gene]
                if dist.shape is ():
                    results.append(
                        {
                            'feature': feature,
                            'gene': gene + "-{}".format(i),
                            'pval': pval,
                            'dist': dist
                        }
                    )
                else:
                    for j,k in zip(*dist.nonzero()):
                        results.append(
                            {
                                'feature': feature,
                                'gene': gene,
                                'pval': pval,
                                'dist': dist[j,k]
                            }
                        )
    return results


@click.command()
@click.option('--ac-file', required=True, type=str)
@click.option('--m-file', required=True, type=str)
@click.option('--exp-file', required=True, type=str)
@click.option('--probe-map-file', required=True, type=str)
@click.option('--peak-map-file', required=True, type=str)
@click.option('--out-name', required=True, type=str)
def main(**opts):
    pc_remove = 10
    # load data
    ac_samples, ac_ids, acetylation = dm.load_acetylation(opts['ac_file'])
    m_samples, m_ids, methylation = dm.load_methylation(opts['m_file'])
    e_samples, e_ids, expression = dm.load_expression(opts['exp_file'])
    m_mapping_data = dm.load_mapping(opts['probe_map_file'])
    ac_mapping_data = dm.load_mapping(opts['peak_map_file'])

    # remove PCs
    acetylation = dm.standardize_remove_pcs(acetylation, pc_remove)
    methylation = dm.standardize_remove_pcs(methylation, pc_remove)
    expression = dm.standardize_remove_pcs(expression, pc_remove)

    mask = ~np.all(np.isnan(acetylation),axis=0)
    acetylation = acetylation[:, mask]
    ac_ids = ac_ids[mask]
    map_mask = np.isin(ac_mapping_data[0],ac_ids)
    ac_mapping_data = ac_mapping_data[0][map_mask], ac_mapping_data[1], ac_mapping_data[2][map_mask,:]

    # run regressions
    with joblib.parallel_backend("loky"):
        # match samples
        m_idx, e_idx = dm.match_samples(m_samples,e_samples)
        methy_data = m_samples[m_idx], m_ids, methylation[m_idx,:]
        exp_data = e_samples[e_idx], e_ids, expression[e_idx,:]
        m_results = joblib.Parallel(n_jobs=-1, verbose=10)(
            joblib.delayed(run_regression)(exp_data,methy_data, m_mapping_data,gene)
            for gene in m_mapping_data[1] if np.isin(gene,e_ids)
        )
        # m_results = [run_regression(exp_data,methy_data,m_mapping_data,gene) for gene in m_mapping_data[1] if np.isin(gene,e_ids)]
        # match samples
        ac_idx, e_idx = dm.match_samples(ac_samples,e_samples)
        acety_data = ac_samples[ac_idx], ac_ids, acetylation[ac_idx,:]
        exp_data = e_samples[e_idx], e_ids, expression[e_idx,:]
        ac_results = joblib.Parallel(n_jobs=-1, verbose=10)(
            joblib.delayed(run_regression)(exp_data,acety_data, ac_mapping_data,gene)
            for gene in ac_mapping_data[1] if np.isin(gene,e_ids)
        )
        # ac_results = [run_regression(exp_data,acety_data,ac_mapping_data,gene) for gene in ac_mapping_data[1] if np.isin(gene,e_ids)]
        m_results = [elem for sublist in m_results for elem in sublist]
        ac_results = [elem for sublist in ac_results for elem in sublist]

        dm.write_csv(m_results, "methyl_" + opts['out_name'])
        dm.write_csv(ac_results, "acetyl" + opts['out_name'])


    # write output


if __name__ == '__main__':
    main()
