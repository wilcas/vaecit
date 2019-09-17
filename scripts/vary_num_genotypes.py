import numpy as np
import joblib
import data_model as dm
import cit_sm as cit


def run_cit_sim(trait,expr,geno,scenario,lv_method,num_latent,vae_depth,num_bootstrap):
    z = dm.reduce_genotype(geno,lv_method,num_latent,"",vae_depth=vae_depth)
    if type(z) != np.ndarray:
        z = z.numpy().astype(np.float64)
    result = cit.cit(trait,expr,z,num_bootstrap=num_bootstrap)
    result_rev = cit.cit(trait,z,expr,num_bootstrap=num_bootstrap)
    result_rev = {f'rev_{k}': result_rev[k] for k in result_rev}
    result.update(result_rev)
    result['method'] = lv_method
    result['scenario'] = scenario
    return result


def main():
    model_str = {
        'null': dm.generate_null, 
        'causal': dm.generate_caus1,
        'independent':dm.generate_ind1, 
        'causal_independent': dm.generate_caus_ind,
        'causal_hidden': dm.generate_caus_hidden,
        'independent_hidden': dm.generate_ind_hidden
    }
    params = {
        'models': model_str.keys(),
        'num_genotypes': [1],#25,50,100,200,400,600],
        'lv_method': [
            'pca', 
            'mmdvae',
            'lfa', 
            'kernelpca', 
            'fastica', 
            'mmdvae_warmup', 
            'mmdvae_batch', 
            'mmdvae_batch_warmup', 
            'ae',
            'ae_batch'
        ],
        'num_latent': 1,
        'num_simulations': 1000,
        'num_samples': 500,
        'num_bootstrap': None,
        'vae_depth': 5
    }
    data = {
        f'{num_genotype}genotypes_{model}': [
            model_str[model](params['num_samples'],num_genotype) 
            for i in range(params['num_simulations'])
        ]
        for num_genotype in params['num_genotypes'] 
        for model in params['models']
    }
    with joblib.parallel_backend('loky', n_jobs=8):
        results = joblib.Parallel(verbose=10)(
            joblib.delayed(run_cit_sim)(trait,expr,geno, k, lv_method, params['num_latent'], params['vae_depth'], params['num_bootstrap']) 
                for lv_method in params['lv_method']
                for k in data
                for (trait,expr,geno) in data[k]
                
        )
    cit.write_csv(results, "simulation_num_genotype_varied.csv" )
    

if __name__ == "__main__":
    main()    
