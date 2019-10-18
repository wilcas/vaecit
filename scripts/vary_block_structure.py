import numpy as np
import sys
import joblib
import data_model as dm
import cit_sm as cit
import os

def run_cit_sim(trait,expr,geno,scenario,lv_method,num_latent,vae_depth,num_bootstrap):
    z = dm.reduce_genotype(geno,lv_method,num_latent,"",vae_depth=vae_depth)
    if type(z) != np.ndarray:
        z = z.numpy().astype(np.float64)
    result = cit.cit(trait,expr,z,num_bootstrap=num_bootstrap)
    result_rev = cit.cit(expr,trait,z,num_bootstrap=num_bootstrap)
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
    # block_structures = {
    #     "100": [1.0],
    #     "80-20": [0.8,0.2],
    #     "50-50": [0.5,0.5],
    #     "33-33-34": [0.33,0.33,0.34],
    #     "all-25": [0.25,0.25,0.25,0.25],
    #     "all-20": [0.2,0.2,0.2,0.2,0.2],
    #     "all": None
    # }
    block_structures = range(1,6)
    params = {
        'models': model_str.keys(),
        'num_genotypes': [200],
        'lv_method': [
            'pca',
            'lfa',
            'kernelpca',
            'fastica',
        #    'mmdvae_warmup',
        #    'mmdvae_batch',
        #    'mmdvae_batch_warmup',
            'mmdvae',
            'ae'
        #    'ae_batch'
        ],
        'fix_effects': sys.argv[1] == "fix_effects" if len(sys.argv) > 1 else False,
        'num_latent': 1,
        'num_simulations': 1000,
        'num_samples': 500,
        'num_bootstrap': None,
        'vae_depth': 5
    }
    np.random.seed(42)
    data = {
        f'{num_genotype}genotypes{structure}_{model}': [
            model_str[model](
                params['num_samples'],
                num_genotype,
                # dm.block_genotype(
                #     params['num_samples'],
                #     num_genotype,
                #     block_structures[structure]),
                dm.generate_block(
                    params['num_samples'],
                    num_genotype,
                    num_blocks=structure),
                fix_effects=params['fix_effects'])
            for i in range(params['num_simulations'])
        ]
        for num_genotype in params['num_genotypes']
        for model in params['models']
        for structure in block_structures
    }
    with joblib.parallel_backend('loky', n_jobs=8):
        for lv_method in params['lv_method']:
            for k in data:
                fname = f"{sys.argv[1] if len(sys.argv) > 1 else ''}simulation_{lv_method}_{k}.csv"
                cur_files = [f for (_,_,files) in os.walk("/home/wcasazza/scratch/vaecit/data/") for f in files]
                if fname not in cur_files and not os.path.isfile(fname):
                    results = joblib.Parallel(verbose=10)(
                        joblib.delayed(run_cit_sim)(trait,expr,geno, k, lv_method, params['num_latent'], params['vae_depth'], params['num_bootstrap'])
                            for (trait,expr,geno) in data[k]
                    )
                    cit.write_csv(results, fname)


if __name__ == "__main__":
    main()
