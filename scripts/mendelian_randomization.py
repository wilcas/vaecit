""" Api code for Mendelian randomization analyses

First off, there will be functions for running Bidirectional Mendelian 
randomization and summary statistics versions on three-four sets of variables in 
the ROSMAP dataset: DNAm, H3K9 peaks, RNAseq, and genotypes. 

Summary statistics version, will be either called to from here in which case the 
location of the executable will be provided. MR-Egger regression, if tested, 
doesn't fit a complicated LMM so will likely be implemented here if used.

@TODO before running:
  check initial filtering set overlapping eQTLs and m/acQTLs meet the necessary
  criteria. 
   -Steiger filtering

"""
import numpy as np
import pandas as pd
from statsmodels.sandbox.regression.gmm import IV2SLS 


def write_csv(results, filename):
    with open(filename, 'w') as f:
        names = results[0].keys()
        writer = csv.DictWriter(f, names)
        writer.writeheader()
        writer.writerows(results)


def MR(exposure, outcome, SNP):
    model = IV2SLS(outcome,exposure,SNP)
    res = model.fit()
    return {
      'pMR': res.pvalues[-1],
      'bMR': res.params[-1]
    }


def steiger(exposure, outcome, SNP):
    rho_gy = np.corcoef(SNP,outcome)
    rho_gx = np.corcoef(SNP,exposure)
    rho_xy = np.corcoef(exposure,outcome)
    Z_gy = 0.5 * np.log((1.0 + rho_gy) / (1.0 - rho_gy))
    Z_gx = 0.5 * np.log((1.0 + rho_gx) / (1.0 - rho_gx))
    rm_2 = 0.5 * (rho_gx**2 + rho_gy**2)
    f = (1.0 - rho_xy) / (2.0 *(1 - rm_2))
    h = (1.0 - f*rm_2) /(1.0 - rm_2)
    Z = (Z_gx - Z_gy) * (np.sqrt(np.size(exposure)[1] - 3.0) / np.sqrt(2.0*(1 - rho_xy)*h))
    return Z


def steiger_MR(exposure,outcome,SNP):
    mr = MR(exposure,outcome, SNP)
    Z = steiger(exposure, outcome, SNP)
    mr['Z'] = Z
    return mr
