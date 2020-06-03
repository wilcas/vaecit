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

import statsmodels.api as sm
import data_model as dm
