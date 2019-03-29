# Data Processing Pipeline
The general goal of this project is to take in a series of genotypes associated with a set of molecular traits and gene expression and test for mediation of a genotype-trait association by those molecular traits using a causal inference test. The difference here from past analyses is that instead of using single genotypes I'm going to use latent representations of many genotypes, then test mediation by single molecular traits or the first PC of a handful of molecular traits.

The data processing steps that need to take place are:
1. Loading in data and matching samples
1. Appropriate accounting for known confounds in data
1. Obtaining latent genotype variables on per QTLgene basis
1. Causal inference of with each epigenetic mark/PC as a mediator

The experimental steps I'm taking are:
1. comparison of p values using LVs of all QTL SNPs for a given gene in place of one qtl per probe
1. Expansion of previous analysis by finding individual probes/PC of handfuls of probes mediating LV QTL gene relationship
1. selection of probes to expand LV to Gene associations/ meaningful selection of SNPs to summarize (i.e. associations a that exist at with non-QTL SNPs when we consider them all at once with LVs)
