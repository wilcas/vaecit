"""
This Module serves to generate datasets under different causal structures. It mimics the simulation file in the R
version of this project.
"""
import numpy as np


def generate_null(n=100, p=200):
    trait = np.random.normal(size=(n, 1))
    genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    gene_exp = np.random.normal(size=(n, 1))
    return trait, gene_exp, genotype


def generate_caus1(n=100, p=200):
    genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    gene_exp = np.array([[1] for i in range(n)])+(genotype@np.array([[10] for i in range(p)])) + np.random.normal(size=(n, 1))
    trait = np.array([[2] for i in range(n)]) + 10 * gene_exp + np.random.normal(size=(n,1))
    return trait, gene_exp, genotype


def generate_ind1(n=100, p=200):
    genotype = np.random.binomial(n=2, p=0.25, size=(n, p))
    gene_exp = np.array([[1] for i in range(n)])+(genotype@np.array([[10] for i in range(p)])) + np.random.normal(size=(n, 1))
    trait = np.array([[2] for i in range(n)])+(genotype@np.array([[10] for i in range(p)])) + np.random.normal(size=(n, 1))
    return trait, gene_exp, genotype


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
    return trait, gene_exp, genotype
