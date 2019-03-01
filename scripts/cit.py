import numpy as np
from scipy import stats
from numba import jit


@jit
def ftest(fit1, fit2, n):
    RSS1 = fit1['rss']
    RSS2 = fit2['rss']
    p1 = fit1['beta'].size
    p2 = fit2['beta'].size
    fstat = ((RSS1 - RSS2) / (p2-p1)) / (RSS2 / (n - p2))
    return 1 - stats.f.cdf(fstat, p2-p1,n-p2), fstat


@jit    
def linreg_with_stats(y, X=None):
    if X is None:
        X = np.ones(shape = (y.shape[0],1))
    n = X.shape[0]
    data = np.c_[np.ones(n), X]
    beta, _, _, _ = np.linalg.lstsq(data,y, rcond=None)
    y_bar = np.mean(y)
    TSS = np.sum(np.square(y - y_bar))
    RSS = np.sum(np.square(y - data@beta))
    var_beta = RSS * np.linalg.pinv(data.T@data)
    se = np.sqrt(np.diag(var_beta))
    res_dict = {
        'beta': beta,
        'rss': RSS,
        'se': se,
        't': beta / se,
        'p' :  (1 - stats.t.cdf(beta / se, df = n - 2)),
        'r2': 1 - (RSS / TSS)
    }
    return res_dict


@jit
def test1(T, L):
    n = T.shape[0]
    fit1 = linreg_with_stats(T, L)
    fit2 = linreg_with_stats(T)
    p, _ = ftest(fit2, fit1, n)
    return fit1, p


@jit
def test2(T, G, L):
    n = T.shape[0]
    fit1 = linreg_with_stats(G, np.c_[T, L])
    fit2 = linreg_with_stats(G, T)
    p, _ = ftest(fit2, fit1, n)
    return fit1, p


@jit
def test3(T, G, L):
    n = T.shape[0]
    fit1 = linreg_with_stats(T, np.c_[G, L])
    fit2 = linreg_with_stats(T, G)
    p, _ = ftest(fit2, fit1, n)
    return fit1, p
    
    
@jit
def test4(T, G, L, num_bootstrap):
    n = T.shape[0]
    fit = linreg_with_stats(G, L)
    beta = fit['beta']
    residual = G - np.c_[np.ones(n),L]@beta
    f_list = []
    for i in range(num_bootstrap):
        np.random.shuffle(residual)
        fitA = linreg_with_stats(T,np.c_[residual, L])
        fitB = linreg_with_stats(T, residual)
        _, fstat = ftest(fitB, fitA, n)
        f_list.append(fstat)
    fit1 = linreg_with_stats(T, np.c_[G, L])
    fit2 = linreg_with_stats(T, G)
    _, fstat = ftest(fit2, fit1, n)
    p = sum([fstat > f for f in f_list]) / num_bootstrap
    return fit1, p



def cit(target, mediator, instrument, num_bootstrap=10000):
    # run tests
    stats1, p1 = test1(target, instrument)
    stats2, p2 = test2(target, mediator, instrument)
    stats3, p3 = test3(target, mediator, instrument)
    stats4, p4 = test4(target, mediator, instrument, num_bootstrap)
    omni_p = max(p1, p2, p3, p4)
    
    # merge stats into one table
    res = {
        'test1': stats1,
        'p1': p1,
        'test2': stats2,
        'p2': p2,
        'test3': stats3,
        'p3': p3,
        'test4': stats4,
        'p4': p4,
        'omni_p': omni_p
    }
    return res