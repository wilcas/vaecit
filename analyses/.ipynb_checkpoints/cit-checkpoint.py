import numpy as np
from scipy import stats
from numba import jit


def stats_dict(beta, RSS, TSS, se, t, n):
    res_dict = {
        'beta': beta,
        'rss': RSS,
        'se': se,
        't': beta / se,
        'p' :  (1 - stats.t.cdf(beta / se, df = n - 2)),
        'r2': 1 - (RSS / TSS)
    }
    return res_dict



def ftest(fit1, fit2, n):
    beta1, RSS1, _, _, _ = fit1
    beta2, RSS2, _, _, _ = fit2
    p1 = beta1.shape[0]
    p2 = beta2.shape[0]
    fstat = ((RSS1 - RSS2) / (p2-p1)) / (RSS2 / (n - p2))
#     print('''
#         test1: {}
#         test2: {}
#         fstat: {}'''.format(fit1,fit2,fstat))
    return 1 - stats.f.cdf(fstat, p2-p1,n-p2), fstat


@jit(nopython=True, cache=True)
def linreg_with_stats(y, X):
    n = y.shape[0]
    beta = np.linalg.pinv(X.T@X)@X.T@y
    y_bar =np.mean(y)
    TSS = np.sum(np.square(y - y_bar))
    RSS = np.sum(np.square(y - X@beta))
    var_beta = RSS * np.linalg.pinv(X.T@X)
    se = np.sqrt(np.diag(var_beta))
    t = beta / se
    return beta.reshape(-1), RSS, TSS, se.reshape(-1), t.reshape(-1)


def test_association(y,design_null,design_full):
    n = y.shape[0]
    print(y.shape, design_null.shape, design_full.shape)
    fit1 = linreg_with_stats(y, design_full)
    fit2 = linreg_with_stats(y, design_null)
    p, _ = ftest(fit2, fit1, n)
    return fit1, p


@jit(nopython=True, cache=True)
def run_bootstraps(T, residual, L, n, num_bootstrap):
    fit_list = []
    for i in range(num_bootstrap):
        np.random.shuffle(residual)
        fitA = linreg_with_stats(T,np.concatenate((np.ones((n,1)), residual, L), axis= 1))
        fitB = linreg_with_stats(T, np.concatenate((np.ones((n,1)), residual),axis = 1))
        fit_list.append((fitB, fitA))
    return fit_list


def test_independence(T, G, L, num_bootstrap):
    n = T.shape[0]
    fit = linreg_with_stats(G, np.c_[np.ones(n),L])
    beta, _, _,_,_ = fit
    test = np.c_[np.ones((n,1)),L]@beta
    residual = G - (np.c_[np.ones((n,1)),L]@beta).reshape((n,1))
    bootstraps = run_bootstraps(T, residual, L, n, num_bootstrap)
    f_list = [ftest(fitB, fitA, n) for (fitB,fitA) in bootstraps]
    fit1 = linreg_with_stats(T, np.c_[np.ones(n), G, L])
    fit2 = linreg_with_stats(T, np.c_[np.ones(n), G])
    _, fstat = ftest(fit2, fit1, n)
    p = np.sum([fstat > f for (p, f) in f_list]) / float(num_bootstrap)
    return fit1, p


def cit(target, mediator, instrument, num_bootstrap=10000):
    # run tests
    n = target.shape[0]
    stats1, p1 = test_association(
        target, 
        np.ones((n, 1)),
        np.c_[np.ones((n, 1)), instrument]
    )
    stats2, p2 = test_association(
        mediator,
        np.c_[np.ones((n, 1)), target],
        np.c_[np.ones((n, 1)), target, instrument]
    )
    stats3, p3 = test_association(
        target,
        np.c_[np.ones((n, 1)), mediator],
        np.c_[np.ones((n, 1)), mediator, instrument]
    )
    stats4, p4 = test_independence(target, mediator, instrument, num_bootstrap)
    omni_p = max(p1, p2, p3, p4)
    
    # merge stats into one table
    res = {'test1': stats_dict(*stats1,n),
        'p1': p1,
        'test2': stats_dict(*stats2,n),
        'p2': p2,
        'test3': stats_dict(*stats3,n),
        'p3': p3,
        'test4': stats_dict(*stats4,n),
        'p4': p4,
        'omni_p': omni_p
    }
    return res
