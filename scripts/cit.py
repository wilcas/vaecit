import csv
import numpy as np
from scipy import stats
from ctypes import CDLL, c_int, c_void_p, c_double, POINTER, byref

def write_csv(results, filename):
    out_rows = []
    for res in results:
        cur_row = {}
        for j in range(1,5):
            cur_test = 'test{}'.format(j)
            cur_p = 'p{}'.format(j)
            for key in res[cur_test]:
                if key in ['rss', 'r2']: #single value for test
                    cur_key = '{}_{}'.format(cur_test,key)
                    cur_row[cur_key] = res[cur_test][key]
                else:
                    for k in range(len(res[cur_test][key])):
                        cur_key = '{}_{}{}'.format(cur_test,key,k)
                        cur_row[cur_key] = res[cur_test][key][k]
            cur_row[cur_p] = res[cur_p]
        cur_row['omni_p'] = res['omni_p']
        out_rows.append(cur_row)
    with open(filename, 'w') as f:
        names = out_rows[0].keys()
        writer = csv.DictWriter(f, names)
        writer.writeheader()
        writer.writerows(out_rows)


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
    return (1. - stats.f.cdf(fstat, p2-p1,n-p2)), fstat


def linreg_with_stats(y,X):
    lib = CDLL('./cit_functions.so')
    lib.linreg_with_stats.argtypes = [
        c_int, c_int, np.ctypeslib.ndpointer(dtype=np.float),
        np.ctypeslib.ndpointer(dtype=np.float),
        np.ctypeslib.ndpointer(dtype=np.float),POINTER(c_double), POINTER(c_double),
        np.ctypeslib.ndpointer(dtype=np.float),
        np.ctypeslib.ndpointer(dtype=np.float)]
    lib.linreg_with_stats.restype = c_void_p
    n, p = X.shape
    beta =  np.empty(p, np.float)
    se = np.empty(p, np.float)
    t = np.empty(p, np.float)
    RSS = c_double(0.)
    TSS = c_double(0.)
    lib.linreg_with_stats(n,p,y,X,beta,byref(RSS),byref(TSS),se,t)
    return (beta.flatten(),RSS.value,TSS.value,se.flatten(),t.flatten())


def test_association(y,design_null,design_full):
    n = y.shape[0]
    fit1 = linreg_with_stats(y, design_full)
    fit2 = linreg_with_stats(y, design_null)
    p, _ = ftest(fit2, fit1, n)
    return fit1, p


def run_bootstraps(T, residual, L, n, num_bootstrap):
    lib = CDLL('./cit_functions.so')
    lib.run_bootstraps.argtypes = [
        c_int, c_int, np.ctypeslib.ndpointer(dtype=np.float),
        np.ctypeslib.ndpointer(dtype=np.float),
        np.ctypeslib.ndpointer(dtype=np.float),
        np.ctypeslib.ndpointer(dtype=np.float)
    ]
    lib.run_bootstraps.restype = c_void_p
    tstats = np.empty(num_bootstrap, np.float)
    lib.run_bootstraps(n, num_bootstrap,T,residual,L,tstats)
    return tstats


def test_independence(T, G, L, num_bootstrap):
    n = T.shape[0]
    fit = linreg_with_stats(G, np.concatenate((np.ones((n,1)),L),axis=1))
    beta, _, _,_,_ = fit
    test = np.concatenate((np.ones((n,1)),L), axis=1)@beta
    residual = G - (np.concatenate((np.ones((n,1)),L), axis=1)@beta).reshape((n,1))
    bootstraps = run_bootstraps(T, residual, L, n, num_bootstrap)
    f_list = [t**2 for t in bootstraps]
    fit1 = linreg_with_stats(T, np.concatenate((np.ones((n,1)), G, L),axis=1))
    fstat = fit1[-1][-1]**2
    p = np.sum(f_list <= fstat) / num_bootstrap
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
        np.c_[np.ones((n, 1)), instrument],
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
