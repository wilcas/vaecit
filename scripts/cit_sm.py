import csv
import statsmodels.api as sm
import numpy as np
from scipy import stats

def write_csv(results, filename):
    with open(filename, 'w') as f:
        names = results[0].keys()
        writer = csv.DictWriter(f, names)
        writer.writeheader()
        writer.writerows(results)


def test_independence(T, G, L, num_bootstrap):
    n = T.shape[0]
    fit_orig = sm.OLS(T, np.c_[np.ones((n,1)), G, L]).fit()
    f_orig = fit_orig.tvalues[-1] ** 2
    tmp_fit = sm.OLS(G, np.c_[np.ones((n,1)),L]).fit()
    residual = tmp_fit.resid.reshape((n,1))
    Ge = tmp_fit.fittedvalues.reshape((n,1))
    if num_bootstrap is None:
        num_iter = 100
    else:
        num_iter = num_bootstrap
    fstats = np.zeros(num_iter)
    for i in range(num_iter):
        np.random.shuffle(residual)
        G_star = Ge + residual
        fit = sm.OLS(T, np.c_[np.ones((n,1)), G_star, L]).fit()
        if L.shape[1] > 1:
            A = np.identity(len(fit.params))[2:,:]
            f_test = fit.f_test(A)
            fstats[i] = f_test.pvalue.item()
        else:
            fstats[i] = fit.tvalues[-1] ** 2
    if num_bootstrap is None:
        if L.shape[1] == 1:
            v1 = 1
            v2 = n - 3
        else:
            v1 = f_test.df_num
            v2 = f_test.df_denom
        delta = ((np.mean(fstats) * v1 * (v2 - 2)) / v2) - v1
        pperm = stats.ncf.cdf(fstats,v1,v2,delta)
        zperm = stats.norm.ppf(pperm)
        porig = stats.ncf.cdf(f_orig,v1,v2,delta)
        zorig = stats.norm.ppf(porig)
        p = stats.norm.cdf(zorig,scale=np.std(zperm))
        if p == np.nan:
            return 1
    else:
        p = np.sum(fstats <= f_orig) / num_bootstrap
    return p


def cit(target, mediator, instrument, num_bootstrap=10000):
    # run tests
    n = target.shape[0]
    num_instruments = instrument.shape[1]
    if num_instruments > 0:
        p1_fit =  sm.OLS(target,np.c_[np.ones((n, 1)), instrument]).fit()
        A = np.identity(len(p1_fit.params))[1:,:]
        p1 = p1_fit.f_test(A).pvalue.item()
    else:
        p1 = sm.OLS(target,np.c_[np.ones((n, 1)), instrument]).fit().pvalues[-1]
    p2 = sm.OLS(mediator, np.c_[np.ones((n, 1)), target, instrument]).fit().pvalues[-1]
    p3_fit = sm.OLS(target,np.c_[np.ones((n, 1)), instrument, mediator]).fit()
    p3 = p3_fit.pvalues[-1]
    p4 = test_independence(target, mediator, instrument, num_bootstrap)
    omni_p = max(p1, p2, p3, p4)
    return {"p1":p1, "p2":p2,"p3":p3,"p4":p4,"omni_p": omni_p, "mediation_coef": p3_fit.params[-1] }
