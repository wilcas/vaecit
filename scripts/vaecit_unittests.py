import data_model as dm
import cit
import numpy as np
import os
import pandas as pd
import unittest

from scipy import io,stats


class SimulationDataTests(unittest.TestCase):
    def setUp(self):
        self.num_tests = 100
        self.null_data = [dm.generate_null() for i in range(num_tests)]
        self.ind1_data = [dm.generate_ind1() for i in range(num_tests)]
        self.caus1_data = [dm.generate_caus1() for i in range(num_tests)]


    def test_default_sizes(self):
        null_trait, null_exp, null_genotype  = self.null_data[0]
        ind1_trait, ind1_exp, ind1_genotype  = self.ind1_data[0]
        caus1_trait, caus1_exp, caus1_genotype  = self.caus1_data[0]
        self.assertEqual(null_trait.shape, ind1_trait.shape)
        self.assertEqual(ind1_trait.shape, caus1_trait.shape)

        self.assertEqual(null_exp.shape, ind1_exp.shape)
        self.assertEqual(null_exp.shape, ind1_exp.shape)

        self.assertEqual(null_genotype.shape, ind1_genotype.shape)
        self.assertEqual(null_genotype.shape, ind1_genotype.shape)

        self.assertEqual(null_genotype.shape[0], null_trait.shape[0])
        self.assertEqual(null_trait.shape[0], null_exp.shape[0])
        self.assertEqual(null_trait.shape[1], 1)
        self.assertEqual(null_exp.shape[1], 1)
        self.assertFalse(null_genotype.shape[0] > null_genotype.shape[1])

    def test_associations(self):
        null_trait_genotype = [
            stats.pearsonr(trait,np.sum(genotype,1))[0]
            for (trait,_,genotype) in self.null_data]
        ind1_trait_genotype = [
            stats.pearsonr(trait,np.sum(genotype,1))[0]
            for (trait,_,genotype) in self.ind1_data]
        caus1_trait_genotype = [
            stats.pearsonr(trait,np.sum(genotype,1))[0]
            for (trait,_,genotype) in self.caus1_data]
        self.assertTrue((sum(null_trait_genotype) / self.num_tests) < 0.3)
        self.assertTrue((sum(ind1_trait_genotype) / self.num_tests) > 0.3)
        self.assertTrue((sum(caus1_trait_genotype) / self.num_tests) > 0.3)

        null_trait_exp = [
            stats.pearsonr(trait,exp)[0]
            for (trait,exp,_) in self.null_data]
        ind1_trait_exp = [
            stats.pearsonr(trait,exp)[0]
            for (trait,exp,_) in self.ind1_data]
        caus1_trait_exp = [
            stats.pearsonr(trait,exp)[0]
            for (trait,exp,_) in self.caus1_data]
        self.assertTrue((sum(null_trait_genotype) / self.num_tests) < 0.3)
        self.assertTrue((sum(ind1_trait_genotype) / self.num_tests) < 0.3)
        self.assertTrue((sum(caus1_trait_genotype) / self.num_tests) > 0.3)

        null_exp_genotype = [
            stats.pearsonr(gene_exp,np.sum(genotype,1))[0]
            for (_,_,genotype) in self.null_data]
        ind1_exp_genotype = [
            stats.pearsonr(gene_exp,np.sum(genotype,1))[0]
            for (_,_,genotype) in self.ind1_data]
        caus1_exp_genotype = [
            stats.pearsonr(gene_exp,np.sum(genotype,1))[0]
            for (_,_,genotype) in self.caus1_data]
        self.assertTrue((sum(null_trait_genotype) / self.num_tests) < 0.3)
        self.assertTrue((sum(ind1_trait_genotype) / self.num_tests) > 0.3)
        self.assertTrue((sum(caus1_trait_genotype) / self.num_tests) > 0.3)


class CausalInferenceTests(unittest.TestCase):
    def setUp(self):
        self.n = 100
        self.p = 1
        self.num_tests = 100
        self.null_data =  [dm.generate_null(n=n, p=p) for i in range(num_tests)]
        self.caus1_data = [dm.generate_caus1(n=n, p=p) for i in range(num_tests)]
        self.ind1_data = [dm.generate_ind1(n=n, p=p) for i in range(num_tests)]
        self.ind1_reg_results = [
            cit.linreg_with_stats(trait,np.c_[np.ones((n,1)),genotype])
            for (trait,_,genotype) in ind1_data]
        self.ind1_reg_results_null = [
            cit.linreg_with_stats(trait,np.c_[np.ones((n,1))])
            for (trait,_,genotype) in ind1_data]
        self.ind1_neg_reg_results = [
            cit.linreg_with_stats(trait,np.c_[np.ones((n,1)),gene_exp])
            for (trait,gene_exp,_) in ind1_data]

    def test_linreg_with_stats(self):
        beta,RSS,TSS,se,t = self.ind1_reg_results[0]
        self.assertEqual(beta.shape,(2,))
        self.assertEqual(type(RSS), float)
        self.assertEqual(type(TSS), float)
        self.assertEqual(se.shape,(2,))
        self.assertEqual(t.shape,(2,))
        self.assertAlmostEqual(100.,beta[1])

    def test_ftest_association(self):
        results = [
            cit.ftest(fit2,fit1,self.n)[0]
            for (fit1,fit2) in
            zip(self.ind1_reg_results, self.ind1_reg_results_null)]
        self.assertEqual(sum(results) / self.n,0)

        results_neg = [
            cit.ftest(fit2,fit1,self.n)[0]
            for (fit1,fit2) in
            zip(self.ind1_neg_reg_results, self.ind1_reg_results_null)]
        self.assertTrue(sum(results_neg) / self.n > 0.3)
        design_full_neg = [
            np.c_[trait,np.ones((n,1)), gene_exp]
            for (trait,gene_exp,_) in self.ind1_data]
        results_association = [
            cit.test_association(data[:,0],data[:,1:],data[:,2:])[1]
            for data in design_full_neg]
        for test1,test2 in  zip(results_neg, results_association):
            self.assertEqual(test1,test2)

    def test_independence_test(self):
        results_ind1 = [
            cit.test_independence(T,G,L,100)[1]
            for (T,G,L) in self.ind1_data]
        self.assertTrue(sum(results_ind1)/ self.n > 0.8)
        results_caus1 = [
            cit.test_independence(T,G,L,100)[1]
            for (T,G,L) in self.caus1_data]
        self.assertTrue(sum(results_caus1)/ self.n < 0.2)

    def test_cit(self):
        results_ind1 = [
            cit.test_independence(T,G,L,100)['omni_p']
            for (T,G,L) in self.ind1_data]
        self.assertTrue(sum(results_ind1)/ self.n > 0.8)
        results_caus1 = [
            cit.test_independence(T,G,L,100)[1]
            for (T,G,L) in self.caus1_data]
        self.assertTrue(sum(results_caus1)/ self.n < 0.2)['omni_p']


class DataProcessingTests(unittest.TestCase):
    def setUp(self):
        self.fname = "{}.csv".format(hash("My testing file"))


    def tearDown(self):
        os.remove(self.fname)


    def test_write_csv(self):
        results = [{
            'test1': {
                'beta': np.array([-0.0302261, -0.026906 ]),
                'rss': 89.18929626082245,
                'se': np.array([1.23926206, 1.51399957]),
                't': np.array([-0.0243904 , -0.01777147]),
                'p': np.array([0.5097046 , 0.50707135]),
                'r2': 0.0003157255308469109},
            'p1': 0.8607130553386586,
            'test2': {
                'beta': np.array([-0.04961563, -0.07693241, -0.0753968 ]),
                'rss': 121.04793906838377,
                'se': np.array([1.44415772, 1.16499036, 1.76407341]),
                't': np.array([-0.0343561 , -0.06603695, -0.04274017]),
                'p': np.array([0.51366847, 0.52625853, 0.51700219]),
                'r2': 0.006052367175423012},
            'p2': 0.6747287550139232,
            'test3': {'beta': np.array([-0.03289509, -0.05643842, -0.03104445]),
                'rss': 88.80204135042904,
                'se': np.array([1.23722905, 0.85464918, 1.51200843]),
                't': np.array([-0.02658771, -0.06603695, -0.02053193]),
                'p': np.array([0.51057868, 0.52625853, 0.50816961]),
                'r2': 0.004656298451157115},
            'p3': 0.8401713172629797,
            'test4': {'beta': np.array([-0.03289509, -0.05643842, -0.03104445]),
                'rss': 88.80204135042904,
                'se': np.array([1.23722905, 0.85464918, 1.51200843]),
                't': np.array([-0.02658771, -0.06603695, -0.02053193]),
                'p': np.array([0.51057868, 0.52625853, 0.50816961]),
                'r2': 0.004656298451157115},
            'p4': 0.8,
            'omni_p': 0.8607130553386586
        }]
        def count(d):
            return sum([count(v) if isinstance(v, dict) else 1 for v in d.values()])
        cit.write_csv(results, self.fname)
        df = pd.read_csv(self.fname)
        self.assertEqual(df.shape,(1,count(results)))
        self.assertTrue('test1_beta0' in df.keys())
        self.assertTrue('test2_beta1' in df.keys())
        self.assertTrue('p3' in df.keys())


    def test_stats_dict(self):
        test_result = {
            'beta': np.array([-0.03289509, -0.05643842, -0.03104445]),
            'rss': 88.80204135042904,
            'se': np.array([1.23722905, 0.85464918, 1.51200843]),
            't': np.array([-0.02658771, -0.06603695, -0.02053193]),
            'p': np.array([0.51057868, 0.52625853, 0.50816961]),
            'r2': 0.004656298451157115}
        tss = test_result['rss'] / (-1 * test_result['r2'] + 1)
        result = stats_dict(
        test_result['beta'],
        test_result['rss'],
        tss,
        test_result['se'],
        test_result['t'],
        100)
        self.assertAlmostEqual(test_result,result)


@unittest.skipIf(os.path.exists("/zfs3/scratch/saram_lab/ROSMAP/"))
class LoadingDataTests(unittest.TestCase):

if __name__ == '__main__':
    main()
