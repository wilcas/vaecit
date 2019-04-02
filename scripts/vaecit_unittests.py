import data_model as dm
import cit
import numpy as np
import os
import pandas as pd
import random
import unittest

from scipy import io,stats


class SimulationDataTests(unittest.TestCase):
    def setUp(self):
        self.num_tests = 100
        self.null_data = [dm.generate_null() for i in range(self.num_tests)]
        self.ind1_data = [dm.generate_ind1() for i in range(self.num_tests)]
        self.caus1_data = [dm.generate_caus1() for i in range(self.num_tests)]


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
            stats.pearsonr(trait.flatten(),np.sum(genotype,1))[0]
            for (trait,_,genotype) in self.null_data]
        ind1_trait_genotype = [
            stats.pearsonr(trait.flatten(),np.sum(genotype,1))[0]
            for (trait,_,genotype) in self.ind1_data]
        caus1_trait_genotype = [
            stats.pearsonr(trait.flatten(),np.sum(genotype,1))[0]
            for (trait,_,genotype) in self.caus1_data]
        self.assertTrue(abs(sum(null_trait_genotype) / self.num_tests) < 0.3)
        self.assertTrue(abs(sum(ind1_trait_genotype) / self.num_tests) > 0.3)
        self.assertTrue(abs(sum(caus1_trait_genotype) / self.num_tests) > 0.3)

        null_trait_exp = [
            stats.pearsonr(trait.flatten(),exp.flatten())[0]
            for (trait,exp,_) in self.null_data]
        ind1_trait_exp = [
            stats.pearsonr(trait.flatten(),exp.flatten())[0]
            for (trait,exp,_) in self.ind1_data]
        caus1_trait_exp = [
            stats.pearsonr(trait.flatten(),exp.flatten())[0]
            for (trait,exp,_) in self.caus1_data]
        self.assertTrue((sum(null_trait_genotype) / self.num_tests) < 0.3)
        self.assertTrue((sum(ind1_trait_genotype) / self.num_tests) > 0.3)
        self.assertTrue((sum(caus1_trait_genotype) / self.num_tests) > 0.3)

        null_exp_genotype = [
            stats.pearsonr(gene_exp.flatten(),np.sum(genotype,1))[0]
            for (_,gene_exp,genotype) in self.null_data]
        ind1_exp_genotype = [
            stats.pearsonr(gene_exp.flatten(),np.sum(genotype,1))[0]
            for (_,gene_exp,genotype) in self.ind1_data]
        caus1_exp_genotype = [
            stats.pearsonr(gene_exp.flatten(),np.sum(genotype,1))[0]
            for (_,gene_exp,genotype) in self.caus1_data]
        self.assertTrue((sum(null_trait_genotype) / self.num_tests) < 0.3)
        self.assertTrue((sum(ind1_trait_genotype) / self.num_tests) > 0.3)
        self.assertTrue((sum(caus1_trait_genotype) / self.num_tests) > 0.3)


class CausalInferenceTests(unittest.TestCase):
    def setUp(self):
        self.n = 100
        self.p = 1
        self.num_tests = 100
        self.null_data =  [dm.generate_null(n=self.n, p=self.p) for i in range(self.num_tests)]
        self.caus1_data = [dm.generate_caus1(n=self.n, p=self.p) for i in range(self.num_tests)]
        self.ind1_data = [dm.generate_ind1(n=self.n, p=self.p) for i in range(self.num_tests)]
        self.ind1_reg_results = [
            cit.linreg_with_stats(trait,np.c_[np.ones((self.n,1)),genotype])
            for (trait,_,genotype) in self.ind1_data]
        self.ind1_reg_results_null = [
            cit.linreg_with_stats(trait,np.c_[np.ones((self.n,1))])
            for (trait,_,genotype) in self.ind1_data]
        self.ind1_neg_reg_results = [
            cit.linreg_with_stats(trait,np.c_[np.ones((self.n,1)),gene_exp])
            for (trait,gene_exp,_) in self.ind1_data]

    def test_linreg_with_stats(self):
        beta,RSS,TSS,se,t = self.ind1_reg_results[0]
        self.assertEqual(beta.shape,(2,))
        self.assertTrue(isinstance(RSS, float))
        self.assertTrue(isinstance(TSS, float))
        self.assertEqual(se.shape,(2,))
        self.assertEqual(t.shape,(2,))
        self.assertAlmostEqual(20.,round(beta[1]))

    def test_ftest_association(self):
        results = [
            cit.ftest(fit2,fit1,self.n)[0]
            for (fit1,fit2) in
            zip(self.ind1_reg_results, self.ind1_reg_results_null)]
        self.assertAlmostEqual(sum(results) / self.n,0)

        results_neg = [
            cit.ftest(fit2,fit1,self.n)[0]
            for (fit1,fit2) in
            zip(self.ind1_neg_reg_results, self.ind1_reg_results_null)]
        design_full_neg = [
            np.c_[trait,np.ones((n,1)), gene_exp]
            for (trait,gene_exp,_) in self.ind1_data]
        results_association = [
            cit.test_association(data[:,0],data[:,1:],data[:,2:])[1]
            for data in design_full_neg]
        for test1,test2 in  zip(results_neg, results_association):
            self.assertAlmostEqual(test1,test2)

    def test_independence_test(self):
        results_ind1 = [
            cit.test_independence(T,G,L,100)[1]
            for (T,G,L) in self.ind1_data]
        self.assertTrue(sum(results_ind1)/ self.n < 0.2)
        results_caus1 = [
            cit.test_independence(T,G,L,100)[1]
            for (T,G,L) in self.caus1_data]
        self.assertTrue(sum(results_caus1)/ self.n < 0.2)
        results_null = [
            cit.test_independence(T,G,L,100)[1]
            for (T,G,L) in self.null_data]
        self.assertTrue(sum(results_null)/ self.n > 0.2)

    def test_cit(self):
        results_ind1 = [
            cit.cit(T,G,L,100)['omni_p']
            for (T,G,L) in self.ind1_data]
        print(sum(results_ind1)/ self.n)
        self.assertTrue(sum([elem > 0.3 for elem in results_ind1]) > 50)
        results_caus1 = [
            cit.cit(T,G,L,100)['omni_p']
            for (T,G,L) in self.caus1_data]
        print(sum(results_caus1)/ self.n)
        self.assertTrue(sum([elem < 0.3 for elem in results_caus1]) < 50)


class DataProcessingTests(unittest.TestCase):

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
            if isinstance(d,np.ndarray):
                return len(d)
            return sum([count(v) if isinstance(v, (dict,np.ndarray)) else 1 for v in d.values()])
        fname = "tmp{}.csv".format(hash("William Casazza"))
        cit.write_csv(results, fname)
        df = pd.read_csv(fname)
        os.remove(fname)
        self.assertEqual(df.shape,(1,count(results[0])))
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
        result = cit.stats_dict(
        test_result['beta'],
        test_result['rss'],
        tss,
        test_result['se'],
        test_result['t'],
        100)
        for (test1,test2) in zip(test_result.values(),result.values()):
            if isinstance(test1,int) or isinstance(test1,float):
                self.assertAlmostEqual(test1,test2)
            else:
                for (testA,testB) in zip(test1,test2):
                    self.assertAlmostEqual(testA,testB)



    def test_match_samples(self):
        num = "1234567890"
        id_size = 10
        num_samples = 5
        ids_present = 9
        population = ["".join([random.choice(num) for i in range(id_size)])
            for i in range(id_size)]
        samples = [np.array(random.sample(population,ids_present)) for i in range(num_samples)]
        shared_idx = dm.match_samples(*samples)
        self.assertTrue(len(shared_idx[0]) > 0)
        for i in range(1,num_samples):
            for (test1,test2) in zip(samples[i-1][shared_idx[i-1]], samples[i][shared_idx[i]]):
                self.assertEqual(test1,test2)

    def test_get_mediator(self):# TODO: Finish
        raise NotImplementedError("No test case")


    @unittest.skipUnless(os.path.exists("/zfs3/scratch/saram_lab/ROSMAP/"),
        "Meant for issues with reading in specific dataset")
    def test_get_snp_groups(self):# TODO: Finish
        raise NotImplementedError("No test case")


    def test_compute_pcs(self):# TODO: finish this
        raise NotImplementedError("No test case")


    def test_reduce_genotype(self):# TODO: finish this
        raise NotImplementedError("No test case")


@unittest.skipUnless(os.path.exists("/zfs3/scratch/saram_lab/ROSMAP/"),
    "Meant for issues with reading in specific dataset")
class LoadingDataTests(unittest.TestCase):
    def setUp(self):
        base_path = "/zfs3/scratch/saram_lab/ROSMAP/data/"
        self.gene_exp_file = os.path.join(base_path,"expressionAndPhenotype.mat")
        self.methyl_file = os.path.join(base_path,"methylationSNMnorm.mat")
        self.acetyl_file = os.path.join(base_path,"acetylationNormNoNaN.mat")
        self.geno_file_hrc = os.path.join(base_path,"genotypeImputed/hrc/snpMatrix/hrc/chr20.raw")
        self.geno_file_1kg = os.path.join(base_path,"genotypeImputed/1kg/snpMatrix/1kg/snpMatrixChr20a.csv")

    def test_load_genotype(self):
        rsids = ['rs11907414', 'rs73121632', 'rs11907414','rs6016785']
        res_1kg = dm.load_genotype(self.geno_file_1kg, rsids)
        res_hrc = dm.load_genotype(self.geno_file_hrc, rsids)
        self.assertEqual(res_1kg[0].shape[1],3)
        self.assertEqual(res_1kg[0],res_hrc[0])
        self.assertEqual(res_1kg[1],res_hrc[1])
        self.assertEqual(res_1kg[2].shape, res_hrc[2].shape)
        self.assertRaises(NotImplementedError,dm.load_genotype("blah.tch", rsids))
        self.assertRaises(LookupError,dm.load_genotype(self.geno_file_1kg,["rs119"]))


    def test_load_methylation(self):
        raise NotImplementedError("No test case")


    def test_load_acetylation(self):# TODO: finish this
        raise NotImplementedError("No test case")


    def test_reduce_genotype(self):# TODO: finish this
        raise NotImplementedError("No test case")


if __name__ == '__main__':
    unittest.main()
