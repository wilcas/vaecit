PCA vs MMD VAE 2
================
William Casazza
March 10, 2019

Testing Dimensionality Reduction in Causal Inference (SECOND ANALYSIS)
======================================================================

PCA in Place of Genotype
------------------------

Step 1: Load in Data

``` r
dir()
```

    ## [1] "latex_equations.md"        "latex_equations.Rmd"      
    ## [3] "mnist_test_weights.h5"     "pca_cit_exploration.ipynb"
    ## [5] "pca_mmd_vae_cmp_2.Rmd"     "pca_mmd_vae_cmp.Rmd"      
    ## [7] "remake_xQTL_figure.Rmd"    "tf_exploration.ipynb"     
    ## [9] "vae_genotype_test.ipynb"

``` r
pca_test_list <- list()
for(f in dir("../data/csvs")){
  model <- gsub("cit_(.*)_([0-9])_PCs.*.csv","\\1", f)
  num_pc <- as.numeric(gsub("cit_(.*)_([0-9])_PCs.*.csv","\\2", f))
  tmp <- read.csv(sprintf("../data/csvs/%s", f))
  tmp$model <- model
  tmp$num_pc <- num_pc
  pca_test_list[[f]] <- tmp
}
tail(pca_df <- bind_rows(pca_test_list))
```

    ##     test1_beta0 test1_beta1 test1_rss test1_se0 test1_se1    test1_t0
    ## 295 -0.22148058 -0.06021317  509.3192  5.829853  1.605445 -0.03799077
    ## 296  0.05866757  0.02980633  477.5957  5.689744  1.586761  0.01031111
    ## 297  0.38031456  0.11551821  459.6966  5.767643  1.622023  0.06593934
    ## 298  0.32766881  0.06877564  500.5351  6.199544  1.731978  0.05285370
    ## 299 -0.12554579 -0.02769802  469.3132  5.671778  1.581736 -0.02213517
    ## 300  0.11572159  0.03798028  466.5794  5.857034  1.635249  0.01975771
    ##        test1_t1  test1_p0  test1_p1     test1_r2        p1   test2_beta0
    ## 295 -0.03750560 0.5151449 0.5149515 0.0014046937 0.4030100 -0.2177772354
    ## 296  0.01878439 0.4958886 0.4925103 0.0003527288 0.6752575  0.1396578925
    ## 297  0.07121859 0.4737263 0.4716262 0.0050464913 0.1126255  0.3250976957
    ## 298  0.03970930 0.4789348 0.4841704 0.0015743458 0.3759648 -0.2829597788
    ## 299 -0.01751115 0.5088255 0.5069821 0.0003065464 0.6961290 -0.0008662909
    ## 300  0.02322599 0.4921223 0.4907397 0.0005391559 0.6044727  0.0825461546
    ##      test2_beta1 test2_beta2 test2_rss test2_se0 test2_se1 test2_se2
    ## 295 -0.034239780 -0.07634508  474.8029  5.632906 0.9655209  1.551181
    ## 296  0.022167103  0.05810826  500.3034  5.823745 1.0234969  1.624331
    ## 297  0.008180319  0.07536093  473.0259  5.863369 1.0143943  1.649539
    ## 298 -0.021829386 -0.05495173  477.3213  6.062526 0.9765357  1.692672
    ## 299  0.056678389  0.02334458  463.1454  5.635766 0.9934073  1.571549
    ## 300  0.006654321  0.03089740  510.4338  6.127303 1.0459404  1.710834
    ##          test2_t0     test2_t1    test2_t2  test2_p0  test2_p1  test2_p2
    ## 295 -0.0386616163 -0.035462494 -0.04921740 0.5154122 0.5141374 0.5196171
    ## 296  0.0239807703  0.021658203  0.03577365 0.4904388 0.4913646 0.4857386
    ## 297  0.0554455433  0.008064239  0.04568606 0.4779029 0.4967845 0.4817894
    ## 298 -0.0466735760 -0.022353903 -0.03246450 0.5186039 0.5089127 0.5129427
    ## 299 -0.0001537131  0.057054534  0.01485451 0.5000613 0.4772623 0.4940771
    ## 300  0.0134718567  0.006362046  0.01805984 0.4946284 0.4974632 0.4927992
    ##         test2_r2        p2 test3_beta0  test3_beta1 test3_beta2 test3_rss
    ## 295 0.0035415161 0.2730731 -0.22919106 -0.036682740 -0.06293809  508.6794
    ## 296 0.0017754042 0.4255301  0.05568615  0.021151065  0.02856331  477.3718
    ## 297 0.0022108758 0.3089350  0.37770553  0.007949290  0.11491164  459.6667
    ## 298 0.0016111704 0.4695615  0.32103115 -0.022879591  0.06748401  500.2851
    ## 299 0.0034354127 0.7406641 -0.12508884  0.057246824 -0.02894455  467.7904
    ## 300 0.0003720315 0.6874031  0.11521483  0.006082362  0.03779082  466.5605
    ##     test3_se0 test3_se1 test3_se2    test3_t0     test3_t1    test3_t2
    ## 295  5.830246 1.0344095  1.606275 -0.03931070 -0.035462494 -0.03918263
    ## 296  5.690076 0.9765845  1.587427  0.00978654  0.021658203  0.01799346
    ## 297  5.776522 0.9857458  1.623714  0.06538632  0.008064239  0.07077087
    ## 298  6.205104 1.0235166  1.732509  0.05173663 -0.022353903  0.03895160
    ## 299  5.662575 1.0033703  1.579319 -0.02209045  0.057054534 -0.01832723
    ## 300  5.857457 0.9560388  1.635487  0.01966977  0.006362046  0.02310677
    ##      test3_p0  test3_p1  test3_p2     test3_r2        p3 test4_beta0
    ## 295 0.5156708 0.5141374 0.5156198 0.0026589384 0.4295644 -0.22919106
    ## 296 0.4960978 0.4913646 0.4928256 0.0008214212 0.6294240  0.05568615
    ## 297 0.4739463 0.4967845 0.4718043 0.0051111909 0.8573985  0.37770553
    ## 298 0.4793796 0.5089127 0.4844723 0.0020730069 0.6184602  0.32103115
    ## 299 0.5088077 0.4772623 0.5073074 0.0035502095 0.2039875 -0.12508884
    ## 300 0.4921573 0.4974632 0.4907872 0.0005796081 0.8872700  0.11521483
    ##      test4_beta1 test4_beta2 test4_rss test4_se0 test4_se1 test4_se2
    ## 295 -0.036682740 -0.06293809  508.6794  5.830246 1.0344095  1.606275
    ## 296  0.021151065  0.02856331  477.3718  5.690076 0.9765845  1.587427
    ## 297  0.007949290  0.11491164  459.6667  5.776522 0.9857458  1.623714
    ## 298 -0.022879591  0.06748401  500.2851  6.205104 1.0235166  1.732509
    ## 299  0.057246824 -0.02894455  467.7904  5.662575 1.0033703  1.579319
    ## 300  0.006082362  0.03779082  466.5605  5.857457 0.9560388  1.635487
    ##        test4_t0     test4_t1    test4_t2  test4_p0  test4_p1  test4_p2
    ## 295 -0.03931070 -0.035462494 -0.03918263 0.5156708 0.5141374 0.5156198
    ## 296  0.00978654  0.021658203  0.01799346 0.4960978 0.4913646 0.4928256
    ## 297  0.06538632  0.008064239  0.07077087 0.4739463 0.4967845 0.4718043
    ## 298  0.05173663 -0.022353903  0.03895160 0.4793796 0.5089127 0.4844723
    ## 299 -0.02209045  0.057054534 -0.01832723 0.5088077 0.4772623 0.5073074
    ## 300  0.01966977  0.006362046  0.02310677 0.4921573 0.4974632 0.4907872
    ##         test4_r2     p4    omni_p model num_pc
    ## 295 0.0026589384 0.7284 0.7284000  null      1
    ## 296 0.0008214212 0.4197 0.6752575  null      1
    ## 297 0.0051111909 0.9252 0.9252000  null      1
    ## 298 0.0020730069 0.6498 0.6498000  null      1
    ## 299 0.0035502095 0.3657 0.7406641  null      1
    ## 300 0.0005796081 0.4678 0.8872700  null      1

Step 2 is pulling apart the data and plotting \#\#\# Differences in performance using different numbers of PCs

``` r
pca_df <- rbind_list(pca_test_list)
pca_df <- pca_df %>% mutate(Model = plyr::mapvalues(model,from = c("caus1","ind1","null"), to = c("Full Mediation", "Independent Association", "Null"))) %>% rename(nPC = num_pc)
```

### Interesting Notes

<!-- Caus1 Appears to work better in the way we constructed the data, but it's important to note that it's about a 50 50 shot to call independent association, as constructed, as causal mediation. In the case where there's no association between variables we don't call causal mediation at all, which is a good behavior. -->
MMD VAE
-------

Step 1: Load in the data

``` r
mmd_test_list <- list()
for(f in dir("../data/mmd_vae_tests_4/")){
  model <- gsub("cit_(.*)_mmdvae_([0-9]*)_depth_([0-9]*)_latent.*.csv","\\1", f)
  num_hidden <- as.numeric(gsub("cit_(.*)_mmdvae_([0-9]*)_depth_([0-9]*)_latent.*.csv","\\2", f))
  num_latent <- as.numeric(gsub("cit_(.*)_mmdvae_([0-9]*)_depth_([0-9]*)_latent.*.csv","\\3", f))
  tmp <- read.csv(sprintf("../data/mmd_vae_tests_4/%s", f))
  tmp$model <- model
  tmp$num_hidden <- num_hidden
  tmp$num_latent <- num_latent
  mmd_test_list[[f]] <- tmp
}
tail(mmd_df <- rbind_list(mmd_test_list))
```

    ## # A tibble: 6 x 60
    ##   test1_beta0 test1_beta1 test1_rss test1_se0 test1_se1 test1_t0 test1_t1
    ##         <dbl>       <dbl>     <dbl>     <dbl>     <dbl>    <dbl>    <dbl>
    ## 1      0.0336   -0.0127        537.     1.04      0.943   0.0324 -1.34e-2
    ## 2      0.108     0.00344       521.     1.02      0.938   0.105   3.67e-3
    ## 3     -0.0301    0.00105       488.     0.987     0.895  -0.0305  1.17e-3
    ## 4     -0.0443    0.0220        456.     0.955     0.877  -0.0464  2.51e-2
    ## 5      0.0633    0.000393      577.     1.07      1.01    0.0589  3.88e-4
    ## 6      0.107    -0.0154        527.     1.03      0.974   0.104  -1.58e-2
    ## # … with 53 more variables: test1_p0 <dbl>, test1_p1 <dbl>,
    ## #   test1_r2 <dbl>, p1 <dbl>, test2_beta0 <dbl>, test2_beta1 <dbl>,
    ## #   test2_beta2 <dbl>, test2_rss <dbl>, test2_se0 <dbl>, test2_se1 <dbl>,
    ## #   test2_se2 <dbl>, test2_t0 <dbl>, test2_t1 <dbl>, test2_t2 <dbl>,
    ## #   test2_p0 <dbl>, test2_p1 <dbl>, test2_p2 <dbl>, test2_r2 <dbl>,
    ## #   p2 <dbl>, test3_beta0 <dbl>, test3_beta1 <dbl>, test3_beta2 <dbl>,
    ## #   test3_rss <dbl>, test3_se0 <dbl>, test3_se1 <dbl>, test3_se2 <dbl>,
    ## #   test3_t0 <dbl>, test3_t1 <dbl>, test3_t2 <dbl>, test3_p0 <dbl>,
    ## #   test3_p1 <dbl>, test3_p2 <dbl>, test3_r2 <dbl>, p3 <dbl>,
    ## #   test4_beta0 <dbl>, test4_beta1 <dbl>, test4_beta2 <dbl>,
    ## #   test4_rss <dbl>, test4_se0 <dbl>, test4_se1 <dbl>, test4_se2 <dbl>,
    ## #   test4_t0 <dbl>, test4_t1 <dbl>, test4_t2 <dbl>, test4_p0 <dbl>,
    ## #   test4_p1 <dbl>, test4_p2 <dbl>, test4_r2 <dbl>, p4 <dbl>,
    ## #   omni_p <dbl>, model <chr>, num_hidden <dbl>, num_latent <dbl>

Step 2 is pulling apart the data and plotting \#\#\# Differences in performance using different numbers of latent variables and hidden layers

``` r
mmd_df <- rbind_list(mmd_test_list)
mmd_df <- mmd_df %>%  mutate(Model = plyr::mapvalues(model,from = c("caus1","ind1","null"), to = c("Full Mediation", "Independent Association", "Null"))) %>%rename(nLatent = num_latent, nHidden = num_hidden)
```

VanBUG Draft plots
------------------

``` r
# condition 1 capture association
ggplot(pca_df, aes(p1, fill = Model)) + geom_histogram(position = "identity", alpha = 0.6) + scale_fill_manual(values = c("blue", "red", "gold")) + labs(y = "Count (100 simulations per model)",x = "P Value") + ggtitle("Directed Association Using Genotype PC 1")+ theme_minimal(base_family = "Arial")+ scale_y_continuous(limits=c(0,100))
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](pca_mmd_vae_cmp_2_files/figure-markdown_github/pca%20VanBUG-1.png)

``` r
ggplot(pca_df, aes(p2, fill = Model)) + geom_histogram(position = "identity", alpha = 0.6) + scale_fill_manual(values = c("blue", "red", "gold")) + labs(y = "Count (100 simulations per model)",x = "P Value") + ggtitle("Association PC 1 with Epigenetics Given Expression")+ theme_minimal(base_family = "Arial")+ scale_y_continuous(limits=c(0,100))
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](pca_mmd_vae_cmp_2_files/figure-markdown_github/pca%20VanBUG-2.png)

``` r
ggplot(pca_df, aes(p3 , fill = Model)) + geom_histogram(position = "identity", alpha = 0.6) + scale_fill_manual(values = c("blue", "red", "gold"))+ labs(y = "Count (100 simulations per model)",x = "P Value ") + ggtitle("Association of Expression with Epigenetic Variable Given Genotype PC1")+ theme_minimal(base_family = "Arial") + scale_y_continuous(limits=c(0,100))
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](pca_mmd_vae_cmp_2_files/figure-markdown_github/pca%20VanBUG-3.png)

``` r
ggplot(pca_df, aes(p4 , fill = Model)) + geom_histogram(position = "identity", alpha = 0.6) + scale_fill_manual(values = c("blue", "red", "gold"))+ labs(y = "Count (100 simulations per model)",x = "P Value") + ggtitle("Independence of Genotype PC1 and Expression Given Epigenetic Variable")+ theme_minimal(base_family = "Arial") + scale_y_continuous(limits=c(0,100))
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](pca_mmd_vae_cmp_2_files/figure-markdown_github/pca%20VanBUG-4.png)

``` r
ggplot(pca_df, aes(omni_p , fill = Model)) + geom_histogram(position = "identity", alpha = 0.6) + scale_fill_manual(values = c("blue", "red", "gold"))+ labs(y = "Count (100 simulations per model)",x = "P Value Epigenetic Mediation") + ggtitle("Capturing Epigenetic Mediation of Genotype PC1")+ theme_minimal(base_family = "Arial") + scale_y_continuous(limits=c(0,100))
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](pca_mmd_vae_cmp_2_files/figure-markdown_github/pca%20VanBUG-5.png)

``` r
to_plot_pca <- pca_df %>% gather(key = "test", value = "test_pvalue",p1,p2,p3,p4,omni_p)
ggplot(to_plot_pca, aes(test_pvalue, fill = Model)) + 
  geom_histogram(position = "identity", alpha = 0.6) + 
  scale_fill_manual(values = c("blue", "red", "gold")) +
  labs(y = "Count (100 simulations per model)",x = "P Value Epigenetic Mediation") +
  ggtitle("Capturing Epigenetic Mediation of Genotype PC1")+ 
  theme_minimal(base_family = "Arial") + 
  scale_y_continuous(limits=c(0,100)) + 
  facet_wrap(~test,ncol=2)
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](pca_mmd_vae_cmp_2_files/figure-markdown_github/pca%20debugging-1.png)

``` r
ggplot(mmd_df, aes(p1, fill = Model)) + geom_histogram(position = "identity", alpha = 0.6) + scale_fill_manual(values = c("blue", "red", "gold")) + labs(y = "Count (100 simulations per model)",x = "P Value") + ggtitle("Directed Association 1 Autoencoder LV with Expression")+ theme_minimal(base_family = "Arial")+ scale_y_continuous(limits=c(0,100))
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](pca_mmd_vae_cmp_2_files/figure-markdown_github/mmdvae%20VanBUG-1.png)

``` r
ggplot(mmd_df, aes(p2, fill = Model)) + geom_histogram(position = "identity", alpha = 0.6) + scale_fill_manual(values = c("blue", "red", "gold")) + labs(y = "Count (100 simulations per model)",x = "P Value ") + ggtitle("Association 1 Autoencoder LV with Epigenetic Given Expression")+ theme_minimal(base_family = "Arial")+ scale_y_continuous(limits=c(0,100))
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](pca_mmd_vae_cmp_2_files/figure-markdown_github/mmdvae%20VanBUG-2.png)

``` r
ggplot(mmd_df, aes(p3, fill = Model)) + geom_histogram(position = "identity", alpha = 0.6) + scale_fill_manual(values = c("blue", "red", "gold")) + labs(y = "Count (100 simulations per model)",x = "P Value ") + ggtitle("Association of Epigenetic with Expression Given 1 Autoencoder LV")+ theme_minimal(base_family = "Arial")+ scale_y_continuous(limits=c(0,100))
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](pca_mmd_vae_cmp_2_files/figure-markdown_github/mmdvae%20VanBUG-3.png)

``` r
ggplot(mmd_df, aes(p4, fill = Model)) + geom_histogram(position = "identity", alpha = 0.6) + scale_fill_manual(values = c("blue", "red", "gold")) + labs(y = "Count (100 simulations per model)",x = "P Value ") + ggtitle("Independence of 1 Autoencoder LV and Expression Given Epigenetic")+ theme_minimal(base_family = "Arial")+ scale_y_continuous(limits=c(0,100))
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](pca_mmd_vae_cmp_2_files/figure-markdown_github/mmdvae%20VanBUG-4.png)

``` r
ggplot(mmd_df, aes(omni_p , fill = Model)) + geom_histogram(position = "identity", alpha = 0.6) + scale_fill_manual(values = c("blue", "red", "gold")) + labs(y = "Count (100 simulations per model)",x = "P Value Epigenetic Mediation") + ggtitle("Capturing Epigenetic Mediation of 1 Autoencoder LV")+ theme_minimal(base_family = "Arial")+ scale_y_continuous(limits=c(0,100))
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](pca_mmd_vae_cmp_2_files/figure-markdown_github/mmdvae%20VanBUG-5.png)

``` r
to_plot_mmd <- mmd_df %>% gather(key = "test", value = "test_pvalue",p1,p2,p3,p4,omni_p)
ggplot(to_plot_mmd, aes(test_pvalue, fill = Model)) + 
  geom_histogram(position = "identity", alpha = 0.6) + 
  scale_fill_manual(values = c("blue", "red", "gold")) +
  labs(y = "Count (100 simulations per model)",x = "P Value Epigenetic Mediation") +
  ggtitle("Capturing Epigenetic Mediation of Genotype PC1")+ 
  theme_minimal(base_family = "Arial") + 
  scale_y_continuous(limits=c(0,100)) + 
  facet_wrap(~test,ncol=2)
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](pca_mmd_vae_cmp_2_files/figure-markdown_github/mmd%20vae%20debug-1.png)

``` r
unif_tests <- matrix(runif(4000),1000,4)
ggplot(data.frame(P = apply(unif_tests,1,max)), aes(x = P)) + geom_histogram(fill="gold", alpha = 0.8)+theme_minimal()
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](pca_mmd_vae_cmp_2_files/figure-markdown_github/distribution%20max%204%20uniform%20RV-1.png)
