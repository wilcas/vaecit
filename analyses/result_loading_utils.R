# Scripts used for reading in result sets 

# @knitr load_cit_data
require(tidyverse)
cit_df <- read.table("/home/wcasazza/vaecit/data/rosmap_cit_pca/CIT_groupby_order.txt", header = TRUE) %>% select(snp,probes,peaks,gene,pCausal,pReact) %>% rename(pub_omni_p=pCausal,rev_pub_omni_p=pReact)

alt_lv_list <- list()
for(f in dir("/home/wcasazza/vaecit/data/gsea_data/", full.names = TRUE)){
  model <- gsub(".*/perm_test.*_(lfa|pca|fastica|kernelpca|mmdvae_batch_warmup|mmdvae_batch|mmdvae_warmup|ae|ae_batch)_1_latent_depth_5_cit.csv$","\\1",f)
  if(grepl("rev",f)){
    model <- paste0("rev_", model)
  }
  df <- read.csv(f) %>% rename_all(function(x) paste0(model,"_",x))
  alt_lv_list[[f]] <- df
}
alt_lv_df <- bind_cols(alt_lv_list) # MAIN DF



multisnp_n <- nrow(cit_df %>% select(probes,peaks,gene) %>% unique())
classify_p_value <- function(x,y,n){
  ifelse(x < (0.05 / n) & y > (0.05 / n),
         "Epigenetic Mediation",
         ifelse(x > (0.05 / n) & y < (0.05/ n),
                "Transcriptional Mediation",
                ifelse(x > (0.05 / n) & y > (0.05/ n),
                       "Independent Association",
                       "Unclassified")))
  
}
mediation_df <- cbind(cit_df %>% select(probes,peaks,gene),alt_lv_df) %>% group_by(probes,peaks,gene)%>% summarize_all(max) %>% unique() %>% 
  mutate(
    ae_model = classify_p_value(ae_omni_p,rev_ae_omni_p,multisnp_n),
    ae_batch_model = classify_p_value(ae_batch_omni_p,rev_ae_batch_omni_p, multisnp_n),
    mmdvae_warmup_model = classify_p_value(mmdvae_warmup_omni_p,rev_mmdvae_warmup_omni_p,multisnp_n),
    mmdvae_batch_model = classify_p_value(mmdvae_batch_omni_p, rev_mmdvae_batch_omni_p, multisnp_n),
    pca_model = classify_p_value(pca_omni_p, rev_pca_omni_p, multisnp_n),
    lfa_model = classify_p_value(lfa_omni_p,rev_lfa_omni_p, multisnp_n),
    fastica_model = classify_p_value(fastica_omni_p,rev_fastica_omni_p, multisnp_n),
    kernelpca_model = classify_p_value(kernelpca_omni_p,rev_kernelpca_omni_p, multisnp_n)
  )
