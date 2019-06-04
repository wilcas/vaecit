#!/usr/bin/env python
# coding: utf-8

# # Testing new training
# I've added a warm-up period and batch normalization after every layer except for output layers a la 
# >Sønderby, Casper Kaae, et al. How to Train Deep Variational Autoencoders and Probabilistic Ladder Networks. no. Icml, 2016, http://arxiv.org/abs/1602.02282.
# 
# Additionally I'm experimenting with the same architecture *without* using any sort of variational inference (i.e. just a 
# regular autoencoder with $\mathcal{NLL}$ loss. I'm using this space to test small examples since I have no GPU access for now.
# 
# First we'll load up some of the relevant packages and switch to where I'm keeping the python source code for this project:

# In[1]:


import os
os.chdir("../scripts")
import data_model as dm
import vae_torch as vt


# In[2]:


import numpy as np
import torch 
import pandas as pd
torch.manual_seed(0)
np.random.seed(0)
trait,gene_exp,genotype = dm.generate_caus1(500,300)


# ## Variational Stuff
# ### Baseline: the training I've already been using

# In[3]:


lv_baseline = dm.reduce_genotype(genotype, "mmdvae",1,'testing',vae_depth=5,model_dir="/home/wcasazza/projects/def-saram/wcasazza/vaecit/analyses/tmp/")


#   ![image.png](attachment:image.png) <!--![testing_mmdvaeloss_5_depth.png](tmp/testing_mmdvaeloss_5_depth.png)-->

# In[ ]:


lv_batch = dm.reduce_genotype(genotype, "mmdvae_batch",1,'testing',vae_depth=5,model_dir="/home/wcasazza/projects/def-saram/wcasazza/vaecit/analyses/tmp/")


# ![testing_mmdvae_batchloss_5_depth.png](tmp/testing_mmdvae_batchloss_5_depth.png)

# In[ ]:


lv_warmup = dm.reduce_genotype(genotype, "mmdvae_warmup",1,'testing',vae_depth=5,model_dir="/home/wcasazza/projects/def-saram/wcasazza/vaecit/analyses/tmp/")


# ![testing_mmdvae_warmuploss_5_depth.png](tmp/testing_mmdvae_warmuploss_5_depth.png)

# In[ ]:


lv_batch_warmup = dm.reduce_genotype(genotype, "mmdvae_batch_warmup",1,'testing',vae_depth=5,model_dir="/home/wcasazza/projects/def-saram/wcasazza/vaecit/analyses/tmp/")


# ![testing_mmdvae_batch_warmuploss_5_depth.png](tmp/testing_mmdvae_batch_warmuploss_5_depth.png)

# ## No variational loss
# Just a plain autoencoder

# In[ ]:


lv_no_var = dm.reduce_genotype(genotype, "ae",1,'testing',vae_depth=5,model_dir="/home/wcasazza/projects/def-saram/wcasazza/vaecit/analyses/tmp/")


# ![testing_aeloss_5_depth.png](tmp/testing_aeloss_5_depth.png)

# In[ ]:


lv_no_var_batch = dm.reduce_genotype(genotype, "ae_batch",1,'testing',vae_depth=5,model_dir="/home/wcasazza/projects/def-saram/wcasazza/vaecit/analyses/tmp/")


# ![testing_ae_batchloss_5_depth.png](tmp/testing_ae_batchloss_5_depth.png)
