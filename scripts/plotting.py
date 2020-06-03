

#%%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import zscore
phen_df = pd.read_csv("/home/wcasazza/ROSMAP_PHEN.csv", dtype={'Unnamed: 0':np.str_}).set_index('Unnamed: 0')
phen_df[["age death"]].hist()
plt.show()
print(phen_df[["age death"]].mean())
print(phen_df[["age death"]].median())
print((phen_df[["age death"]] > phen_df[["age death"]].mean()).sum())
print((phen_df[["age death"]] < phen_df[["age death"]].mean()).sum())
#%%
phen_df[["age death"]] = zscore(phen_df[["age death"]])
phen_df[["age death"]].hist()
plt.show()
print(phen_df[["age death"]].mean())
print(phen_df[["age death"]].median())
print((phen_df[["age death"]] > phen_df[["age death"]].mean()).sum())
print((phen_df[["age death"]] < phen_df[["age death"]].mean()).sum())




#%%
