# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
import numpy as np
import pandas as pd
import os
import scipy
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from plotnine import *
# %matplotlib inline

# %% [markdown]
# ## Paths

# %%
eqtl_file = 'susie_backmerged_dedup.csv'

# %% [markdown]
# ## Load data

# %%
eqtl_df = pd.read_csv(eqtl_file)

# %% [markdown]
# ## Subset to high-confidence sc-eQTLs

# %%
pip_cut=0.9
effect_cut = 0.01

# %%
eqtl_df = eqtl_df[eqtl_df.pip > pip_cut].copy()
eqtl_df_pred = eqtl_df[eqtl_df.abs_matched_score > effect_cut].copy()

len(eqtl_df), len(eqtl_df_pred)

# %% [markdown]
# # Predicting the beta for positive variants

# %%
print("Decima")
print(scipy.stats.pearsonr(eqtl_df['beta'], eqtl_df['matched_score']))
print(scipy.stats.pearsonr(eqtl_df_pred['beta'], eqtl_df_pred['matched_score']))

print("Borzoi whole blood")
print(scipy.stats.pearsonr(eqtl_df['beta'], eqtl_df['borzoi_wholeblood_score']))
print(scipy.stats.pearsonr(eqtl_df_pred['beta'], eqtl_df_pred['borzoi_wholeblood_score']))

print("Borzoi matched")
print(scipy.stats.pearsonr(eqtl_df['beta'], eqtl_df['borzoi_matched_score']))
print(scipy.stats.pearsonr(eqtl_df_pred['beta'], eqtl_df_pred['borzoi_matched_score']))

# %%
print(accuracy_score(eqtl_df['beta'] > 0, eqtl_df['matched_score'] > 0))
print(accuracy_score(eqtl_df_pred['beta'] > 0, eqtl_df_pred['matched_score'] > 0))

# %% [markdown]
# ## Visualize

# %%
(
    ggplot(eqtl_df, aes(x='beta', y = 'matched_score')) +\
    geom_pointdensity(size=.3) + theme_classic() + theme(figure_size=(2.9, 2.5)) 
    + xlab("Beta") + ylab("   Predicted logFC\n(cell-type matched)")
    + geom_vline(xintercept=0, color='grey', linetype='dashed') 
    + geom_hline(yintercept=0, color='grey', linetype='dashed') 
)

# %%
