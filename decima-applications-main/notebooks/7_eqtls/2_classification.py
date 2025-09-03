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
import numpy as np
import pandas as pd
import os
import scipy
import yaml
from tqdm import tqdm
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt
from plotnine import *
# %matplotlib inline

# %% [markdown]
# # Paths

# %%
eqtl_file = 'susie_backmerged_dedup.csv'

# %% [markdown]
# ## Load data

# %%
eqtl_df = pd.read_csv(eqtl_file)

# %% [markdown]
# ## Precision at differentiating high pip from negatives

# %%
pip_cut=0.9

# %%
rows = []

for celltype in tqdm(set(eqtl_df.celltype)):
    sub = eqtl_df[eqtl_df.celltype == celltype].copy()
    positive_genes = sub[sub.pip > pip_cut].gene_id
    negative_genes = sub[sub.cs_id == "negative"].gene_id
    sub = sub[sub.gene_id.isin(set(positive_genes).intersection(negative_genes))]
    sub = sub[(sub.pip > pip_cut) | (sub.cs_id == "negative")]
    labels = sub.pip > pip_cut
    positives = labels.sum()
    negatives = len(sub[sub.cs_id == "negative"])
    if positives == 0 or negatives == 0:
        continue
    decima_ap = average_precision_score(labels, sub['abs_matched_score'])
    borzoi_blood_ap = average_precision_score(labels, sub['abs_borzoi_wholeblood_score'])
    borzoi_ap = average_precision_score(labels, sub['abs_borzoi_matched_score'])
    distance_ap = average_precision_score(labels, -sub['abspos_rel_TSS'])
    rows.append({'celltype':celltype, 'decima_ap':decima_ap,
                'borzoi_blood_ap':borzoi_blood_ap, 'borzoi_ap':borzoi_ap,
                'distance_ap':distance_ap, 'positives':positives, 'negatives':negatives,
                })

metrics = pd.DataFrame(rows)

# %% [markdown]
# ## Plot Decima performance

# %%
metrics.celltype = metrics.celltype.apply(
    lambda x: x[0].upper() + x[1:])

# %%
metrics.celltype = pd.Categorical(metrics.celltype, categories=metrics.sort_values('decima_ap', ascending=False).celltype.tolist())
(
    ggplot(metrics, aes(x='celltype', y='decima_ap')) + geom_col() + theme_classic()
    + theme(figure_size=(3.5, 3)) 
    + theme(axis_text_x=element_text(rotation=90, hjust=.5))
     + xlab("") + ylab("AUPRC")
)

# %% [markdown]
# ## Compare to baseline

# %%
pval = scipy.stats.wilcoxon(metrics['decima_ap'],metrics['distance_ap'])[1]
(
    ggplot(metrics, aes(x='distance_ap', y = 'decima_ap')) +\
    geom_point(size=.5) + theme_classic() + theme(figure_size=(3,2.5)) + geom_abline(intercept=0, slope=1) 
    + xlab("AUPRC per cell type:\n        Distance") + ylab("AUPRC per cell type:\n        Decima")
    + geom_text(x=0.5, y=0.125, label=f"P-value = {np.round(pval, 4)}") 
)

# %% [markdown]
# ## Compare to Borzoi

# %%
pval = scipy.stats.wilcoxon(metrics['decima_ap'], metrics['borzoi_blood_ap'])[1]
(
    ggplot(metrics, aes(x='borzoi_blood_ap', y = 'decima_ap')) + geom_point(size=.5) 
    + theme_classic() + theme(figure_size=(3, 2.5)) + geom_abline(intercept=0, slope=1) 
    + xlab("AUPRC per celltype:\nBorzoi (Whole Blood)") + ylab("AUPRC per celltype:\n          Decima ")
    + geom_text(x=0.5, y=0.125, label=f"P-value = {np.round(pval, 4)}") 
)

# %%
pval = scipy.stats.wilcoxon(metrics['decima_ap'], metrics['borzoi_ap'])[1]
(
    ggplot(metrics, aes(x='borzoi_ap', y = 'decima_ap')) + geom_point(size=.5) 
    + theme_classic() + theme(figure_size=(3, 2.5)) + geom_abline(intercept=0, slope=1) 
    + xlab("AUPRC per celltype:\nBorzoi (Matched)") + ylab("AUPRC per celltype:\n          Decima ")
    + geom_text(x=0.6, y=0.125, label=f"P-value = {np.round(pval, 4)}") 
)
