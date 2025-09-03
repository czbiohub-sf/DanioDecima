# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Evaluate Decima's performance on held-out genes

# %%
import numpy as np
import pandas as pd
import anndata
import os
from grelu.visualize import plot_distribution
from plotnine import *
# %matplotlib inline

# %% [markdown]
# ## Paths

# %%
#save_dir="/gstore/data/resbioai/grelu/decima/20240823"
save_dir="/hpc/mydata/mathias.voges/Projects/research/zf-decima/outputs/grelu/decima"
matrix_file = os.path.join(save_dir, "data_out_zf-Decima_Random_Rep0.h5ad")
h5_file = os.path.join(save_dir, "data.h5")

# %% [markdown]
# ## Load data

# %%
ad = anndata.read_h5ad(matrix_file)

# %%
ad

# %%
for dataset in ad.var.dataset.unique():
    print(dataset)
    print(ad.var.loc[ad.var.dataset==dataset, 'pearson'].mean().round(2))
    print(ad.var.loc[ad.var.dataset==dataset, 'size_factor_pearson'].mean().round(2))
    print(ad.obs[f"{dataset}_pearson"].mean().round(2))

# %% [markdown]
# ## Plot correlations on test genes

# %%
(
    ggplot(ad.obs, aes(x="val_pearson"))
    + geom_histogram(fill="white", color="forestgreen", bins=50)
    + theme_classic() + theme(figure_size=(5,1.5))
    + xlab("Pearson correlation per pseudobulk")
    + ylab("Count")
)

# %%
(
    ggplot(ad.var[ad.var.dataset=="val"], aes(x="pearson"))
    + geom_histogram(fill="white", color="mediumslateblue", bins=50)
    + theme_classic() + theme(figure_size=(5,1.5))
    + xlab("Pearson correlation per gene")
    + ylab("Count")
)

# %%
