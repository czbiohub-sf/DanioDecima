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
import anndata
import os, sys

sys.path.append('/code/decima/src/decima')

from evaluate import compare_marker_zscores, compute_marker_metrics
from grelu.visualize import plot_distribution
from plotnine import *

# %matplotlib inline

# %% [markdown]
# ## Paths

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823"
matrix_file = os.path.join(save_dir, "data.h5ad")

# %% [markdown]
# ## Load data

# %%
ad = anndata.read_h5ad(matrix_file)
ad = ad[:, ad.var.dataset == "test"].copy()
ad.shape

# %% [markdown]
# ## Compute z-scores per cell type

# %%
marker_zscores = compare_marker_zscores(ad, key='cell_type')

# %% [markdown]
# ## Compute metrics for marker prediction

# %%
marker_metrics = compute_marker_metrics(marker_zscores, key='cell_type')
print(len(marker_metrics))

# %%
marker_metrics[['pearson', 'auprc', 'auroc']].aggregate(['mean', 'median']) 

# %%
(
    plot_distribution(marker_metrics['auroc'], method="histogram", 
                      binwidth=0.01, fill='white', color='black', bins=50) 
    +theme(figure_size=(2.8,2)) 
    + xlab("         AUROC per cell-type\n(specific vs. nonspecific genes)")
    + ylab("Count")
)
