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
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import os, sys

sys.path.append('/code/decima/src/decima/')
import preprocess

from plotnine import *
# %matplotlib inline

# %% [markdown]
# ## Paths

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823"
matrix_file = os.path.join(save_dir, "processed_pseudobulks/combined_inner.h5ad")

# %% [markdown]
# ## Load count matrix

# %%
ad = anndata.read_h5ad(matrix_file)
print(ad.shape)

# %% [markdown]
# ## Check NaNs

# %%
ad.var['frac_nan'] = np.isnan(ad.X).mean(0)

# %%
print(ad.shape)
ad = ad[:, ad.var.frac_nan < .33]
print(ad.shape)

# %%
ad.obs['frac_nan'] = np.isnan(ad.X).mean(1)

# %%
print(ad.shape)
ad = ad[ad.obs.frac_nan < .25]
print(ad.shape)

# %%
ad.X = np.nan_to_num(ad.X)

# %% [markdown]
# ## Aggregate

# %%
# %%time
ad = preprocess.aggregate_anndata(ad)
print(ad.shape)

# %%
ad.obs.loc[ad.obs.dataset!="skin_atlas", 'celltype_coarse'] = None

# %% [markdown]
# ## Calculate per-track statistics

# %%
ad.obs['total_counts'] = ad.X.sum(1)
ad.obs['n_genes'] = np.sum(ad.X > 0, axis=1)

# %% [markdown]
# ## Drop extremely low quality tracks

# %%
## Calculate low thresholds for genes, cells and total counts
for col in ["n_genes", "n_cells", "total_counts"]:
    print(col)
    for quantile in [.1, .2]:
        print(quantile, np.quantile(ad.obs[col], quantile))
    print("")

# %%
drop = (ad.obs.n_cells < 50) & (ad.obs.n_genes < 7670) & (ad.obs.total_counts < 76505)
ad = ad[~drop]
ad.shape

# %% [markdown]
# ## Normalize data

# %%
ad.layers['counts'] = ad.X.copy()

# %%
sc.pp.normalize_total(ad, target_sum=1e6)

# %%
ad.layers['norm'] = ad.X.copy()

# %%
sc.pp.log1p(ad)

# %% [markdown]
# ## Calculate reintroduced size factor

# %%
ad.obs['size_factor'] = ad.X.sum(1)

# %%
(
    ggplot(ad.obs, aes(x="size_factor")) + geom_density() + theme(figure_size=(5, 2))
)

# %% [markdown]
# ## Add per-gene statistics

# %%
ad.var['mean_counts'] = ad.X.mean(0)
ad.var['n_tracks'] = np.sum(ad.X > 0, axis=0)

# %% [markdown]
# ## z-score

# %%
ad_scaled = sc.pp.scale(ad, copy=True)
ad.layers['scaled'] = ad_scaled.X.copy()
del ad_scaled

# %% [markdown]
# ## Count number of tuples

# %%
for col in ['dataset', 'study', 'cell_type', 'tissue', 'disease']:
    print(col)
    print(len(ad.obs[col].unique()))

# %%
print(len(ad.obs[['cell_type', 'tissue', 'disease', 'study']].drop_duplicates()))
print(len(ad.obs[['cell_type', 'tissue', 'disease']].drop_duplicates()))
print(len(ad.obs[['cell_type', 'tissue']].drop_duplicates()))

# %% [markdown]
# ## Save

# %%
out_file = os.path.join(save_dir, "aggregated.h5ad")
#ad.write_h5ad(out_file)
#ad = sc.read_h5ad(out_file)

# %%
