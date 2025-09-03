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
import os, sys
import bioframe as bf

# %% [markdown]
# ## Paths

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823/"
matrix_file = os.path.join(save_dir, "aggregated.h5ad")
splits_file = '/gstore/data/resbioai/grelu/borzoi-data/hg38/sequences.bed'

# %% [markdown]
# ## Read inputs

# %%
splits = pd.read_table(splits_file, header=None, names=['chrom', 'start', 'end', 'fold'])

# %%
ad = anndata.read_h5ad(matrix_file)

# %% [markdown]
# ## Overlap gene intervals with Borzoi folds

# %%
overlaps = bf.overlap(ad.var.reset_index(names="gene"), splits, how='left')
overlaps = overlaps[['gene', 'fold_']].drop_duplicates().astype(str)
overlaps.columns=['gene', 'fold']
overlaps.head(2)

# %% [markdown]
# ## List all overlapping folds for each interval

# %%
overlaps = overlaps.groupby('gene').fold.apply(list).reset_index()
overlaps.loc[overlaps.fold.apply(lambda x: x[0] is None), 'fold'] = "none"
overlaps.head(2)

# %%
# Add this to ad.var
ind = ad.var.index
ad.var = ad.var.merge(overlaps, left_index=True, right_on='gene', how='left')
ad.var = ad.var.drop(columns='gene')
ad.var.index = ind

# %% [markdown]
# ## Split datasets into train, val, test based on fold 3

# %%
test_fold='fold3'
val_fold='fold4'
train_folds = [f'fold{f}' for f in range(8) if f'fold{f}' not in [val_fold, test_fold]]

# %%
# The important thing is that the model should not be validated/tested on any gene in the training folds.
# Therefore, we will first assign every gene to test and then remove all that overlap with the other folds
ad.var["dataset"] = "test"

# %%
# If the gene overlaps with val -> move it to val
ad.var.loc[ad.var.fold.apply(lambda x: val_fold in x), "dataset"] = "val"

# If the gene overlaps with train -> Move it to train
ad.var.loc[ad.var.fold.apply(
    lambda x: len(set(x).intersection(train_folds)) > 0), "dataset"] = "train"

# If the gene does not overlap with any folds -> Move it to train
ad.var.loc[ad.var.fold.apply(lambda x: x == "none"), "dataset"] = "train"

# %%
ad.var["fold"] = ad.var["fold"].astype(str)

# %% [markdown]
# ## Check

# %%
ad.var.dataset.value_counts()

# %%
ad.var.gene_name.value_counts().max()

# %%
ad.var = ad.var.reset_index().set_index('gene_name')

# %%
ad.var.index = ad.var.index.tolist()
ad.var_names = ad.var.index.tolist()

# %% [markdown]
# ## Save

# %%
ad.write_h5ad(matrix_file)

# %%
