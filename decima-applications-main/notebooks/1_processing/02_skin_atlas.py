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
import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import sys
sys.path.append('/code/decima/src/decima/')

import preprocess
import resources

# %% [markdown]
# ## Paths

# %%
matrix_file = "/gstore/data/resbioai/grelu/decima/pseudobulks/skin-pseudobulk.h5ad"
save_dir="/gstore/data/resbioai/grelu/decima/20240823/processed_pseudobulks"

# %% [markdown]
# ## Load

# %%
ad = anndata.read_h5ad(matrix_file)
ad.obs.index = ad.obs.index.astype(str)
ad.obs_names_make_unique()

print(ad.shape)
display(ad.obs.head(1))
display(ad.var.head(1))

# %% [markdown]
# ## Process .obs

# %%
ad.obs = ad.obs.drop(columns='study_ID')
ad.obs = ad.obs.rename(columns = {'sample_ID':'sample', 'disease_status':'disease', 'study_accession':'study',
       'celltype_granular': 'cell_type'})

# %%
ad.obs['tissue'] = 'skin'
ad.obs['organ'] = 'skin'

# %%
# Match cell type terms to scimilarity
disease_dict = {'Healthy':'healthy'}
ad.obs = preprocess.change_values(ad.obs, col='disease', value_dict=disease_dict)

# %% [markdown]
# ## Save

# %%
ad.write_h5ad(os.path.join(save_dir, "skin_processed.h5ad"))

# %%
