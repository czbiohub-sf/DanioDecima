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
matrix_file='/gstore/data/resbioai/grelu/decima/pseudobulks/bca_pseudobulk.h5ad'
save_dir="/gstore/data/resbioai/grelu/decima/20240823/processed_pseudobulks"

# %% [markdown]
# ## Load

# %%
ad = anndata.read_h5ad(matrix_file)

# %%
ad.obs.index = ad.obs.index.astype(str)
ad.obs_names_make_unique()

print(ad.shape)
display(ad.obs.head(1))
display(ad.var.head(1))

# %% [markdown]
# ## Process .obs

# %%
ad.obs = ad.obs[['project_code', 'sample_ID', 'cell_type', 'sample_status', 'treatment', 'n_cells', 'region', 'subregion']]

# %%
ad.obs = ad.obs.rename(columns={'project_code':'study', 'sample_ID':'sample', 'sample_status':'disease'})

# %%
ad.obs['tissue'] = ad.obs['region'].astype(str) + '_' + ad.obs['subregion'].astype(str)

# %%
# Remove unannotated cells and artifacts

print(ad.shape)
ad = ad[~ad.obs.cell_type.isin(['unannoted'])]
print(ad.shape)

# %%
# Drop cancers
print(ad.shape)
ad = ad[ad.obs.region!="Tumour", :]
print(ad.shape)

# %%
# Match cell type terms to scimilarity
br_sc_cell_type_dict = {
    'Fibroblast':'fibroblast'
}

ad.obs = preprocess.change_values(ad.obs, col='cell_type', value_dict=br_sc_cell_type_dict)

# %%
# Match cell type terms to scimilarity
br_disease_dict = {
    "Alzheimer disease":"Alzheimer's disease",
    "Alzheimer’s disease":"Alzheimer's disease",
    'Multiple sclerosis':'multiple sclerosis',
    'Healthy':'healthy',
}

ad.obs = preprocess.change_values(ad.obs, col='disease', value_dict=br_disease_dict)

# %%
ad.obs['organ'] = 'CNS'

# %%
ad.obs.disease = ad.obs.disease.astype(str)
ad.obs.treatment = ad.obs.treatment.astype(str)

# %%
ad.obs.loc[(ad.obs.disease == "Temporal lobe epilepsy"), 'disease'] = ad.obs.loc[(ad.obs.disease == "Temporal lobe epilepsy")].disease + '_' + ad.obs.loc[(ad.obs.disease == "Temporal lobe epilepsy")].treatment
ad.obs = ad.obs.drop(columns='treatment')

# %% [markdown]
# ## Save

# %%
ad.write_h5ad(os.path.join(save_dir, "brain_processed.h5ad"))
