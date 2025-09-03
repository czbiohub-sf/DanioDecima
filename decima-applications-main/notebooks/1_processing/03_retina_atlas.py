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

import os
import sys
sys.path.append('/code/decima/src/decima/')

import preprocess
import resources

# %% [markdown]
# ## Paths

# %%
matrix_file = '/gstore/data/resbioai/grelu/decima/pseudobulks/retina-snrna-seq-atlas.h5ad'
save_dir="/gstore/data/resbioai/grelu/decima/20240823/processed_pseudobulks"

# %% [markdown]
# ## Load

# %%
ad = anndata.read_h5ad(matrix_file)

# %% [markdown]
# ## Process .obs

# %%
ad.obs = ad.obs.rename(columns={
    'sample_uuid':'sample',
    'study_name':'study'
})

# %%
ad.obs = preprocess.change_values(
    ad.obs, col='cell_type', value_dict={
        'astrocyte':'Astrocyte', 'microglial cell':'Microglia',
    })

# %%
ad.obs['disease'] = "healthy"
ad.obs['organ'] = "retina"

# %% [markdown]
# ## Save

# %%
ad.write_h5ad(os.path.join(save_dir, "retina_processed.h5ad"))

# %%
