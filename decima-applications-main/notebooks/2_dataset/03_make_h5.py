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

sys.path.append('/code/decima/src/decima/')
import write_hdf5

# %% [markdown]
# ## Paths

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823/"
matrix_file = os.path.join(save_dir, "aggregated.h5ad")

# %% [markdown]
# ## Load matrix

# %%
ad = anndata.read_h5ad(matrix_file)
ad.shape

# %% [markdown]
# ## Write h5 file

# %%
out_file = os.path.join(save_dir, "data.h5")

# %%
ad.var = ad.var[ad.var.columns[1:].tolist() + ['gene_id']]

# %%
# %%time

write_hdf5.write_hdf5(file=out_file, ad=ad, pad=5000)
