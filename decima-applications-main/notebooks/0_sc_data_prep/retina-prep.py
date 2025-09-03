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

# %% [markdown]
# # Pseudobulk the Retina (HRCA) cell atlas data

# %%
import scanpy as sc
import pandas as pd
pd.options.display.max_columns = None

# %% [markdown]
# ## Load data

# %%
ad = sc.read('cellxgene-retina-snRNA-seq.h5ad') # from https://cellxgene.cziscience.com/collections/4c6eaf5c-6d57-4c76-b1e9-60df8c655f1e sn all cells
ad = ad.raw.to_adata()

# %%
ident_cols = ['cell_type', 'majorclass', 'donor_id', 'sex', 'donor_age', 'sample_uuid', 'tissue', 'sample_source', 'study_name', 'development_stage']

# %%
ad.obs = ad.obs[ident_cols]

# %% [markdown]
# ## Filter

# %%
sc.pp.filter_genes(ad, min_cells=50)

# %%
ad.var.index = ad.var.feature_name

# %%
ad.var.index.name = None

# %% jupyter={"outputs_hidden": true}
ad.var = ad.var.iloc[:, 0:0]

# %%
ad.obs.donor_id = ad.obs.donor_id.astype(str) + '-' + ad.obs.sex.astype(str) + '-' + ad.obs.donor_age.astype(str)

# %%
ad.obs.drop(columns=['sex', 'donor_age'], inplace=True)

# %%
ident_cols = ['cell_type', 'donor_id', 'sample_uuid', 'tissue', 'study_name']
ad.obs = ad.obs[ident_cols]

# %% [markdown]
# ## Pseudobulk

# %%
adp = sc.get.aggregate(ad, ad.obs.columns, func='sum')

# %%
adp.X = adp.layers['sum']
del adp.layers['sum']

# %% [markdown]
# ## Add cell counts

# %%
counts = ad.obs[ident_cols].value_counts().reset_index()
counts = counts.rename(columns={'count': 'n_cells'})

# %%
adp.obs = adp.obs.merge(counts, how='left')

# %%
adp = adp[adp.obs.n_cells>=10].copy()

# %% [markdown]
# ## Save

# %%
adp.write('retina-snrna-seq-atlas.h5ad')
