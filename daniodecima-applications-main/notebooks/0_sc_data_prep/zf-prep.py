# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: torch-py310
#     language: python
#     name: torch-py310
# ---

# %% [markdown]
# # Pseudobulk of neural crest cells in zebrafish

# %%
import scanpy as sc
import pandas as pd

pd.options.display.max_columns = None

# %% [markdown]
# ## Load the raw data

# %%
# downloaded from https://www.heartcellatlas.org/ # Heart Global, raw
ad = sc.read('/hpc/mydata/mathias.voges/Projects/research/zf-decima/data/zf_atlas_neuralcrest_v4_release.h5ad')

# %%
#ad.obs.region_finest = ad.obs.region_finest.astype(str)

# %%
# ad.obs.loc[ad.obs.region_finest == 'na', 'region_finest'] = 'SAN_unknown'
# ad.obs.loc[ad.obs.region_finest == 'IVS MID LV', 'region_finest'] = 'SP IVS MID LV'
# ad.obs.loc[ad.obs.region_finest == 'IVS MID RV', 'region_finest'] = 'SP IVS MID RV'

# %%
ad._sanitize()

# %% [markdown]
# ## Filtering 

# %%
ad.obs

# %%
ident_cols = ['annotation']  # adjust these columns as needed
counts = ad.obs[ident_cols].value_counts().reset_index()
counts = counts.rename(columns={'count': 'n_cells'})


# %%
adp = sc.get.aggregate(ad, ident_cols, func='sum')


# %%
adp.X = adp.layers['sum'].astype(int)


# %%
del adp.layers['sum']  # cleanup
adp.obs = adp.obs.merge(counts, how='left')

# %%
adp.obs

# %%
adp.X

# %%
#ad = ad[(ad.obs.cell_or_nuclei == 'Nuclei') &  (ad.obs.cell_state!='unclassified')].copy()

# %%
#ad.obs.cell_state = ad.obs.cell_type.astype(str) + '-' + ad.obs.cell_state.astype(str)

# %%
#ident_cols = ['sample_ID', 'region_finest', 'cell_state', 'cell_type']
#ad.obs = ad.obs[ident_cols].copy()

# %%
#for c in ident_cols:
#    ad.obs[c] = ad.obs[c].astype(str)

# %%
ad._sanitize()

# %% [markdown]
# ## Now the pseudobulking

# %%
ad.var

# %%
counts = ad.obs[ident_cols].value_counts().reset_index()
counts = counts.rename(columns={'count': 'n_cells'})

# %%
ad = ad[ad.obs.merge(counts, how='left').n_cells>=10].copy()

# %%
sc.pp.filter_genes(ad, min_cells=50)

# %%
ad.var['gene_id'] = ad.var.index
ad.var.index = ad.var['gene_name-new']
ad.var.index.name = None

# %%
counts = ad.obs[ident_cols].value_counts().reset_index()
counts = counts.rename(columns={'count': 'n_cells'})

# %%
adp = sc.get.aggregate(ad, ident_cols, func='sum')

# %%
adp.X = adp.layers['sum'].astype(int)

# %%
del adp.layers['sum']

# %% [markdown]
# ### Add cell counts

# %%
adp.obs = adp.obs.merge(counts, how='left')

# %%
adp.obs.rename(columns={'region_finest': 'region'}, inplace=True)

# %%
adp = adp.copy()

# %% [markdown]
# ## Save

# %%
adp.write('heart-pseudobulk.h5ad')
