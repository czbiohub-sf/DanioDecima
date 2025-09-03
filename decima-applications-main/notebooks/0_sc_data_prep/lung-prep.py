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
# # Pseudobulk the HLCA (Human lung cell atlas) data

# %%
import scanpy as sc
import pandas as pd

pd.options.display.max_columns = None

# %% [markdown]
# ## Load the data

# %%
# downloaded from https://data.humancellatlas.org/hca-bio-networks/lung/atlases/lung-v1-0 (full atlas = core + extension)
ad = sc.read('hlca.h5ad') # 2,282,447 cells 
ad = ad.raw.to_adata()

# %% [markdown]
# ## Filtering 

# %%
ad = ad[ad.obs.assay.str.startswith('10x') & (ad.obs.suspension_type == 'cell') & (ad.obs["3'_or_5'"] == "3'")].copy()

# %%
ad = ad[~ad.obs.ann_level_4.isin(['Unknown', 'None'])].copy()

# %%
ad = ad[ad.obs.ann_level_4.notna()].copy()

# %%
ad.obs['tissue'] = ad.obs.tissue.astype(str) + ['' if x == 'nan' else f'-{x}' for x in ad.obs.tissue_level_2.astype(str)]

# %%
ad.obs['donor_id'] = ad.obs.donor_id.astype(str) + '-' + ad.obs['sample'].astype(str)

# %%
ident_cols = ['donor_id', 'ann_level_4', 'dataset', 'lung_condition', 'tissue']

# %%
ad.obs = ad.obs[ident_cols].copy()

# %%
for c in ident_cols:
    ad.obs[c] = ad.obs[c].astype(str)

# %%
ad._sanitize()

# %% [markdown]
# ## Now the pseudobulking

# %%
counts = ad.obs[ident_cols].value_counts().reset_index()
counts = counts.rename(columns={'count': 'n_cells'})

# %%
ad = ad[ad.obs.merge(counts, how='left').n_cells>=10].copy()

# %%
sc.pp.filter_genes(ad, min_cells=50)

# %%
ad.var['gene_id'] = ad.var.index
ad.var.index = ad.var.feature_name
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
adp.obs.rename(columns={'ann_level_4': 'celltype', 'lung_condition': 'condition'}, inplace=True)

# %% [markdown]
# ## Save

# %%
adp = adp.copy()

# %%
adp.write('lung-pseudobulk.h5ad')
