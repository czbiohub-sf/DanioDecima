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
from tqdm import tqdm
import json

sys.path.append('/code/decima/src/decima/')
import preprocess

# %% [markdown]
# ## paths

# %%
save_dir = '/gstore/data/resbioai/grelu/decima/20240823/processed_pseudobulks/'

# %%
sc_file = os.path.join(save_dir, 'scimilarity_processed.h5ad')
br_file = os.path.join(save_dir, 'brain_processed.h5ad')
sk_file = os.path.join(save_dir, 'skin_processed.h5ad')
ret_file = os.path.join(save_dir, 'retina_processed.h5ad')

# %% [markdown]
# ## Load

# %%
# %%time
sc = anndata.read_h5ad(sc_file)
br = anndata.read_h5ad(br_file)
sk = anndata.read_h5ad(sk_file)
ret = anndata.read_h5ad(ret_file)

# %%
# %%time
gtf = resources.load_gtf(
    file='/gstore/data/resbioai/grelu/decima/refdata-gex-GRCh38-2020-A/genes/genes.gtf',
    feature="transcript")

genes20 = preprocess.merge_transcripts(gtf)

gtf = resources.load_gtf(
    file='/gstore/data/resbioai/grelu/decima/refdata-gex-GRCh38-2024-A/genes/genes.gtf',
    feature="transcript")

genes24 = preprocess.merge_transcripts(gtf)

# %%
genes24 = genes24[~(genes24.index.isin(genes20.index))]
print(len(genes20), len(genes24))

# %% [markdown]
# ## Process scimilarity data

# %% [markdown]
# ### Match gene names to cellranger

# %%
sc.var = sc.var.merge(genes20, left_index=True, right_index=True, how="left")
sc.var.head(2)

# %%
preprocess.match_cellranger_2024(sc, genes24=genes24)

# %% [markdown]
# ### Match remaining gene names to NCBI

# %%
sc.var['symbol'] = None

# %%
unm = sc.var.index[sc.var.chrom.isna()].tolist()
len(unm)

# %%
# %%time
# ncbi = !datasets summary gene symbol {" ".join(unm)} --report gene

# %%
ncbi = preprocess.load_ncbi_string(ncbi)
ncbi = ncbi[(ncbi.gene_id is None) or (~ncbi.gene_id.isin(sc.var.gene_id))]
print(len(ncbi), ncbi['gene_id'].value_counts().max(), ncbi.symbol.value_counts().max())

# %%
preprocess.match_ncbi(sc, ncbi)

# %%
sc.var.chrom.isna().sum(), sc.var.gene_id.value_counts().max()

# %% [markdown]
# ## Process skin atlas data

# %% [markdown]
# ### match gene names to cellranger

# %%
sk.var = sk.var.merge(genes20, left_index=True, right_index=True, how="left")
display(sk.var.head(2))
print(sk.var.chrom.isna().sum())

# %%
preprocess.match_cellranger_2024(sk, genes24=genes24)

# %%
sk.var['symbol'] = None

# %%
preprocess.match_ref_ad(sk, sc)

# %%
print(sk.var.chrom.isna().sum()), sk.var.gene_id.value_counts().max()

# %% [markdown]
# ### match remaining gene names to NCBI

# %%
unm = sk.var.index[sk.var.chrom.isna()].tolist()
len(unm)

# %%
unm_arrs = np.array_split(unm, 50)
df = []

for unm in tqdm(unm_arrs):
    # ncbi = !datasets summary gene symbol {" ".join(list(unm))} --report gene
    try:
        curr_df = preprocess.load_ncbi_string(ncbi)
        df.append(curr_df)
    except:
        print(ncbi)

ncbi = pd.concat(df)

# %%
ncbi = ncbi[(ncbi.gene_id is None) or (~ncbi.gene_id.isin(sk.var.gene_id))]
print(len(ncbi), ncbi['gene_id'].value_counts().max(), ncbi.symbol.value_counts().max())

# %%
ncbi = ncbi[ncbi.gene_id.isin(
    ncbi.gene_id.value_counts()[ncbi.gene_id.value_counts()==1].index
)]

# %%
print(len(ncbi), ncbi['gene_id'].value_counts().max(), ncbi.symbol.value_counts().max())

# %%
preprocess.match_ncbi(sk, ncbi)
sk.var.chrom.isna().sum(), sk.var.gene_id.value_counts().max()

# %% [markdown]
# ## Process retina data

# %% [markdown]
# ### Match gene names to cellranger

# %%
ret.var = ret.var.merge(genes20, left_index=True, right_index=True, how="left")
display(ret.var.head(2))
print(ret.var.chrom.isna().sum())

# %%
preprocess.match_cellranger_2024(ret, genes24=genes24)

# %%
ret.var['symbol'] = None

# %%
preprocess.match_ref_ad(ret, sc)

# %%
preprocess.match_ref_ad(ret, sk)

# %% [markdown]
# ### Match remaining gene names to NCBI

# %%
unm = ret.var.index[ret.var.chrom.isna()].tolist()
len(unm)

# %%
unm_arrs = np.array_split(unm, 100)
df = []

for unm in tqdm(unm_arrs):
    # ncbi = !datasets summary gene symbol {" ".join(list(unm))} --report gene
    try:
        curr_df = preprocess.load_ncbi_string(ncbi)
        df.append(curr_df)
    except:
        print(ncbi)

ncbi = pd.concat(df)

# %%
ncbi = ncbi[(ncbi.gene_id is None) or (~ncbi.gene_id.isin(ret.var.gene_id))]
print(len(ncbi), ncbi['gene_id'].value_counts().max(), ncbi.symbol.value_counts().max())

# %%
ncbi = ncbi[ncbi.symbol!='EFCAB3P1']

# %%
print(len(ncbi), ncbi['gene_id'].value_counts().max(), ncbi.symbol.value_counts().max())

# %%
preprocess.match_ncbi(ret, ncbi)
ret.var.chrom.isna().sum(), ret.var.gene_id.value_counts().max()

# %% [markdown]
# ## Process Brain data

# %% [markdown]
# ### match gene names to cellranger

# %%
br.var = br.var.merge(genes20, left_index=True, right_index=True, how="left")
print(br.var.chrom.isna().sum())

# %%
preprocess.match_cellranger_2024(br, genes24=genes24)

# %%
br.var['symbol'] = None

# %%
preprocess.match_ref_ad(br, sc)

# %%
preprocess.match_ref_ad(br, sk)

# %%
preprocess.match_ref_ad(br, ret)

# %%
print(len(br), br.var['gene_id'].value_counts().max(), br.var.symbol.value_counts().max())

# %% [markdown]
# ## Drop unannotated genes from all datasets

# %%
print(sc.shape)
sc = sc[:, ~sc.var.chrom.isna()]
print(sc.shape)

# %%
print(sk.shape)
sk = sk[:, ~sk.var.chrom.isna()]
print(sk.shape)

# %%
print(ret.shape)
ret = ret[:, ~ret.var.chrom.isna()]
print(ret.shape)

# %%
print(br.shape)
br = br[:, ~br.var.chrom.isna()]
print(br.shape)

# %% [markdown]
# ## Combine all datasets

# %%
sc.var = sc.var.reset_index(names='gene_name').set_index('gene_id')
sk.var = sk.var.reset_index(names='gene_name').set_index('gene_id')
ret.var = ret.var.reset_index(names='gene_name').set_index('gene_id')
br.var = br.var.reset_index(names='gene_name').set_index('gene_id')

# %%
sc.var.index = sc.var.index.astype(str)
sc.var_names = sc.var.index.astype(str)

sk.var.index = sk.var.index.astype(str)
sk.var_names = sk.var.index.astype(str)

ret.var.index = ret.var.index.astype(str)
ret.var_names = ret.var.index.astype(str)

br.var.index = br.var.index.astype(str)
br.var_names = br.var.index.astype(str)

# %%
common_genes = list(set(
    sc.var_names).intersection(
    sk.var_names).intersection(
    ret.var_names).intersection(
    br.var_names)
)

len(common_genes)

# %%
# %%time
sc_common = sc[:, common_genes].copy()
sk_common = sk[:, common_genes].copy()
ret_common = ret[:, common_genes].copy()
br_common = br[:, common_genes].copy()

# %%
sc_common.var.start = sc_common.var.start.astype(int)
sc_common.var.end = sc_common.var.end.astype(int)

# %%
ad_inner = anndata.concat(
    [sc_common, sk_common, ret_common, br_common], join='inner', label='dataset',
    keys=['scimilarity', 'skin_atlas', 'retina_atlas', 'brain_atlas'],
    merge='same'
)

# %% [markdown]
# ## Format the combined pseudobulk matrix

# %% [markdown]
# ### Combine .var

# %%
np.all(ad_inner.var.index == sc_common.var.index)

# %%
ad_inner.var = sc_common.var.copy().drop(columns='symbol')

# %%
for gene_id in tqdm(ad_inner.var.index):
    names = []
    gene_name = ad_inner.var.loc[gene_id, 'gene_name']
    sk_name = sk.var.loc[gene_id, 'gene_name']      
    ret_name = ret.var.loc[gene_id, 'gene_name']
    br_name = br.var.loc[gene_id, 'gene_name']
    for name in [sk_name, ret_name, br_name]:
        if (name != gene_name) and (name not in names):
            names.append(name)
    if len(names) > 0:
        ad_inner.var.loc[gene_id, 'other_names'] = ",".join(names)
    else:
        ad_inner.var.loc[gene_id, 'other_names'] = None

# %%
ad_inner.var = preprocess.change_values(ad_inner.var, col="gene_type", value_dict={
    'PROTEIN_CODING':'protein_coding',
    'ncRNA':'lncRNA',
    'PSEUDO':'pseudogene'
})

# %% [markdown]
# ### Combine .obs

# %%
ad_inner.obs[['study', 'dataset']].drop_duplicates().value_counts().max()

# %%
all_obs = pd.concat([
    sc_common.obs,
    sk_common.obs,
    ret_common.obs,
    br_common.obs
])

# %%
np.all(all_obs.index == ad_inner.obs.index)

# %%
all_obs.loc[all_obs.tissue=="head of femur", "organ"] = "bone"

# %%
all_obs['dataset'] = ad_inner.obs.dataset.tolist()

# %%
ad_inner.obs = all_obs

# %% [markdown]
# ## Save

# %%
out_file = os.path.join(save_dir, "combined_inner.h5ad")
ad_inner.write_h5ad(out_file)

# %%
