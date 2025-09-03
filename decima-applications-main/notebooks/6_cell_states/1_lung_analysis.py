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
import numpy as np
import pandas as pd
import anndata
import h5py
import os
import scipy
import tqdm
import sys

from grelu.interpret.motifs import trim_pwm
from grelu.visualize import plot_attributions
from grelu.sequence.format import indices_to_strings

import torch
import seaborn as sns
import wandb
from plotnine import *

sys.path.append('/code/decima/src/decima')
sys.path.append('.')

from lightning import LightningModel
from interpret import read_meme_file
from visualize import plot_logo
from motif_meta import clustername_mapping, bad_motifs

# %matplotlib inline

# %% [markdown]
# ## Paths

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823"
matrix_file = os.path.join(save_dir, "data.h5ad")
meme_file = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/data/jaspar/H12CORE_meme_format.meme"
motif_json = '/gstore/data/resbioai/karollua/Decima/scborzoi/decima/data/jaspar/H12CORE_annotation.jsonl'
motifcluster_path = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/data/jaspar/cluster/cluster_key.txt"

# %% [markdown]
# ## Load data

# %%
ad = anndata.read_h5ad(matrix_file)
ad = ad[ad.obs.organ.isin(['lung', 'airway'])]
hmotifs, hnames = read_meme_file('../H12CORE_meme_format.meme')

# %% [markdown]
# ## Subset to epithelial cell types

# %%
cts = ['respiratory basal cell','type II pneumocyte','type I pneumocyte','lung secretory cell',
      'club cell','ciliated cell','goblet cell']

# %%
ad = ad[ad.obs.cell_type.isin(cts)]

# %% [markdown]
# ## Load motif metadata

# %%
jsonObj = pd.read_json(motif_json, lines=True)
tf_to_family_dict = jsonObj[['name','masterlist_info']].set_index('name').to_dict()['masterlist_info']

# %%
motif_clusters = pd.read_csv(motifcluster_path, sep="\t",names=['cluster_name','hits'])
motif_clusters['hits'] = motif_clusters['hits'].apply(lambda x: x.split(','))
tf_to_motifcluster_dict = {}
for _,row in motif_clusters.iterrows():
    for hit in row['hits']:
        tf_to_motifcluster_dict[hit] = row['cluster_name']

# %% [markdown]
# ## Match TOMTOM results to HOCOMOCO clusters

# %%
rows = []
for ct in cts:
    print(ct)
    ct_ = ct.replace(' ', '_')
    modisco_dir = f'lung_modisco/{ct_}'
    modisco_h5 = os.path.join(modisco_dir, f'{ct_}.h5')
    f = h5py.File(modisco_h5, 'r')
    for i in range(len(f['pos_patterns'])):
        num_seqlets = f['pos_patterns'][f'pattern_{i}']['seqlets']['n_seqlets'][0]
        row = {'cell_type':ct,'pattern':i,'count':num_seqlets}
        tomtom = pd.read_csv(os.path.join(modisco_dir,'tomtom',f"pos_patterns.pattern_{i}.tomtom.tsv"),sep="\t").dropna()
        tomtom = tomtom[tomtom['q-value'] < 0.05]
        if len(tomtom) > 0:
            target = tomtom.sort_values('q-value')['Target_ID'].iloc[0]
            top_target = tf_to_family_dict[target]['tf']
            top_cluster = tf_to_motifcluster_dict[target]
            if top_target is not None:
                row['target'] = top_target
                row['cluster'] = top_cluster
                rows.append(row)
        
ct_motif_df = pd.DataFrame(rows)

# %% [markdown]
# ## Drop uninformative motif clusters

# %%
low_count_clusters = set(ct_motif_df.groupby('cluster')['count'].sum()[ct_motif_df.groupby('cluster')['count'].sum() < 250].index)

# %%
ct_motif_df = ct_motif_df[~ct_motif_df.cluster.isin(low_count_clusters|bad_motifs)]

# %% [markdown]
# ## Compute motif cluster enrichment per cell type

# %%
ct_cluster_df = ct_motif_df.groupby(['cell_type', 'cluster'])['count'].sum().reset_index()
seqlet_counts = ct_cluster_df.groupby('cell_type')['count'].sum().reset_index().set_index('cell_type').to_dict()['count']
ct_cluster_df = ct_cluster_df.sort_values(['cell_type','count'], ascending=False)
ct_cluster_df['total_count'] = ct_cluster_df['cell_type'].apply(lambda x: seqlet_counts[x])
ct_cluster_df['weight'] = ct_cluster_df['count']/ct_cluster_df['total_count']
ct_cluster_df['weight_sum'] = ct_cluster_df.groupby(['cluster'])['weight'].transform('sum')
ct_cluster_df['enrichment'] = ct_cluster_df['weight']/(ct_cluster_df['weight_sum']/len(seqlet_counts))
ct_cluster_df['cluster_name'] = ct_cluster_df['cluster'].apply(lambda x: clustername_mapping[x] if x in clustername_mapping else x)

# %%
ct_cluster_pivot = ct_cluster_df[['cell_type','cluster_name','weight']].pivot(
    index='cell_type', columns='cluster_name').fillna(0.001)
ct_cluster_pivot.columns = [x[1] for x in ct_cluster_pivot.columns]
cbar_min, cbar_max = np.array(ct_cluster_pivot).min(),np.array(ct_cluster_pivot).max()
ct_cluster_melt = ct_cluster_pivot.reset_index().melt(
    id_vars='cell_type', var_name="cluster_name", value_name='Motif\nweight').fillna(cbar_min)

# %%
((ct_cluster_pivot - ct_cluster_pivot.mean(0))/ct_cluster_pivot.mean(0)).max(0).sort_values(ascending=False)

# %% [markdown]
# ## Plot TF expression for top hits

# %%
cluster_name = "TEAD-like"
tf_genes = ['TEAD1','TEAD2','TEAD3','TEAD4']
ad_sub = ad[:,ad.var.reset_index()['index'].isin(tf_genes)].copy()
ad_sub.obs['expr'] = ad_sub.X.mean(1)
plot_df = ad_sub.obs.merge(ct_cluster_melt.query('cluster_name == @cluster_name'), on="cell_type")
(
    ggplot(plot_df, aes(x="cell_type", y="expr", fill='Motif\nweight')) 
    + geom_boxplot(outlier_size=.1, size=.3, width=.5) 
    + scale_fill_cmap('coolwarm', limits=(cbar_min, .14))
    + theme_classic() + theme(figure_size=(6, 1.5))
    + xlab("") + ylab("")
)

# %%
cluster_name = "P53-like"
tf_genes = ['TP63']
ad_sub = ad[:,ad.var.index.isin(tf_genes)].copy()
ad_sub.obs['expr'] = ad_sub.X.mean(1)
plot_df = ad_sub.obs.merge(ct_cluster_melt.query('cluster_name == @cluster_name'), on="cell_type")
(
    ggplot(plot_df, aes(x="cell_type", y="expr", fill='Motif\nweight')) 
    + geom_boxplot(outlier_size=.1, size=.3, width=.5) 
    + scale_fill_cmap('coolwarm', limits=(cbar_min, .14))
    + theme_classic() + theme(figure_size=(6, 1.5)) + xlab("") + ylab("")
)

# %%
cluster_name = "RFX"
tf_genes = ['RFX2','RFX3']
ad_sub = ad[:,ad.var.index.isin(tf_genes)].copy()
ad_sub.obs['expr'] = ad_sub.X.mean(1)
plot_df = ad_sub.obs.merge(ct_cluster_melt.query('cluster_name == @cluster_name'), on="cell_type")
(
    ggplot(plot_df, aes(x="cell_type", y="expr", fill='Motif\nweight'))
    + geom_boxplot(outlier_size=.1, size=.3, width=.5)
    + scale_fill_cmap('coolwarm', limits=(cbar_min, .14))
    + theme_classic() + theme(figure_size=(6, 1.5)) +xlab("") + ylab("")
)

# %%
cluster_name = "SOX-like"
tf_genes = ['SOX2']
ad_sub = ad[:,ad.var.index.isin(tf_genes)].copy()
ad_sub.obs['expr'] = ad_sub.X.mean(1)
plot_df = ad_sub.obs.merge(ct_cluster_melt.query('cluster_name == @cluster_name'), on="cell_type")
(
    ggplot(plot_df, aes(x="cell_type", y="expr", fill='Motif\nweight')) 
    + geom_boxplot(outlier_size=.1, size=.3, width=.5) 
    + scale_fill_cmap('coolwarm', limits=(cbar_min, .14))
    + theme_classic() + theme(figure_size=(6, 1.5)) + xlab("") + ylab("")
)

# %% [markdown]
# ## Plot motif logos from tf-modisco

# %%
ct='type_I_pneumocyte'
i=1
modisco_h5 = f'lung_modisco/{ct}/{ct}.h5'
f = h5py.File(modisco_h5, 'r')
m = trim_pwm(np.array(f['pos_patterns'][f'pattern_{i}']['contrib_scores']), 0.1)
display(plot_attributions(m.T, figsize=(2,.8)))

# %%
ct='ciliated_cell'
i=6
modisco_h5 = f'lung_modisco/{ct}/{ct}.h5'
f = h5py.File(modisco_h5, 'r')
m = trim_pwm(np.array(f['pos_patterns'][f'pattern_{i}']['contrib_scores']), 0.1)
s = indices_to_strings(m.argmax(1))
display(plot_attributions(m.T, figsize=(2,.8)))

# %%
ct='respiratory_basal_cell'
i=2
modisco_h5 = f'lung_modisco/{ct}/{ct}.h5'
f = h5py.File(modisco_h5, 'r')
m = trim_pwm(np.array(f['pos_patterns'][f'pattern_{i}']['contrib_scores']), 0.1)
s = indices_to_strings(m.argmax(1))
display(plot_attributions(m.T, figsize=(2,.8)))

# %%
ct='ciliated_cell'
i=5
modisco_h5 = f'lung_modisco/{ct}/{ct}.h5'
f = h5py.File(modisco_h5, 'r')
m = trim_pwm(np.array(f['pos_patterns'][f'pattern_{i}']['contrib_scores']), 0.1)
display(plot_attributions(m.T, figsize=(2,.8)))

# %% [markdown]
# ## Plot logos from HOCOMOCO

# %%
for i in np.where(['P63' in x for x in hnames])[0]:
    print(hnames[i])
    plot_logo(hmotifs[i])

# %%
