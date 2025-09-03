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
import os, sys
from scipy.stats import pearsonr, mannwhitneyu
sys.path.append('/code/decima/src/decima')

from visualize import plot_marker_box, plot_gene_scatter
from plotnine import *

# %matplotlib inline

# %% [markdown]
# ## Paths

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823"
matrix_file = os.path.join(save_dir, "data.h5ad")
h5_file = os.path.join(save_dir, "data.h5")

# %% [markdown]
# ## Load data

# %%
ad = anndata.read_h5ad(matrix_file)
ad = ad[:, ad.var.dataset == "test"].copy()
ad.shape

# %% [markdown]
# ### DNAH6

# %%
gene = 'DNAH6'
pearsonr(np.array(ad[:, gene].X).squeeze(),
         np.array(ad[:, gene].layers['preds']).squeeze())

# %%
plot_gene_scatter(ad, gene, show_corr=False) + theme(
    figure_size=(2.6,2))+ xlab('Measured Expression')

# %%
gene = 'DNAH6'
filter_df = {'cell_type':['Ependymal', 'Choroid plexus', 'ciliated cell']}
display(plot_marker_box(
    gene, ad_, filter_df, split_col='organ', split_values=['CNS', 'lung'],
    label_name='Cell type') 
        + theme(figure_size=(4, 2))
       )

# %%
gene = 'DNAH6'
cts = ['Ependymal', 'Choroid plexus', 'ciliated cell']
print(mannwhitneyu(
    np.array(ad[ad.obs.cell_type.isin(cts), gene].X).squeeze(),
    np.array(ad[~ad.obs.cell_type.isin(cts), gene].X).squeeze(),
    alternative='greater'
))
print(mannwhitneyu(
    np.array(ad[ad.obs.cell_type.isin(cts), gene].layers['preds']).squeeze(),
    np.array(ad[~ad.obs.cell_type.isin(cts), gene].layers['preds']).squeeze(),
    alternative='greater'
))

# %% jupyter={"source_hidden": true}
ad_ = ad.copy()
ad_.obs.cell_type = ad_.obs.cell_type.astype(str)
ad_.obs.loc[(ad_.obs.organ=='airway') &\
(~ad_.obs.cell_type.str.contains('pneumo')), 'cell_type'] = 'other lung'

for gene in ['EVA1A']:
    filter_df = {'cell_type':['type I pneumocyte', 'type II pneumocyte', 'other lung']}
    inp = extract_gene_data(h5_file, gene)
    display(plot_marker_box(gene, ad_, filter_df) + theme(figure_size=(4,2)))
    display(plot_gene_scatter(ad, gene, corrx=2))

# %% [markdown]
# ### FABP1

# %%
gene = 'FABP1'
pearsonr(np.array(ad[:, gene].X).squeeze(),
         np.array(ad[:, gene].layers['preds']).squeeze())

# %%
plot_gene_scatter(ad, gene, show_corr=False) + theme(
    figure_size=(2.6,2))+ xlab('Measured Expression')

# %%
gene = 'FABP1'
cts = ['enterocyte', 'hepatocyte']
print(mannwhitneyu(
    np.array(ad[ad.obs.cell_type.isin(cts), gene].X).squeeze(),
    np.array(ad[~ad.obs.cell_type.isin(cts), gene].X).squeeze(),
    alternative='greater'
))
print(mannwhitneyu(
    np.array(ad[ad.obs.cell_type.isin(cts), gene].layers['preds']).squeeze(),
    np.array(ad[~ad.obs.cell_type.isin(cts), gene].layers['preds']).squeeze(),
    alternative='greater'
))

# %%
gene = 'FABP1'
filter_df = {'cell_type':['enterocyte', 'hepatocyte']}
plot_marker_box(gene, ad, filter_df, split_col='organ', 
     split_values=['gut', 'liver'],
    order=['enterocyte', 'hepatocyte', 'Other gut', 'Other liver', 'Other'], 
    label_name='Cell type') + theme(figure_size=(4,2))

# %%
ad_.obs.loc[(ad_.obs.organ=='CNS') & (ad_.obs.cell_type.isin(['Oligodendrocyte',
 'Oligodendrocyte precursor', 'Astrocyte', 'Bergmann glia', 'Committed oligodendrocyte precursor'
                           ])), 'cell_type'] = 'other glia'

for gene in ['TAB2', 'PRAM1', 'QSER1']:
    filter_df = {'cell_type':['Microglia', 'macrophage', 'other glia']}
    inp = extract_gene_data(h5_file, gene)
    display(plot_marker_box(gene, ad_, filter_df) + theme(figure_size=(4,2)))
    #display(plot_gene_scatter(ad, gene, corrx=2))

# %% [markdown]
# ## SPI1

# %%
gene = 'SPI1'
pearsonr(np.array(ad[:, gene].X).squeeze(),
         np.array(ad[:, gene].layers['preds']).squeeze())

# %%
plot_gene_scatter(ad, gene, show_corr=False) + theme(
    figure_size=(2.6,2))+ xlab('Measured Expression')

# %%
gene = 'SPI1'
cts = ['non-classical monocyte', 'classical monocyte', 
            'intermediate monocyte', 'macrophage', 'Microglia']
print(mannwhitneyu(
    np.array(ad[ad.obs.cell_type.isin(cts), gene].X).squeeze(),
    np.array(ad[~ad.obs.cell_type.isin(cts), gene].X).squeeze(),
    alternative='greater'
))
print(mannwhitneyu(
    np.array(ad[ad.obs.cell_type.isin(cts), gene].layers['preds']).squeeze(),
    np.array(ad[~ad.obs.cell_type.isin(cts), gene].layers['preds']).squeeze(),
    alternative='greater'
))

# %%
gene = 'SPI1'
filter_df = {'cell_type':['non-classical monocyte', 'classical monocyte', 
                          'intermediate monocyte', 'macrophage', 'Microglia']}
display(plot_marker_box(
    gene, ad_, filter_df, split_col='organ', split_values=['blood', 'CNS'],
    label_name='Cell type') 
        + theme(figure_size=(6, 2)) + guides(fill=guide_legend(ncol=2))
       )

# %%
