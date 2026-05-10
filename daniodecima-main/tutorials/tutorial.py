# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How to use Decima

# %%
import numpy as np
import pandas as pd
import anndata
import torch
import os, sys

# Import Decima functions
sys.path.append('../src/')
#sys.path.append(os.path.dirname(__file__))
sys.path.insert(0, '/code/decima/src/decima')

# %matplotlib inline

# %% [markdown]
# ## Important file paths

# %%
# Folder
#latest_version = '20240823'
#save_dir = f"/gstore/data/resbioai/grelu/decima/{latest_version}"
save_dir = "/hpc/mydata/mathias.voges/Projects/research/zf-decima/outputs/grelu/decima"

# Data
matrix_file = os.path.join(save_dir, "data_out_zf-Decima_Random_Rep0.h5ad") # Includes predictions
h5_file = os.path.join(save_dir, "data.h5")

# Model checkpoints
ckpt_dir = os.path.join(save_dir, 'lightning_logs')
ckpts = [
     os.path.join(ckpt_dir, '5ly2p4gy/checkpoints/epoch=9-step=9190.ckpt'),
    #  os.path.join(ckpt_dir, 'i68hdsdk/checkpoints/epoch=2-step=2190.ckpt'),
    #  os.path.join(ckpt_dir, '0as9e8of/checkpoints/epoch=7-step=5840.ckpt'),
    #  os.path.join(ckpt_dir, 'i9zsp4nm/checkpoints/epoch=8-step=6570.ckpt'),
    ]

primary_ckpt = ckpts[0]

print(f'Anndata containing data and predictions: {matrix_file}')
print(f'Gene sequences: {h5_file}')
print(f'Primary model checkpoint: {primary_ckpt}')

# %% [markdown]
# ## How to find predictions for a gene of interest

# %% [markdown]
# Both the true and predicted log(CPM+1) matrices are saved in `matrix_file`.

# %%
ad = anndata.read_h5ad(matrix_file)

# %%
ad

# %%
ad.obs.cell_type.unique()

# %%
ad.obs.sort_values(by='test_pearson', ascending=False)

# %%
ad.var.sort_values(by='pearson', ascending=False)[:50]#[13000:13100]

# %%
ad.X[:5, :5]

# %%
ad.layers['preds'][:5, :5]

# %% [markdown]
# To find true values and predictions for a specific gene (SPI1):

# %%
ad.var.loc[['neurog1']]

# %%
np.array(ad[:, 'neurog1'].X).squeeze()

# %%
np.array(ad[:, 'neurog1'].layers['preds']).squeeze()

# %% [markdown]
# ## How to visualize the model's predictions for a gene of interest

# %%
from decima.visualize import plot_gene_scatter, plot_marker_box

# %%
plot_gene_scatter(gene="zmat2", ad=ad, size=2, figure_size=(6, 5))

# %% [markdown]
# We can also compare the predictions in specific cell types vs. others:

# %%
# spi1_cell_types = ['classical monocyte', 'intermediate monocyte', 'non-classical monocyte', 
#                  'alveolar macrophage', 'macrophage']

neurod1_cell_types = ['central_nervous_system', 'endoderm', 'lateral_mesoderm', 'notochord', 'paraxial_mesoderm', 'periderm', 'hematopoietic_system', 'intermediate_mesoderm', 'mesenchyme', 'neural_crest']

# %%
marker_features = {'cell_type': neurod1_cell_types}
#marker_features = {'timepoint': ['10hpf', '12hpf', '14hpf', '16hpf', '19hpf', '24hpf', '2dpf', '3dpf', '5dpf', '10dpf']}
plot_marker_box(gene='neurog1', ad=ad, marker_features=marker_features)

# %% [markdown]
# ## How to load the model 

# %%
from decima.lightning import LightningModel

# %%
model = LightningModel.load_from_checkpoint(primary_ckpt)
model = model.eval()

# %% [markdown]
# ## How to get the sequence input for the gene of interest 

# %% [markdown]
# The sequence input for each gene is stored in `h5_file`.

# %%
from decima.read_hdf5 import extract_gene_data

# %%
neurod1_input = extract_gene_data(h5_file, "neurog1")

# %%
neurod1_input.shape

# %%
neurod1_input[:, :10]

# %% [markdown]
# This input corresponds to the sequence between the start and end of SPI1 in `ad.var`, shown below. Note that for negative strand genes such as SPI1, the input is the reverse complemented sequence.
# The first 4 channels contain the one-hot encoded sequence. The 5th channel is the 'gene mask' that indicates the boundaries of the gene. It is 0 for positions outside the gene and 1 for positions inside. `gene_mask_start` and `gene_mask_end` contain the positions of the gene boundaries on the input.

# %%
ad.var.loc[['neurog1']]

# %% [markdown]
# We can check that both the sequence and the gene mask are correct:

# %%
from grelu.sequence.format import convert_input_type

# %%
torch.all(neurod1_input[:4] == convert_input_type(ad.var.loc[['neurod1']], "one_hot", genome="hg38")[0])

# %%
np.where(spi1_input[4] == 1)

# %% [markdown]
# ## How to interpret the model's predictions

# %% [markdown]
# To interpret the model's predictions with respect to cell type specificity, we define a set of 'on_tasks' and 'off_tasks' (background tasks).

# %%
ad.obs.cell_type

# %%
on_tasks = ad.obs.index[ad.obs.cell_type.isin(neurod1_cell_types)].tolist()
off_tasks = ad.obs.index[(~ad.obs.cell_type.isin(neurod1_cell_types))]# & (ad.obs.organ=="blood")].tolist()

# on_tasks = ['central_nervous_system']
# off_tasks = ad.obs.index[(~ad.obs.cell_type.isin(on_tasks))]# & (ad.obs.organ=="blood")].tolist()

len(on_tasks), len(off_tasks)

# %%
import interpret
from interpret import attributions#, find_attr_peaks

seq, tss_pos, attr =  interpret.attributions(
    gene="neurog1", h5_file=h5_file, model=model, device=0, 
    tasks=on_tasks,
    off_tasks=off_tasks,
    transform="aggregate")

# %% [markdown]
# Note that if we only want to interpret the model's predictions for some set of tasks without caring about specificity, we should use `transform="aggregate" and not define `off_tasks`.

# %% [markdown]
#

# %%
attr.shape

# %% [markdown]
# In addition to the attributions, this function also returns the position of the TSS on the input. This is normally 163840 but this is not the case for all genes.

# %%
tss_pos

# %% [markdown]
# First, we can visualize the importance across the entire 500 kb interval:

# %%
from decima.visualize import plot_attributions

# %%
plot_attributions(attr, tss_pos)

# %% [markdown]
# We can also identify the peaks or regions of high importance:

# %%
peaks = find_attr_peaks(attr, tss_pos=tss_pos, n=10, min_dist=6)
peaks

# %% [markdown]
# We will visualize the attributions close to the TSS (which contains the top two peaks:

# %%
from decima.visualize import plot_attributions

# %%
plot_attributions(attr[:, tss_pos-50:tss_pos+50], figsize=(10, 2))

# %% [markdown]
# We can also match these top 2 peaks to motifs:

# %%
from decima.interpret import motifs_to_df, scan_attributions

# %%
motifs = motifs_to_df('jaspar')

# %%
results = scan_attributions(attr, motifs, peaks.iloc[:2], window=18)
results.sort_values('score', ascending=False).groupby('peak').head(3).sort_values('height', ascending=False)

# %% [markdown]
# ## How to predict variant impact

# %% [markdown]
# To predict variant impact, Decima requires a dataframe containing variant columns (chrom, pos, ref, alt), as well as the gene name. Optionally, the cell type of interest can also be included.

# %% [markdown]
# We will use a few fine-mapped eQTLs in this example.

# %%
from decima.resources.eqtl import load_susie, filter_susie

# %%
variant_df = load_susie(susie_dir='/gstore/data/resbioai/grelu/decima/onek1k/susie/QTS000038')
variant_df = filter_susie(variant_df, ad)

# %%
variant_df = variant_df.head()
variant_df

# %% [markdown]
# We first need to ensure that these variants are close enough to a gene.

# %%
from decima.variant import process_variants

# %%
variant_df = process_variants(variant_df, ad, min_from_end=5000)
variant_df

# %% [markdown]
# Now we can create the reference and alternate allele containing inputs:

# %%
from decima.read_hdf5 import VariantDataset

# %%
dataset = VariantDataset(variant_df, h5_file)

# %%
ref = dataset[0]
alt = dataset[1]

# %%
ref.shape, alt.shape

# %%
np.where(ref!=alt)

# %%
1000018-837298

# %% [markdown]
# Also, suppose we only want to make predictions in selected cell types, we can subset those tracks using the `Aggregate` transform.

# %%
relevant_tasks = ad.obs.index[(ad.obs.tissue=="blood") & (ad.obs.disease.isin(["healthy", "NA"])) & (ad.obs.cell_type.isin(variant_df.cell_type))].tolist()
len(relevant_tasks)

# %%
from grelu.transforms.prediction_transforms import Aggregate

# %%
agg_transform = Aggregate(tasks=relevant_tasks, model=model)
model.add_transform(agg_transform)

# %% [markdown]
# Now we can compute variant effects:

# %%
preds = model.predict_on_dataset(dataset, devices=1, batch_size=8, num_workers=16)
preds = anndata.AnnData(X=preds, obs=variant_df.set_index('rsid'), var=ad.obs.loc[relevant_tasks])
preds.shape

# %%
preds

# %%
preds.obs

# %%
preds.var.head()

# %%
preds.X.shape

# %%
