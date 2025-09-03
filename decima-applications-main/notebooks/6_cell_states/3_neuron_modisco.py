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
import os
import tqdm
import sys
import torch
import wandb

from grelu.transforms.prediction_transforms import Aggregate, Specificity

sys.path.append('/code/decima/src/decima')
from lightning import LightningModel
from evaluate import marker_zscores

# %matplotlib inline

# %% [markdown]
# ## Paths

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823"
matrix_file = os.path.join(save_dir, "data.h5ad")
h5_file = os.path.join(save_dir, "data.h5")
ckpt_dir = os.path.join(save_dir, 'lightning_logs')
meme_file = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/data/jaspar/H12CORE_meme_format.meme"

# %% [markdown]
# ## Load data and models

# %%
ad = anndata.read_h5ad(matrix_file)
ad = ad[ad.obs.dataset=='brain_atlas']

# %%
wandb.login(host="https://genentech.wandb.io")
ckpts=[
'/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/kugrjb50/checkpoints/epoch=3-step=2920.ckpt',
'/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/i68hdsdk/checkpoints/epoch=2-step=2190.ckpt',
'/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/0as9e8of/checkpoints/epoch=7-step=5840.ckpt',
'/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/i9zsp4nm/checkpoints/epoch=8-step=6570.ckpt',
]
models = [LightningModel.load_from_checkpoint(ckpt).eval() for ckpt in ckpts]

# %% [markdown]
# ## combine all neuronal subtypes into a single group

# %%
ad.obs['Group'] = ad.obs.cell_type.tolist()
ad.obs.loc[ad.obs.cell_type.isin([
    'Amygdala excitatory', 'CGE interneuron', 'Cerebellar inhibitory', 'Deep-layer corticothalamic and 6b',
    'Deep-layer intratelencephalic', 'Deep-layer near-projecting', 'Eccentric medium spiny neuron', 'Hippocampal CA1-3',
 'Hippocampal CA4', 'Hippocampal dentate gyrus', 'LAMP5-LHX6 and Chandelier', 'Lower rhombic lip', 'MGE interneuron',
    'Mammillary body', 'Medium spiny neuron','Midbrain-derived inhibitory','Splatter', 'Thalamic excitatory',
 'Upper rhombic lip','Upper-layer intratelencephalic']), 'Group'] = 'Neuron'

# %%
neg_groups = ['Astrocyte', 'Bergmann glia','Choroid plexus','Ependymal','Microglia','Oligodendrocyte']
pos_groups = ['Neuron']

# %% [markdown]
# ## Select top neuron-specific genes

# %%
gene_df = marker_zscores(ad[ad.obs.Group.isin(pos_groups + neg_groups)], key='Group', layer='preds')
genes = gene_df[gene_df.Group=='Neuron'].sort_values('score', ascending=False).head(250)
genes = genes.gene.tolist()

# %% [markdown]
# ## Calculate differential attributions for these genes and run modisco

# %%
on_tasks = ad.obs_names[ad.obs.Group.isin(pos_cts)].tolist()
off_tasks = ad.obs_names[ad.obs.Group.isin(neg_cts)].tolist()

# %%
sequences = []
attributions = []
with torch.no_grad():
    for gene in tqdm.tqdm(genes):
        gene_attrs = []
        for model in models:
            model = model.eval()
            seq, tss_pos, attr = get_attr(gene=gene, h5_file=h5_file, model=model, device=0, 
                tasks=on_tasks, off_tasks=off_tasks, transform='specificity', method=Saliency, abs=False)
            gene_attrs.append(attr)

        gene_attrs = np.stack(gene_attrs).mean(0)
        attributions.append(gene_attrs[:4, tss_pos-10000:tss_pos+10000])
        sequences.append(seq[:4, tss_pos-10000:tss_pos+10000])

sequences = np.stack(sequences)
attributions = np.stack(attributions)

# %%
attributions = attributions - attributions.mean(1, keepdims=True)
out_dir = 'neuron_vs_glia_modisco'
seq_path = os.path.join(out_dir, 'sequences.npy')
attr_path = os.path.join(out_dir, 'attributions.npy')
np.save(attr_path, attributions)
np.save(seq_path, sequences)

# %%
print(f'python modisco_simple.py -seq_file {seq_path} -attr_file {attr_path} -meme_file {meme_file} -out_dir {out_dir}')

# %% [markdown]
# ## Modisco on neurons only

# %%
sequences = []
attributions = []
with torch.no_grad():
    for gene in tqdm.tqdm(genes):
        gene_attrs = []
        for model in models:
            model = model.eval()
            seq, tss_pos, attr = get_attr(gene=gene, h5_file=h5_file, model=model, device=0, 
                tasks=on_tasks, transform='aggregate', method=Saliency, abs=False)
            gene_attrs.append(attr)

        gene_attrs = np.stack(gene_attrs).mean(0)
        attributions.append(gene_attrs[:4, tss_pos-10000:tss_pos+10000])
        sequences.append(seq[:4, tss_pos-10000:tss_pos+10000])

sequences = np.stack(sequences)
attributions = np.stack(attributions)

# %%
attributions = attributions - attributions.mean(1, keepdims=True)
out_dir = 'neuron_modisco'
seq_path = os.path.join(out_dir, 'sequences.npy')
attr_path = os.path.join(out_dir, 'attributions.npy')
np.save(attr_path, attributions)
np.save(seq_path, sequences)

# %%
print(f'python modisco_simple.py -seq_file {seq_path} -attr_file {attr_path} -meme_file {meme_file} -out_dir {out_dir}')

# %% [markdown]
# ## Modisco on non-neurons only

# %%
sequences = []
attributions = []
with torch.no_grad():
    for gene in tqdm.tqdm(genes):
        gene_attrs = []
        for model in models:
            model = model.eval()
            seq, tss_pos, attr = get_attr(gene=gene, h5_file=h5_file, model=model, device=0, 
                tasks=off_tasks, transform='aggregate', method=Saliency, abs=False)
            gene_attrs.append(attr)

        gene_attrs = np.stack(gene_attrs).mean(0)
        attributions.append(gene_attrs[:4, tss_pos-10000:tss_pos+10000])
        sequences.append(seq[:4, tss_pos-10000:tss_pos+10000])

sequences = np.stack(sequences)
attributions = np.stack(attributions)

# %%
attributions = attributions - attributions.mean(1, keepdims=True)
out_dir = 'glia_modisco'
seq_path = os.path.join(out_dir, 'sequences.npy')
attr_path = os.path.join(out_dir, 'attributions.npy')
np.save(attr_path, attributions)
np.save(seq_path, sequences)

# %%
print(f'python modisco_simple.py -seq_file {seq_path} -attr_file {attr_path} -meme_file {meme_file} -out_dir {out_dir}')

# %%
