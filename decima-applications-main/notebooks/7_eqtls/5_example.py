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
import sys
import numpy as np
import pandas as pd
import os
import anndata
import wandb
import torch
from tqdm import tqdm

from scipy.ndimage import gaussian_filter1d
from grelu.visualize import plot_attributions, plot_tracks, add_highlights
from captum.attr import InputXGradient
from grelu.transforms.prediction_transforms import Aggregate

import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
# %matplotlib inline

sys.path.append('/code/decima/src/decima/')
from interpret import extract_gene_data
from lightning import LightningModel

# %% [markdown]
# ## Paths

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823"
matrix_file = os.path.join(save_dir, "data.h5ad")
h5_file = os.path.join(save_dir, "data.h5")
eqtl_ad_file = 'eqtl.h5ad'

# %% [markdown]
# ## Load

# %%
ad = anndata.read_h5ad(matrix_file)
eqtl_ad = anndata.read_h5ad(eqtl_ad_file)

# %%
wandb.login(host="https://genentech.wandb.io")
ckpts=[
'/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/kugrjb50/checkpoints/epoch=3-step=2920.ckpt',
'/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/i68hdsdk/checkpoints/epoch=2-step=2190.ckpt',
'/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/0as9e8of/checkpoints/epoch=7-step=5840.ckpt',
'/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/i9zsp4nm/checkpoints/epoch=8-step=6570.ckpt',
]
models = [LightningModel.load_from_checkpoint(ckpt).eval() for ckpt in ckpts]

# %%
from eqtl_meta import cell_type_mapping
cell_type_mapping = pd.read_table(StringIO(cell_type_mapping))
all_modelcelltypes = set(cell_type_mapping['model_celltype'])

# %% [markdown]
# ## Subset to matched cell types

# %%
gene = 'JAZF1'
rsid = 'rs2158799'
for gene in ['CEBPA', 'JAZF1']:
    eqtl_ad.var[gene] = np.array(ad[:, gene].X).squeeze()

# %%
ad = ad[(ad.obs.organ=='blood') & (ad.obs.cell_type.isin(all_modelcelltypes))]
eqtl_ad = eqtl_ad[(eqtl_ad.obs.gene_symbol==gene) & (eqtl_ad.obs.rsid==rsid), 
                (eqtl_ad.var.organ=='blood') & (eqtl_ad.var.cell_type.isin(all_modelcelltypes))]

print(ad.shape, eqtl_ad.shape)

# %%
eqtl_ad.obs

# %%
eqtl_ad.var['effect'] = np.array(eqtl_ad.X).squeeze()
eqtl_ad.var['is_mono'] = eqtl_ad.var.cell_type.str.contains('monocyte')

# %% [markdown]
# ## Visualize predictions

# %%
eqtl_ad.var.cell_type = eqtl_ad.var.cell_type.astype(str)
eqtl_ad.var.loc[eqtl_ad.var.cell_type == 'CD4-positive, alpha-beta T cell', 'cell_type'] = 'CD4+ alpha-beta T cell'
eqtl_ad.var.loc[eqtl_ad.var.cell_type == 'CD8-positive, alpha-beta T cell', 'cell_type'] = 'CD8+ alpha-beta T cell'
eqtl_ad.var.cell_type = eqtl_ad.var.cell_type.apply(lambda x:x[0].upper() + x[1:])
eqtl_ad.var.cell_type = pd.Categorical(eqtl_ad.var.cell_type,
        categories = eqtl_ad.var.groupby('cell_type').effect.mean().sort_values().index.tolist())

# %%
(
    ggplot(eqtl_ad.var, aes(x='cell_type', y='effect', color='is_mono')) + geom_boxplot(outlier_size=0.1, size=.3)
    + theme_classic() + theme(figure_size=(2.8, 3.5)) + scale_color_manual(values=['black', 'blue'])
    + theme(axis_text_x=element_text(rotation=90, hjust=.5)) + ylab("") + xlab("")
    + geom_hline(yintercept=0, linetype='--')
)

# %%
(
    ggplot(eqtl_ad.var, aes(x='cell_type', y='CEBPA', color='is_mono')) + geom_boxplot(outlier_size=0.1, size=.3)
    + theme_classic() + theme(figure_size=(2.8,3.5)) + scale_color_manual(values=['black', 'blue'])
    + theme(axis_text_x=element_text(rotation=90, hjust=.5)) + ylab("") + xlab("")
)

# %%
(
    ggplot(eqtl_ad.var, aes(x='cell_type', y='JAZF1', color='is_mono')) + geom_boxplot(outlier_size=0.1, size=.3)
    + theme_classic() + theme(figure_size=(2.8,3.5)) + scale_color_manual(values=['black', 'blue'])
    + theme(axis_text_x=element_text(rotation=90, hjust=.5)) + ylab("Measured Expression") + xlab("")
    + ggtitle('JAZF1') + theme(plot_title = element_text(face = "italic")) + theme(legend_position='none')
)

# %% [markdown]
# ## Extract inputs

# %%
seq, mask = extract_gene_data(h5_file, gene, merge=False)
tss_pos = np.where(mask[0] == 1)[0][0] - 2
device = torch.device(0)

# %%
print(seq[:, 107147])

ref_seq = seq.clone()
alt_seq = seq.clone()
alt_seq[2, 107147] = 0
alt_seq[1, 107147] = 1
print(ref_seq[:, 107147])
print(alt_seq[:, 107147])

ref_inputs = torch.vstack([ref_seq, mask]).to(device)
alt_inputs = torch.vstack([alt_seq, mask]).to(device)

# %% [markdown]
# ## Attributions in monocytes

# %%
# %%time
on_tasks = ad.obs_names[ad.obs.cell_type.str.contains('monocyte')].tolist()

attr_ref_on = []
attr_alt_on = []

for model in models:
    model.add_transform(Aggregate(tasks=on_tasks, task_aggfunc="mean", model=model))
    attributer = InputXGradient(model.to(device))
    with torch.no_grad():
        attr_ref_on.append(attributer.attribute(ref_inputs).cpu().numpy())
        attr_alt_on.append(attributer.attribute(alt_inputs).cpu().numpy())

attr_ref_on = np.stack(attr_ref_on).mean(0)
attr_alt_on = np.stack(attr_alt_on).mean(0)

# %%
st = -58000
en = 4000
a_r_on = gaussian_filter1d(np.abs(attr_ref_on.mean(0)[tss_pos+st:tss_pos+en]), 5)
a_a_on = gaussian_filter1d(np.abs(attr_alt_on.mean(0)[tss_pos+st:tss_pos+en]), 5)
fig, axes = plt.subplots(2, 1, figsize=(6, 2), sharex=True, tight_layout=True)
axes[0].fill_between(np.linspace(st, en, num=en-st), a_r_on, color="black")
sns.despine(top=True, right=True, bottom=True)
axes[0].set_ylim(0, 0.021)
axes[1].fill_between(np.linspace(st, en, num=en-st), a_a_on, color="black")
sns.despine(top=True, right=True, bottom=True)
axes[1].set_ylim(0, 0.021)

# %%
display(plot_attributions(attr_ref_on[:-1, tss_pos-56720:tss_pos-56640], ylim=(-.1, .2), figsize=(5,1)))
display(plot_attributions(attr_alt_on[:-1, tss_pos-56720:tss_pos-56640], ylim=(-.1, .2), figsize=(5,1)))

# %% [markdown]
# ## Attributions in off-target cell types

# %%
# %%time
off_tasks = ad.obs_names[ad.obs.cell_type == 'natural killer cell'].tolist()

attr_ref_off = []
attr_alt_off = []

for model in models:
    model.add_transform(Aggregate(tasks=off_tasks, task_aggfunc="mean", model=model))
    attributer = InputXGradient(model.to(device))
    with torch.no_grad():
        attr_ref_off.append(attributer.attribute(ref_inputs).cpu().numpy())
        attr_alt_off.append(attributer.attribute(alt_inputs).cpu().numpy())

attr_ref_off = np.stack(attr_ref_off).mean(0)
attr_alt_off = np.stack(attr_alt_off).mean(0)

# %%
st = -58000
en = 4000
a_r_off = gaussian_filter1d(np.abs(attr_ref_off.mean(0)[tss_pos+st:tss_pos+en]), 5)
a_a_off = gaussian_filter1d(np.abs(attr_alt_off.mean(0)[tss_pos+st:tss_pos+en]), 5)
fig, axes = plt.subplots(2, 1, figsize=(6, 2), sharex=True, tight_layout=True)
axes[0].fill_between(np.linspace(st, en, num=en-st), a_r_off, color="black")
sns.despine(top=True, right=True, bottom=True)
axes[0].set_ylim(0, 0.021)
axes[0].set_title('Attributions of JAZF1 expression in Natural Killer cells', fontsize=12)
axes[0].text(-56000, 0.015, 'Reference', fontsize=9)
add_highlights(axes[0], starts=-56600, ends=-56300, ymin=0, ymax=0.02, facecolor='skyblue', alpha=.3)
axes[1].fill_between(np.linspace(st, en, num=en-st), a_a_off, color="black")
sns.despine(top=True, right=True, bottom=True)
axes[1].set_ylim(0, 0.021)
axes[1].text(-56000, 0.015, 'Alternate', fontsize=9)
axes[1].set_xlabel('Distance from TSS')
add_highlights(axes[1], starts=-56600, ends=-56300, ymin=0, ymax=0.02, facecolor='skyblue', alpha=.3)

# %%
