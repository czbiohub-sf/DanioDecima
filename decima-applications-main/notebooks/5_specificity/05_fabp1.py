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
import wandb
import torch
import os, sys
sys.path.append('/code/decima/src/decima')

from lightning import LightningModel
from visualize import plot_marker_box
from interpret import extract_gene_data, read_meme_file, scan

from grelu.visualize import plot_attributions, add_highlights
from grelu.transforms.prediction_transforms import Aggregate

from captum.attr import InputXGradient
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823"
matrix_file = os.path.join(save_dir, "data.h5ad")
h5_file = os.path.join(save_dir, "data.h5")
ckpt_dir = os.path.join(save_dir, 'lightning_logs')
meme_file = '../H12CORE_meme_format.meme'

# %% [markdown]
# ## Load data

# %%
ad = anndata.read_h5ad(matrix_file)

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
motifs, names = read_meme_file(meme_file)

# %% [markdown]
# ## Gene

# %%
gene = "FABP1"
interval_start = ad.var.loc[gene, 'start']
interval_end = ad.var.loc[gene, 'end']
ad.var.loc[[gene]]

# %% [markdown]
# ## Define tasks

# %%
e_tasks = ad.obs_names[(ad.obs.organ=='gut') & (ad.obs.cell_type=='enterocyte')].tolist()
h_tasks = ad.obs_names[(ad.obs.organ=='liver') & (ad.obs.cell_type=='hepatocyte')].tolist()

# %% [markdown]
# ## Get attributions

# %%
seq, mask = extract_gene_data(h5_file, gene, merge=False)
inputs = torch.vstack([seq, mask])
tss_pos = np.where(mask[0] == 1)[0][0] - 2
device = torch.device(6)
inputs = inputs.to(device)

# %%
# %%time

attr_e = []
for model in models:
    model.add_transform(Aggregate(tasks=e_tasks, task_aggfunc="mean", model=model))
    attributer = InputXGradient(model.to(device))
    with torch.no_grad():
        x = attributer.attribute(inputs).cpu().numpy()
        attr_e.append(x)

attr_e = np.stack(attr_e).mean(0)

# %%
# %%time

attr_h = []
for model in models:
    model.add_transform(Aggregate(tasks=h_tasks, task_aggfunc="mean", model=model))
    attributer = InputXGradient(model.to(device))
    with torch.no_grad():
            x = attributer.attribute(inputs).cpu().numpy()
            attr_h.append(x)

attr_h = np.stack(attr_h).mean(0)

# %% [markdown]
# ## View large region

# %%
sc = 88116000
ec = 88180000
print(ec-sc)

start_pos = sc - interval_start
end_pos = ec - interval_start
a_e = gaussian_filter1d(np.abs(attr_e[:, ::-1].mean(0)[start_pos:end_pos]), 5)
a_h = gaussian_filter1d(np.abs(attr_h[:, ::-1].mean(0)[start_pos:end_pos]), 5)

fig, axes = plt.subplots(2, 1, figsize=(12, 3), sharex=True, tight_layout=True)

axes[0].fill_between(np.linspace(sc, ec, num=ec-sc), a_e, color="darkred")
sns.despine(top=True, right=True, bottom=True)
p = find_peaks(a_e, height=.01)[0]
add_highlights(
    axes[0], starts=[x-50+sc for x in p], ends=[x+50+sc for x in p],
    facecolor='darkred', ymin=-.1, ymax=-.04, alpha=1)

axes[1].fill_between(np.linspace(sc, ec, num=ec-sc), a_h, color="darkblue")
sns.despine(top=True, right=True, bottom=True)
p = find_peaks(a_h, height=.01)[0]
add_highlights(
    axes[1], starts=[x-50+sc for x in p], ends=[x+50+sc for x in p],
    facecolor='darkblue', ymin=-.1, ymax=-.04, alpha=1)


# %% [markdown]
# ## View promoter

# %%
plot_attributions(attr_e[:, tss_pos-50:tss_pos], figsize=(10,1.5))

# %%
plot_attributions(attr_h[:, tss_pos-50:tss_pos], figsize=(10,1.5))

# %% [markdown]
# ## View a distal enhancer

# %%
start_coord=88179510
end_coord=88179590

end_pos = ad.var.loc[gene, 'end']- start_coord
start_pos = ad.var.loc[gene, 'end'] - end_coord
start_pos, end_pos

# %%
plot_attributions(attr_e[:, start_pos:end_pos], figsize=(10, 1.5))

# %%
plot_attributions(attr_h[:, start_pos:end_pos], figsize=(10, 1.5))

# %%
s = scan('AGTGACACAATCA', motifs=motifs, names=motif_names, bg=bg, pthresh=1e-3)
print(s)
for m in s.motif.tolist():
    if m.upper() in ad.var_names:
        m_on=ad[e_tasks, m.upper()].X.mean() 
        m_off=ad[off_tasks, m.upper()].X.mean()
        if m_on-m_off >= 1:
            print(m, m_on-m_off)

# %%
s = scan('ATTTTATAGCTC', motifs=motifs, names=motif_names, bg=bg, pthresh=2.5e-3)
print(s)
for m in s.motif.tolist():
    if m.upper() in ad.var_names:
        m_on=ad[e_tasks, m.upper()].X.mean() 
        m_off=ad[off_tasks, m.upper()].X.mean()
        if m_on-m_off >= 1:
            print(m, m_on-m_off)

# %%
s = scan('TAGCTCAAAGGTTGAG', motifs=motifs, names=motif_names, bg=bg, pthresh=1e-3)
print(s)
for m in s.motif.tolist():
    if m.upper() in ad.var_names:
        m_on=ad[e_tasks, m.upper()].X.mean() 
        m_off=ad[off_tasks, m.upper()].X.mean()
        if m_on-m_off >= 1:
            print(m, m_on-m_off)

# %% [markdown]
# ## View logos

# %%
for m, n in zip(motifs, motif_names):
    if 'CEBPG' in n:
        break

plot_logo(m, rc=True)

# %%
for m, n in zip(motifs, motif_names):
    if 'CDX1' in n:
        break

plot_logo(m)

# %%
for m, n in zip(motifs, motif_names):
    if 'HNF4A' in n:
        break

plot_logo(m)

# %% [markdown]
# ## View TF abundance

# %%
for gene in ['CEBPA', 'CDX1', 'HNF4A']:
    p=plot_marker_box(
        gene=gene, ad=ad[ad.obs.organ.isin(['liver', 'gut'])], 
        marker_features={'cell_type':['hepatocyte', 'enterocyte']},
        split_col='organ', split_values=['gut', 'liver'], label_name='Cell type',
        order=['enterocyte', 'Other gut', 'hepatocyte', 'Other liver'],
        include_preds=False, fill=False) + theme(figure_size=(1.9,2.1))  +\
    theme(axis_title_y=element_blank())
    display(p)

# %%
