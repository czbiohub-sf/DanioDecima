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
# # Calculate attributions for all genes

# %%
import numpy as np
import pandas as pd
import anndata
import wandb
from tqdm import tqdm
import h5py
import torch
import os, sys
sys.path.append('/code/decima/src/decima')

from lightning import LightningModel
from interpret import extract_gene_data, attributions
from plotnine import *
from captum.attr import InputXGradient
from grelu.transforms.prediction_transforms import Aggregate

# %matplotlib inline

# %% [markdown]
# ## Paths

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823"
matrix_file = os.path.join(save_dir, "data.h5ad")
h5_file = os.path.join(save_dir, "data.h5")
ckpt_dir = os.path.join(save_dir, 'lightning_logs')

# %% [markdown]
# ## Load

# %%
ad = anndata.read_h5ad(matrix_file)
ad = ad[:, ad.var.dataset == "test"].copy()
ad.shape

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
# ## Identify expression cutoff

# %%
x = np.ravel(ad.X)
(
    ggplot(pd.DataFrame({'x':np.random.choice(x[x>0], 100000)}), aes(x='x'))
    + geom_histogram() + theme_classic() + theme(figure_size=(6,2))
)

# %% [markdown]
# ## Compute and save attributions

# %%
device = torch.device(1)
models = [model.to(device) for model in models]

# %%
tasks = []
seqs = []
for gene in tqdm(ad.var_names):
    tasks.append(ad.obs_names[np.array(ad[:, gene].X).squeeze() > 1.5].tolist())
    seqs.append(extract_gene_data(h5_file, gene, merge=True))

# %%
file=os.path.join(save_dir, 'attr.h5')
with h5py.File(file, "w") as f:
    for g, t, s in tqdm(zip(ad.var_names, tasks, seqs)):
        s = s.to(device)
        attr = []
        
        for model in models:
            model.add_transform(Aggregate(tasks=t, task_aggfunc="mean", model=model))
            attributer = InputXGradient(model)
            with torch.no_grad():
                attr.append(attributer.attribute(s)[:4].cpu().numpy())

        f.create_dataset(g, shape=(524288,), data=np.stack(attr).mean(0).sum(0)  )

# %%
