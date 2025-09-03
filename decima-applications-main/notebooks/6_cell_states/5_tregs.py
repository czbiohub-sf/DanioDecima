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
import sys
from grelu.interpret.motifs import trim_pwm
from grelu.visualize import plot_attributions
from plotnine import *

sys.path.append('/code/decima/src/decima')
from interpret import read_meme_file
from visualize import plot_logo
# %matplotlib inline

# %% [markdown]
# ## Paths

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823"
matrix_file = os.path.join(save_dir, "data.h5ad")
h5_file = os.path.join(save_dir, "data.h5")
ckpt_dir = os.path.join(save_dir, 'lightning_logs')

# %% [markdown]
# ## Load data

# %%
ad = anndata.read_h5ad(matrix_file)
ad = ad[ad.obs.dataset=='skin_atlas']
ad = ad[ad.obs.cell_type.isin(['Treg cycling','Treg'])]

# %%
motifs, names = read_meme_file('../H12CORE_meme_format.meme')

# %% [markdown]
# ## Predict differential expression

# %%
ad.var['diff_true'] = ad[ad.obs.cell_type == 'Treg cycling'].X.mean(0) - ad[ad.obs.cell_type =='Treg'].X.mean(0)
ad.var['diff_pred'] =  ad[ad.obs.cell_type == 'Treg cycling'].layers['preds'].mean(0) - ad[ad.obs.cell_type =='Treg'].layers['preds'].mean(0)

# %%
print(scipy.stats.pearsonr(ad.var.loc[ad.var.dataset=='test', 'diff_true'], ad.var.loc[ad.var.dataset=='test', 'diff_pred']))

# %%
(
    ggplot(ad.var[ad.var.dataset=='test'], aes(x='diff_true', y='diff_pred')) 
    + geom_pointdensity(size=.1) + theme_classic() + theme(figure_size=(2.6, 2.5))
    + xlab('Measured log FC') + ylab('Predicted logFC')
    + ggtitle('     Treg cycling vs. Treg')
    + geom_abline(slope=1, intercept=0)
    + geom_vline(xintercept = 0, linetype='--')
    + geom_hline(yintercept = 0, linetype='--')
)

# %% [markdown]
# ## Plot logos

# %%
i=2

modisco_h5 = f'Treg_cycling__vs__tregnoncycling/modisco_full/modisco_report.h5'
f = h5py.File(modisco_h5, 'r')
m = trim_pwm(np.array(f['pos_patterns'][f'pattern_{i}']['contrib_scores']), 0.1)
display(plot_attributions(np.flip(m.T, (0, 1)), figsize=(4, 1)))

# %%
plot_logo(motifs[np.where(np.array(names)=='E2F4')[0][0]], figsize=(3, 1))

# %%
