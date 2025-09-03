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
import h5py

from grelu.visualize import plot_attributions
from grelu.interpret.motifs import trim_pwm

sys.path.append('/code/decima/src/decima')

from interpret import read_meme_file
from visualize import plot_logo

from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# %% [markdown]
# ## Paths

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823"
matrix_file = os.path.join(save_dir, "data.h5ad")
h5_file = os.path.join(save_dir, "data.h5")
ckpt_dir = os.path.join(save_dir, 'lightning_logs')

# %% [markdown]
# ## Read data

# %%
ad = anndata.read_h5ad(matrix_file)
ad = ad[ad.obs.organ=='CNS']

# %%
motifs, names = read_meme_file('../H12CORE_meme_format.meme')

# %% [markdown]
# ## Group brain cells

# %%
ad.obs['Group'] = None
ad.obs.loc[ad.obs.cell_type.isin([
    'Amygdala excitatory', 'CGE interneuron', 'Cerebellar inhibitory', 'Deep-layer corticothalamic and 6b',
    'Deep-layer intratelencephalic', 'Deep-layer near-projecting', 'Eccentric medium spiny neuron', 'Hippocampal CA1-3',
 'Hippocampal CA4', 'Hippocampal dentate gyrus', 'LAMP5-LHX6 and Chandelier', 'Lower rhombic lip', 'MGE interneuron',
    'Mammillary body', 'Medium spiny neuron','Midbrain-derived inhibitory','Splatter', 'Thalamic excitatory',
 'Upper rhombic lip','Upper-layer intratelencephalic',
]), 'Group'] = 'Neuron'

ad.obs.loc[ad.obs.cell_type.isin([
'Astrocyte', 'Bergmann glia','Microglia', 'Oligodendrocyte', 'Oligodendrocyte precursor','Committed oligodendrocyte precursor',
]), 'Group'] = 'Glia'

ad.obs.loc[ad.obs.cell_type.isin(['Choroid plexus', 'Ependymal']), 'Group'] = 'BBB'
ad.obs.loc[ad.obs.cell_type.isin(['Vascular', 'fibroblast']), 'Group'] = 'Other'

# %% [markdown]
# ## Predict differential expression (neurons vs. others)

# %%
ad.var['diff_true'] = ad[ad.obs.Group == 'Neuron'].X.mean(0) - ad[ad.obs.Group != 'Neuron'].X.mean(0)
ad.var['diff_pred'] =  ad[ad.obs.Group == 'Neuron'].layers['preds'].mean(0) - ad[ad.obs.Group != 'Neuron'].layers['preds'].mean(0)
print(scipy.stats.pearsonr(ad.var.loc[ad.var.dataset=='test', 'diff_true'], ad.var.loc[ad.var.dataset=='test', 'diff_pred']))
(
    ggplot(ad.var[ad.var.dataset=='test'], aes(x='diff_true', y='diff_pred')) 
    + geom_pointdensity(size=.1) + theme_classic() + theme(figure_size=(2.5, 2.7))
    + xlab('Measured log FC') + ylab('Predicted logFC')
    + ggtitle('      Neurons vs.\n  non-neurons (brain)')
    + geom_abline(slope=1, intercept=0)
    + geom_vline(xintercept = 0, linetype='--')
    + geom_hline(yintercept = 0, linetype='--')
)

# %% [markdown]
# ## Plot tf expression

# %%
gene = 'MYT1L'
ad.obs.cell_type = pd.Categorical(ad.obs.cell_type,
        categories=ad.obs.groupby('cell_type')[gene].median().sort_values(
            ascending=False).index.tolist())
(
    ggplot(ad.obs, aes(x='cell_type', y=gene, fill='Group'))
    + geom_boxplot(outlier_size=.1, size=.3) + theme_classic() + theme(figure_size=(3.6, 1.8))
    +theme(axis_text_x=element_blank()) + ylab('Measured Expression') + xlab('Cell type')
)

# %%
gene = 'REST'
ad.obs.cell_type = pd.Categorical(ad.obs.cell_type,
        categories=ad.obs.groupby('cell_type')[gene].median().sort_values().index.tolist())
(
    ggplot(ad.obs, aes(x='cell_type', y=gene, fill='Group'))
    + geom_boxplot(outlier_size=.1, size=.3) + theme_classic() + theme(figure_size=(4, 1.8))
    +theme(axis_text_x=element_blank()) + ylab('Measured\nExpression') + xlab('Cell type')
)

# %% [markdown]
# ## Plot modisco logos

# %%
i=0
modisco_h5 = f'brain_modisco/neuron_vs_glia_modisco/modisco_report.h5'
f = h5py.File(modisco_h5, 'r')
m = trim_pwm(np.array(f['neg_patterns'][f'pattern_{i}']['contrib_scores']), 0.1)
display(plot_attributions(m.T, figsize=(2,.8)))

# %%
i=12
modisco_h5 = f'brain_modisco/neuron_vs_glia_modisco/modisco_report.h5'
f = h5py.File(modisco_h5, 'r')
m = trim_pwm(np.array(f['pos_patterns'][f'pattern_{i}']['contrib_scores']), 0.1)
display(plot_attributions(np.flip(m.T, (0, 1)), figsize=(4, 1)))

# %% [markdown]
# ## Plot hocomoco logos

# %%
plot_logo(motifs[np.where(np.array(names)=='MYT1L')[0][0]])

# %%
plot_logo(motifs[np.where(np.array(names)=='REST')[0][0]], figsize=(4,1))

# %%
