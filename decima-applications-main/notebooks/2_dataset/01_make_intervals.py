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
import pandas as pd
import numpy as np
import anndata
import os
import sys
from plotnine import *

sys.path.append('/code/decima/src/decima/')
import preprocess

from grelu.data.preprocess import filter_chromosomes

# %% [markdown]
# ## Paths

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823"
matrix_file = os.path.join(save_dir, "aggregated.h5ad")

# %% [markdown]
# ## Load matrix

# %%
# %%time
ad = anndata.read_h5ad(matrix_file)
print(ad.shape)

# %% [markdown]
# ## Format .var

# %%
ad.var = ad.var[['chrom', 'start', 'end','strand','gene_name','gene_type', 'frac_nan',
                 'mean_counts', 'n_tracks']]

# %%
ad.var['gene_start'] = ad.var.start.tolist()
ad.var['gene_end'] = ad.var.end.tolist()
ad.var['gene_length'] = ad.var.end - ad.var.start

# %% [markdown]
# ## Filter chromosomes

# %%
ad = filter_chromosomes(ad, "autosomesX")

# %% [markdown]
# ## Make intervals

# %%
# %%time

ad = preprocess.var_to_intervals(ad.copy(), chr_end_pad = 10000, genome="hg38")
print(ad.shape)
print(ad.var.start.min())

# %% [markdown]
# ## Drop intervals with too many Ns

# %%
# %%time
ad.var["frac_N"] = ad.var.apply(lambda row: preprocess.get_frac_N(row), axis=1)

# %%
print(ad.shape)
ad = ad[:, ad.var.frac_N < 0.4]
print(ad.shape)

# %% [markdown]
# ## How many intervals don't contain the gene end?

# %%
(ad.var.gene_mask_end == 524288).sum()

# %% [markdown]
# ## Visualize number of upstream and downstream bases

# %%
ad.var.loc[:, 'Upstream bases'] = ad.var.gene_mask_start
ad.var.loc[:, 'Downstream bases'] = 524288 - ad.var.gene_mask_end 

# %%
(
    ggplot(ad.var, aes(x='Upstream bases')) 
    + geom_histogram(fill='white', color='black', bins=50)
    + theme_classic() + theme(figure_size=(4, 2)) + ylab('Count')
    + scale_y_log10(labels = label_value) 
    + xlab("Number of bases upstream of TSS")
)

# %%
(
    ggplot(ad.var, aes(x='Downstream bases')) 
    + geom_histogram(fill='white', color='black', bins=50)
    + theme_classic() + theme(figure_size=(4, 2)) + ylab('Count')
    +xlab("Number of bases downstream of gene")
)

# %% [markdown]
# ## Save filtered anndata

# %%
ad.write_h5ad(matrix_file)
#ad = anndata.read_h5ad(out_file)

# %%
