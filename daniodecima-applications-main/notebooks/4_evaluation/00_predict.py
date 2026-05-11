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

# %%
import numpy as np
import pandas as pd
import anndata
import os, sys
sys.path.append('/code/decima/src/decima')

#import read_hdf5
# from lightning import LightningModel

# from grelu.visualize import plot_distribution
# from plotnine import *
# # %matplotlib inline

# %% [markdown]
# ## Paths

# %%
save_dir="/hpc/mydata/mathias.voges/Projects/research/zf-decima/outputs/grelu/decima"
matrix_file = os.path.join(save_dir, "zebrahub_aggregated.h5ad")
h5_file = os.path.join(save_dir, "data.h5")
ckpt_dir = os.path.join(save_dir, 'lightning_logs/umv5p24k') 

# %%
# ckpts = !find {ckpt_dir} -name e*.ckpt
ckpts

# %% [markdown]
# ## Get predictions

# %%
out_file = os.path.join(save_dir, "data_out_decima_v20250319_pretrained_rep0_forecast_horizon_7_lstm.h5ad")

# %%
CMD = f"python /hpc/mydata/mathias.voges/Projects/research/zf-decima/daniodecima-main/scripts/predict_genes_temporal.py --device 1 --ckpts {' '.join(ckpts)} --h5_file {h5_file}  --matrix_file {matrix_file} --out_file {out_file} --max_seq_shift 3"
print(CMD)

# %%
