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
import anndata
import os, sys
import argparse
import wandb

import importlib.util

# Get the full path to the module
module_path = '/hpc/mydata/mathias.voges/Projects/research/zf-decima/daniodecima-main/src/decima/read_hdf5.py'

# Load the module
spec = importlib.util.spec_from_file_location("read_hdf5", module_path)
read_hdf5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(read_hdf5)

# src_dir = f'{os.path.dirname(__file__)}/../src/decima/'
# sys.path.append(src_dir)
#from read_hdf5 import HDF5Dataset
# from lightning import LightningModel

# %%


# # Parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--name", type=str)
# parser.add_argument("--dir", type=str)
# parser.add_argument("--lr", type=float)
# parser.add_argument("--weight", type=float)
# parser.add_argument("--grad", type=int)
# parser.add_argument("--replicate", type=int, default=0)
# parser.add_argument("--bs", type=int, default=4)
# args = parser.parse_args()



# Get paths
data_dir = "/hpc/mydata/mathias.voges/Projects/research/zf-decima/outputs/grelu/decima/"
matrix_file = os.path.join(data_dir, "zebrahub_aggregated.h5ad")
h5_file = os.path.join(data_dir, "data.h5")
print(f"Data paths: {matrix_file}, {h5_file}")

# Load data
print("Reading anndata")
#print(os.getcwd())
ad = anndata.read_h5ad(matrix_file)
print(ad)

# Make datasets
# print("Making dataset objects")
train_dataset = read_hdf5.GeneForecastDataset(h5_file=h5_file, ad=ad, key="train", max_seq_shift=5000, augment_mode="random", seed=0, history_length=7, forecast_horizon=3)
val_dataset = read_hdf5.GeneForecastDataset(h5_file=h5_file, ad=ad, key="val", max_seq_shift=0, history_length=7, forecast_horizon=3)

# %%
train_dataset[4]

# %%

# %%
# Get paths
# data_dir = "/hpc/mydata/mathias.voges/Projects/research/zf-decima/outputs/grelu/decima/"
# matrix_file = os.path.join(data_dir, "Supplementary_file_1.h5ad")
# #h5_file = os.path.join(data_dir, "data.h5")
# print(f"Data paths: {matrix_file}, {h5_file}")

# # Load data
# print("Reading anndata")
# #print(os.getcwd())
# ad = anndata.read_h5ad(matrix_file)
# print(ad)

# %%
ad.obs

# %%
cns_cells = ad[ad.obs['cell_type'] == 'central_nervous_system'].copy()
notochord_cells = ad[ad.obs['cell_type'] == 'notochord'].copy()
periderm_cells = ad[ad.obs['cell_type'] == 'periderm'].copy()


# %%
notochord_cells.obs['timepoint']

# %%
cns_cells.var

# %%
import pandas as pd

expression_df = pd.DataFrame(cns_cells[:, ['neurog1']].X.toarray(), 
                            index=cns_cells.obs.index,
                            columns=['neurog1'])




# %%
expression_df

# %%
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming cns_cells is your AnnData object with CNS cells
# and it has a column in .obs called 'timepoint' or similar

# 1. Define the genes you want to plot
genes_of_interest = ['sox2', 'pax6', 'neurog1', 'neurod1', 'gfap', 'gpr19', 'ptpn12', 'chmp1a']  # Replace with your genes

# Make sure all genes exist in the dataset
genes_to_plot = [gene for gene in genes_of_interest if gene in periderm_cells.var_names]
if len(genes_to_plot) < len(genes_of_interest):
    missing_genes = set(genes_of_interest) - set(genes_to_plot)
    print(f"Warning: Some genes not found in dataset: {missing_genes}")

# 2. Extract expression data for these genes
expression_df = pd.DataFrame(periderm_cells[:, genes_to_plot].X.toarray(), 
                            index=periderm_cells.obs.index,
                            columns=genes_to_plot)

# Add timepoint information
expression_df['timepoint'] = periderm_cells.obs['timepoint'].values  # Replace 'timepoint' with your actual column name

# 3. Melt the dataframe for easier plotting
melted_df = pd.melt(expression_df, 
                   id_vars=['timepoint'],
                   value_vars=genes_to_plot,
                   var_name='Gene', 
                   value_name='Expression')

# 4. Create the plot
plt.figure(figsize=(12, 6))

# Option 1: Line plot showing mean expression by timepoint
sns.lineplot(data=melted_df, x='timepoint', y='Expression', hue='Gene', errorbar='se')
plt.title('Gene Expression Over Time in CNS Cells')
plt.xlabel('Developmental Timepoint')
plt.ylabel('Mean Expression')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Option 2: Violin plot to show distribution at each timepoint
plt.figure(figsize=(14, 8))
for i, gene in enumerate(genes_to_plot):
    plt.subplot(1, len(genes_to_plot), i+1)
    sns.violinplot(data=melted_df[melted_df['Gene'] == gene], 
                  x='timepoint', y='Expression')
    plt.title(f'{gene}')
    plt.xlabel('Timepoint')
    if i == 0:
        plt.ylabel('Expression')
    else:
        plt.ylabel('')
plt.tight_layout()
plt.show()

# Option 3: Heatmap of expression over time
pivot_df = melted_df.pivot_table(index='Gene', columns='timepoint', values='Expression', aggfunc='mean')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, cmap='viridis', annot=True, fmt='.2f', linewidths=.5)
plt.title('Mean Gene Expression by Timepoint')
plt.tight_layout()
plt.show()

# %%
print(ad.X is not None)


# %%
import scanpy as sc
sc.pl.violin(ad, keys=["STRADA", "ETV4"], groupby="celltype_coarse")


# %%
print(ad.raw)


# %%
