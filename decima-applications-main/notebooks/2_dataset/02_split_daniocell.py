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
# # DanioCell: Train/Val/Test Split
#
# Assign genes to train/val/test by chromosome, replicating the Zebrahub
# `chrom_split_v1` chromosome assignments for cross-atlas comparability.
#
# Input:  `daniocell_aggregated.h5ad` (with interval columns)
# Output: updated `daniocell_aggregated.h5ad` (with `dataset` column in var)

# %%
import argparse
import sys
import os
import pandas as pd
import numpy as np
import anndata

# %% [markdown]
# ## Args & resume check

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true', help='Re-run even if dataset column exists')
args, _ = parser.parse_known_args()

# %% [markdown]
# ## Paths

# %%
scratch_dir   = '/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/celltypes_chrom_split_v1'
matrix_file   = os.path.join(scratch_dir, 'daniocell_aggregated.h5ad')
zebrahub_file = '/hpc/projects/data.science/yangjoon.kim/zebrafish-seq2func-data/celltypes_chrom_split_v1/zebrahub_aggregated.h5ad'

# %%
_ad_check = anndata.read_h5ad(matrix_file)
_has_split = 'dataset' in _ad_check.var.columns
del _ad_check

if _has_split and not args.force:
    print(f"'dataset' column already present in {matrix_file}")
    print("Skipping Phase 4. Use --force to re-run.")
    sys.exit(0)

# %% [markdown]
# ## Read Zebrahub chromosome → split mapping

# %%
print("Loading Zebrahub h5ad to extract chromosome split mapping...")
zh = anndata.read_h5ad(zebrahub_file)
print(f"Zebrahub shape: {zh.shape}")

chrom_split_tab = pd.crosstab(zh.var['chrom'], zh.var['dataset'])
print("\nZebrahub chromosome × dataset cross-tabulation:")
print(chrom_split_tab)

# Assign each chromosome to the split that contains the most of its genes
chrom_to_split = chrom_split_tab.idxmax(axis=1).to_dict()

# Zebrahub stores chromosomes without 'chr' prefix (e.g. '1','2',...);
# DanioCell uses 'chr1','chr2',... — normalise the keys to match.
if not any(k.startswith('chr') for k in chrom_to_split.keys()):
    chrom_to_split = {f'chr{k}': v for k, v in chrom_to_split.items()}

print("\nChromosome → dataset mapping:")
for chrom in sorted(chrom_to_split.keys()):
    print(f"  {chrom}: {chrom_to_split[chrom]}")

del zh

# %% [markdown]
# ## Load DanioCell aggregated matrix

# %%
ad = anndata.read_h5ad(matrix_file)
print(f"\nDanioCell shape: {ad.shape}")

# %% [markdown]
# ## Assign dataset split

# %%
ad.var['dataset'] = ad.var['chrom'].map(chrom_to_split).fillna('train')

print("\nDanioCell chromosome × dataset:")
print(pd.crosstab(ad.var['chrom'], ad.var['dataset']))

# %% [markdown]
# ## Rename index to gene_name (pipeline convention)

# %%
if 'gene_name' in ad.var.columns and ad.var.index.name != 'gene_name':
    ad.var = ad.var.reset_index().set_index('gene_name')
    ad.var.index = ad.var.index.tolist()
    ad.var_names = ad.var.index.tolist()

# %% [markdown]
# ## Sanity check

# %%
assert 'dataset' in ad.var.columns, "FAIL: dataset column missing"
split_counts = ad.var['dataset'].value_counts()
for split in ['train', 'val', 'test']:
    assert split in split_counts.index, f"FAIL: split '{split}' has 0 genes"

total = len(ad.var)
print(f"\nSanity checks:")
print(f"  Shape: {ad.shape}  OK")
print(f"  Dataset split:")
for split, count in split_counts.sort_index().items():
    print(f"    {split}: {count:,} genes ({count/total:.1%})")
print(f"  All three splits present: OK")

# %% [markdown]
# ## Save

# %%
print(f"\nSaving to {matrix_file} ...")
ad.write_h5ad(matrix_file)
print("Done.")
