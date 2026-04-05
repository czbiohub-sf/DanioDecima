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
# # DanioCell Pseudobulk Preparation
#
# Pseudobulk the DanioCell atlas by cell type (`identity.super`) × developmental
# stage (`stage.group`). Produces one track per combination with ≥ 10 cells.
#
# Input:  `Daniocell2023_varnames_fixed.h5ad`
# Output: `daniocell_pseudobulk.h5ad`

# %%
import argparse
import sys
import os
import scanpy as sc
import pandas as pd
import numpy as np

pd.options.display.max_columns = None

# %% [markdown]
# ## Args & resume check

# %%
# parse_known_args handles the case when run as a notebook (extra kernel args)
parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true', help='Re-run even if output exists')
args, _ = parser.parse_known_args()

# %% [markdown]
# ## Paths

# %%
input_h5ad  = '/hpc/projects/data.science/yangjoon.kim/daniocell-seq2func-data/Daniocell2023_varnames_fixed.h5ad'
output_h5ad = '/hpc/projects/data.science/yangjoon.kim/daniocell-seq2func-data/daniocell_pseudobulk.h5ad'
ident_cols  = ['identity.super', 'stage.group']

# %%
if os.path.exists(output_h5ad) and not args.force:
    print(f"Output already exists: {output_h5ad}")
    print("Skipping Phase 1. Use --force to re-run.")
    sys.exit(0)

# %% [markdown]
# ## Load data

# %%
print("Loading DanioCell h5ad (var_names fixed)...")
ad = sc.read_h5ad(input_h5ad)
print(f"Shape: {ad.shape}")
print(f"Available layers: {list(ad.layers.keys())}")

# %% [markdown]
# ## Use raw counts

# %%
if 'counts' in ad.layers:
    print("Setting ad.X = ad.layers['counts'] (raw counts)")
    ad.X = ad.layers['counts'].copy()
    del ad.layers['counts']
else:
    print("WARNING: No 'counts' layer found — using ad.X as-is")

ad.layers = {}

# %% [markdown]
# ## Count cells per group

# %%
counts = ad.obs[ident_cols].value_counts().reset_index()
counts = counts.rename(columns={'count': 'n_cells'})
print(f"Total cell-type × stage combinations: {len(counts)}")
print(f"Combinations with >= 10 cells: {(counts.n_cells >= 10).sum()}")

# %% [markdown]
# ## Filter: keep groups with >= 10 cells

# %%
cells_before = ad.n_obs
merged = ad.obs[ident_cols].merge(counts, on=ident_cols, how='left')
ad = ad[merged['n_cells'].values >= 10].copy()
print(f"Cells before filter: {cells_before:,}")
print(f"Cells after filter:  {ad.n_obs:,}")

# %% [markdown]
# ## Filter: remove lowly expressed genes

# %%
genes_before = ad.n_vars
sc.pp.filter_genes(ad, min_cells=50)
print(f"Genes before filter: {genes_before:,}")
print(f"Genes after filter:  {ad.n_vars:,}")

# %% [markdown]
# ## Pseudobulk by identity.super × stage.group

# %%
counts = ad.obs[ident_cols].value_counts().reset_index()
counts = counts.rename(columns={'count': 'n_cells'})

print("Running pseudobulk aggregation (sum)...")
adp = sc.get.aggregate(ad, ident_cols, func='sum')
print(f"Pseudobulk shape: {adp.shape}")

adp.X = adp.layers['sum'].astype(int)
del adp.layers['sum']

adp.obs = adp.obs.merge(counts, on=ident_cols, how='left')
adp = adp.copy()

# %% [markdown]
# ## Sanity check

# %%
assert adp.n_obs > 0,  f"FAIL: pseudobulk has 0 tracks"
assert adp.n_vars > 0, f"FAIL: pseudobulk has 0 genes"
assert 'n_cells' in adp.obs.columns, "FAIL: n_cells missing from obs"
assert adp.obs['n_cells'].isna().sum() == 0, "FAIL: n_cells has NaN values"

# Check X is raw integer counts
x_sample = np.array(adp.X[:5, :200]) if not hasattr(adp.X, 'toarray') else adp.X[:5, :200].toarray()
is_integer = np.allclose(x_sample, np.round(x_sample), atol=1e-5)
print(f"\nSanity checks:")
print(f"  Shape:               {adp.shape}  OK")
print(f"  n_cells in obs:      OK")
print(f"  X dtype:             {adp.X.dtype}")
print(f"  X values are int:    {is_integer}  {'OK' if is_integer else 'WARNING: non-integer values!'}")
print(f"  X min / max:         {x_sample.min():.1f} / {x_sample.max():.1f}")
print(f"  Tracks per stage:\n{adp.obs['stage.group'].value_counts().sort_index().to_string()}")

# %% [markdown]
# ## Save

# %%
print(f"\nSaving to {output_h5ad} ...")
adp.write_h5ad(output_h5ad)
print(f"Done. Final shape: {adp.shape}")
