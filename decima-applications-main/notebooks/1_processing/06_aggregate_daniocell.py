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
# # DanioCell Aggregation & QC
#
# QC filtering, normalization, and statistics for the DanioCell pseudobulk matrix.
#
# Input:  `daniocell_pseudobulk.h5ad` (raw summed counts)
# Output: `daniocell_aggregated.h5ad` (QC-filtered, log-CPM normalized)

# %%
import argparse
import sys
import os
import pandas as pd
import numpy as np
import anndata
import scanpy as sc

REPO_DIR = '/hpc/projects/data.science/yangjoon.kim/step'
sys.path.append(os.path.join(REPO_DIR, 'decima-main/src/decima/'))
import preprocess

# %% [markdown]
# ## Args & resume check

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true', help='Re-run even if output exists')
args, _ = parser.parse_known_args()

# %% [markdown]
# ## Paths

# %%
input_h5ad  = '/hpc/projects/data.science/yangjoon.kim/daniocell-seq2func-data/daniocell_pseudobulk.h5ad'
scratch_dir = '/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/celltypes_chrom_split_v1'
output_h5ad = os.path.join(scratch_dir, 'daniocell_aggregated.h5ad')
by_cols     = ['identity.super', 'stage.group']

os.makedirs(scratch_dir, exist_ok=True)

# %%
if os.path.exists(output_h5ad) and not args.force:
    print(f"Output already exists: {output_h5ad}")
    print("Skipping Phase 2. Use --force to re-run.")
    sys.exit(0)

# %% [markdown]
# ## Load pseudobulk

# %%
ad = anndata.read_h5ad(input_h5ad)
print(f"Shape: {ad.shape}")

if hasattr(ad.X, 'toarray'):
    ad.X = ad.X.toarray()
ad.X = ad.X.astype(float)

# %% [markdown]
# ## Verify raw counts
#
# X should contain raw integer counts (summed pseudobulk).
# Values like 1.0, 2.0, ... confirm this; fractional values would indicate
# the input was already normalized.

# %%
print("=== Raw counts verification ===")
x_check = ad.X[:10, :500]
is_integer    = np.allclose(x_check, np.round(x_check), atol=1e-5)
is_nonneg     = (x_check >= 0).all()
sample_nonzero = x_check[x_check > 0][:20]

print(f"  X dtype:                {ad.X.dtype}")
print(f"  Values non-negative:    {is_nonneg}")
print(f"  Values are integers:    {is_integer}  <-- should be True for raw counts")
print(f"  X min / max:            {ad.X.min():.2f} / {ad.X.max():.2f}")
print(f"  Mean of nonzero values: {ad.X[ad.X > 0].mean():.2f}  (raw counts: typically 10-10000)")
print(f"  Sample nonzero values:  {sample_nonzero}")

if not is_integer:
    print("\n  WARNING: X values are not integers — input may already be normalized.")
    print("  Proceeding, but verify that daniocell-prep.py used raw counts (layers['counts']).")
else:
    print("\n  OK: X contains raw integer counts.")
print("================================")

# %% [markdown]
# ## NaN check

# %%
ad.var['frac_nan'] = np.isnan(ad.X).mean(0)
print(f"Genes with any NaN: {(ad.var.frac_nan > 0).sum()}")

print(f"Shape before NaN gene filter:  {ad.shape}")
ad = ad[:, ad.var.frac_nan < 0.33]
print(f"Shape after NaN gene filter:   {ad.shape}")

ad.obs['frac_nan'] = np.isnan(ad.X).mean(1)
print(f"Tracks with any NaN: {(ad.obs.frac_nan > 0).sum()}")

print(f"Shape before NaN track filter: {ad.shape}")
ad = ad[ad.obs.frac_nan < 0.25]
print(f"Shape after NaN track filter:  {ad.shape}")

ad.X = np.nan_to_num(ad.X)

# %% [markdown]
# ## Aggregate (consolidate obs columns)

# %%
ad = preprocess.aggregate_anndata(ad, by_cols=by_cols, sum_cols=['n_cells'])
print(f"Shape after aggregation: {ad.shape}")

# %% [markdown]
# ## Per-track QC statistics

# %%
ad.obs['total_counts'] = ad.X.sum(1)
ad.obs['n_genes']      = np.sum(ad.X > 0, axis=1)

print("Per-track quantiles (10th, 20th percentile):")
for col in ['n_genes', 'n_cells', 'total_counts']:
    p10 = np.quantile(ad.obs[col], 0.10)
    p20 = np.quantile(ad.obs[col], 0.20)
    print(f"  {col}: p10={p10:.0f}, p20={p20:.0f}")

# %% [markdown]
# ## Drop low-quality tracks

# %%
n_genes_thr       = np.quantile(ad.obs.n_genes, 0.10)
total_counts_thr  = np.quantile(ad.obs.total_counts, 0.10)
n_cells_thr       = 50

drop = (
    (ad.obs.n_cells < n_cells_thr) &
    (ad.obs.n_genes < n_genes_thr) &
    (ad.obs.total_counts < total_counts_thr)
)
print(f"Tracks to drop (all three thresholds met): {drop.sum()}")
ad = ad[~drop]
print(f"Shape after low-quality track removal: {ad.shape}")

# %% [markdown]
# ## Normalize

# %%
ad.layers['counts'] = ad.X.copy()          # raw counts layer
sc.pp.normalize_total(ad, target_sum=1e6)  # CPM
ad.layers['norm'] = ad.X.copy()
sc.pp.log1p(ad)                            # log1p(CPM)

ad.var['mean_counts'] = ad.X.mean(0)
ad.var['n_tracks']    = np.sum(ad.X > 0, axis=0)

ad_scaled = sc.pp.scale(ad, copy=True)
ad.layers['scaled'] = ad_scaled.X.copy()
del ad_scaled

# %% [markdown]
# ## Sanity check

# %%
required_var_cols = ['frac_nan', 'mean_counts', 'n_tracks']
required_obs_cols = ['identity.super', 'stage.group', 'n_cells', 'total_counts', 'n_genes']

assert ad.n_obs > 0,  "FAIL: aggregated has 0 tracks"
assert ad.n_vars > 0, "FAIL: aggregated has 0 genes"
for c in required_var_cols:
    assert c in ad.var.columns, f"FAIL: var missing column '{c}'"
for c in required_obs_cols:
    assert c in ad.obs.columns, f"FAIL: obs missing column '{c}'"
assert 'counts' in ad.layers, "FAIL: 'counts' layer missing"
assert (ad.layers['counts'] >= 0).all(), "FAIL: negative values in counts layer"

# Confirm X is log-normalized (values should be in ~0–15 range, not large integers)
x_max = ad.X.max()
assert x_max < 100, f"FAIL: X max is {x_max:.1f} — expected log-normalized values (<100)"

print(f"\nSanity checks:")
print(f"  Shape:               {ad.shape}  OK")
print(f"  Required var cols:   OK  {required_var_cols}")
print(f"  Required obs cols:   OK  {required_obs_cols}")
print(f"  Counts layer:        OK")
print(f"  X (log-CPM) range:   {ad.X.min():.2f} – {ad.X.max():.2f}  OK")
print(f"  Cell types:          {ad.obs['identity.super'].nunique()}")
print(f"  Stage groups:        {ad.obs['stage.group'].nunique()}")

# %% [markdown]
# ## Save

# %%
print(f"\nSaving to {output_h5ad} ...")
ad.write_h5ad(output_h5ad)
print("Done.")
