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
# # DanioCell: HDF5 Dataset Generation
#
# Encode DNA sequences for all gene intervals and write the training HDF5.
# Uses the local GRCz11+Lawson FASTA for sequence encoding.
#
# Input:  `daniocell_aggregated.h5ad` (intervals + split assigned)
# Output: `data.h5`

# %%
import argparse
import sys
import os
import pandas as pd
import numpy as np
import anndata
import h5py

REPO_DIR = '/hpc/projects/data.science/yangjoon.kim/step'
sys.path.append(os.path.join(REPO_DIR, 'decima-main/src/decima/'))
import write_hdf5

# %% [markdown]
# ## Args & resume check

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true', help='Re-run even if data.h5 exists')
args, _ = parser.parse_known_args()

# %% [markdown]
# ## Paths

# %%
scratch_dir = '/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/celltypes_chrom_split_v1'
matrix_file = os.path.join(scratch_dir, 'daniocell_aggregated.h5ad')
out_file    = os.path.join(scratch_dir, 'data.h5')
FASTA = '/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11_v4.3.2/fasta/genome.fa'

# %%
if os.path.exists(out_file) and not args.force:
    print(f"Output already exists: {out_file}")
    print("Skipping Phase 5. Use --force to re-run.")
    sys.exit(0)

# %% [markdown]
# ## Load matrix

# %%
ad = anndata.read_h5ad(matrix_file)
print(f"Shape: {ad.shape}")
print(f"Dataset split: {ad.var.dataset.value_counts().to_dict()}")

# Verify required columns
required = ['chrom', 'start', 'end', 'strand', 'gene_mask_start', 'gene_mask_end', 'dataset']
for c in required:
    assert c in ad.var.columns, f"Missing required var column: {c}"

# %% [markdown]
# ## Select var columns for HDF5

# %%
keep_cols = ['chrom', 'start', 'end', 'strand',
             'gene_name', 'gene_type', 'frac_nan',
             'gene_start', 'gene_end', 'gene_length',
             'gene_mask_start', 'gene_mask_end',
             'frac_N', 'mean_counts', 'n_tracks', 'dataset']
keep_cols = [c for c in keep_cols if c in ad.var.columns]

extra = [c for c in ad.var.columns if c not in keep_cols]
if extra:
    print(f"Dropping extra var columns: {extra}")
ad.var = ad.var[keep_cols]

# %% [markdown]
# ## Write HDF5

# %%
print(f"Writing HDF5 to {out_file} ...")
print(f"Genome FASTA: {FASTA}")
write_hdf5.write_hdf5(file=out_file, ad=ad, pad=5000, genome=FASTA)
print("Done.")

# %% [markdown]
# ## Sanity check HDF5 structure

# %%
expected_datasets = {'sequences', 'masks', 'labels', 'tasks', 'genes',
                     'pad', 'seq_len', 'padded_seq_len'}

with h5py.File(out_file, 'r') as f:
    present = set(f.keys())
    print(f"\nHDF5 datasets:")
    for key in sorted(f.keys()):
        item = f[key]
        if hasattr(item, 'shape') and item.shape:
            print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
        else:
            print(f"  {key}: {item[()]}")

missing = expected_datasets - present
assert not missing, f"FAIL: missing HDF5 datasets: {missing}"

with h5py.File(out_file, 'r') as f:
    n_genes = f['sequences'].shape[0]
    n_tasks = f['tasks'].shape[0]
    padded_len = int(f['padded_seq_len'][()])

assert n_genes > 0,  "FAIL: 0 genes in HDF5"
assert n_tasks > 0,  "FAIL: 0 tasks in HDF5"
assert padded_len == 524288 + 2*5000, f"FAIL: unexpected padded_seq_len={padded_len}"

print(f"\nSanity checks:")
print(f"  All expected datasets present:  OK")
print(f"  Genes (sequences):              {n_genes}")
print(f"  Tasks (tracks):                 {n_tasks}")
print(f"  Padded seq len:                 {padded_len} (=524288+10000)  OK")
