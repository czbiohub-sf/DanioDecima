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
# # DanioCell: Gene Annotation & Interval Creation
#
# Parse the GRCz11+Lawson GTF to annotate genes with genomic coordinates,
# then create 524,288bp intervals centered on each gene's TSS.
#
# Input:  `daniocell_aggregated.h5ad`
# Output: updated `daniocell_aggregated.h5ad` (interval columns added to var)

# %%
import argparse
import sys
import gzip
import os
import pandas as pd
import numpy as np
import anndata

REPO_DIR = '/hpc/projects/data.science/yangjoon.kim/step'
sys.path.append(os.path.join(REPO_DIR, 'decima-main/src/decima/'))
import preprocess

from grelu.data.preprocess import filter_chromosomes

# %% [markdown]
# ## Args & resume check

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true', help='Re-run even if output has interval columns')
args, _ = parser.parse_known_args()

# %% [markdown]
# ## Paths

# %%
scratch_dir  = '/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/celltypes_chrom_split_v1'
matrix_file  = os.path.join(scratch_dir, 'daniocell_aggregated.h5ad')
GTF   = '/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11_v4.3.2/genes/genes.gtf.gz'
FASTA = '/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11_v4.3.2/fasta/genome.fa'
ZF_CHROMS = [f'chr{i}' for i in range(1, 26)]

# %%
# Resume check: skip if interval columns already present
_ad_check = anndata.read_h5ad(matrix_file)
_has_intervals = 'gene_mask_start' in _ad_check.var.columns and 'frac_N' in _ad_check.var.columns
del _ad_check

if _has_intervals and not args.force:
    print(f"Interval columns already present in {matrix_file}")
    print("Skipping Phase 3. Use --force to re-run.")
    sys.exit(0)

# %% [markdown]
# ## Load aggregated matrix

# %%
ad = anndata.read_h5ad(matrix_file)
print(f"Shape: {ad.shape}")
print(f"Var columns: {ad.var.columns.tolist()}")

# %% [markdown]
# ## Parse GTF for gene coordinates

# %%
print(f"Parsing GTF: {GTF} ...")
records = []
with gzip.open(GTF, 'rt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9 or parts[2] != 'gene':
            continue
        chrom  = parts[0]
        start  = int(parts[3]) - 1  # 1-based → 0-based
        end    = int(parts[4])
        strand = parts[6]
        gene_name = None
        gene_type = None
        for field in parts[8].split(';'):
            field = field.strip()
            if field.startswith('gene_name'):
                gene_name = field.split('"')[1]
            elif field.startswith('gene_biotype'):
                gene_type = field.split('"')[1]
        if gene_name is not None:
            records.append({
                'gene_name': gene_name,
                'chrom': chrom, 'start': start, 'end': end,
                'strand': strand, 'gene_type': gene_type or '',
            })

gtf_df = pd.DataFrame(records)
print(f"GTF gene records total: {len(gtf_df)}")
gtf_df = gtf_df[gtf_df.chrom.isin(ZF_CHROMS)]
gtf_df = gtf_df.drop_duplicates('gene_name', keep='first')
print(f"GTF genes on chr1-25 (unique): {len(gtf_df)}")

# %% [markdown]
# ## Merge GTF coordinates into ad.var

# %%
n_genes_before = ad.n_vars
var_df = ad.var.copy()
var_df['gene_name'] = var_df.index

var_df = var_df.merge(
    gtf_df[['gene_name', 'chrom', 'start', 'end', 'strand', 'gene_type']],
    on='gene_name', how='left'
)
var_df = var_df.set_index('gene_name')

n_matched = var_df['chrom'].notna().sum()
print(f"Genes with GTF match: {n_matched} / {n_genes_before} ({n_matched/n_genes_before:.1%})")

var_df = var_df[var_df['chrom'].notna()]
ad = ad[:, var_df.index]
ad.var = var_df
print(f"Shape after dropping unmatched genes: {ad.shape}")

# %% [markdown]
# ## Set gene start/end/length

# %%
ad.var['gene_start']  = ad.var['start'].astype(int)
ad.var['gene_end']    = ad.var['end'].astype(int)
ad.var['gene_length'] = (ad.var['end'] - ad.var['start']).astype(int)

# %% [markdown]
# ## Filter to zebrafish autosomes (chr1–chr25)

# %%
print(f"Shape before chromosome filter: {ad.shape}")
ad = filter_chromosomes(ad, include=ZF_CHROMS)
print(f"Shape after chromosome filter:  {ad.shape}")

# %% [markdown]
# ## Create 524,288bp intervals

# %%
print(f"Creating intervals (genome: {FASTA})...")
ad = preprocess.var_to_intervals(ad.copy(), chr_end_pad=10000, genome=FASTA)
print(f"Shape after interval creation: {ad.shape}")

# %% [markdown]
# ## Drop N-rich intervals

# %%
print("Calculating frac_N (reads FASTA for each gene — may take several minutes)...")
ad.var['frac_N'] = ad.var.apply(
    lambda row: preprocess.get_frac_N(row, genome=FASTA), axis=1
)
print(f"Shape before N filter: {ad.shape}")
ad = ad[:, ad.var.frac_N < 0.4]
print(f"Shape after N filter:  {ad.shape}")

# %% [markdown]
# ## Sanity check

# %%
required_cols = ['chrom', 'start', 'end', 'strand',
                 'gene_start', 'gene_end', 'gene_length',
                 'gene_mask_start', 'gene_mask_end', 'frac_N']

assert ad.n_obs > 0,  "FAIL: 0 tracks"
assert ad.n_vars > 0, "FAIL: 0 genes"
for c in required_cols:
    assert c in ad.var.columns, f"FAIL: var missing '{c}'"

# Interval length should be exactly 524,288
interval_lengths = ad.var['end'] - ad.var['start']
assert (interval_lengths == 524288).all(), \
    f"FAIL: not all intervals are 524,288bp (got {interval_lengths.unique()})"

# gene_mask_start should be >= 0 and < 524288
assert (ad.var.gene_mask_start >= 0).all(), "FAIL: negative gene_mask_start"
assert (ad.var.gene_mask_end <= 524288).all(), "FAIL: gene_mask_end > 524288"

print(f"\nSanity checks:")
print(f"  Shape:             {ad.shape}  OK")
print(f"  Interval length:   524,288bp   OK")
print(f"  gene_mask_start:   {ad.var.gene_mask_start.min()} – {ad.var.gene_mask_start.max()}  OK")
print(f"  frac_N range:      {ad.var.frac_N.min():.3f} – {ad.var.frac_N.max():.3f}")
print(f"  Chromosomes:       {sorted(ad.var.chrom.unique())}")

# %% [markdown]
# ## Save

# %%
print(f"\nSaving to {matrix_file} ...")
ad.write_h5ad(matrix_file)
print(f"Done. Final shape: {ad.shape}")
