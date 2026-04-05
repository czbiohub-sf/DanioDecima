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
# # Fix DanioCell var_names to match Lawson GTF naming conventions
#
# DanioCell was processed with Seurat in R, which applied `make.names()` character
# sanitization. This caused 3 categories of gene name divergence from the Lawson 4.3.2 GTF:
#
# 1. **`XLOC-XXXXXX` → `XLOC_XXXXXX`** (1,948 genes): Novel loci — Seurat converted `_` to `-`
# 2. **`NC-002333.N` → `NC_002333.N`** (17 genes): Mitochondrial genes — same `_` to `-` conversion
# 3. **`gene.N` → `gene`** (230 genes): CellRanger added `.N` version suffix for duplicate symbols
# 4. **`GENE` → `GENE (1 of many)`** (76 genes): CellRanger stripped the `(1 of many)` suffix
# 5. **9 genes** with no GTF match — dropped (uncharacterized/unmapped loci)
#
# Output: corrected h5ad saved to the DanioCell data directory (original not modified).

# %%
import gzip
import anndata
import pandas as pd
import numpy as np

# %% [markdown]
# ## Paths

# %%
input_h5ad  = '/hpc/projects/zebrahub/zebrahub_revision/scRNAseq/DanioCell/Daniocell2023_SeuratV4_annotated.h5ad'
output_h5ad = '/hpc/projects/data.science/yangjoon.kim/daniocell-seq2func-data/Daniocell2023_varnames_fixed.h5ad'
gtf_path    = '/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11_v4.3.2/genes/genes.gtf.gz'

# %% [markdown]
# ## Load GTF gene names

# %%
print("Parsing GTF...")
gtf_genes = set()
with gzip.open(gtf_path, 'rt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9 or parts[2] != 'gene':
            continue
        for field in parts[8].split(';'):
            field = field.strip()
            if field.startswith('gene_name'):
                gtf_genes.add(field.split('"')[1])
print(f"GTF gene names loaded: {len(gtf_genes)}")

# %% [markdown]
# ## Load DanioCell h5ad

# %%
print("Loading DanioCell h5ad...")
ad = anndata.read_h5ad(input_h5ad)
print(f"Shape: {ad.shape}")

# %% [markdown]
# ## Build gene name renaming map

# %%
rename_map = {}   # old_name -> new_name
drop_genes = []   # genes with no GTF match

for g in ad.var_names:
    if g in gtf_genes:
        continue  # already correct

    fixed = None

    # Fix 1: NC-002333.N -> NC_002333.N  (mito genes, underscore converted to dash by Seurat)
    if g.startswith('NC-'):
        candidate = g.replace('NC-', 'NC_', 1)
        if candidate in gtf_genes:
            fixed = candidate

    # Fix 2: XLOC-XXXXXX -> XLOC_XXXXXX  (novel loci, same conversion)
    if fixed is None and g.startswith('XLOC-'):
        candidate = g.replace('XLOC-', 'XLOC_', 1)
        if candidate in gtf_genes:
            fixed = candidate

    # Fix 3: gene.N -> gene  (CellRanger version suffix on duplicate gene symbols)
    if fixed is None and '.' in g:
        parts = g.rsplit('.', 1)
        if len(parts) == 2 and parts[1].isdigit():
            candidate = parts[0]
            if candidate in gtf_genes:
                fixed = candidate

    # Fix 4: GENE -> GENE (1 of many)  (CellRanger stripped the multi-mapping suffix)
    if fixed is None:
        candidate = g + ' (1 of many)'
        if candidate in gtf_genes:
            fixed = candidate

    if fixed is not None:
        rename_map[g] = fixed
    else:
        drop_genes.append(g)

print(f"Genes already matching GTF:  {ad.n_vars - len(rename_map) - len(drop_genes)}")
print(f"Genes to rename:             {len(rename_map)}")
print(f"  Fix 1 (NC- -> NC_):              {sum(1 for k in rename_map if k.startswith('NC-'))}")
print(f"  Fix 2 (XLOC- -> XLOC_):          {sum(1 for k in rename_map if k.startswith('XLOC-'))}")
print(f"  Fix 3 (strip .N suffix):         {sum(1 for k,v in rename_map.items() if not k.startswith(('NC-','XLOC-')) and '(1 of many)' not in v)}")
print(f"  Fix 4 (add '(1 of many)'):       {sum(1 for v in rename_map.values() if '(1 of many)' in v)}")
print(f"Genes with no GTF match (drop):  {len(drop_genes)}")
print(f"  Dropped: {drop_genes}")

# %% [markdown]
# ## Apply fixes

# %%
# Resolve collisions: when renaming gene.1 -> gene would collide with an existing gene,
# the already-correct gene takes priority — drop the versioned duplicate.
existing_genes = set(ad.var_names)
collision_drops = set()
for old, new in rename_map.items():
    if new in existing_genes:
        # The target name already exists as a valid gene — drop the duplicate
        collision_drops.add(old)

print(f"Collision duplicates to drop: {len(collision_drops)}")
print(f"  Examples: {sorted(list(collision_drops))[:10]}")

# %%
# Drop unfixable genes AND collision duplicates
to_drop = set(drop_genes) | collision_drops
ad = ad[:, [g for g in ad.var_names if g not in to_drop]].copy()
print(f"Shape after dropping unfixable/collision genes: {ad.shape}")

# %%
# Rename var_names using the map (only for non-dropped genes)
new_var_names = [rename_map.get(g, g) for g in ad.var_names]

# Sanity check: no duplicates after renaming
assert len(set(new_var_names)) == len(new_var_names), \
    "Renaming still has duplicate var_names! Check for remaining collisions."

ad.var_names = new_var_names
print(f"var_names updated.")

# %% [markdown]
# ## Verify

# %%
still_unmatched = [g for g in ad.var_names if g not in gtf_genes]
print(f"Genes still not matching GTF after fix: {len(still_unmatched)}")
if still_unmatched:
    print(f"  {still_unmatched[:20]}")

matched_frac = (ad.n_vars - len(still_unmatched)) / ad.n_vars
print(f"GTF match rate after fix: {matched_frac:.4%}")

# %%
# Spot check a few renamed genes
import random
sample = random.sample(list(rename_map.items()), min(10, len(rename_map)))
print("\nSample renames (old -> new):")
for old, new in sorted(sample):
    print(f"  {old!r:45s} -> {new!r}")

# %% [markdown]
# ## Save

# %%
print(f"\nSaving to {output_h5ad} ...")
ad.write_h5ad(output_h5ad)
print("Done.")
print(f"Final shape: {ad.shape}")
