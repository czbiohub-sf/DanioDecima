# Plan: Fine-tune Decima on DanioCell Dataset

## Context

The Decima model has been fine-tuned on the Zebrahub zebrafish atlas (85 cell-type x timepoint tracks). We want to repeat this fine-tuning with DanioCell (Sur & Farrell), a different zebrafish single-cell atlas. Both use the same species/genome, so the genomic infrastructure is shared. The goal is to produce DanioCell-trained Decima models that can predict gene expression across DanioCell cell types.

The existing Zebrahub training data lives at:
- `/hpc/projects/data.science/yangjoon.kim/zebrafish-seq2func-data/celltypes_chrom_split_v1/` (`zebrahub_aggregated.h5ad` + `data.h5`)

---

## Confirmed Paths & Parameters

| Item | Value |
|------|-------|
| DanioCell h5ad (original) | `/hpc/projects/zebrahub/zebrahub_revision/scRNAseq/DanioCell/Daniocell2023_SeuratV4_annotated.h5ad` |
| DanioCell h5ad (var_names fixed) | `/hpc/projects/data.science/yangjoon.kim/daniocell-seq2func-data/Daniocell2023_varnames_fixed.h5ad` — **use this as input to Phase 1** |
| GRCz11+Lawson GTF | `/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11_v4.3.2/genes/genes.gtf.gz` |
| Pseudobulk grouping | `['identity.super', 'stage.group']` |
| Gene name format | Mixed case gene symbols — **100% GTF match** after var_names fix |
| Genome (FASTA) | `/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11_v4.3.2/fasta/genome.fa` (GRCz11+Lawson v4.3.2) |
| Genome (GTF) | `/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11_v4.3.2/genes/genes.gtf.gz` |
| Data + checkpoints scratch | `/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/` |

---

## DanioCell Dataset Summary (from inspection)

**Shape:** 489,686 cells × 36,250 genes

**Key `obs` columns:**
| Column | N unique | Notes |
|--------|----------|-------|
| `tissue` | 43 | Coarse tissue category (neural, eye, mesenchyme, periderm, ...) |
| `identity.super` | 158 | **Coarse cell type** — recommended for pseudobulking |
| `identity.sub` | 441 | Fine cell type; labels embed stage info (e.g. "36-60 hpf [glutamatergic]") |
| `stage.group` | 14 | Time window in hpf (e.g. "3-4", "48-58", "120") |
| `cluster` | 522 | Too granular for pseudobulking |
| `hpf` | 65 | Continuous hours post-fertilization |

**Pseudobulk grouping:** `['identity.super', 'stage.group']`
- `identity.super` (158) = cell type axis; `stage.group` (14) = temporal axis
- Mirrors Zebrahub structure: cell-type × developmental timepoint
- Max possible combinations: 158 × 14 = 2,212 — but most won't have cells
- After filtering for minimum cells (≥10), expect a few hundred pseudobulk tracks

---

## Gene Name Notes (IMPORTANT)

Inspected 2026-04-04. Key findings:

### Format
- Gene names are **mixed case gene symbols** (NOT Ensembl IDs)
- Lowercase zebrafish genes (e.g. `a1cf`, `aak1a`) — standard ZFIN format
- Uppercase entries (e.g. `ABCD2`, `ACKR2`) — human ortholog naming used by Lawson GTF
- 93 genes have `(1 of many)` suffix (e.g. `CDHR1 (1 of many)`) — multi-mapping marker from CellRanger

### GTF compatibility
- **93.7% of DanioCell genes (33,970/36,250) match the Lawson GTF by exact gene name**
- The 6.3% mismatch (2,280 genes) is **entirely due to Seurat/CellRanger processing artifacts**, not a real annotation incompatibility:
  1. **`NC-002333.X` / `XLOC-XXXXXX`** (~1,000+ genes): Seurat's `make.names()` in R converts `_` → `-`, so `NC_002333.10` (GTF) becomes `NC-002333.10` (DanioCell). These are mitochondrial genes and novel loci — filtered out during standard preprocessing anyway.
  2. **`.1`/`.2` version suffixes** (216 genes): CellRanger appends version numbers to disambiguate duplicate gene symbols (e.g. `LOC100149554` → `LOC100149554.1`). Fixable by stripping the suffix.
  3. **`(1 of many)` stripping** (64 genes): CellRanger sometimes drops the `(1 of many)` suffix when writing the count matrix even though the GTF retains it.
- **Conclusion: DanioCell and the Lawson GTF are fully compatible.** After filtering mito/XLOC genes (standard practice), effective mismatch is negligible.

### Seurat dot-replacement check
- Zebrafish paralog numbering uses genuine dots: `acot9.1`, `abcc6b.2` etc. — **NOT Seurat artifacts**
- **Confirmed: no gene has both a dot version AND a dash version coexisting** → no Seurat `"-"→"."` conversion detected
- The 18 non-numeric dot genes (`nkx2.2a`, `rxfp3.2a` etc.) are legitimate zebrafish TF names

### Genes to filter out during preprocessing
- Drop genes with `(1 of many)` that have ambiguous coordinates
- Drop the 2,280 genes not found in the GTF (they'll fail coordinate lookup anyway)
- Final expected gene count after filtering: ~33,000–34,000 genes

### Zebrahub comparison
- Zebrahub has 31,767 genes; 26,859 (84.6%) overlap with DanioCell
- Genes only in Zebrahub (not DanioCell): largely uppercase loci not captured in DanioCell

---

## Phase 0: Git Workflow

1. Merge `model-checkpoints` into `main` (or create PR)
2. Create new branch `daniocell-pipeline` from `main`
3. Optionally create a worktree for isolated development

---

## Phase 1: Single-Cell Data Prep

**Create:** `decima-applications-main/notebooks/0_sc_data_prep/daniocell-prep.py`

Template: `zf-prep.py` (same directory)

Steps:
1. Load **var_names-fixed** DanioCell h5ad (100% GTF match):
   ```python
   ad = sc.read('/hpc/projects/data.science/yangjoon.kim/daniocell-seq2func-data/Daniocell2023_varnames_fixed.h5ad')
   ```
   Pre-processing applied by `notebooks/0_sc_data_prep/daniocell-fix-varnames.py`:
   - Fixed 1,948 `XLOC-` → `XLOC_` (Seurat `_`→`-` artifact)
   - Fixed 17 `NC-002333` → `NC_002333` (mito genes, same artifact)
   - Fixed 230 `gene.N` → `gene` (CellRanger version suffixes, kept original)
   - Fixed 76 `GENE` → `GENE (1 of many)` (CellRanger stripped suffix)
   - Dropped 314 genes (9 no-GTF-match + 305 versioned collision duplicates)
   - Final shape: 489,686 cells × 35,936 genes (100% GTF match)
2. Set `ident_cols = ['identity.super', 'stage.group']`
   - `identity.super` = cell type, `stage.group` = timepoint
   - DanioCell does not have a `celltype` column — `identity.super` is the coarse cell type label
3. Filter low-count groups: `n_cells >= 10`
4. Filter lowly-expressed genes: `sc.pp.filter_genes(ad, min_cells=50)`
5. Pseudobulk via `sc.get.aggregate(ad, ident_cols, func='sum')`
6. Save as `daniocell_pseudobulk.h5ad`

**Note:** DanioCell uses `ad.layers['counts']` for raw counts (not `ad.X` directly). Use `ad.layers['counts']` for pseudobulking if `ad.X` is normalized.

---

## Phase 2: Aggregation & QC

**Create:** `decima-applications-main/notebooks/1_processing/06_aggregate_daniocell.py`

Template: `05_aggregate.py` (same directory)

Steps:
1. Load pseudobulked h5ad from Phase 1
2. Filter NaN-heavy genes (<33% NaN) and samples (<25% NaN)
3. Calculate QC metrics: `total_counts`, `n_genes`, `n_cells`
4. Drop low-quality samples
5. Add counts layer and CPM normalization
6. Add per-gene statistics (`mean_counts`, `n_tracks`)
7. Save as `daniocell_aggregated.h5ad`

**Note:** Call `preprocess.aggregate_anndata()` with explicit `by_cols=['identity.super', 'stage.group']` — the default is hardcoded for human tissue columns.

---

## Phase 3: Gene Annotation & Intervals

**Create:** `decima-applications-main/notebooks/2_dataset/01_make_intervals_daniocell.py`

Template: `01_make_intervals.py` (same directory)

Steps:
1. Parse GRCz11+Lawson GTF to get genomic coordinates per gene:
   ```
   GTF: /hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11_v4.3.2/genes/genes.gtf.gz
   ```
2. Match `ad.var_names` → GTF `gene_name` field (93.7% match rate expected)
3. Populate `ad.var` with: `chrom`, `start`, `end`, `strand`, `gene_name`, `gene_type`, `gene_start`, `gene_end`, `gene_length`
4. Drop unmatched genes (~6.3%, mostly `(1 of many)` base duplicates)
5. Filter chromosomes to zebrafish autosomes (1-25) — **NOT** `autosomesX`
6. Create 524,288bp intervals:
   ```python
   preprocess.var_to_intervals(ad, chr_end_pad=10000, genome="danRer11")
   ```
   Pass local FASTA path directly (genomepy alias not required):
   ```python
   FASTA = '/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11_v4.3.2/fasta/genome.fa'
   preprocess.var_to_intervals(ad, chr_end_pad=10000, genome=FASTA)
   ```
7. Filter intervals with >40% Ns:
   ```python
   preprocess.get_frac_N(row, genome=FASTA)
   ```
8. Save updated h5ad

---

## Phase 4: Train/Val/Test Split

**Create:** `decima-applications-main/notebooks/2_dataset/02_split_daniocell.py`

**Strategy:** Chromosome-based split, replicating Zebrahub `chrom_split_v1`.

**First step:** Inspect Zebrahub split to determine which chromosomes → which split:
```python
import anndata, pandas as pd
zh = anndata.read_h5ad('.../celltypes_chrom_split_v1/zebrahub_aggregated.h5ad')
print(pd.crosstab(zh.var['chrom'], zh.var['dataset']))
```
Then hardcode the same chromosome → split mapping in the DanioCell script.

Steps:
1. Load DanioCell aggregated h5ad
2. Assign `dataset` column by chromosome (matching Zebrahub scheme)
3. Verify gene counts: expect similar train/val/test proportions to Zebrahub (~58%/19%/23%)
4. Save updated h5ad

---

## Phase 5: HDF5 Generation

**Create:** `decima-applications-main/notebooks/2_dataset/03_make_h5_daniocell.py`

Template: `03_make_h5.py` (same directory)

Steps:
1. Load interval-annotated, split-assigned h5ad
2. Call:
   ```python
   FASTA = '/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11_v4.3.2/fasta/genome.fa'
   write_hdf5.write_hdf5(file=out_file, ad=ad, pad=5000, genome=FASTA)
   ```
3. Save as `data.h5` in DanioCell data directory

**Required library fix:**
- **File:** `decima-main/src/decima/write_hdf5.py` line 7 + 50
- **Change:** Add `genome="hg38"` parameter to `write_hdf5()` and pass it through to `convert_input_type()`:
  ```python
  # Current (line 7):
  def write_hdf5(file, ad, pad=0):
  # New:
  def write_hdf5(file, ad, pad=0, genome="hg38"):
  
  # Current (line 50):
  arr = convert_input_type(arr, "indices", genome="hg38")
  # New:
  arr = convert_input_type(arr, "indices", genome=genome)
  ```
  For DanioCell, pass the local FASTA path as `genome`; default `"hg38"` preserved for Zebrahub/human pipelines.

---

## Phase 6: Training

### 6a. Parameterize training script

**Modify:** `decima-main/scripts/decima_finetune.py`

- Line 64: Make matrix filename configurable via `--matrix_name` arg (default: `"zebrahub_aggregated.h5ad"`)
- Line 44: Replace hardcoded scratch path with `--scratch_dir` arg (default: `"/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell"`)

### 6b. Create DanioCell SLURM submission script

**Create:** `decima-main/scripts/submit_daniocell_finetune.sh`

Template: `submit_decima_finetune.sh`

Key changes:
- `BASE_DIR="/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell"`
- `DATA_DIR="/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/celltypes_chrom_split_v1"`
- Pass `--matrix_name daniocell_aggregated.h5ad`
- `--array=0-11` (12 experiments, or `0-7` for base set only)
- Same SLURM resources: 1 node, 8 CPUs, 1 GPU (H100/H200), 100G, 60hr

### 6c. Experiment matrix (12 experiments)

**What `decima-human` means:** Loads the original human Decima backbone (trained on human single-cell data) from:
`/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/decima_checkpoints_lal/rep{replicate}.ckpt`
The head is dropped; only the backbone (embedding + transformer) is reused. This is NOT the Zebrahub-fine-tuned model.

| ID | Init Mode | Source | Replicate | LR | Seed | Notes |
|----|-----------|--------|-----------|-----|------|-------|
| 0-3 | pretrained | decima-human | 0-3 | 3e-5 | 42 | **Primary**: Human Decima backbone → DanioCell |
| 4-7 | random | — | — | 3e-6 | 42-45 | Baseline: random init comparison |
| 8-11 | pretrained | local | 0-3 | 3e-5 | 42 | **Optional**: Zebrahub-fine-tuned → DanioCell (3-stage transfer) |

**On the optional 3-stage transfer (IDs 8-11):**
Using Human Decima → Zebrahub fine-tuned checkpoints as the starting point for DanioCell fine-tuning. This tests whether Zebrahub training provides a useful intermediate representation for DanioCell. These use `init_mode="pretrained", pretrained_source="local"` with paths to the best Zebrahub training checkpoints (paths TBD once Zebrahub runs are complete).

**Start with:** `--array=0-7` (8 core experiments). Add `8-11` later if Zebrahub checkpoints are available and the 3-stage transfer is worth investigating.

---

## Phase 7: Output Organization

```
Data (intermediate h5ad files):
/hpc/projects/data.science/yangjoon.kim/daniocell-seq2func-data/
  Daniocell2023_varnames_fixed.h5ad       ← DONE

Scratch (data + training outputs):
/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/
  celltypes_chrom_split_v1/
    daniocell_aggregated.h5ad
    data.h5
  experiments/
    decima_experiments_{timestamp}/
      pretrained_decima-human_rep{0-3}_lr3e-05_seed42/
      random_lr3e-06_seed{42-45}/
      pretrained_local_rep{0-3}_lr3e-05_seed42/   ← optional 3-stage
  genomepy_cache/
  wandb_cache/
```

---

## Summary of Files to Create/Modify

**New files (6):**
1. `decima-applications-main/notebooks/0_sc_data_prep/daniocell-prep.py`
2. `decima-applications-main/notebooks/1_processing/06_aggregate_daniocell.py`
3. `decima-applications-main/notebooks/2_dataset/01_make_intervals_daniocell.py`
4. `decima-applications-main/notebooks/2_dataset/02_split_daniocell.py`
5. `decima-applications-main/notebooks/2_dataset/03_make_h5_daniocell.py`
6. `decima-main/scripts/submit_daniocell_finetune.sh`

**Modified files (2):**
1. `decima-main/src/decima/write_hdf5.py` — add `genome` parameter (2-line change)
2. `decima-main/scripts/decima_finetune.py` — make matrix filename and scratch dir configurable

---

## Verification Checklist

1. After Phase 1: Check pseudobulk shape — expect ~200-400 tissue×stage tracks
2. After Phase 3: Verify ~93% gene retention after GTF matching
3. After Phase 5: Verify `data.h5` structure (sequences, masks, labels, tasks, genes datasets)
4. Smoke test: Run experiment_id=4 with `max_epochs=2` before full job submission
5. Full training: Submit `--array=0-7`, monitor with `squeue` and `seff`

---

## Remaining TODOs

- [ ] **Inspect Zebrahub chromosome split** — run `pd.crosstab(zh.var['chrom'], zh.var['dataset'])` to get exact chrom→split mapping for Phase 4
- [ ] **Verify grelu accepts local FASTA path** — test that `preprocess.var_to_intervals(genome=FASTA_PATH)` and `convert_input_type(genome=FASTA_PATH)` work with a filepath string, not just genomepy aliases
- [ ] **Check `ad.X` vs `ad.layers['counts']`** — confirm raw counts layer for pseudobulking (DanioCell has `layers['counts']`)
- [ ] **Check pseudobulk track count** — after Phase 1, verify how many `identity.super × stage.group` combos survive the ≥10 cells filter
- [ ] **Zebrahub checkpoint paths for optional experiments 8-11** — identify best Zebrahub training run checkpoints for 3-stage transfer (once Zebrahub runs complete)
- [ ] **Create scratch directory** — `mkdir -p /hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/`
