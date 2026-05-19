# Manuscript figures — central folder

All figures for the DanioDecima manuscript, plus their render scripts, in one place.

Each render script saves both `.pdf` (Type-42 fonts, editable in Illustrator/Inkscape) and `.png` (300 DPI) into `figures/`.

## Status

| Fig | File stem | Source notebook | Status | Style |
|-----|-----------|-----------------|--------|-------|
| 1   | `fig01_training_curves` | `decima-main/scripts/results_celltype_models.ipynb` L929 | not started | — |
| 2   | `fig02_loss_curves_lr_sweep` | same notebook L16 | not started | — |
| 3   | `fig03_model_comparison` | `decima-applications-main/notebooks/4_evaluation/01_evaluate_celltypes.ipynb` L825 | **done** | original (user-approved) |
| 4   | `fig04_timepoint_enrichment` | same notebook L3334 | **done** | publication.mplstyle |
| 5   | `fig05_developmental_performance` | same notebook L4344 | **done** | publication.mplstyle |
| 6   | `fig06_conservation_expression` | `decima-applications-main/notebooks/5_specificity/04_analyze_attributions_celltypes_decima.ipynb` L1377 | not started | will be publication |
| 7   | `fig07_attribution_by_region` | same notebook L646 | not started | will be publication |
| 8   | `fig08_atac_attribution_overlap` | same notebook L214 | not started | will be publication |
| 9+10 | `fig09_10_motif_clustering` | `decima-applications-main/notebooks/9_design/designs_clustering_analysis.ipynb` L95 + L156 | **done** (merged) | original (user-approved) |

## Re-running

All scripts depend on the `gReLu` conda env and need `LD_LIBRARY_PATH` prefixed to avoid the `GLIBCXX_3.4.30` issue from the R module:

```bash
LD_LIBRARY_PATH=/hpc/mydata/yang-joon.kim/.conda/envs/gReLu/lib:$LD_LIBRARY_PATH \
    /hpc/mydata/yang-joon.kim/.conda/envs/gReLu/bin/python \
    manuscript_figures/scripts/figXX_render.py
```

## Data dependencies

| Fig | Input data |
|-----|------------|
| 1, 2 | TensorBoard event files under `/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/` |
| 3, 4, 5 | 16 × `data_out_*.h5ad` files under the same experiments dir (~240 MB each, only `.obs` is read) |
| 6, 7 | 16 × `*_fulltestset_cres_all.pkl` (alongside the h5ads) + `highly_variable_genes.txt` at `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/2_dataset/highly_variable_genes.txt` |
| 8 | Same as 6/7 plus possibly the ATAC bigWig (path in notebook is currently broken — verify on render) |
| 9, 10 | `motif_occurrence_by_celltype_combined_normalized.csv` under `…/design/analysis_25ct_20250619/comprehensive_summary_ism_90/` |

## Style policy

- **New figures (4, 5, 6, 7, 8)** → `figure_helpers.apply_style()` (the bundled `publication.mplstyle`: 6 pt body, 8 pt title, no top/right spines, no gridlines, lowercase axis labels, Type-42 PDF fonts).
- **Already-approved figures (3, 9/10)** → kept as-is to preserve the user-approved iterations. Bumping them to the publication style would be a one-line `apply_style()` change if uniform style becomes a priority later.
- All saves go through `figure_helpers.save_figure()` which writes PNG + PDF and verifies PDF text editability via `pypdf`.
