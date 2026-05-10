# daniodecima-applications-main — Fork Notes

This directory is a fork of [Genentech/decima-applications](https://github.com/Genentech/decima-applications).

**Upstream fork point:** commit `d39cc08` ("finish uploading preprint code")
**Fork maintainer:** Chan Zuckerberg Biohub (DanioDecima authors)
**License:** Genentech Non-Commercial Software License v1.0 (see `LICENSE.txt`)

---

## Summary

| Category | Count |
|---|---|
| Pristine (byte-equal to upstream HEAD) | 1 |
| Modified | 4 |
| New (added by DanioDecima authors) | 35 |
| Deleted (upstream-only material removed for our scope) | 49 |

The DanioDecima fork is concerned with zebrafish single-cell developmental
data and zebrafish-specific regulatory-element analyses. Most of the upstream
notebooks (which target human/mouse cell-type analyses, eQTLs, and disease
variants) do not apply to that scope and were removed.

---

## Modified files (4)

Each file below differs from upstream HEAD. A change-notice cell or header
in the file documents the modification.

| File | What we changed |
|---|---|
| `notebooks/4_evaluation/01_evaluate.ipynb` | Adapted evaluation logic for zebrafish cell-type fine-tuning checkpoints. |
| `notebooks/4_evaluation/00_predict.ipynb` | Adapted prediction pipeline; removed upstream-specific scaffolding. |
| `notebooks/3_training/01_finetune.ipynb` | Minor edits to the upstream training notebook for zebrafish data. |
| `notebooks/6_cell_states/modisco_simple.py` | Minor adjustment for our motif-attribution workflow. |

## New files (35)

Files added by the DanioDecima authors. These constitute the genuinely novel
zebrafish work in this fork.

### Zebrafish data preparation
- `notebooks/0_sc_data_prep/zf-prep.ipynb`, `zf-prep.py`

### Zebrafish dataset construction
- `notebooks/2_dataset/zebrafish.ipynb`, `zebrafish.py`
- `notebooks/2_dataset/zebrafish_data_exploration.ipynb`

### Training (jupytext export of modified upstream notebook)
- `notebooks/3_training/01_finetune.py`

### Evaluation
- `notebooks/4_evaluation/00_predict.py`, `00_submit_predict_decima.sh`
- `notebooks/4_evaluation/01_evaluate.py`, `01_evaluate_celltypes.ipynb`
- `notebooks/4_evaluation/ckpt_dirs_celltypes.txt`
- `notebooks/4_evaluation/README_daniodecima_predictions.md`

### Combined attribution analysis
- `notebooks/5_specificity/00_combined_attribution_analysis.py`, `00_submit_combined_attributions.sh`
- `notebooks/5_specificity/04_analyze_attributions_celltypes_decima.ipynb`
- `notebooks/5_specificity/ensemble_orthologies.ipynb`
- `notebooks/5_specificity/README_attribution_analysis.md`

### Cell-type motif attribution (TF-MoDISco)
- `notebooks/6_cell_states/celltype_motif_attribution.py`
- `notebooks/6_cell_states/environment_modisco.yml`, `environment_pytorch.yml`
- `notebooks/6_cell_states/submit_celltype_motifs.sh`
- `notebooks/6_cell_states/README_analysis_TF-MoDISco.md`

### Directed-evolution design of zebrafish regulatory elements
- `notebooks/9_design/00_evolve_combined.py`, `00_submit_evolve_combined.sh`
- `notebooks/9_design/02_summarize_results_designs.py`, `02_summarize_results_designs_test.py`
- `notebooks/9_design/02_summarize_results.py`, `02_summarize_results_spec.py`
- `notebooks/9_design/02_submit_summarize.sh`, `02_submit_summarize_results.sh`, `02_submit_summarize_spec.sh`
- `notebooks/9_design/1_read_celltypes.py`
- `notebooks/9_design/designs_clustering_analysis.ipynb`, `designs_volcano_plot.ipynb`
- `notebooks/9_design/README_design.md`

## Deleted files (49)

The following upstream files were removed because they do not apply to
the DanioDecima zebrafish work. Refer to upstream Genentech/decima-applications
at commit `d39cc08` for these analyses.

### Whole directories removed
- `notebooks/1_processing/` — atlas-building for human/mouse cell types (7 files)
- `notebooks/7_eqtls/` — human eQTL analysis (9 files)
- `notebooks/8_disease/` — human disease-variant analysis (5 files)

### Partial removals (only zebrafish work retained)
- `notebooks/0_sc_data_prep/` — removed `bca_prep.{ipynb,py}`, `heart-prep.{ipynb,py}`, `lung-prep.{ipynb,py}`, `retina-prep.{ipynb,py}`, `skin-prep.{ipynb,py}` (10 files; non-zebrafish prep)
- `notebooks/2_dataset/` — removed `01_make_intervals.{ipynb,py}`, `02_split.{ipynb,py}`, `03_make_h5.{ipynb,py}` (6 files; their zebrafish counterparts live alongside)
- `notebooks/3_training/__main__.py` — upstream entry point not used in our pipeline
- `notebooks/5_specificity/` — removed `01_evaluate_specific.{ipynb,py}`, `02_examples.{ipynb,py}`, `03_calculate_attributions.{ipynb,py}`, `04_analyze_attributions.{ipynb,py}`, `05_fabp1.{ipynb,py}` (10 files; non-zebrafish)
- `notebooks/6_cell_states/` — removed `0_lung_run.{ipynb,py}`, `1_lung_analysis.{ipynb,py}`, `2_neurons_analysis.{ipynb,py}`, `3_neuron_modisco.{ipynb,py}`, `4_others_run.{ipynb,py}`, `5_tregs.{ipynb,py}`, `6_fibroblasts.{ipynb,py}`, `Interpret.py`, `InterpretModisco.py`, `motif_meta.py` (17 files)
- `notebooks/9_design/` — removed `0_evolve.{ipynb,py}`, `1_read.{ipynb,py}` (4 files; replaced by our `00_evolve_combined.py` and `1_read_celltypes.py`)

## Pristine upstream files retained (1)

- `LICENSE.txt` (Genentech NC license — required by §4)

The `README.md` was modified to reflect the DanioDecima scope (fork-specific
content rather than upstream's one-line project README).
