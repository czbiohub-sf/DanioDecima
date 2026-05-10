# daniodecima-main — Fork Notes

This directory is a fork of [Genentech/decima](https://github.com/Genentech/decima).

**Upstream fork point:** commit `d04961e` ("copied code")
**Fork maintainer:** Chan Zuckerberg Biohub (DanioDecima authors)
**License:** Genentech Non-Commercial Software License v1.0 (see `LICENSE.txt`)

This document enumerates the modifications, additions, and removals relative
to the upstream Decima fork point. Per Genentech NC §4 (Redistribution), modified
files also carry per-file change-notices in their headers.

---

## Summary

| Category | Count |
|---|---|
| Pristine (byte-equal to upstream HEAD) | 11 |
| Modified | 8 |
| New (added by DanioDecima authors) | 10 |
| Deleted (upstream-only material removed for our scope) | 19 |

---

## Modified files (8)

Each file below differs from upstream HEAD. The change-notice header in each
file states the modification scope.

| File | What we changed |
|---|---|
| `src/decima/decima_model.py` | Added support for multiple `init_mode` options (random, xavier, kaiming, zeros) and multiple `pretrained_source` options. Migrated Borzoi weight loading from WandB to HuggingFace. Added explicit `_set_seed()` for reproducibility. |
| `src/decima/lightning.py` | Added `temporal_loss_grouped()` for time-series forecasting variants. Exposed weight-decay hyperparameters. Split Poisson and multinomial loss components for logging. Set `CUBLAS_WORKSPACE_CONFIG` for deterministic training. |
| `src/decima/read_hdf5.py` | Pure addition: `GeneForecastDataset` class for time-series gene-expression forecasting. |
| `src/decima/loss.py` | Split Poisson and multinomial loss components for separate logging during training. |
| `src/decima/evaluate.py` | Bug fix: `auprc, auroc = np.nan` → `np.nan, np.nan` (tuple-unpack crash when all labels are positive or all negative for a cell type). |
| `src/decima/visualize.py` | Trivial axis/label tweak. |
| `scripts/finetune.py` | `sys.path.append` → `sys.path.insert(0, ...)` so the local `src/decima/` shadows the installed `lightning` package. Removed previously-exposed WandB API key. |
| `scripts/predict_genes.py` | Same `sys.path` ordering fix. |

## New files (10)

Files added by the DanioDecima authors that have no upstream counterpart.

| File | Purpose |
|---|---|
| `src/decima/lightning_temporal.py` | Temporal `LightningModel` variant for time-series gene-expression forecasting. |
| `src/decima/genome.py` | Genome-handling utilities (zebrafish-specific). |
| `scripts/decima_finetune.py` | Top-level fine-tuning entry point used by DanioDecima training jobs. |
| `scripts/decima_predictions.py` | Top-level prediction entry point. |
| `scripts/finetune_temporal.py` | Fine-tuning entry point for the temporal model. |
| `scripts/predict_genes_temporal.py` | Prediction entry point for the temporal model. |
| `scripts/submit_decima_finetune.sh` | SLURM job submission script for fine-tuning. |
| `scripts/submit_decima.sh` | SLURM job submission script for inference. |
| `scripts/README_decima_finetune.md` | Fine-tuning documentation. |
| `scripts/results_celltype_models.ipynb` | Results-aggregation notebook. |

## Deleted files (19, vs upstream)

These upstream files were removed because they did not apply to the DanioDecima
fork and would have constituted unnecessary republishing of upstream material.

- `__init__.py` (top-level 0-byte PyScaffold artifact)
- `AUTHORS.rst`, `CHANGELOG.rst`, `CONTRIBUTING.rst` (upstream contributor metadata)
- `README.md`, `README.rst` (upstream's project READMEs; DanioDecima has its own at the repo root)
- `fig1.png` (upstream banner figure)
- `tox.ini` (tox config — not used)
- `docs/*` (10 files: upstream Sphinx docs about installing/using upstream Decima)
- `tutorials/tutorial.{ipynb,py}` (upstream introductory tutorial)

Refer to upstream Genentech/decima at commit `d04961e` for these files.

## Pristine upstream files retained (11)

These files are byte-identical to upstream HEAD and are kept either because
they are imported by our modified/new code, are required for the package to
build/install, or are required by the Genentech NC license redistribution
clause.

- `LICENSE.txt` (Genentech NC license — required by §4)
- `setup.cfg`, `setup.py`, `pyproject.toml` (package metadata)
- `src/decima/__init__.py`
- `src/decima/interpret.py` (imported by attribution-analysis code)
- `src/decima/metrics.py` (imported by training)
- `src/decima/preprocess.py`
- `src/decima/variant.py`
- `src/decima/write_hdf5.py`
- `tests/conftest.py` (empty PyScaffold stub)
