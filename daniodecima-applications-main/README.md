# daniodecima-applications-main

Notebooks and analysis scripts for **DanioDecima**, a zebrafish-specific fork
of [Genentech/decima-applications](https://github.com/Genentech/decima-applications).

This directory contains the zebrafish single-cell developmental analysis,
attribution analysis, and directed-evolution design pipelines that accompany
the DanioDecima manuscript (Voges et al., 2025, in preparation).

## Notebook layout

The notebooks follow a structured zebrafish pipeline:

| Directory | Purpose |
|---|---|
| `0_sc_data_prep/` | Zebrafish single-cell data preprocessing (`zf-prep.{ipynb,py}`) |
| `2_dataset/` | Zebrafish training dataset construction (intervals, splits, HDF5) |
| `3_training/` | Model fine-tuning (adapted from upstream Decima) |
| `4_evaluation/` | Model prediction and evaluation (cell-type-specific) |
| `5_specificity/` | Cell-type specificity analysis and combined attribution analysis |
| `6_cell_states/` | TF-MoDISco cell-type motif attribution |
| `9_design/` | Directed-evolution design of cell-type-specific zebrafish regulatory elements |

Upstream's `1_processing/`, `7_eqtls/`, and `8_disease/` directories targeted
human/mouse analyses and were removed from this fork. See upstream
Genentech/decima-applications for those analyses.

## License & attribution

This directory is licensed under the Genentech Non-Commercial Software
License v1.0 (see `LICENSE.txt`). Modified files carry change-notices per
§4 of the license. Per-file modification details are in `FORK_NOTES.md`.

For the original Decima methodology and human/mouse experiments, see and
cite the Decima manuscript (Lal et al., 2024). For DanioDecima, cite the
manuscript referenced in the repository-root `NOTICE` file.
