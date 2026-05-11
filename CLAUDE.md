# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a machine learning research repository containing **Decima**, a deep learning framework for predicting single-cell RNA-seq data from genomic DNA sequences. The repository consists of two main components:

- **daniodecima-main/**: Core Python package implementing the Decima model
- **daniodecima-applications-main/**: Jupyter notebooks and analysis scripts for experiments and applications

## Environment Setup

Load the anaconda module before working:
```bash
module load anaconda
```

All .ipynb notebook files have been converted to .py format using jupytext. The repository ignores .ipynb files.

## Core Architecture

### Decima Model (daniodecima-main/)
- **Base**: Built on Borzoi model architecture with 5-channel input (4 DNA + 1 gene mask)
- **Sequence Length**: 524,288bp input, cropped to 5,120bp 
- **Architecture**: 7 CNN blocks + 8 Transformer blocks → 1,920 embedding channels
- **Output**: Gene-specific expression counts via exponential activation
- **Loss**: TaskWisePoissonMultinomialLoss (combines Poisson + multinomial terms)

### Key Components
- `decima_model.py`: Core DecimaModel class with pretrained weight loading
- `lightning.py`: PyTorch Lightning training wrapper with custom metrics
- `loss.py`: Count-based loss functions for gene expression data
- `read_hdf5.py`: HDF5Dataset classes for sequence/expression data loading
- `evaluate.py`: Single-cell specific evaluation metrics and marker gene analysis
- `interpret.py`: Model attribution and motif analysis tools

## Development Commands

### Testing
```bash
cd daniodecima-main/
pytest
# Or with coverage:
pytest --cov decima --cov-report term-missing --verbose
```

### Building
```bash
cd daniodecima-main/
# Clean previous builds
python -c 'import shutil; [shutil.rmtree(p, True) for p in ("build", "dist")]'
# Build package
python -m build
```

### Code Quality
```bash
cd daniodecima-main/
# Run flake8 (configured for line length 88, Black-compatible)
flake8 src/
```

## Analysis Workflows

The notebooks in `daniodecima-applications-main/notebooks/` follow a structured pipeline (zebrafish-specific):

1. **0_sc_data_prep/**: Zebrafish single-cell data preprocessing (`zf-prep.{ipynb,py}`)
2. **2_dataset/**: Zebrafish training dataset creation (intervals, splits, HDF5)
3. **3_training/**: Model fine-tuning (adapted from upstream)
4. **4_evaluation/**: Model prediction and evaluation
5. **5_specificity/**: Cell-type specificity analysis and combined attribution analysis
6. **6_cell_states/**: TF-MoDISco cell-type motif attribution
7. **9_design/**: Directed-evolution design of cell-type-specific zebrafish regulatory elements

(Upstream's `1_processing/`, `7_eqtls/`, and `8_disease/` directories were not used in the DanioDecima work and were removed. See upstream Genentech/decima-applications for those analyses.)

## Common Patterns

### Model Initialization
- Pretrained weights loaded via HuggingFace (Genentech/borzoi-model) or local checkpoints
- Support for Decima checkpoints and random initialization
- Xavier, Kaiming, or zeros initialization options

### Data Loading
- HDF5-based datasets for memory efficiency
- Sequence augmentation with shifts (no reverse complement)
- Gene masks enable cell-type-specific predictions
- Support for variant analysis (reference/alternative alleles)

### Training
- Multi-GPU distributed training via PyTorch Lightning
- Adam optimization with configurable learning rates
- Gradient accumulation support
- MSE and Pearson correlation metrics tracked per gene

### Interpretation
- Attribution analysis using InputXGradient and Saliency (Captum)
- FIMO integration for transcription factor binding site analysis
- MEME suite integration for motif discovery
- Specificity transforms for cell-type-focused analysis

## HPC Integration

The codebase is designed for HPC environments with SLURM job submission scripts throughout the applications notebooks. Many scripts use `module load anaconda` and are optimized for GPU cluster execution.

## Key File Types

- `.py` files: All notebook code converted via jupytext
- `.h5`/`.h5ad`: Single-cell data in HDF5/AnnData formats
- `.gtf`: Genome annotation files
- `.bed`: Genomic interval files
- `.ckpt`: Model checkpoint files
- `.meme`: Motif analysis files

## Package Structure

```
daniodecima-main/
├── src/decima/          # Core package code (forked from Genentech/decima; package import name retained as `decima` for compatibility)
├── scripts/             # Training and prediction scripts
├── tests/               # Test suite (currently a stub)
├── setup.cfg            # Package configuration
├── pyproject.toml       # Build system configuration
└── LICENSE.txt          # Genentech Non-Commercial Software License v1.0 (inherited from upstream)
```