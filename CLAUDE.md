# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a machine learning research repository containing **Decima**, a deep learning framework for predicting single-cell RNA-seq data from genomic DNA sequences. The repository consists of two main components:

- **decima-main/**: Core Python package implementing the Decima model
- **decima-applications-main/**: Jupyter notebooks and analysis scripts for experiments and applications

## Environment Setup

Load the anaconda module before working:
```bash
module load anaconda
```

All .ipynb notebook files have been converted to .py format using jupytext. The repository ignores .ipynb files.

## Core Architecture

### Decima Model (decima-main/)
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
cd decima-main/
pytest
# Or with coverage:
pytest --cov decima --cov-report term-missing --verbose
```

### Building
```bash
cd decima-main/
# Clean previous builds
python -c 'import shutil; [shutil.rmtree(p, True) for p in ("build", "dist", "docs/_build")]'
# Build package
python -m build
```

### Code Quality
```bash
cd decima-main/
# Run flake8 (configured for line length 88, Black-compatible)
flake8 src/
```

### Documentation
```bash
cd decima-main/
# Build docs
sphinx-build --color -b html -d "docs/_build/doctrees" "docs" "docs/_build/html"
# Check for broken links
sphinx-build --color -b linkcheck -d "docs/_build/doctrees" "docs" "docs/_build/linkcheck"
```

## Analysis Workflows

The notebooks in `decima-applications-main/notebooks/` follow a structured pipeline:

1. **0_sc_data_prep/**: Single-cell data preprocessing for different tissues
2. **1_processing/**: Atlas generation and data aggregation
3. **2_dataset/**: Training dataset creation (intervals, splits, HDF5 generation)
4. **3_training/**: Model fine-tuning and training scripts
5. **4_evaluation/**: Model prediction and evaluation
6. **5_specificity/**: Cell-type specificity analysis and attribution calculations
7. **6_cell_states/**: Cell state analysis including modisco motif analysis
8. **7_eqtls/**: eQTL analysis and genetic variant effects
9. **8_disease/**: Disease-associated variant analysis
10. **9_design/**: Regulatory element design via directed evolution

## Common Patterns

### Model Initialization
- Pretrained weights loaded via WandB (human/mouse Borzoi models)
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
decima-main/
├── src/decima/          # Core package code
├── scripts/             # Training and prediction scripts  
├── tests/               # Test suite
├── docs/                # Sphinx documentation
├── setup.cfg            # Package configuration
└── pyproject.toml       # Build system configuration
```