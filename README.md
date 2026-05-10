# DanioDecima Deep Learning Framework

A comprehensive deep learning framework for predicting single-cell RNA-seq expression from genomic DNA sequences in zebrafish development.

## Overview

DanioDecima extends the Borzoi architecture to predict cell-type-specific gene expression patterns across zebrafish developmental stages. The framework implements transfer learning from human and mouse models, enabling cross-species regulatory sequence analysis and synthetic regulatory element design.

![DanioDecima Framework](overview_schematic.png)

## Key Features

- **Multi-species Transfer Learning**: Systematic comparison of human vs. mouse pretrained models for zebrafish expression prediction
- **Cell-Type Specificity**: Predictions across 85 cell-type × developmental timepoint combinations
- **Attribution Analysis**: Model interpretability through genomic feature attribution to assess conservation confounding
- **Regulatory Element Design**: Directed evolution pipeline for generating cell-type-specific synthetic promoters
- **Comprehensive Evaluation**: Statistical framework for model performance assessment across developmental stages

## Framework Architecture

### Model Specifications
- **Input**: 5-channel sequences (4 DNA bases + gene mask, 524,288bp windows)
- **Architecture**: 7 CNN blocks + 8 Transformer blocks (1,920 embedding channels)
- **Output**: Cell-type-specific expression predictions via exponential activation
- **Loss Function**: TaskWisePoissonMultinomialLoss (Poisson + multinomial components)

### Experimental Design
- **16 Model Configurations**: 4 initialization strategies × 4 replicates each
- **Initialization Types**: Random, Human-Borzoi, Human-Decima, Mouse-Borzoi pretraining
- **Training**: Early stopping, gradient accumulation, distributed GPU compute via SLURM

## Analytical Workflows

1. **Data Exploration**: Single-cell preprocessing, variance analysis, conservation metrics
2. **Model Training**: Distributed training pipeline with systematic experimental design
3. **Prediction Generation**: Scalable inference with sequence augmentation
4. **Performance Evaluation**: Cell-type-specific metrics and developmental stage analysis
5. **Attribution Analysis**: InputXGradient interpretability and conservation confounding assessment
6. **Regulatory Design**: Evolutionary optimization of synthetic regulatory elements

## Key Results

- **Transfer Learning Effectiveness**: Quantified performance gains from cross-species pretraining
- **Developmental Patterns**: Temporal prediction accuracy across zebrafish embryogenesis
- **Conservation Analysis**: Assessment of model reliance on evolutionary vs. regulatory signals
- **Synthetic Design**: Generated cell-type-specific promoter elements with motif analysis

## Repository Structure

```
daniodecima-applications-main/notebooks/
├── 2_dataset/           # Data exploration and preprocessing
├── 4_evaluation/        # Model evaluation and performance analysis
├── 5_specificity/       # Attribution analysis pipeline
└── 9_design/           # Regulatory element design workflows

daniodecima-main/scripts/
├── decima_finetune.py          # Model training
├── decima_predictions.py       # Prediction generation
└── submit_*.sh                 # SLURM job submission scripts
```

## Requirements

- PyTorch Lightning
- Captum (attribution analysis)
- scanpy (single-cell analysis)
- SLURM (distributed computing)
- GPU compute (H100/H200 recommended)

## Usage

### Model Training
```bash
sbatch daniodecima-main/scripts/submit_decima_finetune.sh
```

### Generate Predictions
```bash
sbatch daniodecima-applications-main/notebooks/4_evaluation/00_submit_predict_decima.sh
```

### Attribution Analysis
```bash
sbatch daniodecima-applications-main/notebooks/5_specificity/00_submit_combined_attributions.sh
```

### Regulatory Element Design
```bash
sbatch daniodecima-applications-main/notebooks/9_design/00_submit_evolve_combined.sh
```

## Citation

**Primary Framework:**
```bibtex
@article{lal2024decoding,
  title={Decoding sequence determinants of gene expression in diverse cellular and disease states},
  author={Lal, Avantika and Karollus, Alexander and Gunsalus, Laura and Garfield, David and Nair, Surag and Tseng, Alex M and Gordon, M Grace and Blischak, John and van de Geijn, Bryce and Bhangale, Tushar and others},
  journal={bioRxiv},
  pages={2024--10},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}

```

**ZebraHub Dataset:**
```bibtex
@article{lange2024multimodal,
  title={A multimodal zebrafish developmental atlas reveals the state-transition dynamics of late-vertebrate pluripotent axial progenitors},
  author={Lange, Merlin and Granados, Alejandro and VijayKumar, Shruthi and Bragantini, Jord{\~a}o and Ancheta, Sarah and Kim, Yang-Joon and Santhosh, Sreejith and Borja, Michael and Kobayashi, Hirofumi and McGeever, Erin and others},
  journal={Cell},
  volume={187},
  number={23},
  pages={6742--6759},
  year={2024},
  publisher={Elsevier}
}
```

**This Work:**
```bibtex
@article{daniodecima2025,
  title={DanioDecima: A DNA sequence-to-function model of zebrafish embryogenesis},
  author={Voges, Mathias, et al.},
  journal={In preparation},
  year={2025}
}
```

## Pre-Submission Checklist

Before making this repository public or submitting to a conference, address the following:

### Security
- [ ] **Rotate WandB API keys**: Previously exposed keys in `scripts/finetune.py` and `scripts/finetune_temporal.py` have been removed from source but remain in git history. Rotate the affected key (`66d3a7...`) on the WandB dashboard (Settings > API keys) and scrub history with `git filter-repo` before making the repo public.

### Reproducibility
- [ ] **Remove hardcoded paths**: ~150+ hardcoded absolute paths across the codebase reference user-specific HPC directories (`/hpc/mydata/mathias.voges/`, `/home/karollua/`, `/home/gunsalul/`, `/hpc/scratch/.../yangjoon.kim/`, `/gstore/data/resbioai/`, `/code/decima/`). Replace with environment variables, config files, or CLI arguments. Priority files:
  - Core library: `src/decima/decima_model.py:109`, `lightning.py:26`, `lightning_temporal.py:23`
  - Scripts: `finetune.py:15`, `finetune_temporal.py:15`, `decima_finetune.py:44,167`
  - Shell: `submit_decima.sh:25-26`, `submit_decima_finetune.sh:4-7`
  - Notebooks: all directories (0_sc_data_prep through 9_design)
- [ ] **Fix sys.path hacks**: `lightning.py`, `lightning_temporal.py`, `interpret.py` manipulate `sys.path` instead of using proper relative imports
- [ ] **Populate `install_requires`** in `daniodecima-main/setup.cfg` (currently empty; 15+ undeclared dependencies)
- [ ] **Standardize seed handling** across scripts (currently inconsistent: some hardcode 0, some are configurable, some omit seeds entirely)

### Correctness
- [ ] **Loss function epsilon**: `loss.py` defines `self.eps` but never uses it in `forward()` — division by zero and `log(0)` are unprotected (lines 34-35)
- [ ] **Unreachable code**: `lightning.py:get_task_idxs` — the `invert` branch is after all return statements
- [ ] **Early stopping overlap**: `00_evolve_combined.py:262-272` — comparison windows overlap

### Code Quality
- [ ] **Add test suite** (currently zero tests; `tests/conftest.py` is a stub)
- [ ] **Remove dead code**: `02_summarize_results_designs_test.py` (~1200 commented lines), `lightning.py:281-341` (commented temporal smoothness)
- [ ] **Fix bare `except` clauses**: `preprocess.py:228`, `setup.py:15`
- [ ] **Update package metadata**: `setup.cfg` description is placeholder text; URL points to PyScaffold template

### Documentation
- [ ] **Update citation**: author list is "Voges, Mathias, et al." — needs full author list, venue, and year
- [ ] **Fix path references** in `documents/DanioDecimaPipeline_README.md:209-237` (13 paths point to wrong user directory)
- [ ] **Fix missing environment file**: `README_analysis_TF-MoDISco.md` references `environment_pytorch_full.yml` which does not exist

## License

BSD-3-Clause license