# Model Checkpoints for Directed Evolution

This document explains the model checkpoints used in the directed evolution pipeline (`9_design/`) and clarifies the relationship between pretrained weights and fine-tuned checkpoints.

## Overview: The Checkpoint Chain

The models used for directed evolution follow this chain:

```
Pretrained Weights → Fine-tuning on Zebrafish Data → Evolution Checkpoints
     (Source)              (Training Step)                (What's Used)
```

**Key Point**: All models used in the evolution pipeline are **fine-tuned on zebrafish data**, but they start from different initialization sources. The checkpoint directories indicate the initialization source, not that they're still pretrained models.

## Model Types and Checkpoints

The evolution pipeline uses **16 models across 4 types** (4 replicates each):

### 1. Human-Decima (4 replicates)

**Checkpoint Directories**:
```
/hpc/scratch/.../experiments/decima_experiments_20250618_111138/
├── pretrained_decima-human_rep0_lr3e-05_seed42/version_0/
├── pretrained_decima-human_rep1_lr3e-05_seed42/version_0/
├── pretrained_decima-human_rep2_lr3e-05_seed42/version_0/
└── pretrained_decima-human_rep3_lr3e-05_seed42/version_0/
```

**Weight Source Chain**:
1. **Pretrained initialization**: Decima checkpoints from `/hpc/scratch/.../decima_checkpoints_lal/rep{0-3}.ckpt`
   - These are Decima models with 5-channel input (4 DNA + 1 gene mask)
   - Likely pre-trained on zebrafish data in an earlier experiment
   - Loaded via `decima_model.py:109` when `pretrained_source="decima-human"`

2. **Fine-tuning**: The pretrained weights are fine-tuned on the current zebrafish dataset
   - Training run dated: 2025-06-18
   - Learning rate: 3e-05
   - Results saved in experiment directories above

3. **Evolution**: The fine-tuned checkpoints are loaded for directed evolution experiments

**Architecture**:
- Borzoi architecture with 5 input channels (4 DNA + 1 mask)
- Already adapted for Decima-style gene masking

### 2. Human-Borzoi (4 replicates)

**Checkpoint Directories**:
```
/hpc/scratch/.../experiments/decima_experiments_20250617_225114/
├── pretrained_wandb-human_rep0_lr3e-05_seed42/version_0/
├── pretrained_wandb-human_rep1_lr3e-05_seed42/version_0/
├── pretrained_wandb-human_rep2_lr3e-05_seed42/version_0/
└── pretrained_wandb-human_rep3_lr3e-05_seed42/version_0/
```

**Weight Source Chain**:
1. **Pretrained initialization**: WandB artifact `grelu/borzoi/human_state_dict_fold{0-3}:latest`
   - Borzoi model pretrained on human ENCODE/Roadmap data
   - Downloaded from WandB via API (`decima_model.py:62`)
   - 4-channel input (DNA only, no mask)

2. **Fine-tuning**: The pretrained human Borzoi weights are fine-tuned on zebrafish data
   - Gene mask channel added during initialization (`decima_model.py:236-243`)
   - Training run dated: 2025-06-17
   - Learning rate: 3e-05

3. **Evolution**: The fine-tuned checkpoints are loaded for directed evolution experiments

**Architecture**:
- Borzoi architecture adapted from 4 channels to 5 channels (mask added)
- Transfer learning from human regulatory data to zebrafish

### 3. Mouse-Borzoi (4 replicates)

**Checkpoint Directories**:
```
/hpc/scratch/.../experiments/decima_experiments_20250617_225114/
├── pretrained_wandb-mouse_rep0_lr3e-05_seed42/version_0/
├── pretrained_wandb-mouse_rep1_lr3e-05_seed42/version_0/
├── pretrained_wandb-mouse_rep2_lr3e-05_seed42/version_0/
└── pretrained_wandb-mouse_rep3_lr3e-05_seed42/version_0/
```

**Weight Source Chain**:
1. **Pretrained initialization**: WandB artifact `grelu/borzoi/mouse_state_dict_fold{0-3}:latest`
   - Borzoi model pretrained on mouse ENCODE data
   - Downloaded from WandB via API (`decima_model.py:162`)
   - 4-channel input (DNA only, no mask)

2. **Fine-tuning**: The pretrained mouse Borzoi weights are fine-tuned on zebrafish data
   - Gene mask channel added during initialization
   - Training run dated: 2025-06-17 (same batch as Human-Borzoi)
   - Learning rate: 3e-05

3. **Evolution**: The fine-tuned checkpoints are loaded for directed evolution experiments

**Architecture**:
- Borzoi architecture adapted from 4 channels to 5 channels
- Cross-species transfer learning from mouse to zebrafish

### 4. Random (4 replicates)

**Checkpoint Directories**:
```
/hpc/scratch/.../experiments/decima_experiments_20250617_225114/
├── random_lr3e-06_seed42/version_0/
├── random_lr3e-06_seed43/version_0/
├── random_lr3e-06_seed44/version_0/
└── random_lr3e-06_seed45/version_0/
```

**Weight Source Chain**:
1. **Random initialization**: No pretrained weights
   - Borzoi architecture with random PyTorch default initialization (`decima_model.py:190-227`)
   - Different random seeds (42, 43, 44, 45) for each replicate
   - 5-channel input from the start

2. **Fine-tuning**: Randomly initialized weights are trained on zebrafish data
   - Training run dated: 2025-06-17
   - **Lower learning rate**: 3e-06 (10x lower than pretrained models)
   - Serves as baseline without transfer learning

3. **Evolution**: The trained checkpoints are loaded for directed evolution experiments

**Architecture**:
- Same Borzoi architecture as other models
- Control condition to assess benefit of transfer learning

## Training Configuration

All fine-tuning runs used:
- **Optimizer**: Adam
- **Loss**: TaskWisePoissonMultinomialLoss (Poisson + Multinomial terms)
- **Batch size**: 4
- **Sequence augmentation**: Random shifts up to 5000bp during training
- **Model architecture**: Borzoi (7 CNN blocks + 8 Transformer blocks)

Note: Random initialization models used 10x lower learning rate (3e-06 vs 3e-05) to compensate for lack of pretrained weights.

## Evolution Pipeline Usage

The SLURM script `00_submit_evolve_combined.sh` handles:
1. **Model selection**: Maps array job index to one of 16 model directories
2. **Model loading**: Loads fine-tuned checkpoint via `00_evolve_combined.py`
3. **Evolution**: Runs directed evolution for specified cell type
4. **Analysis**: Performs ISM and motif analysis on evolved sequences

The evolution script loads checkpoints using:
```python
checkpoint_path = find_checkpoint_file(args.model_dir)
model, data_params = load_model(checkpoint_path, device)
```

## How to Identify Checkpoint Sources

Looking at a checkpoint directory path like:
```
/experiments/decima_experiments_20250618_111138/pretrained_decima-human_rep2_lr3e-05_seed42/
```

You can identify:
- **Experiment batch**: `20250618_111138` (date and time)
- **Initialization source**: `pretrained_decima-human` (started from Decima pretrained weights)
- **Replicate**: `rep2` (replicate #2)
- **Learning rate**: `lr3e-05` (3×10⁻⁵)
- **Random seed**: `seed42`

The actual checkpoint file will be in:
```
version_0/checkpoints/epoch=N-step=M.ckpt
```

## Attribution Analysis Checkpoints

Attribution analysis (in `5_specificity/`) uses a different set of checkpoints from an earlier training run:

```python
ckpts = [
    '/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/kugrjb50/checkpoints/epoch=3-step=2920.ckpt',
    '/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/i68hdsdk/checkpoints/epoch=2-step=2190.ckpt',
    '/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/0as9e8of/checkpoints/epoch=7-step=5840.ckpt',
    '/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/i9zsp4nm/checkpoints/epoch=8-step=6570.ckpt',
]
```

These are 4 training replicates from the **20240823** experiment, representing an ensemble of models for robust attribution calculation.

## Initial Zebrafish Tissue-Level Checkpoints (Historical)

Before the cell-type-specific experiments documented above, the initial directed evolution and ISM work used **tissue-level** zebrafish models. These checkpoints predate the 16-model systematic experiments.

### Tissue-Level Model for ISM/Design

**Run ID**: `5ly2p4gy`

**Checkpoint Path**:
```
/hpc/mydata/mathias.voges/Projects/research/zf-decima/outputs/grelu/decima/lightning_logs/5ly2p4gy/checkpoints/epoch=9-step=9190.ckpt
```

**Key Characteristics**:
- **Training date**: March 2025
- **# Tracks**: 94 pseudobulks (tissue-level granularity)
- **Data**: `data_out_zf-Decima_Random_Rep0.h5ad`
- **Initialization**: Random
- **Referenced in**: `decima-main/tutorials/tutorial.py`
- **Purpose**: Initial ISM analysis and directed evolution experiments

### Time-Course Forecasting Model

**Run ID**: `umv5p24k`

**Checkpoint Directory**:
```
/hpc/mydata/mathias.voges/Projects/research/zf-decima/outputs/grelu/decima/lightning_logs/umv5p24k/checkpoints/
```

**Key Characteristics**:
- **Training date**: March 19, 2025
- **# Tracks**: 7 (temporal series)
- **Data**: `zebrahub_aggregated.h5ad` with `GeneForecastDataset`
- **Purpose**: Time-course gene expression forecasting (NOT for directed evolution)
- **Referenced in**: `4_evaluation/00_predict.py`

### Timeline of Zebrafish Model Evolution

| Date | Checkpoint | Granularity | # Tracks | Purpose |
|------|------------|-------------|----------|---------|
| 2024-08-23 | `/gstore/data/.../20240823/` | Human/Mouse Decima | 8,856 | Original Decima models |
| 2025-03-19 | `5ly2p4gy` | **Tissue-level** | 94 | Initial zebrafish ISM & evolution |
| 2025-03-19 | `umv5p24k` | Temporal | 7 | Time-course forecasting |
| 2025-06-17 | `decima_experiments_20250617_*` | Cell-type level | ~25 | Systematic 16-model experiments |
| 2025-06-18 | `decima_experiments_20250618_*` | Cell-type level | ~25 | Human-Decima replicates |

The `5ly2p4gy` checkpoint represents the transitional model between the original human/mouse Decima and the comprehensive cell-type-specific zebrafish experiments, and was used for the initial tissue-level directed evolution work.

## Summary Table

| Model Type | # Replicates | Pretrained Source | Fine-tuned on Zebrafish | Used in Evolution |
|------------|--------------|-------------------|-------------------------|-------------------|
| Human-Decima | 4 | `/decima_checkpoints_lal/` | ✓ | ✓ |
| Human-Borzoi | 4 | WandB (human ENCODE) | ✓ | ✓ |
| Mouse-Borzoi | 4 | WandB (mouse ENCODE) | ✓ | ✓ |
| Random | 4 | None (random init) | ✓ | ✓ |

**Total**: 16 models, all fine-tuned on zebrafish single-cell data, differing only in their initialization strategy.

## References

- Model initialization: `decima-main/src/decima/decima_model.py`
- Fine-tuning script: `decima-main/scripts/finetune.py`
- Evolution pipeline: `9_design/00_evolve_combined.py`
- SLURM launcher: `9_design/00_submit_evolve_combined.sh`
