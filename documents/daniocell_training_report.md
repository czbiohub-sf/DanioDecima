# DanioCell Decima Fine-Tuning — Training Report

**Date:** 2026-04-06 to 2026-04-07
**SLURM Job:** 30071550 (array 0–7)
**Branch:** `daniocell-pipeline`
**GPU:** NVIDIA H100 80GB HBM3 (single GPU per experiment)

## Overview

Fine-tuned the Decima model (173M parameters, Borzoi backbone) on the **DanioCell atlas**
(Sur & Farrell zebrafish single-cell atlas) to predict pseudobulk gene expression from
524,288bp DNA sequences. The DanioCell atlas contains **1,047 pseudobulk tracks**
(cell type × stage group) across **28,201 genes** on 25 zebrafish chromosomes.

Two initialization strategies were compared:
- **Pretrained** (exp 0–3): Human-Decima backbone (4 replicates), head re-initialized for 1,047 tasks, lr=3e-5
- **Random** (exp 4–7): Full random initialization (4 seeds), lr=3e-6

Learning rates follow the optimal values from Mathias Voges's Zebrahub sweep:
3e-5 for pretrained, 3e-6 for random init.

## Results

### Final Metrics (early-stopped)

| Exp | Init | Replicate/Seed | val_loss | val_pearson | val_mse | Best val_loss | Best Epoch | Total Epochs |
|-----|------|----------------|----------|-------------|---------|---------------|------------|--------------|
| 0 | pretrained | rep0 | 7.914 | **0.763** | 1.092 | 7.899 | 8 | 18 |
| 1 | pretrained | rep1 | 7.911 | **0.755** | 1.111 | 7.898 | 2 | 12 |
| 2 | pretrained | rep2 | 7.906 | **0.758** | 1.117 | 7.901 | 3 | 13 |
| 3 | pretrained | rep3 | 7.911 | **0.758** | 1.097 | 7.902 | 7 | 17 |
| 4 | random | seed42 | 8.004 | 0.505 | 2.080 | 7.968 | 9 | 19 |
| 5 | random | seed43 | 8.117 | 0.357 | 11.433 | 7.967 | 16 | 26 |
| 6 | random | seed44 | 8.115 | 0.434 | 11.148 | 7.966 | 11 | 21 |
| 7 | random | seed45 | 8.065 | 0.490 | 2.252 | 7.968 | 20 | 30 |

### Summary Statistics

| Metric | Pretrained (mean +/- std) | Random (mean +/- std) | Difference |
|--------|--------------------------|----------------------|------------|
| val_loss | 7.910 +/- 0.003 | 8.075 +/- 0.053 | -0.165 |
| val_pearson | **0.758 +/- 0.003** | 0.447 +/- 0.067 | **+0.311** |
| val_mse | 1.104 +/- 0.011 | 6.728 +/- 4.973 | -5.624 |
| Best val_loss | 7.900 +/- 0.002 | 7.967 +/- 0.001 | -0.067 |
| Epochs to best | 5.0 +/- 2.7 | 14.0 +/- 4.8 | -9.0 |

### Key Findings

1. **Cross-species transfer works well.** Pretrained human-Decima backbone achieves Pearson ~0.76 on zebrafish DanioCell, a +0.31 improvement over random initialization. This confirms meaningful transfer of regulatory grammar from human to zebrafish.

2. **Pretrained models are highly consistent.** All 4 replicates converge to nearly identical performance (val_pearson std = 0.003), indicating the result is robust and not dependent on a specific weight initialization.

3. **Random init is noisy and slower.** Random models show 15x higher variance in val_pearson (std = 0.067) and take ~3x longer to reach their best epoch. Seeds 43 and 44 also show unstable MSE (>11), suggesting occasional training instabilities.

4. **Pretrained models converge faster.** Best validation loss is reached at epoch 2–8 for pretrained vs epoch 9–20 for random, meaning the pretrained backbone provides a strong initialization that needs minimal further tuning.

5. **Best val_loss is similar across groups** (~7.90 vs ~7.97), but Pearson diverges significantly. This suggests the pretrained model learns better gene-level ranking (correlation) while the raw loss (Poisson + multinomial) is somewhat saturated.

## Training Configuration

```
Model:            Decima (BorzoiModel backbone + ConvHead)
Parameters:       173,290,935
Input:            524,288bp DNA + gene mask (5 channels)
Output:           1,047 pseudobulk tracks per gene
Loss:             TaskWisePoissonMultinomialLoss (total_weight=1e-4)
Optimizer:        Adam (weight_decay=0)
Batch size:       4 (effective 20 with grad_accum=5)
Precision:        16-mixed (AMP on H100)
Early stopping:   patience=10, min_delta=0.0001, monitor=val_loss
Gradient clip:    norm=1
Max epochs:       40
```

## Data Split

| Split | Genes | Fraction | Chromosomes |
|-------|-------|----------|-------------|
| Train | 16,122 | 57.2% | chr2,6,7,10,11,12,13,15,16,17,18,19,20,22,23 |
| Val | 5,674 | 20.1% | chr3,5,14,24,25 |
| Test | 6,405 | 22.7% | chr1,4,8,9,21 |

Split follows the Zebrahub `chrom_split_v1` scheme for cross-dataset comparability.

## SLURM Resource Usage

| Resource | Requested | Actual | Recommendation |
|----------|-----------|--------|----------------|
| Memory | 100 GB | 5.9–7.8 GB | **20 GB** |
| CPUs | 8 | ~15% efficiency | **4** |
| Walltime | 60 hours | 10–23 hours | **24 hours** |
| GPU | 1x H100 | ~100% utilization | 1x H100 (unchanged) |

Training speed: ~1.5–1.7 it/s, ~45 min/epoch on H100.

## File Locations

```
Data:         /hpc/scratch/.../daniodecima-daniocell/celltypes_chrom_split_v1/
Results:      /hpc/scratch/.../daniodecima-daniocell/experiments/daniocell_experiments_20260406_114453/
SLURM logs:   /hpc/scratch/.../daniodecima-daniocell/experiments/logs/daniocell_exp_30071550_*.out
Checkpoints:  <results_dir>/<exp_name>/{last.ckpt, epoch=N-step=M.ckpt}
CSV metrics:  <results_dir>/<exp_name>/version_0/metrics.csv
```

## Recommendations for Next Steps

1. **Evaluate on test set.** Run test-set prediction with the best pretrained checkpoint (rep2, val_loss=7.901) to get unbiased performance estimates and per-gene Pearson distributions.

2. **Cell-type specificity analysis.** Compute per-cell-type Pearson correlations to identify which of the 1,047 tracks are well-predicted vs poorly-predicted. This will inform which cell types the model captures well.

3. **Marker gene analysis.** Evaluate prediction accuracy on known zebrafish marker genes to validate biological relevance.

4. **Attribution / motif analysis.** Run InputXGradient or ISM on the best model to identify regulatory motifs driving cell-type-specific predictions.

5. **3-stage transfer (optional).** Once Zebrahub fine-tuned checkpoints are available, run experiments 8–11 (Zebrahub→DanioCell transfer) to test whether zebrafish-specific pretraining further improves over human-Decima.

6. **Upgrade logging.** Switch from CSVLogger to WandB for interactive experiment comparison dashboards (credentials already configured; see memory note).
