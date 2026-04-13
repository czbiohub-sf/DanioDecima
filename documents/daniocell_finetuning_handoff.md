# DanioCell Decima Fine-Tuning — Handoff Document

## Executive Summary (for agents and humans)

**What:** Fine-tuned the Decima sequence-to-expression model on the DanioCell single-cell atlas (zebrafish embryogenesis). 8 models trained: 4 pretrained (Human-Decima init, lr=3e-5) and 4 random init (lr=3e-6).

**Result:** Pretrained models achieve gene-level Pearson r = 0.376 +/- 0.004 (test set), matching the Zebrahub-trained DanioDecima (0.38). Lift over random = +0.15 (gene), +0.23 (track). All DanioDecima evaluation findings replicate: conservation independence, attribution in regulatory regions, CRE enrichment, lineage-specific motifs.

**Best checkpoint for downstream use:**
```
/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/experiments/
  daniocell_experiments_20260406_114453/
    pretrained_decima-human_rep0_lr3e-05_seed42/version_0/checkpoints/last.ckpt
```

**Status:** Training, prediction, and evaluation complete. Directed evolution / in silico mutagenesis not yet started.

**Code branch:** `daniocell-pipeline` (PR #5 → main) in `czbiohub-sf/step`

**Master README (with all paths):**
```
/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/README.md
```

---

## 1. Background and Rationale

### Why this project exists

DanioDecima (Mathias Voges) demonstrated that cross-species transfer learning — initializing a Decima model with Borzoi/Decima weights trained on human and mouse, then fine-tuning on zebrafish single-cell RNA-seq — dramatically improves gene expression prediction over random initialization. That work used the **Zebrahub** atlas (304 pseudobulks, 154 cell types, 10 timepoints).

We wanted to:
1. **Validate the framework** — confirm the same findings with a different, independently-generated atlas
2. **Use richer training data** — DanioCell has 3.4x more pseudobulks and 2.5x better cell-type x stage coverage
3. **Produce a model for downstream use** — in silico mutagenesis, directed evolution of regulatory elements, etc.

### Why DanioCell over Zebrahub

| Property | Zebrahub | DanioCell | Advantage |
|----------|----------|-----------|-----------|
| Pseudobulk tracks | 304 | 1,047 | 3.4x more training signal |
| Cell types | 154 | 153 | Comparable count |
| Developmental stages | 10 (10hpf-10dpf) | 14 (3-120 hpf) | Finer temporal resolution |
| Cell-type x stage coverage | 19.7% | 48.9% | 2.5x denser matrix |
| Cell-type annotation quality | ZAO ontology | identity.super | More harmonized across stages |
| Total genes | 31,767 | 28,201 | Comparable |

DanioCell's denser coverage means the model sees more developmental diversity during training — the same cell type is tracked across more stages, providing better learning signal for temporal regulatory dynamics.

### What we learned from DanioDecima (informing our approach)

- **Best init strategy:** Human-Decima >> Human-Borzoi > Mouse-Borzoi >> Random. We therefore only ran Human-Decima and Random.
- **Optimal learning rates:** 3e-5 for pretrained, 3e-6 for random. Used directly.
- **Evaluation framework:** 6-phase pipeline (predict → evaluate → attribute → specificity → modisco → figures). Replicated exactly.
- **Key controls:** Conservation analysis (is the model memorizing conserved sequences?), CRE attribution (does it attend to accessible chromatin?), motif clustering (do designs make biological sense?).

---

## 2. Input Data

### Training dataset

```
/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/celltypes_chrom_split_v1/
├── daniocell_aggregated.h5ad    (907 MB) — pseudobulk expression matrix
└── data.h5                      (71 GB)  — HDF5 sequences + targets for DataLoader
```

- **Genome assembly:** danRer11 (GRCz11)
- **Sequence length:** 524,288 bp per gene (5-channel: ACGT + gene mask)
- **Tracks:** 1,047 pseudobulks (identity.super x stage.group)
- **Genes:** 28,201 total (16,122 train / 5,674 val / 6,405 test)
- **Gene split:** Leave-one-chromosome-out (LOCO), inherited from Zebrahub dataset prep

### Pretrained weights (initialization source)

```
/hpc/scratch/group.data.science/yang-joon.kim/zebrahub-decima/experiments/
  pretrained_decima-human_rep{0-3}_lr3e-05_seed42/best_model.ckpt
```

These are DanioDecima (Zebrahub) checkpoints, themselves initialized from the published Human-Decima weights. The chain is: **Human genomics → Decima → DanioDecima (Zebrahub) → DanioCell fine-tune**.

### External reference files used in evaluation

```
# Ortholog conservation mapping (for Fig 6)
/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/
  decima-applications-main/notebooks/5_specificity/zebrafish_human_orthologs.csv

# Columns: gene, human_ortholog_id, source_perc_id, source_perc_pos
```

---

## 3. Training Configuration

```
Architecture:       Decima (7 CNN blocks + 8 Transformer blocks, 1,920 embedding channels)
Input:              524,288 bp DNA + gene mask (5 channels)
Output:             1,047 pseudobulk expression tracks
Loss:               TaskWisePoissonMultinomialLoss (Poisson + multinomial terms)
Optimizer:          Adam (no weight decay — tested, did not help)
LR (pretrained):    3e-05
LR (random):        3e-06
Batch size:         4 (accumulate_grad_batches=5 → effective batch size 20)
Early stopping:     patience=10, min_delta=0.0001 (monitoring val_loss)
Max epochs:         40
Augmentation:       max_seq_shift=3 (no reverse complement)
Seed (pretrained):  42 (all 4 reps use same seed, different init weights)
Seed (random):      42, 43, 44, 45
Logger:             CSVLogger (not TensorBoard)
GPU:                1x H100
Training time:      ~10-20 hours per model
```

### Training script
```
/hpc/projects/data.science/yangjoon.kim/step/decima-main/scripts/daniocell_finetune.py
/hpc/projects/data.science/yangjoon.kim/step/decima-main/scripts/submit_daniocell_finetune.sh
```

---

## 4. Output: Checkpoints and Experiments

### Experiment directory

```
/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/experiments/
  daniocell_experiments_20260406_114453/
```

### All 8 models

| Experiment | Group | Best val_pearson | Best val_loss | Saved ckpt | Test gene r | Test track r |
|------------|-------|-----------------|---------------|------------|-------------|-------------|
| `pretrained_decima-human_rep0_lr3e-05_seed42` | pretrained | **0.7633** (ep15) | 7.8994 (ep8) | `epoch=8-step=7263.ckpt` | **0.381** | 0.761 |
| `pretrained_decima-human_rep1_lr3e-05_seed42` | pretrained | 0.7552 (ep12) | **7.8975** (ep2) | `epoch=2-step=2421.ckpt` | 0.371 | 0.755 |
| `pretrained_decima-human_rep2_lr3e-05_seed42` | pretrained | 0.7582 (ep12) | 7.9010 (ep3) | `epoch=3-step=3228.ckpt` | 0.377 | **0.771** |
| `pretrained_decima-human_rep3_lr3e-05_seed42` | pretrained | 0.7614 (ep16) | 7.9018 (ep7) | `epoch=7-step=6456.ckpt` | 0.375 | 0.762 |
| `random_lr3e-06_seed42` | random | 0.5138 (ep4) | 7.9683 (ep9) | `epoch=9-step=8070.ckpt` | 0.228 | 0.496 |
| `random_lr3e-06_seed43` | random | 0.5370 (ep19) | 7.9674 (ep16) | `epoch=16-step=13719.ckpt` | 0.228 | 0.539 |
| `random_lr3e-06_seed44` | random | 0.5335 (ep11) | 7.9662 (ep11) | `epoch=11-step=9684.ckpt` | 0.227 | 0.543 |
| `random_lr3e-06_seed45` | random | 0.5411 (ep20) | 7.9681 (ep20) | `epoch=20-step=16947.ckpt` | 0.234 | 0.541 |

### Best checkpoint (recommended for all downstream use)

```
/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/experiments/
  daniocell_experiments_20260406_114453/
    pretrained_decima-human_rep0_lr3e-05_seed42/version_0/checkpoints/
      last.ckpt                    ← 1.94 GB, used for predictions
      epoch=8-step=7263.ckpt       ← 1.94 GB, best val_loss checkpoint
```

**Why rep0:** Highest test gene Pearson (0.381) and highest val_pearson (0.7633). Rep2 has marginally higher test track Pearson (0.771 vs 0.761) but lower gene Pearson.

### Each experiment directory contains

```
<experiment_name>/version_0/
├── checkpoints/
│   ├── epoch=N-step=M.ckpt    ← best val_loss checkpoint
│   └── last.ckpt               ← last training epoch
└── metrics.csv                  ← CSVLogger output (epoch, step, losses, val_pearson)
```

---

## 5. Output: Evaluation Pipeline

All outputs live under:
```
/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/
```

### Phase 1: Predictions (`01_predictions/`)

8 h5ad files (~1 GB each):
```
01_predictions/
├── data_out_daniocell_pretrained_decima-human_rep{0-3}_lr3e-05_seed42.h5ad
├── data_out_daniocell_random_lr3e-06_seed{42-45}.h5ad
└── README_phase1_results.md
```

**h5ad structure:**
- `ad.X` — observed expression (1,047 tracks x 28,201 genes)
- `ad.layers['preds']` — predicted expression
- `ad.var['pearson']` — per-gene Pearson r
- `ad.var['dataset']` — train/val/test split label
- `ad.obs['test_pearson']` — per-track Pearson r (test genes only)
- `ad.obs['identity.super']` — cell type
- `ad.obs['stage.group']` — developmental stage

**Script:** `decima-main/scripts/daniocell-ft/daniocell_predict.py`

### Phase 2: Evaluation (`02_evaluation/`)

```
02_evaluation/
├── figures/                        ← 11 basic evaluation plots
├── manuscript_figures/             ← 9 publication-quality figures (Figs 1,3-10)
│   ├── fig1_training_curves.{pdf,png}
│   ├── fig3_model_comparison.{pdf,png}
│   ├── fig4_timepoint_enrichment.{pdf,png}
│   ├── fig5_developmental_patterns.{pdf,png}
│   ├── fig6_conservation_analysis.{pdf,png}
│   ├── fig7_attribution_by_region.{pdf,png}
│   ├── fig8_atac_fold_change.{pdf,png}
│   ├── fig9_motif_clustering.{pdf,png}
│   └── fig10_celltype_dendrogram.{pdf,png}
├── marker_metrics_per_celltype.csv  ← AUROC/AUPRC per cell type (153 rows)
└── model_summary.csv                ← mean Pearson per experiment (8 rows)
```

**Scripts:**
- `notebooks/daniocell-ft/02_evaluation/daniocell_evaluate.py` (basic evaluation)
- `notebooks/daniocell-ft/02_evaluation/daniocell_manuscript_figures.py` (pub figures)

### Phase 3: Attributions (`03_attributions/`)

```
03_attributions/
├── pretrained_rep{0-3}-attr-th0.5-all.h5   ← full attribution H5 (4 files)
├── pretrained_rep{0-3}_regional_attributions.pkl   ← regional breakdown (4 files)
├── random_seed42-attr-th0.5-all.h5
└── random_seed42_regional_attributions.pkl
```

**pkl structure:** `dict` with key `'genes'` → DataFrame (6,405 test genes x 47 columns):
- Gene metadata: `gene`, `chrom`, `start`, `end`, `strand`, `pearson`, `dataset`
- Base regions: `Promoter`, `Exons`, `Introns`, `Exon/Intron junctions`
- CRE breakdown: `Promoter CREs`, `Promoter non-CREs`, `Intronic CREs`, `Intronic non-CREs`, etc.
- Distance bins: `1k (CREs)`, `1k (non-CREs)`, `1k (all)`, `1k-10k (...)`, `10k-100k (...)`, `>=100k (...)`

**Script:** `notebooks/daniocell-ft/03_attributions/daniocell_attribution.py`

### Phase 4: Specificity (`04_specificity/`)

10 focal cell types, each using `pretrained_rep0`:
```
04_specificity/
└── pretrained_decima-human_rep0_lr3e-05_seed42_{celltype}/
    ├── sequences.npy       ← input DNA sequences
    ├── attributions.npy    ← per-nucleotide attribution scores
    ├── gene_stats.csv      ← per-gene metadata
    └── metadata.json       ← cell type info, model config
```

**Focal cell types:** cardiac_muscle, epidermis, intestine, liver, motor_neurons, neural_crest, neurons, notochord, radial_glia, somite

**Script:** `notebooks/daniocell-ft/04_specificity/daniocell_specificity.py`

### Phase 5: MoDISco (`05_modisco/`)

```
05_modisco/
└── {cardiac_muscle,epidermis,...,somite}/
    ├── modisco_report.h5    ← MoDISco patterns (pos_patterns + neg_patterns)
    └── trimmed_logos/       ← motif logo images
```

**modisco_report.h5 structure:**
- `pos_patterns/pattern_N/contrib_scores` — CWM matrix (50 x 4)
- `pos_patterns/pattern_N/seqlets/n_seqlets` — support count
- ~20-28 positive patterns per cell type, ~20 negative patterns

**Script:** `notebooks/daniocell-ft/05_modisco/daniocell_modisco.py`

---

## 6. Key Results

### Performance summary

| Metric | Pretrained (n=4) | Random (n=4) | Lift |
|--------|-----------------|--------------|------|
| Gene Pearson (test, mean +/- std) | 0.376 +/- 0.004 | 0.229 +/- 0.003 | +0.147 |
| Gene Pearson (test, median) | 0.375 | 0.192 | +0.183 |
| Track Pearson (test, mean +/- std) | 0.762 +/- 0.007 | 0.530 +/- 0.022 | +0.232 |
| MWU p-value (pretrained > random) | — | — | 0.014 |
| % test genes above diagonal | — | — | 80% |
| Marker gene AUROC (best model) | 0.760 mean | — | — |
| Cell types with AUROC > 0.8 | 45 / 153 (29%) | — | — |

### Comparison with DanioDecima (Zebrahub)

| Evaluation | DanioDecima | DanioCell | Status |
|------------|-------------|-----------|--------|
| Gene Pearson (pretrained) | 0.38 | 0.38 | Matched |
| Track Pearson lift | +0.19 | +0.23 | DanioCell larger |
| Conservation independence | rho ~ 0 | rho = -0.07 | Replicated |
| Attribution: promoter >> distal | Yes | Yes | Replicated |
| CRE enrichment (ATAC) | log2 FC > 0 | log2 FC > 0 | Replicated |
| Motif clustering by lineage | 25 cell types | 10 focal types | Replicated |
| Directed evolution designs | Completed | Not yet started | Next step |

### Key biological findings

1. **Not memorizing conserved sequences:** Pretrained lift is independent of zebrafish-human ortholog conservation (Spearman rho = -0.07, p = 6e-5). The model learned regulatory grammar, not conserved promoter sequences.
2. **Pretrained advantage scales with task difficulty:** Lift is largest at late developmental stages (120 hpf) and for specialized cell types, where the prediction task is hardest.
3. **Regulatory region awareness:** Pretrained models attribute significantly more importance to promoters, CREs, and distal regulatory elements (all Mann-Whitney p < 0.001).
4. **Lineage-specific motifs:** MoDISco recovers biologically meaningful clusters: neurons-notochord (neural lineage), epidermis-radial glia, cardiac muscle-somite (mesodermal).

---

## 7. What Was NOT Done (and Next Steps)

1. **Directed evolution / in silico mutagenesis** — The gReLU design loop has not been run on the DanioCell model yet. This is the next major step: generate synthetic regulatory elements for all 153 cell types.
2. **Mouse-Borzoi and Human-Borzoi init** — We only ran Human-Decima and Random (based on DanioDecima finding that Human-Decima is best). If comparing init strategies is needed, add these.
3. **Weight decay sweep** — Not tested for DanioCell (DanioDecima found it didn't help).
4. **MPRA / reporter assay validation** — No wet-lab validation of designed sequences yet.
5. **Model upload to HuggingFace/Zenodo** — Weights are only on scratch storage.

---

## 8. Lessons Learned / Gotchas

1. **CSVLogger, not TensorBoard:** Our training used CSVLogger (not TensorBoard like Zebrahub). Training metrics are in `metrics.csv`, not tfevents files. Scripts that read training curves must parse CSV.

2. **Chromosome naming mismatch:** DanioCell uses `chr1, chr2, ...` while Zebrahub uses `1, 2, ...`. The specificity pipeline (Phase 4) normalizes this with `chrom.str.replace('chr', '')` before cross-dataset operations.

3. **SLURM scripts must use absolute paths:** SLURM copies scripts to a spool directory, breaking `dirname $0` relative paths. All submit scripts use hardcoded absolute paths.

4. **Conda env must be `gReLu`, not `pytorch`:** The `pytorch` env has NumPy 2.4 / numba incompatibility. Always use: `module load anaconda && source activate gReLu`.

5. **Pickle compatibility:** Regional attribution `.pkl` files were saved with the gReLu env's pandas version. They cannot be loaded with the system Python 3.6 — must use the conda env.

6. **evaluate.py bug:** `auprc, auroc = np.nan` (line 60) crashes with ValueError. Fixed to `np.nan, np.nan`. This is committed in the PR.

---

## 9. Code Locations (in repo)

**Repository:** `czbiohub-sf/step` (branch: `daniocell-pipeline`, PR #5)

```
decima-main/
├── src/decima/evaluate.py                          ← bug fix (tuple unpack)
├── scripts/
│   ├── daniocell_finetune.py                       ← Phase 0: training
│   ├── submit_daniocell_finetune.sh
│   └── daniocell-ft/
│       ├── daniocell_predict.py                    ← Phase 1: predictions
│       └── submit_daniocell_predict.sh

decima-applications-main/notebooks/daniocell-ft/
├── 02_evaluation/
│   ├── daniocell_evaluate.py                       ← Phase 2: basic evaluation
│   ├── daniocell_manuscript_figures.py             ← Phase 6: manuscript figures
│   ├── submit_daniocell_evaluate.sh
│   └── submit_manuscript_figures.sh
├── 03_attributions/
│   ├── daniocell_attribution.py                    ← Phase 3: attributions
│   └── submit_daniocell_attribution.sh
├── 04_specificity/
│   ├── daniocell_specificity.py                    ← Phase 4: cell-type specificity
│   └── submit_daniocell_specificity.sh
└── 05_modisco/
    ├── daniocell_modisco.py                        ← Phase 5: motif discovery
    └── submit_daniocell_modisco.sh
```

---

## 10. How to Use the Best Model

### Load checkpoint for inference

```python
import sys
sys.path.insert(0, "/hpc/projects/data.science/yangjoon.kim/step/decima-main/src/decima")
from decima_model import DecimaModel
from lightning import DecimaLightning
import torch

CKPT = ("/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/"
        "experiments/daniocell_experiments_20260406_114453/"
        "pretrained_decima-human_rep0_lr3e-05_seed42/version_0/checkpoints/last.ckpt")

model = DecimaLightning.load_from_checkpoint(CKPT)
model.eval()
model.cuda()
```

### Load predictions (already computed)

```python
import anndata
ad = anndata.read_h5ad(
    "/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/"
    "01_predictions/data_out_daniocell_pretrained_decima-human_rep0_lr3e-05_seed42.h5ad"
)
# ad.X = observed, ad.layers['preds'] = predicted
# ad.var['pearson'] = per-gene correlation
# ad.obs['test_pearson'] = per-track correlation (test genes)
```

### Run directed evolution (next step)

```python
# See Mathias's design scripts for reference:
# decima-applications-main/notebooks/9_design/00_evolve_combined.py
# Adapt for DanioCell: use identity.super for cell types, 1,047 tracks
```

---

## 11. Presentation

Slide deck for team presentation (Loic Royer et al.):
```
/hpc/projects/data.science/yangjoon.kim/step/docs/slides/daniocell-finetuning-2026-04-08.pptx
```
16 slides, Journey/Timeline narrative. Generation script at `scripts/generate_slides_daniocell.py`.
