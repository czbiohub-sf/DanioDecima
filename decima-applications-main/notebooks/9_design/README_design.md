# Motif Occurrence Analysis Scripts

This directory contains four versions of motif occurrence analysis scripts that evolved over time to handle different data structures and analysis requirements, plus the evolution pipeline that generates the data.

## Evolution Pipeline Overview

### Pipeline Components

**1. `00_submit_evolve_combined.sh` (SLURM Launcher)**
- Submits array job with indices 0-399 (16 models × 25 cell types = 400 jobs)
- Maps each array index to model directory and target cell type
- Sets up clean Genomepy caches to prevent corruption
- Runs two-step pipeline: evolution → analysis
- Resources: 1 GPU (H100/H200), 4 CPUs, 100GB RAM, 8 hours

**2. `00_evolve_combined.py` (Evolution Script)**
- Performs directed evolution of promoter sequences for cell type specificity
- Uses simple mutation scanning (no batching for stability)
- Optimizes target_mean - background_mean specificity
- Outputs CSV with full evolution trajectory

**3. `1_read_celltypes.py` (Analysis Script)**
- Analyzes evolved sequences with ISM (In Silico Mutagenesis)
- Performs TF motif scanning with ISM weight calculation
- Creates evolution plots and motif visualizations
- Designed for single-file processing in job arrays

### Pipeline Data Flow

SLURM Array Job → Evolution (00_evolve_combined.py) → Analysis (1_read_celltypes.py) → Motif Analysis Scripts

## Analysis Scripts

### 1. `02_summarize_results.py` (Basic Version)
**Purpose:** Initial basic motif occurrence analysis  
**Data Structure:** Flat directory with simple filenames  
**Analysis Focus:** Model-centric tables and heatmaps  
**Pipeline Compatibility:** ❌ Pre-pipeline, legacy format

**Features:**
- Basic filtering (p-value, ISM weight)
- Simple filename parsing (`HumanBorzoi_rep` format)
- Model-focused occurrence tables
- Basic heatmaps and comparison plots
- Expected: 4 replicates per cell type

---

### 2. `02_summarize_results_spec.py` (Enhanced Version)
**Purpose:** Enhanced analysis with trajectory data and advanced filtering  
**Data Structure:** Flat directory with trajectory files  
**Analysis Focus:** Model-centric with enhanced statistics  
**Pipeline Compatibility:** ⚠️ Partial - needs trajectory files from pipeline

**Features:**
- ✅ **Trajectory data integration** (final sequence specificity)
- ✅ **Per-sequence ISM weight filtering** (percentile-based within each sequence)
- ✅ **Replicate normalization** (accounts for failed evolution runs)
- ✅ **Enhanced statistics** (ISM weight distributions, specificity metrics)
- ✅ **Volcano plot preparation** (averaged normalized tables)
- ✅ **Comprehensive reporting** with detailed summaries

---

### 3. `02_summarize_results_designs.py` (Multi-Model Pipeline Version)
**Purpose:** Analysis for multiple model types from evolution pipeline  
**Data Structure:** Flat directory with pipeline filenames  
**Analysis Focus:** Both model-centric AND cell type-centric  
**Pipeline Compatibility:** ✅ **Designed for current pipeline**

**Features:**
- ✅ **All features from "spec" version**
- ✅ **Multiple model types** (Human-Borzoi, Human-Decima, Mouse-Borzoi, Random)
- ✅ **Pipeline filename parsing** (`evolved_promoter_Model_rep_seed_celltype_timepoint_single`)
- ✅ **Dual analysis approach** (model-focused + cell type-focused tables)
- ✅ **Cell type-specific summaries with model breakdown**
- Expected: 16 replicates per cell type (4 model types × 4 reps)

**Filename Pattern Support:**

evolved_promoter_Human_Decima_0_seed42_neural_crest_16hpf_single_final_motifs.csv
evolved_promoter_Human_Decima_0_seed42_neural_crest_16hpf_single_trajectory.csv


---

### 4. `02_summarize_results_designs_test.py` (Nested Directory Version)
**Purpose:** Analysis for nested directory structure with robust error handling  
**Data Structure:** Nested subdirectories (e.g., `analysis_25ct_20250619/`)  
**Analysis Focus:** Primarily cell type-centric  
**Pipeline Compatibility:** ✅ **For archived/organized pipeline results**

**Features:**
- ✅ **All advanced filtering from previous versions**
- ✅ **Nested directory support** (searches through subdirectories)
- ✅ **Robust filename parsing** (multiple fallback patterns)
- ✅ **Extensive error handling** and debugging capabilities
- ✅ **Debug mode** (`--debug` flag to examine filenames)
- ✅ **Directory-based metadata extraction**


## Quick Selection Guide

| Your Data Source | Recommended Script |
|------------------|-------------------|
| **Current evolution pipeline output** | `02_summarize_results_designs.py` |
| **Archived pipeline results (nested dirs)** | `02_summarize_results_designs_test.py` |
| **Legacy single-model data** | `02_summarize_results_spec.py` |
| **Very basic analysis** | `02_summarize_results.py` |

## Pipeline Output Structure

### Evolution Pipeline Generates:

evolved_${NUM_CELLTYPES}ct_${DATE}/

├── evolved_promoter_Human_Decima_0_seed42_brain_16hpf_single.csv

├── evolved_promoter_Human_Borzoi_1_seed42_neural_crest_16hpf_single.csv

└── ...

analysis_${NUM_CELLTYPES}ct_${DATE}/

├── evolved_promoter_Human_Decima_0_seed42_brain_16hpf_single_trajectory.csv

├── evolved_promoter_Human_Decima_0_seed42_brain_16hpf_single_final_motifs.csv

├── evolved_promoter_Human_Decima_0_seed42_brain_16hpf_single_evolution_plots.png

└── ...


### Analysis Scripts Process:
- **Trajectory files** (`*_trajectory.csv`) - Evolution progress with specificity scores
- **Motif files** (`*_final_motifs.csv`) - TF motifs with ISM weights
- **Plot files** (`*_evolution_plots.png`, `*_ism_*.png`) - Visualizations

## Key Parameters

### Evolution Parameters (`00_evolve_combined.py`):
- `--rounds`: Evolution rounds (default: 100)
- `--sequence_length`: Promoter length (default: 200bp)
- `--early_stopping_patience`: Convergence patience (default: 50)
- `--target_celltype`: Cell type to optimize for

### Analysis Parameters (all `02_summarize_results_*.py`):
- `--results_dir`: Input directory containing motif analysis results
- `--output_dir`: Output directory for tables and plots
- `--max_motif_pval`: Maximum motif p-value (default: 0.05)
- `--min_ism_weight`: Minimum ISM weight threshold (default: 0.0)
- `--ism_percentile`: Per-sequence ISM weight percentile cutoff (default: 75.0)
- `--min_specificity`: Minimum final sequence specificity (default: 1.0)

## Output Files

### Pipeline Outputs:
- Evolution trajectories with specificity scores
- ISM analysis results and heatmaps
- TF motif hits with ISM weights
- Evolution and motif visualization plots

### Analysis Script Outputs:
- Individual model/cell type occurrence tables (raw and normalized)
- Combined occurrence tables across models and cell types
- Comprehensive motif CSV with all details
- Enhanced summary reports with replicate success rates
- Volcano plot instructions and ready-to-use data

## Model Types Supported

The current pipeline supports 16 models across 4 types:
- **Human-Decima** (4 replicates): Decima model pretrained on human data
- **Human-Borzoi** (4 replicates): Borzoi model pretrained on human data  
- **Mouse-Borzoi** (4 replicates): Borzoi model pretrained on mouse data
- **Random** (4 replicates): Randomly initialized models

## Notes

- **Pipeline Integration:** Use `*designs.py` for direct pipeline output analysis
- **Backward Compatibility:** Newer versions are generally NOT backward compatible with older data structures
- **Specificity Calculation:** Pipeline uses `target_mean - background_mean` (not max)
- **ISM Weights:** Calculated as mean absolute ISM values across motif positions