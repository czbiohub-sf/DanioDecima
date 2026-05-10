# Cell Type-Specific Motif Analysis of DanioDecima with TF-MoDISco

This directory contains scripts for performing differential motif analysis on zebrafish developmental data using TF-MoDISco. The analysis identifies motifs that are specifically important for each cell type compared to all other cell types at a given timepoint.

## Environment Setup

This pipeline requires two separate conda environments due to different dependency requirements:

### Quick Setup

```bash
# Create both environments from the provided YAML files
conda env create -f environment_pytorch_full.yml
conda env create -f environment_modisco.yml
```

### Environment Details

**PyTorch Environment** (`pytorch`):
- Used for: Attribution analysis, model loading, data processing
- Key packages: PyTorch 2.5.1, PyTorch Lightning 2.4.0, Captum 0.5.0, AnnData, Scanpy
- GPU support: CUDA 12.4
- File: `environment_pytorch_full.yml`

**MoDISco Environment** (`modisco_env`):
- Used for: Motif discovery and analysis  
- Key packages: modisco-lite 2.3.2, MEME Suite 5.5.7, basic data science stack
- File: `environment_modisco.yml`

### Custom Package Installation

Please include the following to these environments manually:
-  `grelu`, a custom genomics package. Install from source here: https://github.com/Genentech/gReLU
- The Decima and Decima-applications directories are found in the STEP repo on the CZ Biohub SF Github: https://github.com/czbiohub-sf/step

### Environment Testing

Verify your environments are working:

```bash
# Test PyTorch environment
conda activate pytorch
python -c "import torch, captum, anndata, grelu; print('PyTorch env OK')"

# Test MoDISco environment  
conda activate modisco_env
python -c "import modiscolite, numpy, pandas; print('MoDISco env OK')"
```

## Overview

The pipeline performs:
1. **Differential attribution analysis**: Uses `transform='specificity'` to find regulatory elements that are more important for the target cell type than for other cell types
2. **Motif discovery**: Uses TF-MoDISco to identify enriched motifs in the attribution scores
3. **Motif annotation**: Matches discovered motifs against JASPAR database

## Files

- `celltype_motif_attribution.py` - Main attribution analysis script (requires `pytorch` env)
- `submit_celltype_motifs.sh` - SLURM submission script (handles env switching)
- `modisco_simple.py` - TF-MoDISco analysis script (requires `modisco_env`)
- `environment_pytorch.yml` - PyTorch environment specification
- `environment_modisco.yml` - MoDISco environment specification
- `README_analysis_TF-MoDISco.md` - This documentation

## Quick Start

1. **Set up environments** (first time only):
   ```bash
   conda env create -f environment_pytorch_full.yml
   conda env create -f environment_modisco.yml
   ```

2. **Configure your analysis** by editing `submit_celltype_motifs.sh`:
   ```bash
   # Set timepoint (or leave empty for all timepoints)
   TIMEPOINT="16hpf"  # Options: "10hpf", "12hpf", "14hpf", "16hpf", "19hpf", "24hpf", "2dpf", "3dpf", "5dpf", "10dpf", or ""
   
   # Uncomment models and cell types you want to analyze
   ```

3. **Submit the job**:
   ```bash
   cd /path/to/notebooks/6_cell_states/
   sbatch submit_celltype_motifs.sh
   ```

## Configuration Options

### 1. Timepoint Selection

Edit the `TIMEPOINT` variable in `submit_celltype_motifs.sh`:

```bash
# Single timepoint analysis (recommended)
TIMEPOINT="16hpf"    # Analyzes only 16hpf data (1 task per cell type)

# All timepoints analysis
TIMEPOINT=""         # Analyzes across all developmental stages (multiple tasks per cell type)

# Other timepoints
TIMEPOINT="12hpf"    # Early development
TIMEPOINT="3dpf"    # Later development
```

**Recommendation**: Use single timepoint (e.g., `"16hpf"`) for cleaner analysis with exactly 1 task per cell type.

### 2. Model Selection

Edit the `MODEL_NAMES` array in `submit_celltype_motifs.sh`:

```bash
declare -a MODEL_NAMES=(
    "human_decima_rep0" "human_decima_rep1" "human_decima_rep2" "human_decima_rep3"    # Human Decima models
    "human_borzoi_rep0" "human_borzoi_rep1" "human_borzoi_rep2" "human_borzoi_rep3"   # Human Borzoi models  
    "mouse_borzoi_rep0" "mouse_borzoi_rep1" "mouse_borzoi_rep2" "mouse_borzoi_rep3"   # Mouse Borzoi models
    "random_rep0" "random_rep1" "random_rep2" "random_rep3"                           # Random init models
)
```

And update the corresponding `model_dirs` array in `celltype_motif_attribution.py`.

### 3. Cell Type Selection

Edit the `CELL_TYPES` array in `submit_celltype_motifs.sh`:

```bash
declare -a CELL_TYPES=(
    "adaxial_cell" "brain" "common_myeloid_progenitor" "ectodermal_cell" "floor_plate"
    "hatching_gland_cell" "head_mesenchyme" "heart" "hematopoietic_system" "lateral_mesoderm"
    "lens_placode" "midbrain_hindbrain_boundary" "myotome" "neural_crest" "neural_tube"
    "notochord" "optic_vesicle" "otic_placode" "paraxial_mesoderm" "periderm"
    "pronephros" "somite" "spinal_cord_neural_tube" "telencephalon" "trigeminal_placode"
)
```

**Available cell types at 16hpf** (25 total):
- `adaxial cell` - Muscle precursors
- `brain` - Central nervous system
- `common myeloid progenitor` - Blood cell precursors
- `ectodermal cell` - Outer layer cells
- `floor plate` - Ventral neural tube
- `hatching gland cell` - Secretory cells
- `head mesenchyme` - Head connective tissue
- `heart` - Cardiac tissue
- `hematopoietic system` - Blood system
- `lateral mesoderm` - Side mesoderm
- `lens placode` - Eye lens precursor
- `midbrain hindbrain boundary` - Brain region boundary
- `myotome` - Muscle segments
- `neural crest` - Multipotent cells
- `neural tube` - Central nervous system precursor
- `notochord` - Axial skeleton precursor
- `optic vesicle` - Eye precursor
- `otic placode` - Ear precursor
- `paraxial mesoderm` - Somite precursors
- `periderm` - Outer skin layer
- `pronephros` - Primitive kidney
- `somite` - Segmented mesoderm
- `spinal cord neural tube` - Spinal cord
- `telencephalon` - Forebrain
- `trigeminal placode` - Cranial nerve precursor

### 4. Job Array Configuration

Update the SLURM array bounds based on your selection:

```bash
# For troubleshooting (1 model × 1 cell type)
#SBATCH --array=0-0

# For full analysis (16 models × 25 cell types = 400 jobs)
#SBATCH --array=0-399%20

# For subset analysis (e.g., 4 models × 5 cell types = 20 jobs)
#SBATCH --array=0-19%10
```

And update the calculation:
```bash
# Full analysis
MODEL_ID=$((SLURM_ARRAY_TASK_ID / 25))    # Number of cell types
CELL_TYPE_ID=$((SLURM_ARRAY_TASK_ID % 25))

# Subset analysis (adjust denominators accordingly)
MODEL_ID=$((SLURM_ARRAY_TASK_ID / 5))     # If using 5 cell types
CELL_TYPE_ID=$((SLURM_ARRAY_TASK_ID % 5))
```

## Example Configurations

### Configuration 1: Single Model, Single Cell Type (Troubleshooting)
```bash
# In submit_celltype_motifs.sh
TIMEPOINT="16hpf"
#SBATCH --array=0-0

declare -a MODEL_NAMES=("human_decima_rep0")
declare -a CELL_TYPES=("adaxial_cell")

MODEL_ID=$((SLURM_ARRAY_TASK_ID / 1))
CELL_TYPE_ID=$((SLURM_ARRAY_TASK_ID % 1))
```

### Configuration 2: All Models, Neural Cell Types Only
```bash
# In submit_celltype_motifs.sh
TIMEPOINT="16hpf"
#SBATCH --array=0-63%15  # 16 models × 4 cell types = 64 jobs

declare -a MODEL_NAMES=(
    "human_decima_rep0" "human_decima_rep1" "human_decima_rep2" "human_decima_rep3"
    "human_borzoi_rep0" "human_borzoi_rep1" "human_borzoi_rep2" "human_borzoi_rep3"
    "mouse_borzoi_rep0" "mouse_borzoi_rep1" "mouse_borzoi_rep2" "mouse_borzoi_rep3"
    "random_rep0" "random_rep1" "random_rep2" "random_rep3"
)

declare -a CELL_TYPES=(
    "brain" "neural_crest" "neural_tube" "spinal_cord_neural_tube"
)

MODEL_ID=$((SLURM_ARRAY_TASK_ID / 4))
CELL_TYPE_ID=$((SLURM_ARRAY_TASK_ID % 4))
```

### Configuration 3: Full Analysis
```bash
# In submit_celltype_motifs.sh
TIMEPOINT="16hpf"
#SBATCH --array=0-399%20  # 16 models × 25 cell types = 400 jobs

# Uncomment all models and cell types
MODEL_ID=$((SLURM_ARRAY_TASK_ID / 25))
CELL_TYPE_ID=$((SLURM_ARRAY_TASK_ID % 25))
```

## Output Structure

Results are saved to: `/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/celltype_motif_analysis_job_${JOB_ID}_task_${TASK_ID}/`

For each analysis, you'll get:
{model_name}{cell_type}{timepoint}/

├── sequences.npy # Input sequences for MoDISco

├── attributions.npy # Attribution scores for MoDISco

├── metadata.json # Analysis metadata

├── gene_stats.csv # Marker gene statistics

├── modisco_report.h5 # MoDISco results

└── motif_report.html # Final motif report (main output)

#### NOTE: Download the entire directory locally for easy visualization of the motif_report.html in a browser.


## Key Parameters

### Attribution Analysis
- `transform='specificity'` - Differential analysis (target vs background)
- `n_genes=50` - Number of top marker genes to analyze
- `method=Saliency` - Attribution method
- `window_size=10000` - 20kb window around TSS (±10kb)

### MoDISco Analysis
- `max_seqlets_per_metacluster=10000` - Maximum sequences per motif cluster
- `trim_threshold=0.2` - Motif trimming threshold
- `top_n_matches=10` - Number of JASPAR matches to report

## Monitoring Progress

Check job status:
```bash
squeue -u $USER -n celltype_motifs
```

Monitor results:
```bash
# Count completed analyses
find /hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/celltype_motif_analysis_job_*/*/motif_report.html | wc -l

# Check recent completions
find /hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/celltype_motif_analysis_job_* -name "motif_report.html" -newer /tmp/last_check
```

## Troubleshooting

### Common Issues

1. **No observations at timepoint**: Check available timepoints in your data
2. **Attribution calculation fails**: Check that gene names match between H5 file and matrix file
3. **MoDISco fails**: Usually due to insufficient attribution signal - try more genes or different cell type
4. **Out of memory**: Reduce `n_genes` or request more memory

### Debug Mode

For debugging, use the single-job configuration:
```bash
TIMEPOINT="16hpf"
#SBATCH --array=0-0
declare -a MODEL_NAMES=("human_decima_rep0")
declare -a CELL_TYPES=("adaxial_cell")
```

### Checking Data

To verify what's in your data:
```python
import anndata
ad = anndata.read_h5ad("path/to/matrix/file.h5ad")
print("Available timepoints:", sorted(ad.obs['timepoint'].unique()))
print("Available cell types:", sorted(ad.obs['zebrafish_anatomy_ontology_class_fine'].unique()))
print("Cell type counts at 16hpf:", ad[ad.obs['timepoint'] == '16hpf'].obs['zebrafish_anatomy_ontology_class_fine'].value_counts())
```

## Expected Runtime

- **Single analysis**: ~30-60 minutes (attribution + MoDISco)
- **Full analysis (400 jobs)**: ~2-4 hours with 20 concurrent jobs
- **Memory usage**: ~32-64GB per job
- **GPU requirement**: Yes (for attribution calculation)


## Environment Requirements

- `pytorch` environment (for attribution analysis) - see `environment_pytorch_full.yml`
- `modisco_env` environment (for TF-MoDISco) - see `environment_modisco.yml`
- JASPAR motif database
- Access to model checkpoints and data files

## Citation

If you use these scripts, please cite:
- TF-MoDISco: Shrikumar et al., 2018
- JASPAR: Fornes et al., 2020
- Decima: Lal et al., 2024

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Check environment setup with the test commands above
3. Examine the log files in `/hpc/scratch/.../logs/`
4. Contact mathias.voges@