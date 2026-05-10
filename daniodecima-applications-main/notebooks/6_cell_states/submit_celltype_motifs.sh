#!/bin/bash
#SBATCH --job-name=celltype_motifs
#SBATCH --output=/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/logs/celltype_motifs_%A_%a.out
#SBATCH --error=/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/logs/celltype_motifs_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --array=0-0

# TROUBLESHOOTING MODE: 1 model × 1 cell type = 1 job (task ID 0)
# To expand: change array to 0-399%20 and uncomment models/cell types

# Function to check command success
check_success() {
    if [ $? -ne 0 ]; then
        echo "ERROR: $1 failed!"
        exit 1
    fi
}

# Load anaconda module
module load anaconda/latest

# Set up paths
BASE_DIR="/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima"
SCRIPT_DIR="/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/6_cell_states"
OUTPUT_BASE="${BASE_DIR}/celltype_motif_analysis_job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
DATA_DIR="${BASE_DIR}/data/celltypes_chrom_split_v1"
MEME_FILE="/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/9_design/JASPAR2020_CORE_vertebrates_non-redundant_pfms.meme"

# Data files
H5_FILE="${DATA_DIR}/data.h5"

# TIMEPOINT CONFIGURATION
# Set to specific timepoint (e.g., "16hpf") or leave empty for all timepoints
TIMEPOINT="16hpf"  # Change this as needed: "16hpf", "12hpf", "24hpf", etc. or "" for all

# Calculate model_id and cell_type_id from array task ID
# For troubleshooting: 1 model × 1 cell type
MODEL_ID=$((SLURM_ARRAY_TASK_ID / 1))  # Changed from 25 to 1
CELL_TYPE_ID=$((SLURM_ARRAY_TASK_ID % 1))  # Changed from 25 to 1

# Define model names for logging (TROUBLESHOOTING - only 1 model)
declare -a MODEL_NAMES=(
    "human_decima_rep0"
    # "human_decima_rep1" "human_decima_rep2" "human_decima_rep3"
    # "human_borzoi_rep0" "human_borzoi_rep1" "human_borzoi_rep2" "human_borzoi_rep3"
    # "mouse_borzoi_rep0" "mouse_borzoi_rep1" "mouse_borzoi_rep2" "mouse_borzoi_rep3"
    # "random_rep0" "random_rep1" "random_rep2" "random_rep3"
)

# Define cell type names for logging (TROUBLESHOOTING - only 1 cell type)
declare -a CELL_TYPES=(
    "adaxial_cell"
    # "brain" "common_myeloid_progenitor" "ectodermal_cell" "floor_plate"
    # "hatching_gland_cell" "head_mesenchyme" "heart" "hematopoietic_system" "lateral_mesoderm"
    # "lens_placode" "midbrain_hindbrain_boundary" "myotome" "neural_crest" "neural_tube"
    # "notochord" "optic_vesicle" "otic_placode" "paraxial_mesoderm" "periderm"
    # "pronephros" "somite" "spinal_cord_neural_tube" "telencephalon" "trigeminal_placode"
)

MODEL_NAME=${MODEL_NAMES[$MODEL_ID]}
CELL_TYPE_NAME=${CELL_TYPES[$CELL_TYPE_ID]}

# Determine timepoint description
if [ -z "$TIMEPOINT" ]; then
    TIMEPOINT_DESC="all timepoints"
    TIMEPOINT_ARG=""
else
    TIMEPOINT_DESC="$TIMEPOINT only"
    TIMEPOINT_ARG="--timepoint $TIMEPOINT"
fi

echo "========================================================================"
echo "CELLTYPE DIFFERENTIAL MOTIF ANALYSIS (TROUBLESHOOTING MODE)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL_NAME (ID: $MODEL_ID)"
echo "Cell type: $CELL_TYPE_NAME (ID: $CELL_TYPE_ID)"
echo "Timepoint: $TIMEPOINT_DESC"
echo "Analysis: Differential (target vs all other cell types at specified timepoint)"
echo "Transform: specificity"
echo "Total jobs: 1 (1 model × 1 cell type)"
echo "========================================================================"

# Create output directory
mkdir -p "$OUTPUT_BASE"
mkdir -p "${BASE_DIR}/experiments/logs"

echo "========================================================================"
echo "STEP 1: Running differential attribution analysis in PyTorch environment"
echo "========================================================================"

# Activate PyTorch environment
source activate pytorch
check_success "PyTorch environment activation"

# Run attribution analysis with timepoint argument
python "${SCRIPT_DIR}/celltype_motif_attribution.py" \
    --model_id $MODEL_ID \
    --cell_type_id $CELL_TYPE_ID \
    --h5_file "$H5_FILE" \
    --output_base "$OUTPUT_BASE" \
    --n_genes 50 \
    --job_id $SLURM_ARRAY_JOB_ID \
    --task_id $SLURM_ARRAY_TASK_ID \
    $TIMEPOINT_ARG

check_success "Attribution analysis"

echo "========================================================================"
echo "STEP 2: Running MoDISco analysis in modisco_env"
echo "========================================================================"

# Switch to MoDISco environment
conda deactivate
source activate modisco_env
check_success "MoDISco environment activation"

# Find the output directory (with safe cell type name and timepoint)
SAFE_CELL_TYPE=$(echo "$CELL_TYPE_NAME" | tr ' ' '_' | tr '/' '_')
if [ -z "$TIMEPOINT" ]; then
    ANALYSIS_DIR="${OUTPUT_BASE}/${MODEL_NAME}_${SAFE_CELL_TYPE}"
else
    ANALYSIS_DIR="${OUTPUT_BASE}/${MODEL_NAME}_${SAFE_CELL_TYPE}_${TIMEPOINT}"
fi

# Verify files exist
SEQ_FILE="${ANALYSIS_DIR}/sequences.npy"
ATTR_FILE="${ANALYSIS_DIR}/attributions.npy"

if [ ! -f "$SEQ_FILE" ] || [ ! -f "$ATTR_FILE" ]; then
    echo "ERROR: Attribution files not found!"
    echo "Expected: $SEQ_FILE and $ATTR_FILE"
    exit 1
fi

echo "Found attribution files:"
echo "  Sequences: $SEQ_FILE"
echo "  Attributions: $ATTR_FILE"

# Run MoDISco
python "${SCRIPT_DIR}/modisco_simple.py" \
    -seq_file "$SEQ_FILE" \
    -attr_file "$ATTR_FILE" \
    -meme_file "$MEME_FILE" \
    -out_dir "$ANALYSIS_DIR"

check_success "MoDISco analysis"

echo "========================================================================"
echo "DIFFERENTIAL ANALYSIS COMPLETED SUCCESSFULLY! (TROUBLESHOOTING MODE)"
echo "Model: $MODEL_NAME"
echo "Cell type: $CELL_TYPE_NAME"
echo "Timepoint: $TIMEPOINT_DESC"
echo "Analysis: Target vs all other cell types at specified timepoint"
echo "Results saved to: $ANALYSIS_DIR"
echo "HTML report: $ANALYSIS_DIR/motif_report.html"
echo "========================================================================"