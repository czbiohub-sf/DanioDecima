#!/bin/bash
#SBATCH --job-name=evolve_design_pipeline_%a
#SBATCH --array=0-399  # 16 models × 25 cell types = 400 jobs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --output=/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/design/logs/evolve_design_%A_%a.out
#SBATCH --error=/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/design/logs/evolve_design_%A_%a.err

# Create logs directory
mkdir -p logs

# CRITICAL: Set genomepy environment variables BEFORE loading any modules
export GENOMEPY_CACHE_DIR="$TMPDIR/genomepy_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export GENOMEPY_CONFIG_DIR="$TMPDIR/genomepy_config_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

# Clear any existing corrupted cache directories
echo "Cleaning up any existing genomepy caches..."
rm -rf ~/.local/share/genomepy
rm -rf ~/.cache/genomepy
rm -rf /tmp/genomepy*
rm -rf "$GENOMEPY_CACHE_DIR" "$GENOMEPY_CONFIG_DIR"

# Create fresh cache directories
mkdir -p "$GENOMEPY_CACHE_DIR" "$GENOMEPY_CONFIG_DIR"

# Now load environment (after setting genomepy vars)
module load anaconda/latest
source activate pytorch

# Set other environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB_CACHE_DIR="/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/wandb_cache"

echo "Genomepy cache directory: $GENOMEPY_CACHE_DIR"
echo "Cache directory contents:"
ls -la "$GENOMEPY_CACHE_DIR" || echo "Cache directory is empty (good)"

# Define model directories (16 total)
declare -a MODEL_DIRS=(
    # Human Decima (4)
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250618_111138/pretrained_decima-human_rep0_lr3e-05_seed42/version_0"
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250618_111138/pretrained_decima-human_rep1_lr3e-05_seed42/version_0"
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250618_111138/pretrained_decima-human_rep2_lr3e-05_seed42/version_0"
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250618_111138/pretrained_decima-human_rep3_lr3e-05_seed42/version_0"
    # Human Borzoi (4)
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-human_rep0_lr3e-05_seed42/version_0"
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-human_rep1_lr3e-05_seed42/version_0"
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-human_rep2_lr3e-05_seed42/version_0"
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-human_rep3_lr3e-05_seed42/version_0"
    # Mouse Borzoi (4)
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-mouse_rep0_lr3e-05_seed42/version_0"
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-mouse_rep1_lr3e-05_seed42/version_0"
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-mouse_rep2_lr3e-05_seed42/version_0"
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-mouse_rep3_lr3e-05_seed42/version_0"
    # Random (4)
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/random_lr3e-06_seed42/version_0"
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/random_lr3e-06_seed43/version_0"
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/random_lr3e-06_seed44/version_0"
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/random_lr3e-06_seed45/version_0"
)

# Define target cell types (only the active ones)
declare -a CELL_TYPES=(
    "adaxial cell"
    "brain"
    "common myeloid progenitor"
    "ectodermal cell"
    "floor plate"
    "hatching gland cell"
    "head mesenchyme"
    "heart"
    "hematopoietic system"
    "lateral mesoderm"
    "lens placode"
    "midbrain hindbrain boundary"
    "myotome"
    "neural crest"
    "neural tube"
    "notochord"
    "optic vesicle"
    "otic placode"
    "paraxial mesoderm"
    "periderm"
    "pronephros"
    "somite"
    "spinal cord neural tube"
    "telencephalon"
    "trigeminal placode"
)

# Calculate indices based on actual active cell types
NUM_CELLTYPES=${#CELL_TYPES[@]}
echo "Number of cell types: $NUM_CELLTYPES"
echo "Active cell types: ${CELL_TYPES[@]}"

# Validate we have cell types
if [ $NUM_CELLTYPES -eq 0 ]; then
    echo "ERROR: No active cell types found!"
    exit 1
fi

model_idx=$((SLURM_ARRAY_TASK_ID / NUM_CELLTYPES))
celltype_idx=$((SLURM_ARRAY_TASK_ID % NUM_CELLTYPES))

MODEL_DIR=${MODEL_DIRS[$model_idx]}
TARGET_CELLTYPE=${CELL_TYPES[$celltype_idx]}

# Set paths
DATA_DIR="/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/data/celltypes_chrom_split_v1"
TODAY=$(date +%Y%m%d)
TIMESTAMP=$(date +%H%M%S)
NUM_CELLTYPES=${#CELL_TYPES[@]}
EVOLUTION_OUTPUT_DIR="/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/design/evolved_${NUM_CELLTYPES}ct_${TODAY}_${TIMESTAMP}"
ANALYSIS_OUTPUT_DIR="/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/design/analysis_${NUM_CELLTYPES}ct_${TODAY}_${TIMESTAMP}"

# Create output directories
mkdir -p $EVOLUTION_OUTPUT_DIR
mkdir -p $ANALYSIS_OUTPUT_DIR

echo "========================================"
echo "EVOLUTION + ANALYSIS PIPELINE"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model Index: $model_idx"
echo "Cell Type Index: $celltype_idx"
echo "Model: $MODEL_DIR"
echo "Target: $TARGET_CELLTYPE"
echo "Started at: $(date)"
echo "========================================"

# Step 1: Evolution
echo "STEP 1: RUNNING EVOLUTION..."
python /hpc/mydata/mathias.voges/Projects/research/seq2fun/step/daniodecima-applications-main/notebooks/9_design/00_evolve_combined.py \
    --device 0 \
    --model_dir "$MODEL_DIR" \
    --data_dir "$DATA_DIR" \
    --output_dir "$EVOLUTION_OUTPUT_DIR" \
    --target_celltype "$TARGET_CELLTYPE" \
    --timepoint "16hpf" \
    --seed 42 \
    --rounds 100 \
    --sequence_length 200 \
    --early_stopping_patience 50 \
    --min_improvement 0.001

EVOLUTION_EXIT_CODE=$?

if [ $EVOLUTION_EXIT_CODE -eq 0 ]; then
    echo "✓ Evolution completed successfully"
    
    # Find the evolved CSV file
    MODEL_INFO=$(python -c "
import sys, re
model_dir = '$MODEL_DIR'
if 'decima_experiments_20250618_111138' in model_dir and 'pretrained_decima-human' in model_dir:
    rep_match = re.search(r'rep(\d+)', model_dir)
    rep = rep_match.group(1) if rep_match else '0'
    print(f'Human_Decima_{rep}')
elif 'decima_experiments_20250617_225114' in model_dir:
    if 'pretrained_wandb-human' in model_dir:
        rep_match = re.search(r'rep(\d+)', model_dir)
        rep = rep_match.group(1) if rep_match else '0'
        print(f'Human_Borzoi_{rep}')
    elif 'pretrained_wandb-mouse' in model_dir:
        rep_match = re.search(r'rep(\d+)', model_dir)
        rep = rep_match.group(1) if rep_match else '0'
        print(f'Mouse_Borzoi_{rep}')
    elif 'random_lr3e-06' in model_dir:
        seed_match = re.search(r'seed(\d+)', model_dir)
        seed = seed_match.group(1) if seed_match else '42'
        seed_to_rep = {'42': '0', '43': '1', '44': '2', '45': '3'}
        rep = seed_to_rep.get(seed, '0')
        print(f'Random_{rep}')
else:
    print('Unknown_Model')
")
    
    CELLTYPE_CLEAN=$(echo "$TARGET_CELLTYPE" | sed 's/ /_/g' | sed 's/\//_/g')
    EVOLVED_FILE="$EVOLUTION_OUTPUT_DIR/evolved_promoter_${MODEL_INFO}_seed42_${CELLTYPE_CLEAN}_16hpf_single.csv"
    
    if [ -f "$EVOLVED_FILE" ]; then
        echo "✓ Found evolved file: $(basename $EVOLVED_FILE)"
        
        # Step 2: Analysis
        echo "STEP 2: RUNNING ANALYSIS..."
        python /hpc/mydata/mathias.voges/Projects/research/seq2fun/step/daniodecima-applications-main/notebooks/9_design/1_read_celltypes.py \
            --file_path "$EVOLVED_FILE" \
            --model_base_dir "$MODEL_DIR" \
            --data_dir "$DATA_DIR" \
            --output_dir "$ANALYSIS_OUTPUT_DIR" \
            --chrom "4" \
            --tss_start 29480218 \
            --window_size 524288 \
            --tss_offset 163840
        
        ANALYSIS_EXIT_CODE=$?
        
        if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
            echo "✓ Analysis completed successfully"
            echo "🎉 PIPELINE COMPLETED SUCCESSFULLY"
        else
            echo "✗ Analysis failed (exit code: $ANALYSIS_EXIT_CODE)"
        fi
    else
        echo "✗ Evolved file not found: $EVOLVED_FILE"
        ANALYSIS_EXIT_CODE=1
    fi
else
    echo "✗ Evolution failed (exit code: $EVOLUTION_EXIT_CODE)"
    ANALYSIS_EXIT_CODE=$EVOLUTION_EXIT_CODE
fi

echo "========================================"
echo "PIPELINE SUMMARY"
echo "Model: $MODEL_INFO"
echo "Cell Type: $TARGET_CELLTYPE"
echo "Evolution exit code: $EVOLUTION_EXIT_CODE"
echo "Analysis exit code: $ANALYSIS_EXIT_CODE"
echo "Completed at: $(date)"
echo "========================================"

exit $ANALYSIS_EXIT_CODE