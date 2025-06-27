#!/bin/bash
#SBATCH --job-name=combined_attribution_analysis
#SBATCH --array=0-15
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/logs/combined_attribution_%A_%a.out
#SBATCH --error=/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/logs/combined_attribution_%A_%a.err

# Create logs directory if it doesn't exist
mkdir -p /hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/logs

PYTHON_SCRIPT="/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/5_specificity/00_combined_attribution_analysis.py"

declare -a PAIRS=(
"Human_Decima_0|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250618_111138/pretrained_decima-human_rep0_lr3e-05_seed42/version_0"
"Human_Decima_1|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250618_111138/pretrained_decima-human_rep1_lr3e-05_seed42/version_0"
"Human_Decima_2|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250618_111138/pretrained_decima-human_rep2_lr3e-05_seed42/version_0"
"Human_Decima_3|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250618_111138/pretrained_decima-human_rep3_lr3e-05_seed42/version_0"
"Human_Borzoi_0|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-human_rep0_lr3e-05_seed42/version_0"
"Human_Borzoi_1|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-human_rep1_lr3e-05_seed42/version_0"
"Human_Borzoi_2|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-human_rep2_lr3e-05_seed42/version_0"
"Human_Borzoi_3|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-human_rep3_lr3e-05_seed42/version_0"
"Mouse_Borzoi_0|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-mouse_rep0_lr3e-05_seed42/version_0"
"Mouse_Borzoi_1|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-mouse_rep1_lr3e-05_seed42/version_0"
"Mouse_Borzoi_2|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-mouse_rep2_lr3e-05_seed42/version_0"
"Mouse_Borzoi_3|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-mouse_rep3_lr3e-05_seed42/version_0"
"Random_0|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/random_lr3e-06_seed42/version_0"
"Random_1|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/random_lr3e-06_seed43/version_0"
"Random_2|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/random_lr3e-06_seed44/version_0"
"Random_3|/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/random_lr3e-06_seed45/version_0"
)

# Get the task for this array job
entry="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
IFS="|" read -r NAME MODEL_DIR <<< "$entry"

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running: $NAME"
echo "Model dir: $MODEL_DIR"
echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"

# Find the checkpoint file in the model directory
CKPT_PATH=$(find "$MODEL_DIR/checkpoints" -name "epoch*.ckpt" | head -n 1)
if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: No checkpoint found in $MODEL_DIR/checkpoints"
    exit 1
fi
echo "Checkpoint: $CKPT_PATH"

# Load necessary modules (adjust as needed for the cluster)
module load anaconda/latest
source activate pytorch

export GENOMEPY_CACHE_DIR="/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/genomepy_cache/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
export GENOMEPY_CONFIG_DIR="/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/genomepy_config/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
export WANDB_CACHE_DIR="/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/wandb_cache"

# Create directories if they don't exist
mkdir -p "$GENOMEPY_CACHE_DIR"
mkdir -p "$GENOMEPY_CONFIG_DIR"

# Run the attribution analysis
python "$PYTHON_SCRIPT" \
    --name "$NAME" \
    --ckpt_path "$CKPT_PATH" \
    --model_dir "$MODEL_DIR" \
    --device 0

echo "Completed processing for $NAME"