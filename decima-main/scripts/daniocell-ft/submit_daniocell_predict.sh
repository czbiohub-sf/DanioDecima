#!/bin/bash
#SBATCH --job-name=daniocell_predict
#SBATCH --output=/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/experiments/logs/daniocell_predict_%A_%a.out
#SBATCH --error=/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/experiments/logs/daniocell_predict_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --array=0-7

# ---------------------------------------------------------------------------
# DanioCell Decima — Test-set predictions (Phase 1)
#
# Array jobs 0-3: pretrained (human Decima backbone)
# Array jobs 4-7: random initialization baseline
# ---------------------------------------------------------------------------

module load anaconda
source activate gReLu

# Checkpoint base directory
EXP_BASE="/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/experiments/daniocell_experiments_20260406_114453"

# Experiment names and checkpoint paths
declare -a EXP_NAMES=(
    "pretrained_decima-human_rep0_lr3e-05_seed42"
    "pretrained_decima-human_rep1_lr3e-05_seed42"
    "pretrained_decima-human_rep2_lr3e-05_seed42"
    "pretrained_decima-human_rep3_lr3e-05_seed42"
    "random_lr3e-06_seed42"
    "random_lr3e-06_seed43"
    "random_lr3e-06_seed44"
    "random_lr3e-06_seed45"
)

declare -a CKPT_FILES=(
    "${EXP_BASE}/pretrained_decima-human_rep0_lr3e-05_seed42/version_0/checkpoints/epoch=8-step=7263.ckpt"
    "${EXP_BASE}/pretrained_decima-human_rep1_lr3e-05_seed42/version_0/checkpoints/epoch=2-step=2421.ckpt"
    "${EXP_BASE}/pretrained_decima-human_rep2_lr3e-05_seed42/version_0/checkpoints/epoch=3-step=3228.ckpt"
    "${EXP_BASE}/pretrained_decima-human_rep3_lr3e-05_seed42/version_0/checkpoints/epoch=7-step=6456.ckpt"
    "${EXP_BASE}/random_lr3e-06_seed42/version_0/checkpoints/epoch=9-step=8070.ckpt"
    "${EXP_BASE}/random_lr3e-06_seed43/version_0/checkpoints/epoch=16-step=13719.ckpt"
    "${EXP_BASE}/random_lr3e-06_seed44/version_0/checkpoints/epoch=11-step=9684.ckpt"
    "${EXP_BASE}/random_lr3e-06_seed45/version_0/checkpoints/epoch=20-step=16947.ckpt"
)

# Get experiment for this array task
EXP_NAME="${EXP_NAMES[$SLURM_ARRAY_TASK_ID]}"
CKPT="${CKPT_FILES[$SLURM_ARRAY_TASK_ID]}"

echo "============================================"
echo "Array task ID:  $SLURM_ARRAY_TASK_ID"
echo "Experiment:     $EXP_NAME"
echo "Checkpoint:     $CKPT"
echo "============================================"

# Verify checkpoint exists
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

# Run prediction
python /hpc/projects/data.science/yangjoon.kim/step/decima-main/scripts/daniocell-ft/daniocell_predict.py \
    --device 0 \
    --ckpt "$CKPT" \
    --exp_name "$EXP_NAME" \
    --max_seq_shift 3 \
    --batch_size 6 \
    --num_workers 4

echo "Done: $(date)"
