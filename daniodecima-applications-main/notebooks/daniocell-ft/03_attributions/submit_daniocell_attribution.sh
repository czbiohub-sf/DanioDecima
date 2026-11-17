#!/bin/bash
#SBATCH --job-name=daniocell_attr
#SBATCH --output=logs/daniocell_attr_%A_%a.out
#SBATCH --error=logs/daniocell_attr_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --array=0-4
: "${DANIODECIMA_REPO:?set it first: source daniodecima-main/scripts/daniocell_paths.env}"
: "${DANIOCELL_WORKDIR:?set it first: source daniodecima-main/scripts/daniocell_paths.env}"

# ---------------------------------------------------------------------------
# DanioCell Decima — Attribution analysis (Phase 3)
#
# Array jobs 0-3: pretrained replicates
# Array job 4:    random seed42 (negative control)
# ---------------------------------------------------------------------------

module load anaconda
source activate gReLu

EXP_BASE="${DANIOCELL_WORKDIR}/experiments/daniocell_experiments_20260406_114453"
PRED_DIR="${DANIOCELL_WORKDIR}/01_predictions"

declare -a NAMES=(
    "pretrained_rep0"
    "pretrained_rep1"
    "pretrained_rep2"
    "pretrained_rep3"
    "random_seed42"
)

declare -a EXP_NAMES=(
    "pretrained_decima-human_rep0_lr3e-05_seed42"
    "pretrained_decima-human_rep1_lr3e-05_seed42"
    "pretrained_decima-human_rep2_lr3e-05_seed42"
    "pretrained_decima-human_rep3_lr3e-05_seed42"
    "random_lr3e-06_seed42"
)

declare -a CKPT_FILES=(
    "${EXP_BASE}/pretrained_decima-human_rep0_lr3e-05_seed42/version_0/checkpoints/epoch=8-step=7263.ckpt"
    "${EXP_BASE}/pretrained_decima-human_rep1_lr3e-05_seed42/version_0/checkpoints/epoch=2-step=2421.ckpt"
    "${EXP_BASE}/pretrained_decima-human_rep2_lr3e-05_seed42/version_0/checkpoints/epoch=3-step=3228.ckpt"
    "${EXP_BASE}/pretrained_decima-human_rep3_lr3e-05_seed42/version_0/checkpoints/epoch=7-step=6456.ckpt"
    "${EXP_BASE}/random_lr3e-06_seed42/version_0/checkpoints/epoch=9-step=8070.ckpt"
)

NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
EXP_NAME="${EXP_NAMES[$SLURM_ARRAY_TASK_ID]}"
CKPT="${CKPT_FILES[$SLURM_ARRAY_TASK_ID]}"
PRED_FILE="${PRED_DIR}/data_out_daniocell_${EXP_NAME}.h5ad"

echo "============================================"
echo "Array task ID:  $SLURM_ARRAY_TASK_ID"
echo "Name:           $NAME"
echo "Checkpoint:     $CKPT"
echo "Pred file:      $PRED_FILE"
echo "============================================"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi
if [ ! -f "$PRED_FILE" ]; then
    echo "ERROR: Prediction file not found: $PRED_FILE"
    exit 1
fi

python "${DANIODECIMA_REPO}/daniodecima-applications-main/notebooks/daniocell-ft/03_attributions/daniocell_attribution.py" \
    --name "$NAME" \
    --ckpt_path "$CKPT" \
    --pred_file "$PRED_FILE" \
    --device 0

echo "Done: $(date)"
