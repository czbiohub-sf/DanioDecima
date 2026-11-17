#!/bin/bash
#SBATCH --job-name=daniocell_spec
#SBATCH --output=logs/daniocell_spec_%A_%a.out
#SBATCH --error=logs/daniocell_spec_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --array=0-9
: "${DANIODECIMA_REPO:?set it first: source daniodecima-main/scripts/daniocell_paths.env}"
: "${DANIOCELL_WORKDIR:?set it first: source daniodecima-main/scripts/daniocell_paths.env}"

# ---------------------------------------------------------------------------
# DanioCell Decima — Cell-type specificity attributions (Phase 4)
#
# Array 0-9: 10 focal cell types, using best pretrained replicate
#
# Cell types:
#   0: neural crest    5: somite
#   1: neurons         6: cardiac muscle
#   2: motor neurons   7: intestine
#   3: radial glia     8: liver
#   4: notochord       9: epidermis
# ---------------------------------------------------------------------------

module load anaconda
source activate gReLu

# Use best pretrained replicate (update after Phase 2 evaluation)
CKPT="${DANIOCELL_WORKDIR}/experiments/daniocell_experiments_20260406_114453/pretrained_decima-human_rep0_lr3e-05_seed42/version_0/checkpoints/epoch=8-step=7263.ckpt"
PRED_FILE="${DANIOCELL_WORKDIR}/01_predictions/data_out_daniocell_pretrained_decima-human_rep0_lr3e-05_seed42.h5ad"

echo "============================================"
echo "Array task ID:  $SLURM_ARRAY_TASK_ID"
echo "Checkpoint:     $CKPT"
echo "============================================"

python "${DANIODECIMA_REPO}/daniodecima-applications-main/notebooks/daniocell-ft/04_specificity/daniocell_specificity.py" \
    --cell_type_id "$SLURM_ARRAY_TASK_ID" \
    --ckpt_path "$CKPT" \
    --pred_file "$PRED_FILE" \
    --device 0 \
    --n_genes 50

echo "Done: $(date)"
