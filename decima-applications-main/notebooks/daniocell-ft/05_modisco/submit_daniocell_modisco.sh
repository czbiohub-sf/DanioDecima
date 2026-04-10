#!/bin/bash
#SBATCH --job-name=daniocell_modisco
#SBATCH --output=/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/experiments/logs/daniocell_modisco_%A_%a.out
#SBATCH --error=/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/experiments/logs/daniocell_modisco_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-9

# ---------------------------------------------------------------------------
# DanioCell Decima — MoDISco motif discovery (Phase 5)
#
# Array 0-9: 10 focal cell types (from Phase 4 output)
# CPU-only, 8 CPUs, 64 GB
# ---------------------------------------------------------------------------

module load anaconda
source activate gReLu

SPEC_BASE="/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/04_specificity"
EXP_PREFIX="pretrained_decima-human_rep0_lr3e-05_seed42"
OUT_BASE="/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/05_modisco"
MEME_FILE="/home/yang-joon.kim/.conda/envs/gReLu/lib/python3.10/site-packages/grelu/resources/meme/H12CORE_meme_format.meme"

# Cell type names (must match Phase 4 output directory suffixes exactly)
declare -a CELL_TYPES=(
    "neural_crest"
    "neurons"
    "motor_neurons"
    "radial_glia"
    "notochord"
    "somite"
    "cardiac_muscle"
    "intestine"
    "liver"
    "epidermis"
)

CELL_TYPE="${CELL_TYPES[$SLURM_ARRAY_TASK_ID]}"
CELL_DIR="${SPEC_BASE}/${EXP_PREFIX}_${CELL_TYPE}"

if [ ! -d "$CELL_DIR" ]; then
    echo "ERROR: Directory not found: $CELL_DIR"
    exit 1
fi

echo "============================================"
echo "Array task ID:  $SLURM_ARRAY_TASK_ID"
echo "Cell type:      $CELL_TYPE"
echo "Input dir:      $CELL_DIR"
echo "============================================"

python /hpc/projects/data.science/yangjoon.kim/step/decima-applications-main/notebooks/daniocell-ft/05_modisco/daniocell_modisco.py \
    --cell_type_dir "$CELL_DIR" \
    --meme_file "$MEME_FILE" \
    --out_dir "${OUT_BASE}/${CELL_TYPE}"

echo "Done: $(date)"
