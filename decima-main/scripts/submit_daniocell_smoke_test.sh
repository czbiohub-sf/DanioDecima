#!/bin/bash
# Submit DanioCell smoke test to a GPU node.
#
# Runs all 6 checks (or pass --skip_pretrained to skip WandB checks 3-4):
#   1. HDF5 file sanity
#   2. DataLoader batches
#   3. Model init (pretrained, decima-human)
#   4. Forward pass + gradient check
#   5. Model init (random)
#   6. 1-epoch mini-training (5 batches, full Lightning loop)
#
# Usage:
#   bash submit_daniocell_smoke_test.sh [--skip_pretrained]

BASE_DIR="/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell"
LOG_DIR="${BASE_DIR}/pipeline_logs"
SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/step/decima-main/scripts"

mkdir -p "${LOG_DIR}"

EXTRA_ARGS=""
if [[ "$1" == "--skip_pretrained" ]]; then
    EXTRA_ARGS="--skip_pretrained"
    echo "Note: skipping pretrained weight checks (--skip_pretrained)"
fi

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=dc_smoke_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --output=${LOG_DIR}/dc_smoke_test_%j.out
#SBATCH --error=${LOG_DIR}/dc_smoke_test_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yang-joon.kim@czbiohub.org

module load anaconda
source activate gReLu

export WANDB_CACHE_DIR="${BASE_DIR}/wandb_cache"
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
mkdir -p \$WANDB_CACHE_DIR

echo "Host:   \${HOSTNAME}"
echo "Job ID: \${SLURM_JOB_ID}"
echo "GPU:    \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unavailable')"
echo ""

python ${SCRIPT_DIR}/daniocell_smoke_test.py --n_batches 5 ${EXTRA_ARGS}

EXIT_CODE=\$?
if [ \$EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Smoke test PASSED — safe to submit full training array job:"
    echo "  bash ${SCRIPT_DIR}/submit_daniocell_finetune.sh"
else
    echo ""
    echo "Smoke test FAILED (exit code \$EXIT_CODE)"
    echo "Fix the issue before submitting full training."
fi
exit \$EXIT_CODE
EOF
)

echo "Submitted smoke test job: ${JOB_ID}"
echo "Monitor: squeue -j ${JOB_ID}"
echo "Logs:    tail -f ${LOG_DIR}/dc_smoke_test_${JOB_ID}.out"
