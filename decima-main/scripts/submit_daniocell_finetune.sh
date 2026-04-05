#!/bin/bash

# DanioCell Decima fine-tuning — SLURM array job submission
# Experiment matrix: 8 core experiments (expand to 0-11 for 3-stage transfer)
#   [0-3]  Human Decima pretrained → DanioCell (replicates 0-3, lr=3e-5, seed=42)
#   [4-7]  Random initialization   (seeds 42-45, lr=3e-6)
#   [8-11] 3-stage transfer (Zebrahub fine-tuned → DanioCell) — optional

BASE_DIR="/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell"
LOG_BASE="${BASE_DIR}/experiments"
DATA_DIR="${BASE_DIR}/celltypes_chrom_split_v1/"
SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/step/decima-main/scripts"

# Create base directories
mkdir -p ${LOG_BASE}
mkdir -p ${LOG_BASE}/logs

# Experiment timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_SET="daniocell_experiments_${TIMESTAMP}"
LOG_DIR="${LOG_BASE}/${EXPERIMENT_SET}"
mkdir -p ${LOG_DIR}

echo "Submitting DanioCell Decima fine-tuning experiments"
echo "Experiment plan (8 core + 4 optional = up to 12 total):"
echo "  [0-3]  Human-Decima pretrained (replicates 0-3, lr=3e-5, seed=42)"
echo "  [4-7]  Random init             (seeds 42-45, lr=3e-6)"
echo "  [8-11] 3-stage transfer (optional — submit separately once Zebrahub ckpts are ready)"
echo ""
echo "Results will be in: ${LOG_DIR}"

# Submit core 8 experiments (array=0-7)
# To also run 3-stage transfer, change to --array=0-11
JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=daniocell_finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --mem=100G
#SBATCH --time=60:00:00
#SBATCH --array=0-7
#SBATCH --output=${LOG_BASE}/logs/daniocell_exp_%A_%a.out
#SBATCH --error=${LOG_BASE}/logs/daniocell_exp_%A_%a.err
#SBATCH --mail-type=ERROR,END
#SBATCH --mail-user=yang-joon.kim@czbiohub.org

# Load environment
module load anaconda
source activate pytorch

# Environment setup — all in scratch
export GENOMEPY_CACHE_DIR="${BASE_DIR}/genomepy_cache/job_\${SLURM_ARRAY_JOB_ID}_task_\${SLURM_ARRAY_TASK_ID}"
export GENOMEPY_CONFIG_DIR="${BASE_DIR}/genomepy_config/job_\${SLURM_ARRAY_JOB_ID}_task_\${SLURM_ARRAY_TASK_ID}"
export WANDB_CACHE_DIR="${BASE_DIR}/wandb_cache"
export CUDA_VISIBLE_DEVICES=0

mkdir -p \$GENOMEPY_CACHE_DIR \$GENOMEPY_CONFIG_DIR \$WANDB_CACHE_DIR

echo "Starting DanioCell experiment \${SLURM_ARRAY_TASK_ID}/7"
echo "Host: \${HOSTNAME}"
echo "Job ID: \${SLURM_ARRAY_JOB_ID}"
echo "Task ID: \${SLURM_ARRAY_TASK_ID}"
echo ""

python ${SCRIPT_DIR}/daniocell_finetune.py \
    --experiment_id \${SLURM_ARRAY_TASK_ID} \
    --data_dir ${DATA_DIR} \
    --log_dir ${LOG_DIR} \
    --matrix_name daniocell_aggregated.h5ad \
    --scratch_dir ${BASE_DIR}

# Cleanup genomepy temp dirs (keep wandb cache)
rm -rf \$GENOMEPY_CACHE_DIR \$GENOMEPY_CONFIG_DIR

echo "Completed experiment \${SLURM_ARRAY_TASK_ID}"
EOF
)

echo "Submitted SLURM array job: ${JOB_ID}"
echo "Monitor with: squeue -j ${JOB_ID}"
echo "Or monitor all: squeue -u \$USER"

# Save experiment mapping
cat > ${LOG_DIR}/experiment_mapping.txt <<EOF
DanioCell Decima Fine-tuning — ${TIMESTAMP}
==========================================

Job ID: ${JOB_ID}
Core Experiments: 8 (array=0-7)

Storage Locations:
- Base directory:   ${BASE_DIR}
- Data directory:   ${DATA_DIR}
- Results:          ${LOG_DIR}
- Job logs:         ${LOG_BASE}/logs/
- Genomepy cache:   ${BASE_DIR}/genomepy_cache/
- WandB cache:      ${BASE_DIR}/wandb_cache/

Experiment Mapping:
[0]  Human-Decima pretrained, replicate=0, lr=3e-5, seed=42
[1]  Human-Decima pretrained, replicate=1, lr=3e-5, seed=42
[2]  Human-Decima pretrained, replicate=2, lr=3e-5, seed=42
[3]  Human-Decima pretrained, replicate=3, lr=3e-5, seed=42
[4]  Random init, lr=3e-6, seed=42
[5]  Random init, lr=3e-6, seed=43
[6]  Random init, lr=3e-6, seed=44
[7]  Random init, lr=3e-6, seed=45
[8]  3-stage transfer (Zebrahub→DanioCell), replicate=0  [optional]
[9]  3-stage transfer (Zebrahub→DanioCell), replicate=1  [optional]
[10] 3-stage transfer (Zebrahub→DanioCell), replicate=2  [optional]
[11] 3-stage transfer (Zebrahub→DanioCell), replicate=3  [optional]

Data Directory: ${DATA_DIR}
Submitted at: $(date)

Commands to monitor:
- squeue -j ${JOB_ID}
- ls ${LOG_DIR}/
- tail -f ${LOG_BASE}/logs/daniocell_exp_${JOB_ID}_*.out
EOF

echo "Experiment mapping saved to: ${LOG_DIR}/experiment_mapping.txt"
echo ""
echo "Useful commands:"
echo "   Monitor jobs:  squeue -j ${JOB_ID}"
echo "   Check results: ls ${LOG_DIR}/"
echo "   Follow logs:   tail -f ${LOG_BASE}/logs/daniocell_exp_${JOB_ID}_*.out"
echo ""
echo "Storage locations:"
echo "   Base:    ${BASE_DIR}"
echo "   Data:    ${DATA_DIR}"
echo "   Results: ${LOG_DIR}"
