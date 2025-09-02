#!/bin/bash

# Base configuration
BASE_DIR="/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima"
LOG_BASE="${BASE_DIR}/experiments"
DATA_DIR="/hpc/projects/data.science/yangjoon.kim/zebrafish-seq2func-data/celltypes_chrom_split_v1/"
SCRIPT_DIR="/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-main/scripts"

# Create base directories
mkdir -p ${LOG_BASE}
mkdir -p ${LOG_BASE}/logs

# Experiment timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_SET="decima_experiments_${TIMESTAMP}"
LOG_DIR="${LOG_BASE}/${EXPERIMENT_SET}"
mkdir -p ${LOG_DIR}

echo "Submitting all Decima experiments as SLURM array job"
echo "Experiment plan (16 total experiments):"
echo "  [0-3]   Human-Borzoi pretrained (replicates 0-3, lr=3e-5, seed=42)"
echo "  [4-7]   Human-Decima pretrained (replicates 0-3, lr=3e-5, seed=42)"  
echo "  [8-11]  Mouse-Borzoi pretrained (replicates 0-3, lr=3e-5, seed=42)"
echo "  [12-15] Random init (lr=3e-6, seeds=42,43,44,45)"
echo ""
echo "Results will be in: ${LOG_DIR}"

# Submit single SLURM array job for all experiments
JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=decima_finetune_array
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --mem=100G
#SBATCH --time=60:00:00
#SBATCH --array=0-15
#SBATCH --output=${LOG_BASE}/logs/decima_exp_%A_%a.out
#SBATCH --error=${LOG_BASE}/logs/decima_exp_%A_%a.err

# Load environment
module load anaconda/latest
source activate pytorch

# Set up environment - ALL in scratch space, not local home
export GENOMEPY_CACHE_DIR="${BASE_DIR}/genomepy_cache/job_\${SLURM_ARRAY_JOB_ID}_task_\${SLURM_ARRAY_TASK_ID}"
export GENOMEPY_CONFIG_DIR="${BASE_DIR}/genomepy_config/job_\${SLURM_ARRAY_JOB_ID}_task_\${SLURM_ARRAY_TASK_ID}"
export WANDB_CACHE_DIR="${BASE_DIR}/wandb_cache"
export CUDA_VISIBLE_DEVICES=0

# Create directories in scratch space
mkdir -p \$GENOMEPY_CACHE_DIR \$GENOMEPY_CONFIG_DIR \$WANDB_CACHE_DIR

echo "Starting experiment \${SLURM_ARRAY_TASK_ID}/15"
echo "Host: \${HOSTNAME}"
echo "Job ID: \${SLURM_ARRAY_JOB_ID}"
echo "Task ID: \${SLURM_ARRAY_TASK_ID}"
echo "Genomepy cache: \$GENOMEPY_CACHE_DIR"
echo "Genomepy config: \$GENOMEPY_CONFIG_DIR"
echo "WandB cache: \$WANDB_CACHE_DIR"
echo ""

# Run the training
python ${SCRIPT_DIR}/decima_finetune.py --experiment_id \${SLURM_ARRAY_TASK_ID} --data_dir ${DATA_DIR} --log_dir ${LOG_DIR}

# Cleanup genomepy temp directories (but keep wandb cache)
rm -rf \$GENOMEPY_CACHE_DIR \$GENOMEPY_CONFIG_DIR

echo "Completed experiment \${SLURM_ARRAY_TASK_ID}"
EOF
)

echo "Submitted SLURM array job: ${JOB_ID}"
echo "Monitor with: squeue -j ${JOB_ID}"
echo "Or monitor all: squeue -u \$USER"

# Create experiment mapping file
cat > ${LOG_DIR}/experiment_mapping.txt <<EOF
Decima Experiments Array Job - ${TIMESTAMP}
==========================================

Job ID: ${JOB_ID}
Total Experiments: 16

Storage Locations:
- Base directory: ${BASE_DIR}
- Results: ${LOG_DIR}
- Job logs: ${LOG_BASE}/logs/
- Genomepy cache: ${BASE_DIR}/genomepy_cache/
- WandB cache: ${BASE_DIR}/wandb_cache/

Experiment Mapping:
[0]  Human-Borzoi pretrained, replicate=0, lr=3e-5, seed=42
[1]  Human-Borzoi pretrained, replicate=1, lr=3e-5, seed=42
[2]  Human-Borzoi pretrained, replicate=2, lr=3e-5, seed=42
[3]  Human-Borzoi pretrained, replicate=3, lr=3e-5, seed=42
[4]  Human-Decima pretrained, replicate=0, lr=3e-5, seed=42
[5]  Human-Decima pretrained, replicate=1, lr=3e-5, seed=42
[6]  Human-Decima pretrained, replicate=2, lr=3e-5, seed=42
[7]  Human-Decima pretrained, replicate=3, lr=3e-5, seed=42
[8]  Mouse-Borzoi pretrained, replicate=0, lr=3e-5, seed=42
[9]  Mouse-Borzoi pretrained, replicate=1, lr=3e-5, seed=42
[10] Mouse-Borzoi pretrained, replicate=2, lr=3e-5, seed=42
[11] Mouse-Borzoi pretrained, replicate=3, lr=3e-5, seed=42
[12] Random init, lr=3e-6, seed=42
[13] Random init, lr=3e-6, seed=43
[14] Random init, lr=3e-6, seed=44
[15] Random init, lr=3e-6, seed=45

Data Directory: ${DATA_DIR}
Submitted at: $(date)

Commands to monitor:
- squeue -j ${JOB_ID}
- ls ${LOG_DIR}/
- tail -f ${LOG_BASE}/logs/decima_exp_${JOB_ID}_*.out
EOF

echo "Experiment mapping saved to: ${LOG_DIR}/experiment_mapping.txt"
echo ""
echo "Useful commands:"
echo "   Monitor jobs: squeue -j ${JOB_ID}"
echo "   Check results: ls ${LOG_DIR}/"
echo "   Follow logs: tail -f ${LOG_BASE}/logs/decima_exp_${JOB_ID}_*.out"
echo ""
echo "Storage locations:"
echo "   Base: ${BASE_DIR}"
echo "   Results: ${LOG_DIR}"
echo "   Genomepy cache: ${BASE_DIR}/genomepy_cache/"
echo "   WandB cache: ${BASE_DIR}/wandb_cache/"