#!/bin/bash
#SBATCH --job-name=decima_predict
#SBATCH --output=/hpc/scratch/group.data.science/yang-joon.kim/zebrahub-decima/experiments/logs/decima_predict_%A_%a.out
#SBATCH --error=/hpc/scratch/group.data.science/yang-joon.kim/zebrahub-decima/experiments/logs/decima_predict_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"   # Only nodes with GPUs labeled h100 OR h200
#SBATCH --array=0-15   # <-- Set this to number of checkpoints minus 1


# Load Python environment
module load anaconda
conda activate pytorch

export GENOMEPY_CACHE_DIR="/hpc/scratch/group.data.science/yang-joon.kim/zebrahub-decima/genomepy_cache/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
export GENOMEPY_CONFIG_DIR="/hpc/scratch/group.data.science/yang-joon.kim/zebrahub-decima/genomepy_config/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
export WANDB_CACHE_DIR="/hpc/scratch/group.data.science/yang-joon.kim/zebrahub-decima/wandb_cache"

# Create directories if they don't exist
mkdir -p "$GENOMEPY_CACHE_DIR"
mkdir -p "$GENOMEPY_CONFIG_DIR"

# Get the checkpoint directory for this array task
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" ckpt_dirs_celltypes.txt)
# Parse the model identifier and checkpoint directory
IFS='|' read -r model_id ckpt_dir <<< "$line"

# Set up paths
save_dir="/hpc/projects/data.science/yangjoon.kim/zebrafish-seq2func-data/celltypes_chrom_split_v1/"
matrix_file="${save_dir}zebrahub_aggregated.h5ad"
h5_file="${save_dir}data.h5"

# Find checkpoint files
ckpts=$(find "$ckpt_dir" -name 'epoch*.ckpt' | head -1)
if [ -z "$ckpts" ]; then
    echo "No checkpoint files found in $ckpt_dir/checkpoints"
    exit 1
fi

echo "Found checkpoint files:"
echo "$ckpts"

# Example: /path/to/decima_experiments_20250617_225114/pretrained_decima-human_rep0_lr3e-05_seed42/version_0
experiment_name=$(basename "$(dirname "$ckpt_dir")")  # Gets: pretrained_decima-human_rep0_lr3e-05_seed42
today=$(date +%Y%m%d)

# Set output file
out_file="/hpc/scratch/group.data.science/yang-joon.kim/zebrahub-decima/experiments/data_out_decima_${experiment_name}_${today}_${model_id}.h5ad"
# out_file="${ckpt_dir}/data_out_decima_${experiment_name}_${today}_${model_id}.h5ad"
echo "Output file will be: $out_file"

python /hpc/projects/data.science/yangjoon.kim/step/daniodecima-main/scripts/decima_predictions.py \
  --device 0 --ckpts $ckpts --h5_file $h5_file --matrix_file $matrix_file --out_file $out_file --max_seq_shift 3