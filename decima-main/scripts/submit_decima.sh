#!/bin/bash
#SBATCH --job-name=decima
#SBATCH --partition=gpu            # Partition (queue) name
#SBATCH --nodes=1                    # Request 4 nodes
#SBATCH --ntasks=1                   # Request 4 tasks
#SBATCH --gres=gpu:4                # Request 2 GPUs (of any type)
#SBATCH --constraint="h100|h200"      # Only nodes with GPUs labeled h100 OR h200
#SBATCH --cpus-per-task=8          # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=100G                   # Total memory per node (adjust as needed)
#SBATCH --time=12:00:00              # Time limit (2 days)
#SBATCH --output=logs/decima_finetune_%A_%a.out
#SBATCH --error=logs/decima_finetune_%A_%a.err

# Load necessary modules (adjust based on your cluster setup)
module load anaconda/latest

# Activate your conda environment
source activate pytorch  # Replace with your environment name

# Set environment variables for PyTorch DDP
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Set variables
scripts_dir="/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-main/scripts"
save_dir="/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/data/"
lr=3e-5
weight=1e-4
grad=5
bs=2
rep=0

# Use SLURM_ARRAY_TASK_ID as the replicate number
name="decima_v20250319_pretrained_rep${rep}_forecast_horizon_5_lstm"

# Run the fine-tuning script
python ${scripts_dir}/finetune.py \
    --name ${name} \
    --dir ${save_dir} \
    --lr ${lr} \
    --weight ${weight} \
    --grad ${grad} \
    --replicate ${rep} \
    --bs ${bs} \
    --init_mode pretrained \
    --pretrained_source wandb \
    --wandb_project grelu/borzoi \
    --checkpoint_path None