#!/bin/bash
#SBATCH --job-name=daniocell_eval
#SBATCH --output=/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/experiments/logs/daniocell_eval_%j.out
#SBATCH --error=/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/experiments/logs/daniocell_eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ---------------------------------------------------------------------------
# DanioCell Decima — Evaluation metrics and figures (Phase 2)
# CPU-only job. Run after all Phase 1 prediction jobs complete.
# ---------------------------------------------------------------------------

module load anaconda
source activate gReLu

python /hpc/projects/data.science/yangjoon.kim/step/decima-applications-main/notebooks/daniocell-ft/02_evaluation/daniocell_evaluate.py

echo "Done: $(date)"
