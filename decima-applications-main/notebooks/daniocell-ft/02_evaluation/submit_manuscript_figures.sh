#!/bin/bash
#SBATCH --job-name=daniocell-manuscript-figs
#SBATCH --output=/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/02_evaluation/manuscript_figures_%j.log
#SBATCH --error=/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/02_evaluation/manuscript_figures_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

module load anaconda
source activate gReLu

SCRIPT_DIR="/hpc/projects/data.science/yangjoon.kim/step/decima-applications-main/notebooks/daniocell-ft/02_evaluation"

echo "Starting manuscript figure generation at $(date)"
echo "Script: ${SCRIPT_DIR}/daniocell_manuscript_figures.py"

python "${SCRIPT_DIR}/daniocell_manuscript_figures.py"

echo "Finished at $(date)"
