#!/bin/bash
#SBATCH --job-name=daniocell-manuscript-figs
#SBATCH --output=logs/manuscript_figures_%j.log
#SBATCH --error=logs/manuscript_figures_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
: "${DANIODECIMA_REPO:?set it first: source daniodecima-main/scripts/daniocell_paths.env}"

module load anaconda
source activate gReLu

SCRIPT_DIR="${DANIODECIMA_REPO}/daniodecima-applications-main/notebooks/daniocell-ft/02_evaluation"

echo "Starting manuscript figure generation at $(date)"
echo "Script: ${SCRIPT_DIR}/daniocell_manuscript_figures.py"

python "${SCRIPT_DIR}/daniocell_manuscript_figures.py"

echo "Finished at $(date)"
