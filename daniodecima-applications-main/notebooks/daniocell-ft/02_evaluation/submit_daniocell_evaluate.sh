#!/bin/bash
#SBATCH --job-name=daniocell_eval
#SBATCH --output=logs/daniocell_eval_%j.out
#SBATCH --error=logs/daniocell_eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
: "${DANIODECIMA_REPO:?set it first: source daniodecima-main/scripts/daniocell_paths.env}"

# ---------------------------------------------------------------------------
# DanioCell Decima — Evaluation metrics and figures (Phase 2)
# CPU-only job. Run after all Phase 1 prediction jobs complete.
# ---------------------------------------------------------------------------

module load anaconda
source activate gReLu

python "${DANIODECIMA_REPO}/daniodecima-applications-main/notebooks/daniocell-ft/02_evaluation/daniocell_evaluate.py"

echo "Done: $(date)"
