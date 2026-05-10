#!/bin/bash
#SBATCH --job-name=evolution_summary
#SBATCH --output=logs/evolution_summary_%j.out
#SBATCH --error=logs/evolution_summary_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load required modules
module load anaconda/latest
source activate pytorch

# Set environment variables  
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Define paths
RESULTS_DIR="/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/design/analysis_results_bg_max_16hpf_20250610_combined"
OUTPUT_DIR="$RESULTS_DIR/comprehensive_summary"

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "Starting comprehensive evolution analysis..."
echo "Results directory: $RESULTS_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run the comprehensive analysis
python /hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/9_design/02_summarize_results.py \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_motif_pval 0.05 \
    --min_ism_weight 0.0

echo "Comprehensive analysis completed at $(date)"
echo "Check outputs in: $OUTPUT_DIR"