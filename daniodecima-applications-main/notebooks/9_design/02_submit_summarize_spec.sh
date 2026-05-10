#!/bin/bash
#SBATCH --job-name=enhanced_evolution_summary
#SBATCH --output=logs/enhanced_evolution_summary_%j.out
#SBATCH --error=logs/enhanced_evolution_summary_%j.err
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
RESULTS_DIR="/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/design/analysis_results_bg_mean_16hpf_20250612"
OUTPUT_DIR="$RESULTS_DIR/enhanced_comprehensive_summary"

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "Starting enhanced comprehensive evolution analysis..."
echo "Results directory: $RESULTS_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run the enhanced analysis with ISM weight percentiles and specificity filtering
python /hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/9_design/02_summarize_results_spec.py \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --min_specificity 0.5 \
    --ism_method percentile \
    --ism_percentile 75

echo "Enhanced comprehensive analysis completed at $(date)"
echo "Check outputs in: $OUTPUT_DIR"