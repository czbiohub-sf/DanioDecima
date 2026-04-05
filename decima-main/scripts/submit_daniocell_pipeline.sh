#!/bin/bash
# =============================================================================
# DanioCell data prep pipeline — chained SLURM jobs
#
# Phases:
#   1. Pseudobulk          (Phase 1 — large memory, ~4h)
#   2. Aggregate & QC      (Phase 2 — small, fast)
#   3. Intervals + frac_N  (Phase 3 — FASTA reads, ~4-8h)
#   4. Chrom split         (Phase 4 — fast)
#   5. HDF5 generation     (Phase 5 — sequence encoding, ~8-12h)
#
# Each phase checks for existing outputs and skips if already done (resume).
# Pass --force to a specific Python script to redo that step.
#
# Usage:
#   bash submit_daniocell_pipeline.sh           # submit all phases
#   bash submit_daniocell_pipeline.sh --start 3 # resume from Phase 3
# =============================================================================

set -eo pipefail

START_PHASE=1
if [[ "${1:-}" == "--start" && -n "${2:-}" ]]; then
    START_PHASE=$2
fi

BASE_DIR="/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell"
PROJECTS_DIR="/hpc/projects/data.science/yangjoon.kim/step"
NB_DIR="${PROJECTS_DIR}/decima-applications-main/notebooks"
LOG_DIR="${BASE_DIR}/pipeline_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${LOG_DIR}
mkdir -p ${BASE_DIR}/celltypes_chrom_split_v1

ENV_CMD="module load anaconda && source activate pytorch"

echo "====================================="
echo "DanioCell Pipeline Submission"
echo "Start phase: ${START_PHASE}"
echo "Logs: ${LOG_DIR}/"
echo "====================================="

# Helper: submit a job and return job ID
# Usage: submit_job <name> <mem> <time> <cpus> <dep_arg> <python_script>
submit_job() {
    local name=$1
    local mem=$2
    local time=$3
    local cpus=$4
    local dep_arg=$5
    local script=$6

    sbatch --parsable \
        ${dep_arg:+--dependency=afterok:${dep_arg}} \
        --job-name=${name} \
        --partition=gpu \
        --gres=gpu:1 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=${cpus} \
        --mem=${mem} \
        --time=${time} \
        --output=${LOG_DIR}/${name}_%j.out \
        --error=${LOG_DIR}/${name}_%j.err \
        --mail-type=FAIL,END \
        --mail-user=yang-joon.kim@czbiohub.org \
        --wrap="${ENV_CMD} && python ${script}"
}

PREV_JOB=""

# ------------------------------------------------------------------
# Phase 1: Pseudobulk (load 489K × 36K h5ad, aggregate)
# ------------------------------------------------------------------
if [[ ${START_PHASE} -le 1 ]]; then
    JOB1=$(submit_job \
        "dc_p1_prep" "120G" "6:00:00" "8" "" \
        "${NB_DIR}/0_sc_data_prep/daniocell-prep.py")
    PREV_JOB=${JOB1}
    echo "Phase 1 submitted: ${JOB1}"
else
    echo "Phase 1: skipped (--start ${START_PHASE})"
fi

# ------------------------------------------------------------------
# Phase 2: Aggregate & QC (small pseudobulk, fast)
# ------------------------------------------------------------------
if [[ ${START_PHASE} -le 2 ]]; then
    JOB2=$(submit_job \
        "dc_p2_agg" "30G" "2:00:00" "4" "${PREV_JOB}" \
        "${NB_DIR}/1_processing/06_aggregate_daniocell.py")
    PREV_JOB=${JOB2}
    echo "Phase 2 submitted: ${JOB2}"
else
    echo "Phase 2: skipped (--start ${START_PHASE})"
fi

# ------------------------------------------------------------------
# Phase 3: Intervals + frac_N (sequential FASTA reads — can be slow)
# ------------------------------------------------------------------
if [[ ${START_PHASE} -le 3 ]]; then
    JOB3=$(submit_job \
        "dc_p3_intervals" "30G" "8:00:00" "4" "${PREV_JOB}" \
        "${NB_DIR}/2_dataset/01_make_intervals_daniocell.py")
    PREV_JOB=${JOB3}
    echo "Phase 3 submitted: ${JOB3}"
else
    echo "Phase 3: skipped (--start ${START_PHASE})"
fi

# ------------------------------------------------------------------
# Phase 4: Chromosome split (fast)
# ------------------------------------------------------------------
if [[ ${START_PHASE} -le 4 ]]; then
    JOB4=$(submit_job \
        "dc_p4_split" "20G" "1:00:00" "2" "${PREV_JOB}" \
        "${NB_DIR}/2_dataset/02_split_daniocell.py")
    PREV_JOB=${JOB4}
    echo "Phase 4 submitted: ${JOB4}"
else
    echo "Phase 4: skipped (--start ${START_PHASE})"
fi

# ------------------------------------------------------------------
# Phase 5: HDF5 generation (encodes 524K bp × N genes from FASTA)
# ------------------------------------------------------------------
if [[ ${START_PHASE} -le 5 ]]; then
    JOB5=$(submit_job \
        "dc_p5_h5" "100G" "12:00:00" "8" "${PREV_JOB}" \
        "${NB_DIR}/2_dataset/03_make_h5_daniocell.py")
    PREV_JOB=${JOB5}
    echo "Phase 5 submitted: ${JOB5}"
else
    echo "Phase 5: skipped (--start ${START_PHASE})"
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "====================================="
echo "Pipeline submitted successfully."
echo ""
echo "Monitoring:"
echo "  squeue -u \$USER"
echo "  watch -n 30 squeue -u \$USER"
echo ""
echo "Logs:"
echo "  ${LOG_DIR}/"
echo ""
echo "To resume from a specific phase after failure:"
echo "  bash submit_daniocell_pipeline.sh --start 3   # resume from Phase 3"
echo "====================================="
