# SLURM launcher: evolve_design_pipeline.sh

What it does
- Submits an array job with indices 0–399 (16 models × 25 cell types).
- Maps each array index to a model directory and a target cell type.
- Prepares clean, job-scoped Genomepy caches in TMPDIR before loading modules (prevents stale cache issues).
- Loads the Python environment, sets CUDA device, thread count, and WANDB cache.
- Creates per-run output folders under the design/ directory: one for evolved sequences and one for downstream analysis.
- Step 1: calls 00_evolve_combined.py to evolve a promoter element for the chosen model and target cell type.
- Step 2: if evolution succeeds and produces the expected CSV file, calls 1_read_celltypes.py to evaluate the evolved element within a fixed genomic window (chromosome 4, positions defined by tss_start, window_size, tss_offset).
- Writes a concise pipeline summary (model label, cell type, exit codes) at the end of each task.

Key SLURM resources per task
- 1 GPU with constraint h100 or h200, 4 CPUs, 100 GB RAM, 8 hours walltime.
- Logs written to the design/logs directory under zebrahub-decima.

Inputs and configuration inside the script
- MODEL_DIRS: 16 Lightning run directories (4 human-Decima, 4 human-Borzoi, 4 mouse-Borzoi, 4 random). Each must contain a checkpoints subfolder with epoch*.ckpt and the model-specific outputs from prediction steps.
- CELL_TYPES: list of 25 active targets (adaxial cell, brain, …, trigeminal placode).
- DATA_DIR: base data folder for zebrahub-decima celltypes_chrom_split_v1.
- The script computes model_idx and celltype_idx from the array index so every (model, celltype) pair is covered once.

Outputs
- EVOLUTION_OUTPUT_DIR: files named evolved_promoter_{ModelInfo}seed42{Celltype}_{Timepoint}_single.csv containing the per-round mutation trace and specificity trajectory.
- ANALYSIS_OUTPUT_DIR: outputs from 1_read_celltypes.py summarizing predicted effects in the genomic window.
- SLURM stdout and stderr per task in design/logs.

Operational tips
- GENOMEPY_CACHE_DIR and GENOMEPY_CONFIG_DIR are set before module load to avoid corrupt cache reuse; the script also removes any user-level genomepy caches at startup.
- If you change the number of cell types or models, update the SBATCH array range and the CELL_TYPES or MODEL_DIRS lists to keep indices consistent.
- Adjust timepoint, rounds, promoter length, and early-stopping thresholds in the evolution call near “STEP 1: RUNNING EVOLUTION…”.

# Python evolution script: 00_evolve_combined.py

Purpose
- For one model and one target cell type, perform simple directed evolution of a short promoter element placed near a fixed TSS within a 524,288 bp window. The objective is to maximize specificity defined as mean prediction on target cell types minus mean prediction on background cell types at the same timepoint.

Main inputs (CLI flags)
- model_dir: path to a Lightning run directory; the script finds the epoch checkpoint automatically.
- data_dir: directory containing model-specific data_out files or the zebrahub aggregated AnnData.
- output_dir: where the CSV trace is written.
- target_celltype and timepoint: define the positive set.
- device: GPU index.
- rounds, sequence_length: evolution length and promoter size (default 50 rounds and 200 bp).
- early_stopping_patience, min_improvement: stop criteria based on recent specificity improvements.
- genomic anchors chrom, tss_start, window_size, tss_offset: define the fixed sequence context.

What it does internally
- Loads the LightningModel from the checkpoint and freezes parameters (eval mode, no grads).
- Loads model-specific tasks metadata (from checkpoint hyperparameters) to identify target vs background cell types at the requested timepoint.
- Builds the full genomic window using grelu sequence utilities and inserts an initial random promoter element plus a constant cargo sequence (EBFP) at the TSS offset.
- Iterates over rounds; in each round scans all single-base substitutions across the promoter element (A/T/G/C excluding the current base), predicts with the model, and selects the mutation with the largest improvement in target-minus-background mean.
- Records the best mutation each round into a CSV with columns: Round, Position (relative), Base, Specificity, Target_Mean, Background_Mean, Current_Element, Time_Elapsed, Convergence_Status.
- Applies early stopping when the average specificity gain across recent rounds falls below the threshold.
- Prints timing and convergence information and empties CUDA cache periodically to keep memory stable.

Outputs
- A CSV file named evolved_promoter_{ModelInfo}seed{Seed}{Celltype}_{Timepoint}_single.csv in output_dir containing the full mutation trajectory and scores.
- Console logs with per-round summary and final specificity.

Dependencies
- torch, numpy, pandas, anndata, tqdm, argparse; grelu sequence utilities (strings_to_one_hot, intervals_to_strings, mutate).
- Local Decima modules via the decima-main src/decima path (LightningModel).
- CUDA for acceleration when device ≥ 0.

Assumptions and notes
- ModelInfo (Human_Decima_n, Human_Borzoi_n, Mouse_Borzoi_n, Random_n) is inferred from model_dir naming; adjust extract_model_info if your directory pattern changes.
- The specificity objective is a simple difference in means; you can swap in your own objective (e.g., log fold-change, contrastive margin) inside directed_evolution_single.
- The promoter element is evolved in place at tss_offset; cargo sequence remains constant.
- If your AnnData or task metadata schema differs, update filter_celltypes and the way task indices are derived.