# Decima Fine-Tuning Experiment Pipeline

This repository runs a matrix of Zebrahub fine-tuning experiments for Decima-style models on a SLURM GPU cluster. It compares pretrained initializations (human/mouse; Borzoi/Decima) against random-init baselines and logs configs + metrics.

# Contents

- submit_decima_finetune_array.sh — launches one SLURM array job (16 tasks), sets directories and caches, routes logs.

- decima_finetune.py — runs one experiment selected by option “--experiment_id”; loads data, builds the model, trains, and saves metrics/config.

# Experiment matrix (16 tasks)

- IDs 0–3: pretrained, source = wandb-human, replicates 0–3, lr 3e-5, seed 42
- IDs 4–7: pretrained, source = decima-human, replicates 0–3, lr 3e-5, seed 42
- IDs 8–11: pretrained, source = wandb-mouse, replicates 0–3, lr 3e-5, seed 42
- IDs 12–15: random init, lr 3e-6, seeds 42–45

Each run gets a timestamped directory under:

- /hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_YYYYMMDD_HHMMSS/

An experiment_mapping.txt is written there for quick reference.

# Data requirements

Inside the data directory specified by “--data_dir”, the Python script expects:

- zebrahub_aggregated.h5ad (AnnData matrix with tasks/metadata)

- data.h5 (HDF5 with sequences/targets; dataset keys: train and val)

If your filenames differ, edit the variables “matrix_file” and “h5_file” in decima_finetune.py.

# How to run

1.  Edit paths at the top of submit_decima_finetune_array.sh:

- BASE_DIR = /hpc/scratch/group.data.science/mathias.voges/zebrahub-decima

- DATA_DIR = /hpc/projects/data.science/yangjoon.kim/zebrafish-seq2func-data/celltypes_chrom_split_v1/

- SCRIPT_DIR = /hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-main/scripts

2.  Submit the array job:

- Run: bash submit_decima_finetune_array.sh

3.  Monitor progress:

- squeue -j JOB_ID

- tail -f /hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/logs/decima_exp_JOBID_*.out

# Outputs

Per experiment (under the run’s log directory):

- experiment_info.json — configuration and final metrics

- TensorBoard logs — files named events.out.tfevents.*

- Optional model checkpoints (if enabled by your Lightning code)

SLURM stdout/err logs:

- /hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/logs/

# Run a single experiment (without SLURM)

Example:

- python decima_finetune.py --experiment_id 3 --data_dir /path/to/data --log_dir /path/to/output/logs

IDs map to the matrix above (0–15).

# Customization

- In decima_finetune.py, edit the list “experiments = []” to change learning rates, seeds, batch size, epochs, gradient clipping, loss weighting, or to add new pretrained sources.

- In the shell script, adjust the SBATCH resources to fit your cluster.

- If you use Weights & Biases, point the “wandb_project” config to your project (the model code reads it if implemented).

# SLURM resources (defaults used by the launcher)

- partition gpu

- one GPU (constraint h100 or h200)

- memory 100 GB

- CPUs per task 8

- time limit 60:00:00

Environment setup used by the launcher:

- module load anaconda/latest

- source activate pytorch

Adjust as needed for your system.

# Scratch behavior

- Uses directories under /hpc/scratch/... for genomepy and wandb caches.

- Creates per-task cache/config paths to avoid collisions.

- Cleans temp genomepy directories after training (wandb cache is kept).

# Requirements

- Set up the custom pytorch environment, as described:
  - decima-applications-main/notebooks/6_cell_states/README_analysis_TF-MoDISco.md
  - decima-applications-main/notebooks/6_cell_states/environment_pytorch.yml

- Python libraries: torch, pytorch-lightning, anndata, numpy, h5py, argparse

- Your local modules in src/decima/ (e.g., read_hdf5.py and lightning.py) should be importable from scripts/ via the relative path used in decima_finetune.py.

- SLURM cluster with GPU access.

# Troubleshooting

- ImportError for read_hdf5 or lightning → ensure src/decima/ exists relative to scripts/ and contains those modules.

- Data not found → confirm zebrahub_aggregated.h5ad and data.h5 exist in the data directory you pass.

- GPU or memory issues → reduce batch size, or raise the SBATCH memory; mixed precision is enabled on GPU by default in the script.