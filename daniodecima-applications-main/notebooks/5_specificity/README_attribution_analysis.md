# SLURM launcher: combined_attribution_analysis.sh

What it does
- Submits an array job (array 0–15 by default).
- Each array task selects one model entry from an in-script list (NAME|MODEL_DIR), finds the first epoch*.ckpt checkpoint, and runs the Python analysis.
- Uses one GPU per task with h100 or h200 constraints.
- Creates per-job Genomepy caches under scratch and writes logs to the experiments/logs directory.

Key resources requested per task
- 1 GPU, 4 CPUs, 100 GB RAM, 4 hours walltime.

Required files and paths
- The PAIRS array inside the script maps a human-readable name to a model directory (the directory containing a checkpoints subfolder and the model-specific data_out_*.h5ad file produced by the prediction step).
- Python environment reachable via module load anaconda/latest and source activate pytorch (adjust if your cluster differs).
- Scratch locations for GENOMEPY_CACHE_DIR, GENOMEPY_CONFIG_DIR, WANDB_CACHE_DIR.

Outputs
- Attribution HDF5 and a summary pickle written next to each MODEL_DIR.
- SLURM stdout and stderr per task in the experiments/logs folder.

How to adapt
- Edit the PAIRS list to point to your checkpoints.
- Adjust the SLURM resources or CUDA constraint as needed.
- Change PYTHON_SCRIPT if you move the Python file.

# Python script: 00_combined_attribution_analysis.py

Purpose
- Loads a Decima checkpoint and the corresponding model directory.
- Computes Input × Gradient attributions over 524,288 bp windows for genes present in the model-specific data_out_*.h5ad (produced by your decima_predictions.py step).
- Aggregates attributions per genomic region and reports summary statistics.

Inputs
- name: an identifier used in output filenames.
- ckpt_path: path to the epoch*.ckpt file.
- model_dir: path to the Lightning run directory that also contains the model-specific data_out_*.h5ad file.
- device: CUDA device index (default 0).
- External data used inside the script:
  – HDF5 sequence/target file at …/data/celltypes_chrom_split_v1/data.h5.
  – ATAC peaks BED file at …/TDR118reseq/outs/atac_peaks.bed.
  – GTF at …/Danio_rerio.GRCz11.113.gtf.

What it computes
- Builds a Decima LightningModel, loads weights and hyperparameters from the checkpoint.
- Loads the model-specific AnnData file (filters to var.dataset == "test").
- For each gene:
  - Builds the task list and extracts the 524,288 bp sequence window via extract_gene_data.
  - Applies an Aggregate transform to average across selected tasks.
  - Computes Input × Gradient attribution on the first 4 channels, averages models (single model by default), sums nucleotide channels to a length-524,288 vector.
  - Strand-corrects and stores absolute attribution.
- Overlaps gene windows with ATAC-derived candidate CREs using bioframe.
- Splits attribution into regions: promoter, exons, introns, exon–intron junctions, CRE vs non-CRE in those regions, and flanking distance bins (0–1 kb, 1–10 kb, 10–100 kb, ≥100 kb).
- Records mean attribution per region per gene.

Outputs
- An HDF5 file of raw per-gene attribution vectors: written as NAME-attr-th05-all.h5 in model_dir.
- A pickle file with a genes DataFrame of regional attribution summaries: NAME_fulltestset_cres_all.pkl in model_dir.
- Console prints of data shapes and progress.

Dependencies
- torch, numpy, pandas, anndata, h5py, tqdm, bioframe, scipy.
- captum for Input × Gradient.
- Local Decima modules: LightningModel, interpret.extract_gene_data, grelu.transforms.prediction_transforms.Aggregate, genome.read_gtf.
- A CUDA-enabled environment when device ≥ 0.

Assumptions and tips
- The script expects the model_dir to contain the predictions file produced earlier (data_out_… .h5ad).
- If your annotation or peak files live elsewhere, update the hard-coded paths near the bottom of the script (ATAC BED and GTF).
- You can change the promoter window, junction window, and distance bins inside the analysis section to fit your biology.
- For multi-checkpoint ensembles, you could pass multiple ckpts and loop, then average; the scaffolding for multiple models is already present.