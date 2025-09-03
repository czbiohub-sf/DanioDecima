# SLURM Launcher: decima_predict.sh

This script:

- Launches an array job (--array=0-15) where each task corresponds to a checkpoint directory listed in ckpt_dirs_celltypes.txt.

- Sets up environment modules and activates a Python environment.

- Creates job-specific cache/config directories under scratch (GENOMEPY_CACHE_DIR, GENOMEPY_CONFIG_DIR, WANDB_CACHE_DIR).

- Reads the checkpoint directory for the given task ID.

- Locates the first epoch*.ckpt file inside that directory.

- Defines input data:

  - zebrahub_aggregated.h5ad (AnnData matrix with genes)

  - data.h5 (HDF5 dataset with sequences/targets)

- Defines the output path: writes predictions as an .h5ad file alongside the checkpoint.

- Calls the Python script (decima_predictions.py) with device assignment, checkpoint path, data files, and output file.


# Python Prediction Script: decima_predictions.py

This script:

 1. Arguments:

   - --device : GPU ID

   - --ckpts : checkpoint file(s)

   - --h5_file : sequence/target HDF5

   - --matrix_file : AnnData file (genes × cells/tracks)

   - --out_file : path to save predictions

   - --max_seq_shift : optional jitter for data augmentation

 2. Model loading:

   - Loads checkpoint weights and parameters from the .ckpt file.

   - Restores model hyperparameters.

   - Imports Decima code (read_hdf5, LightningModel) from ../src/decima/.

 3. Data:

   - Loads AnnData from .h5ad.

   - Builds an HDF5Dataset object using the .h5 file and AnnData metadata.

 4. Prediction:

   - Instantiates LightningModel with the loaded params.

   - Loads state_dict from checkpoint.

   - Runs .predict_on_dataset to compute predictions.

   - Stores predictions in ad.layers["preds"].

 5. Evaluation:

   - Computes Pearson correlation per gene (between observed values in ad.X and predictions).

   - Computes Pearson correlation with size factors.

   - Computes per-dataset pseudobulk correlations.

 6. Output:

   - Writes predictions and correlation metrics back to .h5ad (given by --out_file).


# Usage

- Checkpoints: Ensure ckpt_dirs_celltypes.txt has one line per model, formatted as model_id|checkpoint_directory.

- Output files: Written alongside checkpoint directories with names like:
  - data_out_decima_<experiment_name>_<date>_<model_id>.h5ad

- Parallelization: The array index (SLURM_ARRAY_TASK_ID) determines which checkpoint directory is used.

- Metrics: Saved directly in the AnnData object for downstream analysis.