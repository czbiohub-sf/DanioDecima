# Given an hdf5 file created by write_hdf5.py, make predictions for all the genes

import numpy as np
import anndata
import os, sys
import torch
import argparse

src_dir = f'{os.path.dirname(__file__)}/../src/decima/'
sys.path.insert(0, src_dir)  # before installed lightning pkg

from read_hdf5 import GeneForecastDataset, list_genes
from lightning_temporal import JointLightningModel


parser = argparse.ArgumentParser()
parser.add_argument("--device", 
                    help="which gpu to use",
                    type=int)
parser.add_argument("--ckpts", help="Path to the model checkpoint", nargs='+')
parser.add_argument("--h5_file", 
                    help="Path to h5 file indexed by genes")
parser.add_argument("--matrix_file", 
                    help="Path to h5ad file containing genes to predict")
parser.add_argument("--out_file", 
                    help="Output file path")
parser.add_argument("--max_seq_shift", 
                    help="Maximum jitter for augmentation", default=0, type=int)

args = parser.parse_args()


torch.set_float32_matmul_precision("medium")
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

print("Loading anndata")
ad = anndata.read_h5ad(args.matrix_file)
assert np.all(list_genes(args.h5_file, key=None) == ad.var_names.tolist())

print("Making test dataset")
ds = GeneForecastDataset(
    h5_file=args.h5_file,
    ad=ad,
    key="test",
    max_seq_shift=args.max_seq_shift,
    history_length=5,
    forecast_horizon=5
)

print("Loading models from checkpoint")
models = [JointLightningModel.load_from_checkpoint(f).eval() for f in args.ckpts]

print("Computing predictions")

preds = [model.predict_on_dataset(ds, devices=0, batch_size=6, num_workers=16) for model in models]
if isinstance(preds, list):
    preds = np.array(preds)

# Print original shape
print(f"Original predictions shape: {preds.shape}")

n_samples, forecast_horizon, n_cell_types = preds[0].shape
cell_type_indices = [0, 1, 2]  # Adjust these indices based on your data
for t in range(forecast_horizon):
    # Create an empty matrix with AnnData dimensions
    forecast_matrix = np.zeros((ad.n_obs, ad.n_vars))
    
    # Fill in the predictions for the the cell types
    for sample_idx in range(n_samples):
        gene_idx = sample_idx  # Assuming direct mapping
        
        # For each of the cell types
        for cell_type_idx in range(min(n_cell_types, len(cell_type_indices))):
            obs_idx = cell_type_indices[cell_type_idx]
            print(f"sample_idx: {sample_idx}, gene_idx: {gene_idx}, cell_type_idx: {cell_type_idx}")
            # Place the prediction in the right spot
            forecast_matrix[obs_idx, gene_idx] = preds[0][sample_idx, t, cell_type_idx]
    
    # Add to AnnData as a separate layer
    layer_name = f'pred_{t+1}'
    ad.layers[layer_name] = forecast_matrix
    print(f"Added forecast for time point {t+1} as layer '{layer_name}'")

print("Saved")
ad.write_h5ad(args.out_file)
