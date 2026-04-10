#!/usr/bin/env python3
"""
Generate test-set predictions for DanioCell Decima models.

Adapted from decima-main/scripts/decima_predictions.py for the DanioCell atlas.
Produces one h5ad per model with predictions, per-gene Pearson, and per-track Pearson.

Usage:
  python daniocell_predict.py \
    --device 0 \
    --ckpt <path_to_checkpoint> \
    --exp_name pretrained_rep0 \
    [--data_dir ...] [--out_dir ...] [--max_seq_shift 3]
"""

import numpy as np
import anndata
import os
import sys
import torch
import argparse
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="DanioCell Decima test-set predictions"
)
parser.add_argument("--device", type=int, default=0,
                    help="GPU device ID")
parser.add_argument("--ckpt", type=str, required=True,
                    help="Path to model checkpoint (.ckpt)")
parser.add_argument("--exp_name", type=str, required=True,
                    help="Experiment name (used in output filename)")
parser.add_argument("--data_dir", type=str,
                    default="/hpc/scratch/group.data.science/yang-joon.kim/"
                            "daniodecima-daniocell/celltypes_chrom_split_v1",
                    help="Directory containing data.h5 and daniocell_aggregated.h5ad")
parser.add_argument("--matrix_name", type=str,
                    default="daniocell_aggregated.h5ad",
                    help="Filename of the aggregated h5ad matrix")
parser.add_argument("--out_dir", type=str,
                    default="/hpc/scratch/group.data.science/yang-joon.kim/"
                            "daniodecima-daniocell/01_predictions",
                    help="Output directory for prediction h5ad files")
parser.add_argument("--max_seq_shift", type=int, default=3,
                    help="Maximum jitter for test-time augmentation (default: 3)")
parser.add_argument("--batch_size", type=int, default=6,
                    help="Batch size for prediction")
parser.add_argument("--num_workers", type=int, default=4,
                    help="Number of data loader workers")

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
torch.set_float32_matmul_precision("medium")
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

# Load decima modules from local source (not installed 'lightning' package)
src_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'decima')
src_dir = os.path.abspath(src_dir)
sys.path.insert(0, src_dir)
from read_hdf5 import HDF5Dataset, list_genes
from lightning import LightningModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
h5_file = os.path.join(args.data_dir, "data.h5")
matrix_file = os.path.join(args.data_dir, args.matrix_name)
os.makedirs(args.out_dir, exist_ok=True)
out_file = os.path.join(args.out_dir, f"data_out_daniocell_{args.exp_name}.h5ad")

print(f"Checkpoint:  {args.ckpt}")
print(f"Data dir:    {args.data_dir}")
print(f"Output file: {out_file}")

# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
print("Loading checkpoint and extracting weights/params")
ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
state_dict = ckpt['state_dict']
model_params = ckpt['hyper_parameters']['model_params']
train_params = ckpt['hyper_parameters']['train_params']
data_params = ckpt['hyper_parameters'].get('data_params', {})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading anndata")
ad = anndata.read_h5ad(matrix_file)
assert np.all(list_genes(h5_file, key=None) == ad.var_names.tolist()), \
    "Gene names in HDF5 do not match AnnData var_names"

print(f"AnnData shape: {ad.shape}  (tracks × genes)")
print(f"Test genes: {(ad.var.dataset == 'test').sum()}")

# ---------------------------------------------------------------------------
# Create dataset — predict ALL genes (key=None), not just test
# ---------------------------------------------------------------------------
print("Creating HDF5 dataset (all genes, with test-time augmentation)")
ds = HDF5Dataset(
    key=None,
    h5_file=h5_file,
    ad=ad,
    seq_len=524288,
    max_seq_shift=args.max_seq_shift,
)

# ---------------------------------------------------------------------------
# Instantiate model
# ---------------------------------------------------------------------------
print("Instantiating model and loading weights")
# Override init_mode to skip pretrained weight download — we load state_dict directly
model_params = {**model_params, "init_mode": "random"}
model = LightningModel(model_params, train_params, data_params)
model.load_state_dict(state_dict)
model.eval()

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
print("Computing predictions")
preds = model.predict_on_dataset(
    ds, devices=0, batch_size=args.batch_size, num_workers=args.num_workers
).T  # shape: (n_tracks, n_genes)
ad.layers['preds'] = preds
print(f"Predictions shape: {preds.shape}")

# ---------------------------------------------------------------------------
# Per-gene Pearson correlation (across tracks)
# ---------------------------------------------------------------------------
print("Computing per-gene Pearson correlations")
pearson_per_gene = []
for i in tqdm(range(ad.shape[1]), desc="Per-gene Pearson"):
    obs_i = np.asarray(ad.X[:, i]).ravel()
    pred_i = np.asarray(ad.layers['preds'][:, i]).ravel()
    if np.std(obs_i) == 0 or np.std(pred_i) == 0:
        pearson_per_gene.append(np.nan)
    else:
        pearson_per_gene.append(np.corrcoef(obs_i, pred_i)[0, 1])
ad.var["pearson"] = pearson_per_gene

# Report per-split mean Pearson
for split in ['train', 'val', 'test']:
    mask = ad.var.dataset == split
    mean_r = ad.var.loc[mask, 'pearson'].mean()
    print(f"  {split:>5s}: mean Pearson = {mean_r:.4f}  (n={mask.sum()} genes)")

# ---------------------------------------------------------------------------
# Per-track Pearson correlation (across genes, split by dataset)
# ---------------------------------------------------------------------------
print("Computing per-track Pearson correlations")
for split in ad.var.dataset.unique():
    key = f"{split}_pearson"
    split_mask = ad.var.dataset == split
    track_pearsons = []
    for i in range(ad.shape[0]):
        obs_i = np.asarray(ad[i, split_mask].X).ravel()
        pred_i = np.asarray(ad[i, split_mask].layers['preds']).ravel()
        if np.std(obs_i) == 0 or np.std(pred_i) == 0:
            track_pearsons.append(np.nan)
        else:
            track_pearsons.append(np.corrcoef(obs_i, pred_i)[0, 1])
    ad.obs[key] = track_pearsons
    print(f"  Mean Pearson per track over {split} genes: "
          f"{np.nanmean(track_pearsons):.4f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
print(f"Saving to {out_file}")
ad.write_h5ad(out_file)
print("Done.")
