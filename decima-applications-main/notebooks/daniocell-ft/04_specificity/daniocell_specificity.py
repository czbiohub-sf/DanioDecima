#!/usr/bin/env python3
"""
DanioCell Decima — Cell-Type Specificity Attributions (Phase 4)

Computes Specificity-transform attributions for cell-type-specific expression.
For each focal cell type: on_tasks (target) vs off_tasks (all others at same stage).
Outputs sequences.npy + attributions.npy for downstream MoDISco (Phase 5).

Adapted from: 6_cell_states/celltype_motif_attribution.py
Key changes:
  - Uses `identity.super` (not `zebrafish_anatomy_ontology_class_fine`)
  - DanioCell data paths and checkpoint loading
  - 10 focal cell types covering major lineages

Usage:
  python daniocell_specificity.py \
    --cell_type_id 0 \
    --ckpt_path <path.ckpt> \
    --pred_file <data_out.h5ad> \
    [--device 0] [--n_genes 50]
"""

import numpy as np
import pandas as pd
import anndata
import os
import sys
import torch
import argparse
import json
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local decima imports
# ---------------------------------------------------------------------------
src_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..',
    'decima-main', 'src', 'decima'
))
sys.path.insert(0, src_dir)
from lightning import LightningModel
from evaluate import marker_zscores
from interpret import attributions as get_attr
from captum.attr import Saliency

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="DanioCell cell-type specificity attributions"
)
parser.add_argument("--cell_type_id", type=int, required=True,
                    help="Cell type index (0-9)")
parser.add_argument("--ckpt_path", type=str, required=True,
                    help="Path to model checkpoint")
parser.add_argument("--pred_file", type=str, required=True,
                    help="Path to prediction h5ad (from Phase 1)")
parser.add_argument("--data_dir", type=str,
                    default="/hpc/scratch/group.data.science/yang-joon.kim/"
                            "daniodecima-daniocell/celltypes_chrom_split_v1",
                    help="Directory containing data.h5")
parser.add_argument("--out_dir", type=str,
                    default="/hpc/scratch/group.data.science/yang-joon.kim/"
                            "daniodecima-daniocell/04_specificity",
                    help="Output directory")
parser.add_argument("--device", type=int, default=0,
                    help="CUDA device (default: 0)")
parser.add_argument("--n_genes", type=int, default=50,
                    help="Number of marker genes per cell type")
parser.add_argument("--window_size", type=int, default=10000,
                    help="Half-window around TSS in bp (default: 10000 = ±10kb)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 10 focal cell types (biologically important, diverse lineages)
# ---------------------------------------------------------------------------
CELLTYPE_KEY = "identity.super"
FOCAL_CELL_TYPES = [
    "neural crest",
    "neurons",
    "motor neurons",
    "radial glia",
    "notochord",
    "somite",
    "cardiac muscle",
    "intestine",
    "liver",
    "epidermis",
]

if args.cell_type_id >= len(FOCAL_CELL_TYPES):
    raise ValueError(f"cell_type_id must be 0-{len(FOCAL_CELL_TYPES)-1}")
target_cell_type = FOCAL_CELL_TYPES[args.cell_type_id]

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"Loading checkpoint: {args.ckpt_path}")
ckpt = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
state_dict = ckpt['state_dict']
model_params = {**ckpt['hyper_parameters']['model_params'], "init_mode": "random"}
train_params = ckpt['hyper_parameters']['train_params']
data_params = ckpt['hyper_parameters'].get('data_params', {})

model = LightningModel(model_params, train_params, data_params)
model.load_state_dict(state_dict)
model.eval()

# ---------------------------------------------------------------------------
# Load prediction data
# ---------------------------------------------------------------------------
print(f"Loading predictions: {args.pred_file}")
ad = anndata.read_h5ad(args.pred_file)
print(f"AnnData shape: {ad.shape}")
print(f"Cell types: {ad.obs[CELLTYPE_KEY].nunique()}")

# Verify target cell type exists
all_cell_types = ad.obs[CELLTYPE_KEY].unique().tolist()
if target_cell_type not in all_cell_types:
    available = [ct for ct in all_cell_types if target_cell_type.lower() in ct.lower()]
    raise ValueError(
        f"'{target_cell_type}' not in obs['{CELLTYPE_KEY}']. "
        f"Close matches: {available}"
    )

# ---------------------------------------------------------------------------
# Define on_tasks / off_tasks
# ---------------------------------------------------------------------------
on_tasks = ad.obs_names[ad.obs[CELLTYPE_KEY] == target_cell_type].tolist()
off_tasks = ad.obs_names[ad.obs[CELLTYPE_KEY] != target_cell_type].tolist()

print(f"Target: {target_cell_type}")
print(f"  on_tasks:  {len(on_tasks)} tracks")
print(f"  off_tasks: {len(off_tasks)} tracks")

# ---------------------------------------------------------------------------
# Find marker genes (using predicted z-scores on test genes)
# ---------------------------------------------------------------------------
print(f"Finding top {args.n_genes} marker genes...")

# Use test genes for marker gene identification
ad_test = ad[:, ad.var.dataset == "test"].copy()

# Add Group column for marker_zscores
ad_test.obs['Group'] = ad_test.obs[CELLTYPE_KEY].copy()
gene_df = marker_zscores(ad_test, key='Group', layer='preds')
top_genes = (gene_df[gene_df.Group == target_cell_type]
             .sort_values('score', ascending=False)
             .head(args.n_genes))
gene_list = top_genes.gene.tolist()
print(f"Selected {len(gene_list)} marker genes")
print(f"Top 5: {gene_list[:5]}")

# ---------------------------------------------------------------------------
# Compute specificity attributions
# ---------------------------------------------------------------------------
h5_file = os.path.join(args.data_dir, "data.h5")
sequences = []
attributions = []

print(f"Computing Specificity attributions ({len(gene_list)} genes)...")
with torch.no_grad():
    for gene in tqdm(gene_list, desc="Saliency"):
        try:
            seq, tss_pos, attr = get_attr(
                gene=gene, h5_file=h5_file, model=model, device=args.device,
                tasks=on_tasks, off_tasks=off_tasks,
                transform="specificity", method=Saliency, abs=False
            )

            # Extract ±window_size around TSS
            start_pos = max(0, tss_pos - args.window_size)
            end_pos = min(seq.shape[1], tss_pos + args.window_size)

            sequences.append(seq[:4, start_pos:end_pos])
            attributions.append(attr[:4, start_pos:end_pos])

        except Exception as e:
            print(f"  Warning: skipping {gene}: {e}")
            continue

if len(sequences) == 0:
    raise RuntimeError("No genes were successfully processed")

sequences = np.stack(sequences)
attributions = np.stack(attributions)

# Center attributions
attributions = attributions - attributions.mean(1, keepdims=True)

print(f"Sequences shape:    {sequences.shape}")
print(f"Attributions shape: {attributions.shape}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
safe_name = target_cell_type.replace(" ", "_").replace("/", "_")
# Extract replicate info from checkpoint path
ckpt_basename = os.path.basename(os.path.dirname(os.path.dirname(
    os.path.dirname(args.ckpt_path))))  # e.g., pretrained_decima-human_rep0...
output_dir = os.path.join(args.out_dir, f"{ckpt_basename}_{safe_name}")
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, 'sequences.npy'), sequences)
np.save(os.path.join(output_dir, 'attributions.npy'), attributions)

# Save metadata
metadata = {
    'target_cell_type': target_cell_type,
    'cell_type_id': args.cell_type_id,
    'ckpt_path': args.ckpt_path,
    'n_genes': len(gene_list),
    'genes': gene_list,
    'n_on_tasks': len(on_tasks),
    'n_off_tasks': len(off_tasks),
    'window_size': args.window_size,
    'sequences_shape': list(sequences.shape),
    'attributions_shape': list(attributions.shape),
}
with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

top_genes.to_csv(os.path.join(output_dir, 'gene_stats.csv'), index=False)

print(f"Saved to: {output_dir}")
print("Ready for MoDISco (Phase 5).")
