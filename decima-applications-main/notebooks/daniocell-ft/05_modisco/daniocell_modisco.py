#!/usr/bin/env python3
"""
DanioCell Decima — TF-MoDISco Motif Discovery (Phase 5)

Discovers enriched TF motifs in cell-type-specific attributions from Phase 4.
Matches motifs against JASPAR H12CORE vertebrate database.

Adapted from: 6_cell_states/modisco_simple.py

Usage:
  python daniocell_modisco.py \
    --cell_type_dir <path_to_specificity_output> \
    --meme_file <H12CORE_meme_format.meme> \
    --out_dir <output_dir>
"""

import numpy as np
import os
import argparse
import modiscolite

parser = argparse.ArgumentParser(
    description="DanioCell MoDISco motif discovery"
)
parser.add_argument("--cell_type_dir", type=str, required=True,
                    help="Directory containing sequences.npy + attributions.npy from Phase 4")
parser.add_argument("--meme_file", type=str,
                    default="/home/yang-joon.kim/.conda/envs/gReLu/lib/python3.10/"
                            "site-packages/grelu/resources/meme/H12CORE_meme_format.meme",
                    help="MEME motif database for TOMTOM matching")
parser.add_argument("--out_dir", type=str, default=None,
                    help="Output directory (default: cell_type_dir/modisco/)")
parser.add_argument("--max_seqlets", type=int, default=10000,
                    help="Max seqlets per metacluster")
args = parser.parse_args()

if args.out_dir is None:
    args.out_dir = os.path.join(args.cell_type_dir, "modisco")
os.makedirs(args.out_dir, exist_ok=True)

# Load data
seq_file = os.path.join(args.cell_type_dir, "sequences.npy")
attr_file = os.path.join(args.cell_type_dir, "attributions.npy")

print(f"Loading sequences: {seq_file}")
sequences = np.load(seq_file)
print(f"Loading attributions: {attr_file}")
attributions = np.load(attr_file)

# MoDISco expects (N, L, 4) — transpose from (N, 4, L)
sequences = sequences.transpose(0, 2, 1).astype("float32")
attributions = attributions.transpose(0, 2, 1).astype("float32")

print(f"Sequences shape:    {sequences.shape}")
print(f"Attributions shape: {attributions.shape}")

# Run MoDISco
print("Running TF-MoDISco...")
pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
    hypothetical_contribs=attributions,
    one_hot=sequences,
    max_seqlets_per_metacluster=args.max_seqlets,
)

# Save H5 report
h5_file = os.path.join(args.out_dir, "modisco_report.h5")
print(f"Saving MoDISco output: {h5_file}")
modiscolite.io.save_hdf5(h5_file, pos_patterns, neg_patterns, window_size=20)

# Generate HTML report with TOMTOM matches
print(f"Generating report with TOMTOM matches...")
print(f"MEME database: {args.meme_file}")
modiscolite.report.report_motifs(
    h5_file,
    args.out_dir,
    is_writing_tomtom_matrix=False,
    top_n_matches=10,
    meme_motif_db=args.meme_file,
    img_path_suffix="./",
    trim_threshold=0.2,
)

print(f"Results saved to: {args.out_dir}")
print("Done.")
