#!/usr/bin/env python3
"""
DanioCell Decima — Attribution Analysis (Phase 3)

Computes Input×Gradient attributions for test genes and analyzes their
overlap with genomic features (promoters, exons, introns, CREs).

Adapted from: 5_specificity/00_combined_attribution_analysis.py
Key changes:
  - DanioCell data paths
  - sys.path.insert(0, ...) for local decima imports
  - init_mode override (skip pretrained download)
  - GTF loaded from file (not genomepy)
  - ATAC peaks: Zebrahub multiome TDR118 (same species/genome, no DanioCell ATAC)

Usage:
  python daniocell_attribution.py \
    --name pretrained_rep0 \
    --ckpt_path <path.ckpt> \
    --pred_file <data_out.h5ad> \
    [--device 0]
"""

import argparse
import torch
import numpy as np
import pandas as pd
import anndata
import h5py
import os
import sys
import pickle
from tqdm import tqdm
import bioframe as bf
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="DanioCell attribution analysis"
)
parser.add_argument('--name', type=str, required=True,
                    help='Name for output files')
parser.add_argument('--ckpt_path', type=str, required=True,
                    help='Path to model checkpoint')
parser.add_argument('--pred_file', type=str, required=True,
                    help='Path to prediction h5ad (from Phase 1)')
parser.add_argument('--device', type=int, default=0,
                    help='CUDA device (default: 0)')
parser.add_argument('--data_dir', type=str,
                    default="/hpc/scratch/group.data.science/yang-joon.kim/"
                            "daniodecima-daniocell/celltypes_chrom_split_v1",
                    help='Directory containing data.h5')
parser.add_argument('--out_dir', type=str,
                    default="/hpc/scratch/group.data.science/yang-joon.kim/"
                            "daniodecima-daniocell/03_attributions",
                    help='Output directory')
parser.add_argument('--gtf_file', type=str,
                    default="/hpc/reference/sequencing_alignment/"
                            "alignment_references/zebrafish_genome_GRCz11_v4.3.2/"
                            "genes/genes.gtf.gz",
                    help='Path to GTF annotation')
parser.add_argument('--atac_bed', type=str,
                    default="/hpc/projects/data.science/yangjoon.kim/"
                            "zebrahub_multiome/data/processed_data/"
                            "00_CRG_arc_processed/TDR118reseq/outs/atac_peaks.bed",
                    help='ATAC peaks BED (Zebrahub multiome, same species/genome)')
parser.add_argument('--expr_threshold', type=float, default=0.5,
                    help='Minimum expression to include track in attribution')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load decima modules
# ---------------------------------------------------------------------------
src_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..',
    'decima-main', 'src', 'decima'
))
sys.path.insert(0, src_dir)
from lightning import LightningModel
from read_hdf5 import extract_gene_data
from captum.attr import InputXGradient
from grelu.transforms.prediction_transforms import Aggregate
import pyranges as pr

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"Loading checkpoint: {args.ckpt_path}")
ckpt = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
state_dict = ckpt['state_dict']
model_params = ckpt['hyper_parameters']['model_params']
train_params = ckpt['hyper_parameters']['train_params']
data_params = ckpt['hyper_parameters'].get('data_params', {})

# Override init_mode to skip pretrained weight download
model_params = {**model_params, "init_mode": "random"}
model = LightningModel(model_params, train_params, data_params)
model.load_state_dict(state_dict)
model.eval()

device = torch.device(args.device)
model = model.to(device)

# ---------------------------------------------------------------------------
# Load prediction data (filter to test genes)
# ---------------------------------------------------------------------------
print(f"Loading predictions: {args.pred_file}")
ad = anndata.read_h5ad(args.pred_file)
ad = ad[:, ad.var.dataset == "test"].copy()
print(f"Test set: {ad.shape[0]} tracks × {ad.shape[1]} genes")

h5_file = os.path.join(args.data_dir, "data.h5")

# ---------------------------------------------------------------------------
# Pre-compute task lists (lightweight — just expression thresholding)
# ---------------------------------------------------------------------------
print("Computing active tasks per gene...")
gene_tasks = {}
for gene in ad.var_names:
    expr = np.array(ad[:, gene].X).squeeze()
    active = ad.obs_names[expr > args.expr_threshold].tolist()
    if active:
        gene_tasks[gene] = active
print(f"Genes with active tasks: {len(gene_tasks)} / {ad.shape[1]}")

# ---------------------------------------------------------------------------
# Compute attributions (load sequences on-the-fly to avoid OOM)
# ---------------------------------------------------------------------------
torch.cuda.set_device(args.device)
attr_outfile = os.path.join(args.out_dir, f'{args.name}-attr-th{args.expr_threshold}-all.h5')
print(f"Computing attributions → {attr_outfile}")

attrs = {}
with h5py.File(attr_outfile, "w") as f:
    for g in tqdm(gene_tasks, desc="Input×Gradient"):
        t = gene_tasks[g]
        s = extract_gene_data(h5_file, g, merge=True).to(device)
        model.add_transform(Aggregate(tasks=t, task_aggfunc="mean", model=model))
        attributer = InputXGradient(model)
        with torch.no_grad():
            attr = attributer.attribute(s)[:4].cpu().numpy()
        attr_data = attr.sum(0)  # sum over channels → (524288,)
        f.create_dataset(g, shape=(524288,), data=attr_data)
        attrs[g] = attr_data

print(f"Attributions computed for {len(attrs)} / {ad.shape[1]} genes")

# ---------------------------------------------------------------------------
# Load genomic annotations
# ---------------------------------------------------------------------------
print("Loading GTF annotations...")

gtf_pr = pr.read_gtf(args.gtf_file)
gtf = gtf_pr.df
gtf = gtf[gtf.Feature == 'exon'].copy()
gtf = gtf.rename(columns={"Chromosome": "chrom", "Start": "start", "End": "end",
                           "gene_name": "gene_name"})
# Ensure chrom has 'chr' prefix to match DanioCell convention
if not gtf.chrom.iloc[0].startswith('chr'):
    gtf['chrom'] = 'chr' + gtf['chrom'].astype(str)
gtf = gtf[gtf.gene_name.isin(ad.var_names)]
print(f"Exons for {gtf.gene_name.nunique()} test genes")

print("Loading ATAC peaks...")
bed_df = pd.read_table(
    args.atac_bed, sep='\t', comment='#', header=None,
    names=['chrom', 'start', 'end'],
    dtype={'chrom': str, 'start': int, 'end': int}
)
# Normalize chrom naming
if not bed_df.chrom.iloc[0].startswith('chr'):
    bed_df['chrom'] = 'chr' + bed_df['chrom'].astype(str)
print(f"ATAC peaks: {len(bed_df)}")

# ---------------------------------------------------------------------------
# Prepare gene coordinates
# ---------------------------------------------------------------------------
genes = ad.var.reset_index()
genes.columns = ['gene'] + list(ad.var.columns)
genes['st'] = genes.gene_start - genes.start
genes['en'] = [min(524287, x) for x in genes.gene_end - genes.start]
genes = genes[genes.gene.isin(gtf.gene_name)]
print(f"Genes with exon annotation: {len(genes)}")

# Apply absolute value (no strand reversal — region analysis uses forward-strand coords)
for gene in ad.var_names:
    if gene in attrs:
        attrs[gene] = np.abs(attrs[gene])

# ---------------------------------------------------------------------------
# Calculate CRE overlaps
# ---------------------------------------------------------------------------
cre_overlap = bf.overlap(genes, bed_df, how='inner')
cre_overlap['st'] = cre_overlap.start_ - cre_overlap.start
cre_overlap['en'] = cre_overlap.end_ - cre_overlap.start
cre_overlap['dist'] = np.abs(np.vstack([
    cre_overlap.start - cre_overlap.gene_start,
    cre_overlap.start - cre_overlap.gene_end
])).min(0)

for lo, hi, label in [(0, 100, '0-100'), (100, 1000, '100-1kb'),
                       (1000, 10000, '1-10kb'), (10000, 100000, '10-100kb')]:
    cre_overlap.loc[(cre_overlap.dist >= lo) & (cre_overlap.dist < hi), 'dist_class'] = label
cre_overlap.loc[cre_overlap.dist >= 100000, 'dist_class'] = '>100kb'

# Build per-gene annotation dict
annot = {}
for gene in tqdm(ad.var_names, desc="Building annotations"):
    exons = gtf[(gtf.gene_name == gene) &
                (gtf.start >= ad.var.start[gene]) &
                (gtf.end <= ad.var.end[gene])].copy()
    exons['st'] = exons['start'] - ad.var.start[gene]
    exons['en'] = exons['end'] - ad.var.start[gene]
    annot[gene] = {
        'exons': exons,
        'cre': cre_overlap[cre_overlap.gene == gene]
    }

# ---------------------------------------------------------------------------
# Analyze attribution patterns per region
# ---------------------------------------------------------------------------
promoter_window = 100
junction_window = 10

def safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else np.nan

print("Analyzing genomic regions...")
genes_analysis = genes.iloc[:, :len(ad.var.columns) + 3].copy()  # gene + var cols + st/en

for row in tqdm(genes_analysis.itertuples(), total=len(genes_analysis),
                desc="Processing genes"):
    if row.gene not in attrs:
        continue

    attr = attrs[row.gene]
    exons = annot[row.gene]['exons']
    cres = annot[row.gene]['cre']

    # Initialize boolean masks
    in_gene = np.zeros(524288, dtype=bool)
    in_promoter = np.zeros(524288, dtype=bool)
    in_exons = np.zeros(524288, dtype=bool)
    in_junctions = np.zeros(524288, dtype=bool)
    in_cre = np.zeros(524288, dtype=bool)
    out_1k = np.zeros(524288, dtype=bool)
    out_1k_10k = np.zeros(524288, dtype=bool)
    out_10k_100k = np.zeros(524288, dtype=bool)
    out_100k = np.zeros(524288, dtype=bool)

    start = int(row.st)
    end = int(row.en)
    in_gene[start:end] = True

    if row.strand == '+':
        in_promoter[max(0, start - promoter_window):start + promoter_window] = True
    else:
        in_promoter[max(0, end - promoter_window):min(524288, end + promoter_window)] = True

    for exon in exons.itertuples():
        s, e = int(exon.st), int(exon.en)
        in_exons[s:e] = True
        in_junctions[max(0, s - junction_window):s + junction_window] = True
        in_junctions[max(0, e - junction_window):e + junction_window] = True

    for cre in cres.itertuples():
        in_cre[int(cre.st):int(cre.en)] = True

    # Distance regions
    out_1k[max(0, start - 1000):start] = True
    out_1k[end:min(524288, end + 1000)] = True
    out_1k_10k[max(0, start - 10000):max(0, start - 1000)] = True
    out_1k_10k[min(524288, end + 1000):min(524288, end + 10000)] = True
    out_10k_100k[max(0, start - 100000):max(0, start - 10000)] = True
    out_10k_100k[min(524288, end + 10000):min(524288, end + 100000)] = True
    out_100k[:max(0, start - 100000)] = True
    out_100k[min(524288, end + 100000):] = True

    # Attribution scores per region
    idx = genes_analysis.gene == row.gene
    genes_analysis.loc[idx, 'Promoter'] = safe_mean(attr[in_promoter])
    genes_analysis.loc[idx, 'Exons'] = safe_mean(attr[in_exons])
    genes_analysis.loc[idx, 'Introns'] = safe_mean(attr[in_gene & ~in_exons])
    genes_analysis.loc[idx, 'Exon/Intron junctions'] = safe_mean(attr[in_junctions])

    genes_analysis.loc[idx, 'Promoter CREs'] = safe_mean(attr[in_promoter & in_cre])
    genes_analysis.loc[idx, 'Promoter non-CREs'] = safe_mean(attr[in_promoter & ~in_cre])
    genes_analysis.loc[idx, 'Intronic CREs'] = safe_mean(attr[in_gene & ~in_exons & in_cre])
    genes_analysis.loc[idx, 'Intronic non-CREs'] = safe_mean(attr[in_gene & ~in_exons & ~in_cre])
    genes_analysis.loc[idx, 'Outer CREs'] = safe_mean(attr[~in_gene & in_cre])
    genes_analysis.loc[idx, 'Outer non-CREs'] = safe_mean(attr[~in_gene & ~in_cre])

    for lo, hi, label in [(0, 1000, '1k'), (1000, 10000, '1k-10k'),
                          (10000, 100000, '10k-100k')]:
        mask = {'1k': out_1k, '1k-10k': out_1k_10k, '10k-100k': out_10k_100k}[label]
        genes_analysis.loc[idx, f'{label} (CREs)'] = safe_mean(attr[mask & in_cre])
        genes_analysis.loc[idx, f'{label} (non-CREs)'] = safe_mean(attr[mask & ~in_cre])
        genes_analysis.loc[idx, f'{label} (all)'] = safe_mean(attr[mask])

    genes_analysis.loc[idx, '>=100k (CREs)'] = safe_mean(attr[out_100k & in_cre])
    genes_analysis.loc[idx, '>=100k (non-CREs)'] = safe_mean(attr[out_100k & ~in_cre])
    genes_analysis.loc[idx, '>=100k (all)'] = safe_mean(attr[out_100k])

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
output_file = os.path.join(args.out_dir, f'{args.name}_regional_attributions.pkl')
with open(output_file, 'wb') as f:
    pickle.dump({'genes': genes_analysis}, f)

print(f"Saved regional analysis to {output_file}")
print("Done!")
