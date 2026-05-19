#!/usr/bin/env python3
"""
Combined Attribution Calculation and Analysis Script
Processes checkpoints to calculate attributions and analyze them against genomic features
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

# Argument parsing
parser = argparse.ArgumentParser(description="Calculate and analyze attributions for a given checkpoint and model directory.")
parser.add_argument('--name', type=str, required=True, help='Name for output file')
parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint file')
parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
parser.add_argument('--device', type=int, default=0, help='CUDA device (default: 0)')
args = parser.parse_args()

print(f"Processing: {args.ckpt_path}\nWith model dir: {args.model_dir}")

# Load checkpoint
ckpt = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
state_dict = ckpt['state_dict']
model_params = ckpt['hyper_parameters']['model_params']
train_params = ckpt['hyper_parameters']['train_params']
data_params = ckpt['hyper_parameters'].get('data_params', {})

# Add Decima to path
sys.path.append('/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/daniodecima-main/src/decima/')
from lightning import LightningModel
from interpret import extract_gene_data
from captum.attr import InputXGradient
from grelu.transforms.prediction_transforms import Aggregate
from genome import read_gtf

# Create model and load weights
model = LightningModel(model_params, train_params, data_params)
model.load_state_dict(state_dict)
model.eval()

# Data paths - for attribution calculation
save_dir = "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/data/celltypes_chrom_split_v1/"
h5_file = os.path.join(save_dir, "data.h5")

# Find the data_out file for this specific model
data_out_files = [f for f in os.listdir(args.model_dir) if f.startswith('data_out_') and f.endswith('.h5ad')]
if not data_out_files:
    raise FileNotFoundError(f"No data_out file found in {args.model_dir}")
data_out_path = os.path.join(args.model_dir, data_out_files[0])
print(f"Using data_out file: {data_out_path}")

# Load the model-specific data (for analysis)
ad = anndata.read_h5ad(data_out_path)
ad = ad[:, ad.var.dataset == "test"].copy()
print(f"Data shape: {ad.shape}")

device = torch.device(args.device)
model = model.to(device)
models = [model]

# Prepare tasks and sequences
tasks = []
seqs = []
for gene in tqdm(ad.var_names, desc="Preparing gene data"):
    tasks.append(ad.obs_names[np.array(ad[:, gene].X).squeeze() > .5].tolist())
    seqs.append(extract_gene_data(h5_file, gene, merge=True))

# Calculate attributions
torch.cuda.set_device(args.device)
attr_outfile = os.path.join(args.model_dir, f'{args.name}-attr-th05-all.h5')

print("Calculating attributions...")
attrs = {}
with h5py.File(attr_outfile, "w") as f:
    for g, t, s in tqdm(zip(ad.var_names, tasks, seqs), desc="Computing attributions"):
        if not t:
            continue
        s = s.to(device)
        attr = []
        for model in models:
            model.add_transform(Aggregate(tasks=t, task_aggfunc="mean", model=model))
            attributer = InputXGradient(model)
            with torch.no_grad():
                attr.append(attributer.attribute(s)[:4].cpu().numpy())
        attr_data = np.stack(attr).mean(0).sum(0)
        f.create_dataset(g, shape=(524288,), data=attr_data)
        attrs[g] = attr_data

print(f"Saved attributions to {attr_outfile}")

# Now analyze attributions using the model-specific data
print("Analyzing attributions...")

# Prepare gene data from the model-specific AnnData
genes = ad.var.reset_index()
genes['gene'] = ad.var_names
genes['st'] = genes.gene_start - genes.start
genes['en'] = [min(524287, x) for x in genes.gene_end - genes.start]

# Load genomic annotations
atac_bed_file = '/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR118reseq/outs/atac_peaks.bed'
bed_df = pd.read_table(atac_bed_file, 
                      sep='\t',
                      comment='#',
                      header=None,
                      names=['chrom', 'start', 'end'],
                      dtype={'chrom': str, 'start': int, 'end': int})

gtf = read_gtf(
    '/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/data/Danio_rerio.GRCz11.113.gtf',
    features='exon')
gtf = gtf[gtf.gene_name.isin(ad.var_names)]

genes = genes[genes.gene.isin(gtf.gene_name)]

# Process attributions with strand correction
for gene in ad.var_names:
    if gene in attrs:
        attr = attrs[gene]
        if ad.var.strand[gene] == '-':
            attr = attr[::-1]
        attrs[gene] = np.abs(attr)

# Calculate CRE overlaps
cre_overlap = bf.overlap(genes, bed_df, how='inner')
cre_overlap['st'] = cre_overlap.start_ - cre_overlap.start
cre_overlap['en'] = cre_overlap.end_ - cre_overlap.start
cre_overlap['dist'] = np.abs(np.vstack([cre_overlap.start - cre_overlap.gene_start, cre_overlap.start - cre_overlap.gene_end])).min(0)
cre_overlap.loc[cre_overlap.dist < 100, 'dist_class'] = '0-100'
cre_overlap.loc[(cre_overlap.dist >= 100) & (cre_overlap.dist < 1000), 'dist_class'] = '100-1kb'
cre_overlap.loc[(cre_overlap.dist >= 1000) & (cre_overlap.dist < 10000), 'dist_class'] ='1-10kb'
cre_overlap.loc[(cre_overlap.dist >= 10000) & (cre_overlap.dist < 100000), 'dist_class'] ='10-100kb'
cre_overlap.loc[cre_overlap.dist >= 100000, 'dist_class'] ='>100kb'

# Prepare annotations
annot = dict()
for gene in tqdm(ad.var_names, desc="Preparing annotations"):
    exons = gtf[(gtf.gene_name==gene) & (gtf.start >= ad.var.start[gene]) & (gtf.end <= ad.var.end[gene])].copy()
    exons['st'] = exons['start'] - ad.var.start[gene]
    exons['en'] = exons['end'] - ad.var.start[gene]
    annot[gene] = {'exons':exons}
    annot[gene]['cre'] = cre_overlap[cre_overlap.gene==gene]

# Analyze attribution patterns
genes = genes.iloc[:, :25]
promoter_window = 100
junction_window = 10

def safe_mean(arr):
    if np.any(arr):
        return np.mean(arr)
    else:
        return np.nan

print("Analyzing genomic regions...")
for row in tqdm(genes.itertuples(), desc="Processing genes"):
    exons = annot[row.gene]['exons']
    cres = annot[row.gene]['cre']
    
    if row.gene not in attrs:
        continue
        
    attr = attrs[row.gene]

    # Initialize boolean arrays
    in_gene = np.array([False]*524288)
    in_promoter = np.array([False]*524288)
    in_exons = np.array([False]*524288)
    in_introns = np.array([False]*524288)
    in_junctions = np.array([False]*524288)
    in_cre = np.array([False]*524288)
    out_1k = np.array([False]*524288)
    out_1k_10k = np.array([False]*524288)
    out_10k_100k = np.array([False]*524288)
    out_100k = np.array([False]*524288)

    # Define regions
    start = int(row.st)
    end = int(row.en)
    in_gene[start: end] = True
    if row.strand=='+':
        in_promoter[start-promoter_window: start+promoter_window] = True
    else:
        in_promoter[end-promoter_window: end+promoter_window] = True
    
    for exon in exons.itertuples():
        in_exons[exon.st:exon.en] = True
        in_junctions[exon.st - junction_window: exon.st+junction_window] = True
        in_junctions[exon.en - junction_window: exon.en+junction_window] = True

    if len(cres) > 0:
        for cre in cres.itertuples():
            in_cre[cre.st:cre.en] = True

    # Define distance regions
    out_1k[start-1000:start] = True
    out_1k[end:end+1000] = True
    out_1k_10k[start-10000:start-1000] = True
    out_1k_10k[end+1000:end+10000] = True
    out_10k_100k[start-100000:start-10000] = True
    out_10k_100k[end+10000:end+100000] = True
    out_100k[:start-100000] = True
    out_100k[end+100000:] = True

    # Calculate attribution scores for each region
    genes.loc[genes.gene==row.gene, 'Promoter'] = safe_mean(attr[in_promoter])
    genes.loc[genes.gene==row.gene, 'Exons'] = safe_mean(attr[in_exons])
    genes.loc[genes.gene==row.gene, 'Introns'] = safe_mean(attr[in_gene & (~in_exons)])
    genes.loc[genes.gene==row.gene, 'Exon/Intron junctions'] = safe_mean(attr[in_junctions])
    
    genes.loc[genes.gene==row.gene, 'Promoter CREs'] = safe_mean(attr[in_promoter & in_cre])
    genes.loc[genes.gene==row.gene, 'Promoter non-CREs'] = safe_mean(attr[in_promoter & (~in_cre)])
    genes.loc[genes.gene==row.gene, 'Exons CREs'] = safe_mean(attr[in_exons & in_cre])
    genes.loc[genes.gene==row.gene, 'Exons non-CREs'] = safe_mean(attr[in_exons & (~in_cre)])
    genes.loc[genes.gene==row.gene, 'Intronic CREs'] = safe_mean(attr[in_gene & (~in_exons) & in_cre])
    genes.loc[genes.gene==row.gene, 'Intronic non-CREs'] = safe_mean(attr[in_gene & (~in_exons) & (~in_cre)])
    genes.loc[genes.gene==row.gene, 'Outer CREs'] = safe_mean(attr[(~in_gene) & in_cre])
    genes.loc[genes.gene==row.gene, 'Outer non-CREs'] = safe_mean(attr[(~in_gene) & (~in_cre)])
    
    genes.loc[genes.gene==row.gene, '1k (CREs)'] = safe_mean(attr[out_1k & in_cre])
    genes.loc[genes.gene==row.gene, '1k (non-CREs)'] = safe_mean(attr[out_1k & (~in_cre)])
    genes.loc[genes.gene==row.gene, '1k (all)'] = safe_mean(attr[out_1k])

    genes.loc[genes.gene==row.gene, '1k-10k (CREs)'] = safe_mean(attr[out_1k_10k & in_cre])
    genes.loc[genes.gene==row.gene, '1k-10k (non-CREs)'] = safe_mean(attr[out_1k_10k & (~in_cre)])
    genes.loc[genes.gene==row.gene, '1k-10k (all)'] = safe_mean(attr[out_1k_10k])
    
    genes.loc[genes.gene==row.gene, '10k-100k (CREs)'] = safe_mean(attr[out_10k_100k & in_cre])
    genes.loc[genes.gene==row.gene, '10k-100k (non-CREs)'] = safe_mean(attr[out_10k_100k & (~in_cre)])
    genes.loc[genes.gene==row.gene, '10k-100k (all)'] = safe_mean(attr[out_10k_100k])

    genes.loc[genes.gene==row.gene, '>=100k (CREs)'] = safe_mean(attr[out_100k & in_cre])
    genes.loc[genes.gene==row.gene, '>=100k (non-CREs)'] = safe_mean(attr[out_100k & (~in_cre)])
    genes.loc[genes.gene==row.gene, '>=100k (all)'] = safe_mean(attr[out_100k])

# Save results
output_file = os.path.join(args.model_dir, f'{args.name}_fulltestset_cres_all.pkl')
with open(output_file, 'wb') as f:
    pickle.dump({
        'genes': genes
    }, f)

print(f"Saved analysis results to {output_file}")
print("Processing complete!")