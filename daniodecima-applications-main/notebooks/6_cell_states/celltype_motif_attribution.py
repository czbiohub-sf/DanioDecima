#!/usr/bin/env python3
import numpy as np
import anndata
import os
import sys
import torch
import argparse
import tqdm
import json

# Add paths
sys.path.append('/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/daniodecima-main/src/decima/')
from lightning import LightningModel
from evaluate import marker_zscores
from interpret import attributions as get_attr
from captum.attr import Saliency

def set_genomepy_paths(job_id, task_id):
    """Set up genomepy paths for this job"""
    base_dir = "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima"
    cache_dir = f"{base_dir}/genomepy_cache/job_{job_id}_task_{task_id}"
    config_dir = f"{base_dir}/genomepy_config/job_{job_id}_task_{task_id}"
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    
    os.environ["GENOMEPY_CACHE_DIR"] = cache_dir
    os.environ["GENOMEPY_CONFIG_DIR"] = config_dir

def define_cell_types():
    """Define cell types to analyze (reduced for troubleshooting)"""
    cell_types = [
        "adaxial cell",
        # "brain",
        # "common myeloid progenitor",
        # "ectodermal cell",
        # "floor plate",
        # "hatching gland cell",
        # "head mesenchyme",
        # "heart",
        # "hematopoietic system",
        # "lateral mesoderm",
        # "lens placode",
        # "midbrain hindbrain boundary",
        # "myotome",
        # "neural crest",
        # "neural tube",
        # "notochord",
        # "optic vesicle",
        # "otic placode",
        # "paraxial mesoderm",
        # "periderm",
        # "pronephros",
        # "somite",
        # "spinal cord neural tube",
        # "telencephalon",
        # "trigeminal placode"
    ]
    return cell_types

def define_all_cell_types():
    """Define ALL 25 cell types for off_tasks (even if we're only testing one)"""
    # We still need all cell types for proper differential analysis
    all_cell_types = [
        "adaxial cell",
        "brain",
        "common myeloid progenitor",
        "ectodermal cell",
        "floor plate",
        "hatching gland cell",
        "head mesenchyme",
        "heart",
        "hematopoietic system",
        "lateral mesoderm",
        "lens placode",
        "midbrain hindbrain boundary",
        "myotome",
        "neural crest",
        "neural tube",
        "notochord",
        "optic vesicle",
        "otic placode",
        "paraxial mesoderm",
        "periderm",
        "pronephros",
        "somite",
        "spinal cord neural tube",
        "telencephalon",
        "trigeminal placode"
    ]
    return all_cell_types

def get_model_info(model_id):
    """Get model directory and metadata based on model_id (reduced for troubleshooting)"""
    model_dirs = [
        # Human Decima (1 for testing)
        "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250618_111138/pretrained_decima-human_rep0_lr3e-05_seed42/version_0",
        # "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250618_111138/pretrained_decima-human_rep1_lr3e-05_seed42/version_0",
        # "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250618_111138/pretrained_decima-human_rep2_lr3e-05_seed42/version_0",
        # "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250618_111138/pretrained_decima-human_rep3_lr3e-05_seed42/version_0",
        # Human Borzoi (commented out)
        # "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-human_rep0_lr3e-05_seed42/version_0",
        # "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-human_rep1_lr3e-05_seed42/version_0",
        # "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-human_rep2_lr3e-05_seed42/version_0",
        # "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-human_rep3_lr3e-05_seed42/version_0",
        # Mouse Borzoi (commented out)
        # "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-mouse_rep0_lr3e-05_seed42/version_0",
        # "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-mouse_rep1_lr3e-05_seed42/version_0",
        # "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-mouse_rep2_lr3e-05_seed42/version_0",
        # "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/pretrained_wandb-mouse_rep3_lr3e-05_seed42/version_0",
        # Random (commented out)
        # "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/random_lr3e-06_seed42/version_0",
        # "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/random_lr3e-06_seed43/version_0",
        # "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/random_lr3e-06_seed44/version_0",
        # "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments/decima_experiments_20250617_225114/random_lr3e-06_seed45/version_0"
    ]
    
    model_names = [
        "human_decima_rep0",
        # "human_decima_rep1", "human_decima_rep2", "human_decima_rep3",
        # "human_borzoi_rep0", "human_borzoi_rep1", "human_borzoi_rep2", "human_borzoi_rep3",
        # "mouse_borzoi_rep0", "mouse_borzoi_rep1", "mouse_borzoi_rep2", "mouse_borzoi_rep3",
        # "random_rep0", "random_rep1", "random_rep2", "random_rep3"
    ]
    
    if model_id >= len(model_dirs):
        raise ValueError(f"Model ID {model_id} out of range (0-{len(model_dirs)-1})")
    
    return model_dirs[model_id], model_names[model_id]

def load_model_and_data(model_dir, _h5_file, timepoint=None):
    """Load model and data, optionally filtering by timepoint"""
    # Find checkpoint
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Use the latest checkpoint (by epoch)
    checkpoint_path = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
    print(f"Loading model from: {checkpoint_path}")
    model = LightningModel.load_from_checkpoint(checkpoint_path).eval()
    
    # Find matrix file
    matrix_files = [f for f in os.listdir(model_dir) if f.startswith("data_out_decima_") and f.endswith(".h5ad")]
    if not matrix_files:
        raise ValueError(f"No matrix file found in {model_dir}")
    
    matrix_file = os.path.join(model_dir, matrix_files[0])
    print(f"Loading data from: {matrix_file}")
    ad = anndata.read_h5ad(matrix_file)
    
    print(f"Before filtering: {ad.n_obs} observations")
    
    # Filter by timepoint if specified
    if timepoint is not None:
        print(f"Filtering to timepoint: {timepoint}")
        if 'timepoint' not in ad.obs.columns:
            raise ValueError("No 'timepoint' column found in data")
        
        # Check available timepoints
        available_timepoints = ad.obs['timepoint'].unique()
        print(f"Available timepoints: {sorted(available_timepoints)}")
        
        if timepoint not in available_timepoints:
            raise ValueError(f"Timepoint '{timepoint}' not found. Available: {sorted(available_timepoints)}")
        
        ad = ad[ad.obs['timepoint'] == timepoint].copy()
        print(f"After filtering to {timepoint}: {ad.n_obs} observations")
        
        if ad.n_obs == 0:
            raise ValueError(f"No observations found at timepoint {timepoint}")
    else:
        print("No timepoint filtering applied - using all timepoints")
    
    return model, ad

def prepare_differential_analysis(ad, target_cell_type, all_cell_types, timepoint=None):
    """Prepare data for differential analysis: target vs all other cell types"""
    # Create Group column
    ad.obs['Group'] = ad.obs.zebrafish_anatomy_ontology_class_fine.copy()
    
    # Filter to only include cell types from our defined list
    ad_filtered = ad[ad.obs.Group.isin(all_cell_types)].copy()
    
    if len(ad_filtered) == 0:
        raise ValueError(f"No cells found for any of the defined cell types")
    
    # Get task IDs for target cell type (on_tasks)
    on_tasks = ad_filtered.obs_names[ad_filtered.obs.Group == target_cell_type].tolist()
    
    if len(on_tasks) == 0:
        raise ValueError(f"No cells found for target cell type: {target_cell_type}")
    
    # Get task IDs for all other cell types (off_tasks)
    other_cell_types = [ct for ct in all_cell_types if ct != target_cell_type]
    off_tasks = ad_filtered.obs_names[ad_filtered.obs.Group.isin(other_cell_types)].tolist()
    
    if len(off_tasks) == 0:
        raise ValueError(f"No cells found for other cell types")
    
    timepoint_str = f" at {timepoint}" if timepoint else ""
    print(f"Target cell type '{target_cell_type}'{timepoint_str}: {len(on_tasks)} tasks")
    print(f"Other cell types{timepoint_str}: {len(off_tasks)} tasks across {len(other_cell_types)} cell types")
    
    # Check for multiple tasks per cell type (should be 1 if single timepoint)
    if timepoint:
        celltype_counts = ad_filtered.obs.Group.value_counts()
        multiple_tasks = celltype_counts[celltype_counts > 1]
        if len(multiple_tasks) > 0:
            print(f"WARNING: Some cell types have multiple tasks at {timepoint}:")
            for ct, count in multiple_tasks.items():
                print(f"  {ct}: {count} tasks")
        else:
            print(f"✓ All cell types have exactly 1 task at {timepoint}")
    
    return ad_filtered, on_tasks, off_tasks

def find_marker_genes(ad_filtered, target_cell_type, n_genes=50):
    """Find marker genes for the target cell type vs all others"""
    print("Finding marker genes...")
    
    # Calculate z-scores for all cell types
    gene_df = marker_zscores(ad_filtered, key='Group', layer='preds')
    
    # Get top genes for the target cell type
    genes = gene_df[gene_df.Group == target_cell_type].sort_values('score', ascending=False).head(n_genes)
    
    gene_list = genes.gene.tolist()
    print(f"Selected {len(gene_list)} marker genes for {target_cell_type}")
    return gene_list, genes

def calculate_attributions(genes, model, h5_file, on_tasks, off_tasks, transform='specificity'):
    """Calculate differential attributions for genes"""
    sequences = []
    attributions = []
    
    print(f"Calculating attributions for {len(genes)} genes...")
    print(f"Using transform: {transform}")
    print(f"On tasks: {len(on_tasks)}, Off tasks: {len(off_tasks)}")
    
    with torch.no_grad():
        for gene in tqdm.tqdm(genes):
            try:
                # Differential analysis (target vs others)
                seq, tss_pos, attr = get_attr(
                    gene=gene, h5_file=h5_file, model=model, device=0,
                    tasks=on_tasks, off_tasks=off_tasks, 
                    transform=transform, method=Saliency, abs=False
                )
                
                # Extract 20kb window around TSS
                window_size = 10000
                start_pos = max(0, tss_pos - window_size)
                end_pos = min(seq.shape[1], tss_pos + window_size)
                
                attributions.append(attr[:4, start_pos:end_pos])
                sequences.append(seq[:4, start_pos:end_pos])
                
            except Exception as e:
                print(f"Warning: Error processing gene {gene}: {e}")
                continue
    
    if len(sequences) == 0:
        raise ValueError("No sequences were successfully processed")
    
    sequences = np.stack(sequences)
    attributions = np.stack(attributions)
    
    # Center attributions
    attributions = attributions - attributions.mean(1, keepdims=True)
    
    print(f"Final shapes - Sequences: {sequences.shape}, Attributions: {attributions.shape}")
    
    return sequences, attributions

def main():
    parser = argparse.ArgumentParser(description="Cell type-specific differential motif attribution analysis")
    parser.add_argument("--model_id", type=int, required=True, 
                       help="Model ID (0 for troubleshooting)")
    parser.add_argument("--cell_type_id", type=int, required=True,
                       help="Cell type ID (0 for troubleshooting)")
    parser.add_argument("--h5_file", type=str, required=True,
                       help="Path to H5 data file")
    parser.add_argument("--output_base", type=str, required=True,
                       help="Base output directory")
    parser.add_argument("--timepoint", type=str, default=None,
                       help="Filter to specific timepoint (e.g., '16hpf'). If not specified, uses all timepoints.")
    parser.add_argument("--n_genes", type=int, default=50,
                       help="Number of marker genes to analyze")
    parser.add_argument("--job_id", type=str, default="0",
                       help="SLURM job ID")
    parser.add_argument("--task_id", type=str, default="0", 
                       help="SLURM task ID")
    
    args = parser.parse_args()
    
    # Set up genomepy paths
    set_genomepy_paths(args.job_id, args.task_id)
    
    # Get cell types and model info
    active_cell_types = define_cell_types()  # Only the ones we're testing
    all_cell_types = define_all_cell_types()  # All 25 for differential analysis
    
    if args.cell_type_id >= len(active_cell_types):
        raise ValueError(f"Cell type ID {args.cell_type_id} out of range (0-{len(active_cell_types)-1})")
    
    target_cell_type = active_cell_types[args.cell_type_id]
    model_dir, model_name = get_model_info(args.model_id)
    
    print(f"Running differential analysis:")
    print(f"  Model: {model_name} (ID: {args.model_id})")
    print(f"  Target cell type: {target_cell_type} (ID: {args.cell_type_id})")
    print(f"  Timepoint: {args.timepoint if args.timepoint else 'all timepoints'}")
    print(f"  Transform: specificity (differential)")
    print(f"  Model directory: {model_dir}")
    print(f"  Active cell types: {len(active_cell_types)}")
    print(f"  All cell types for differential: {len(all_cell_types)}")
    
    # Load model and data (with optional timepoint filtering)
    model, ad = load_model_and_data(model_dir, args.h5_file, args.timepoint)
    
    # Prepare differential analysis data (use ALL cell types for proper differential)
    ad_filtered, on_tasks, off_tasks = prepare_differential_analysis(ad, target_cell_type, all_cell_types, args.timepoint)
    
    # Find marker genes
    genes, gene_stats = find_marker_genes(ad_filtered, target_cell_type, args.n_genes)
    
    # Calculate differential attributions
    sequences, attributions = calculate_attributions(
        genes, model, args.h5_file, on_tasks, off_tasks, transform='specificity'
    )
    
    # Create output directory (include timepoint in name if specified)
    safe_cell_type = target_cell_type.replace(" ", "_").replace("/", "_")
    timepoint_suffix = f"_{args.timepoint}" if args.timepoint else ""
    output_dir = os.path.join(args.output_base, f"{model_name}_{safe_cell_type}{timepoint_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy arrays
    seq_path = os.path.join(output_dir, 'sequences.npy')
    attr_path = os.path.join(output_dir, 'attributions.npy')
    
    np.save(seq_path, sequences)
    np.save(attr_path, attributions)
    
    # Save metadata
    metadata = {
        'model_id': args.model_id,
        'model_name': model_name,
        'model_dir': model_dir,
        'cell_type_id': args.cell_type_id,
        'target_cell_type': target_cell_type,
        'timepoint': args.timepoint,
        'analysis_type': 'differential',
        'transform': 'specificity',
        'n_genes': len(genes),
        'genes': genes,
        'n_on_tasks': len(on_tasks),
        'n_off_tasks': len(off_tasks),
        'on_tasks': on_tasks,
        'off_tasks_count_by_celltype': {
            ct: len([t for t in off_tasks if ad_filtered.obs.loc[t, 'Group'] == ct])
            for ct in all_cell_types if ct != target_cell_type
        },
        'troubleshooting_mode': True,
        'active_cell_types': active_cell_types,
        'all_cell_types_count': len(all_cell_types)
    }
    
    # Save gene statistics
    gene_stats.to_csv(os.path.join(output_dir, 'gene_stats.csv'), index=False)
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Differential attribution analysis completed. Files saved to: {output_dir}")
    print(f"Ready for MoDISco analysis.")

if __name__ == "__main__":
    main()