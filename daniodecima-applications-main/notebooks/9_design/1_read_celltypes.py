#!/usr/bin/env python3
"""
Analyze a single evolved sequence file.
This script is designed to be run as part of a job array.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker
from pathlib import Path
import torch
import warnings
import re
import csv
from collections import deque
import time
import anndata
from datetime import datetime
warnings.filterwarnings('ignore')

def log_progress(message, start_time=None):
    """Log progress with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if start_time:
        elapsed = time.time() - start_time
        print(f"[{timestamp}] {message} (Elapsed: {elapsed:.1f}s)")
    else:
        print(f"[{timestamp}] {message}")
    # Force flush to ensure immediate output
    sys.stdout.flush()

def parse_arguments():
    """Parse command line arguments."""
    log_progress("Parsing command line arguments")
    parser = argparse.ArgumentParser(description='Analyze single evolved sequence file')
    parser.add_argument('--file_path', type=str, required=True,
                        help='Path to evolved sequence CSV file')
    parser.add_argument('--model_base_dir', type=str, required=True,
                        help='Base directory containing model checkpoints')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--chrom', type=str, default='4',
                        help='Chromosome for genomic context')
    parser.add_argument('--tss_start', type=int, default=29480218,
                        help='TSS start position')
    parser.add_argument('--window_size', type=int, default=524288,
                        help='Window size for genomic context')
    parser.add_argument('--tss_offset', type=int, default=163840,
                        help='TSS offset within window')
    args = parser.parse_args()
    log_progress("✓ Command line arguments parsed successfully")
    return args

# def extract_info_from_filename(filename):
#     """Extract model and experiment information from filename."""
#     log_progress(f"Parsing filename: {filename}")
#     # Example: evolved_promoter_HumanBorzoi_rep0_seed123_neural_crest_16hpf_simple.csv
#     pattern = r'evolved_promoter_HumanBorzoi_rep(\d+)_seed(\d+)_(.+)_(\d+hpf)_simple\.csv'
#     match = re.match(pattern, filename)
    
#     if match:
#         rep_num = int(match.group(1))
#         seed = int(match.group(2))
#         celltype = match.group(3).replace('_', ' ')
#         timepoint = match.group(4)
#         log_progress(f"✓ Filename parsed: rep={rep_num}, seed={seed}, celltype='{celltype}', timepoint='{timepoint}'")
#         return rep_num, seed, celltype, timepoint
#     else:
#         raise ValueError(f"Cannot parse filename: {filename}")

def extract_info_from_filename(filename):
    """Extract model and experiment information from filename."""
    log_progress(f"Parsing filename: {filename}")
    
    # Patterns for combined pipeline results
    patterns = [
        # Pattern 1: Human_Borzoi_0, Human_Decima_0, Mouse_Borzoi_0, Random_0
        r'evolved_promoter_([^_]+_[^_]+)_(\d+)_seed(\d+)_(.+)_(\d+hpf)_single\.csv',
        # Pattern 2: fallback for single word model types
        r'evolved_promoter_([^_]+)_(\d+)_seed(\d+)_(.+)_(\d+hpf)_single\.csv',
        # Pattern 3: old format for backwards compatibility
        r'evolved_promoter_HumanBorzoi_rep(\d+)_seed(\d+)_(.+)_(\d+hpf)_simple\.csv'
    ]
    
    for i, pattern in enumerate(patterns):
        match = re.match(pattern, filename)
        if match:
            if i == 2:  # Old format
                rep_num = int(match.group(1))
                seed = int(match.group(2))
                celltype = match.group(3).replace('_', ' ')
                timepoint = match.group(4)
            else:  # New formats
                model_type = match.group(1)
                rep_num = int(match.group(2))
                seed = int(match.group(3))
                celltype = match.group(4).replace('_', ' ')
                timepoint = match.group(5)
            
            log_progress(f"✓ Filename parsed: rep={rep_num}, seed={seed}, celltype='{celltype}', timepoint='{timepoint}'")
            return rep_num, seed, celltype, timepoint
    
    # If no pattern matches
    raise ValueError(f"Cannot parse filename: {filename}")

def find_checkpoint_file(model_dir):
    """Find the epoch checkpoint file in model directory."""
    log_progress(f"Searching for checkpoint in: {model_dir}")
    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    if os.path.exists(checkpoints_dir):
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith("epoch") and f.endswith(".ckpt")]
        if checkpoint_files:
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_files[0])
            log_progress(f"✓ Found checkpoint: {checkpoint_files[0]}")
            return checkpoint_path
    
    raise FileNotFoundError(f"No epoch checkpoint found in {model_dir}")

def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    start_time = time.time()
    log_progress(f"Loading model from checkpoint: {os.path.basename(checkpoint_path)}")
    
    try:
        log_progress("Loading checkpoint file...")
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        log_progress("✓ Checkpoint file loaded")
        
        state_dict = ckpt['state_dict']
        model_params = ckpt['hyper_parameters']['model_params']
        train_params = ckpt['hyper_parameters']['train_params']
        data_params = ckpt['hyper_parameters'].get('data_params', {})
        log_progress("✓ Checkpoint parameters extracted")

        log_progress("Importing grelu modules...")
        from grelu.sequence.format import strings_to_one_hot, intervals_to_strings
        from grelu.sequence.mutate import mutate
        import grelu.sequence.utils
        log_progress("✓ grelu modules imported")

        # Add decima source to path
        src_dir = '/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-main/src/decima/'
        sys.path.insert(0, src_dir)
        log_progress(f"✓ Added decima path: {src_dir}")
        
        log_progress("Importing LightningModel...")
        from lightning import LightningModel
        log_progress("✓ LightningModel imported")
        
        log_progress("Instantiating model...")
        # Instantiate and load model
        model = LightningModel(model_params, train_params, data_params)
        log_progress("✓ Model instantiated")
        
        log_progress("Loading state dictionary...")
        model.load_state_dict(state_dict)
        log_progress("✓ State dictionary loaded")
        
        log_progress(f"Moving model to device: {device}")
        model = model.to(device)
        model.eval()
        log_progress("✓ Model moved to device and set to eval mode")
        
        # Set model to no-grad mode for inference
        for param in model.parameters():
            param.requires_grad = False
        log_progress("✓ Model parameters set to no-grad")
        
        log_progress("Model loaded successfully", start_time)
        return model, data_params
        
    except Exception as e:
        log_progress(f"ERROR in model loading: {e}")
        import traceback
        traceback.print_exc()
        raise

def load_data(data_dir):
    """Load the AnnData object."""
    start_time = time.time()
    log_progress(f"Loading data from directory: {data_dir}")
    
    try:
        # Look for zebrahub_aggregated.h5ad file
        h5ad_files = [f for f in os.listdir(data_dir) if f.endswith('.h5ad')]
        if not h5ad_files:
            raise FileNotFoundError(f"No h5ad files found in {data_dir}")
        
        adata_path = os.path.join(data_dir, h5ad_files[0])
        log_progress(f"Loading AnnData from: {os.path.basename(adata_path)}")
        ad = anndata.read_h5ad(adata_path)
        log_progress(f"✓ AnnData file loaded (shape: {ad.shape})")
        
        log_progress("Filtering for test data...")
        ad = ad[:, ad.var.dataset == "test"]
        log_progress(f"✓ Data loaded successfully (final shape: {ad.shape})", start_time)
        return ad
        
    except Exception as e:
        log_progress(f"ERROR in data loading: {e}")
        raise

def get_full_sequence(chrom, tss_start, window_size, tss_offset):
    """Get genomic sequence using the same approach as the notebook."""
    start_time = time.time()
    log_progress(f"Retrieving genomic sequence for chr{chrom}:{tss_start-tss_offset}-{tss_start-tss_offset+window_size}")
    
    try:
        log_progress("Importing intervals_to_strings...")
        from grelu.sequence.format import intervals_to_strings
        log_progress("✓ intervals_to_strings imported")
        
        sequence_start_location = tss_start - tss_offset
        sequence_end_location = sequence_start_location + window_size
        seqF = pd.DataFrame([[chrom, sequence_start_location, sequence_end_location]])
        seqF.columns = ['chrom', 'start', 'end']
        log_progress("Calling intervals_to_strings with GRCz11 genome...")
        full_sequence = intervals_to_strings(seqF, genome="GRCz11")[0]
        
        log_progress(f"✓ Retrieved sequence of length {len(full_sequence)}", start_time)
        return full_sequence
        
    except Exception as e:
        log_progress(f"ERROR in sequence retrieval: {e}")
        raise

# def place_sequence(full_seq, placed_seq, loc):
#     """Place a sequence at a specific location within another sequence."""
#     left_of_start = full_seq[0:loc]
#     right_of_start = full_seq[loc:len(full_seq) - len(placed_seq)]
#     new_seq = left_of_start + placed_seq + right_of_start
#     return new_seq

def place_sequence(full_seq, placed_seq, loc):
    """Place a sequence at a specific location within another sequence."""
    return full_seq[:loc] + placed_seq + full_seq[loc + len(placed_seq):]

def make_pred(model, ad, full_inserted_sequence, inserted_sequence, window_size=524288, TSS_offset=163840):
    """Make prediction using the same approach as the notebook."""
    from grelu.sequence.format import strings_to_one_hot
    
    # Create mask
    shape = (window_size)
    arr = np.zeros(shape=shape)
    for i, row in enumerate(ad.var.itertuples()):
        arr[TSS_offset:TSS_offset + len(inserted_sequence)] = 1
    
    # Convert to tensors
    full_seq_one_hot = strings_to_one_hot(full_inserted_sequence, add_batch_axis=False)
    arr_reshaped = torch.tensor(arr.reshape(1, -1))
    x = torch.cat((full_seq_one_hot, arr_reshaped), dim=0).float()
    x = x.to(model.device)
    
    # Make prediction
    with torch.no_grad():
        preds = model.forward(x).detach().cpu().numpy()
    preds = preds.squeeze()
    return preds

def filter_celltypes(task_df, target_celltype, timepoint):
    """Filter celltypes for target and background."""
    log_progress(f"Filtering cell types for target: {target_celltype} at {timepoint}")
    
    target_mask = (task_df['zebrafish_anatomy_ontology_class_fine'] == target_celltype) & \
                  (task_df['timepoint'] == timepoint)
    
    #background_mask = ~target_mask  # Simply the inverse of target mask
    background_mask = (task_df['zebrafish_anatomy_ontology_class_fine'] != target_celltype) & \
                     (task_df['timepoint'] == timepoint)
    
    target_celltypes = task_df[target_mask]
    background_celltypes = task_df[background_mask]
    
    log_progress(f"✓ Found {len(target_celltypes)} target and {len(background_celltypes)} background cell types")
    
    return target_celltypes, background_celltypes

def analyze_evolution_trajectory(df, model, data_params, ad, target_celltype, timepoint, 
                               full_sequence, tss_offset, window_size, output_prefix):
    """Analyze the evolution trajectory from the CSV data."""
    
    start_time = time.time()
    log_progress(f"Starting evolution trajectory analysis for {len(df)} sequences")
    
    # Define EBFP cargo sequence
    EBFP_seq = ('ATGGCTAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACACTAGTGACCACCCTGTCCCACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTCGAGTACAACTTCAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGCCAACTTCAAGATCCGCCACAATATTGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGCATCACTCACGGCATGGACGAGCTGTACAAG')
    
    # Create task dataframe and filter celltypes using data_params
    log_progress("Creating task dataframe and filtering cell types...")
    task_df = pd.DataFrame(data_params['tasks'])
    target_celltypes, background_celltypes = filter_celltypes(task_df, target_celltype, timepoint)
    
    if len(target_celltypes) == 0:
        log_progress(f"WARNING: No target celltypes found for '{target_celltype}' at {timepoint}")
        return None, None
    
    # Get indices
    target_indices = target_celltypes.index.values
    background_indices = background_celltypes.index.values
    log_progress(f"Using {len(target_indices)} target and {len(background_indices)} background indices")
    log_progress(f"Specificity calculation: target_mean - background_mean")
    
    # Analyze each sequence
    predictions_list = []
    sequences = []
    
    log_progress("Starting sequence predictions...")
    prediction_start_time = time.time()
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            elapsed = time.time() - prediction_start_time
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(df) - idx) / rate if rate > 0 else 0
            log_progress(f"Processing sequence {idx+1}/{len(df)} (Rate: {rate:.1f} seq/s, ETA: {eta/60:.1f} min)")
        
        current_element = row['Current_Element']
        sequences.append(current_element)
        
        # Create full element with cargo
        full_element = current_element + EBFP_seq
        
        # Place in genomic context
        full_inserted_sequence = place_sequence(full_sequence, full_element, tss_offset)
        
        # Make prediction
        try:
            predictions = make_pred(model, ad, full_inserted_sequence, full_element, window_size, tss_offset)
            
            # Calculate metrics - using MEAN of background, not max
            target_mean = predictions[target_indices].mean()
            background_mean = predictions[background_indices].mean()
            specificity = target_mean - background_mean
            # background_max = predictions[background_indices].max()
            # specificity = target_mean - background_max
            
            predictions_list.append({
                'round': row['Round'],
                'target_mean': target_mean,
                'background_mean': background_mean,
                'specificity': specificity,
                'sequence': current_element
            })
            
        except Exception as e:
            log_progress(f"Error in prediction for sequence {idx+1}: {e}")
            predictions_list.append({
                'round': row['Round'],
                'target_mean': np.nan,
                'background_mean': np.nan,
                'specificity': np.nan,
                'sequence': current_element
            })
    
    log_progress("✓ All sequences processed")
    
    # Create results dataframe
    results_df = pd.DataFrame(predictions_list)
    
    # Save trajectory results
    trajectory_file = f"{output_prefix}_trajectory.csv"
    results_df.to_csv(trajectory_file, index=False)
    log_progress(f"✓ Saved trajectory results to: {os.path.basename(trajectory_file)}")
    
    # Create evolution plots
    log_progress("Creating evolution plots...")
    create_evolution_plots(results_df, output_prefix)
    
    # Get final evolved sequence for detailed analysis
    final_sequence_row = results_df.iloc[-1]  # Last sequence is the final evolved one
    final_sequence = final_sequence_row['sequence']
    
    log_progress(f"Final evolved sequence: {final_sequence}")
    log_progress(f"Final specificity: {final_sequence_row['specificity']:.4f}")
    log_progress(f"Final target activity: {final_sequence_row['target_mean']:.4f}")
    log_progress(f"Final background activity: {final_sequence_row['background_mean']:.4f}")
    
    log_progress("Evolution trajectory analysis completed", start_time)
    return results_df, final_sequence_row

def create_evolution_plots(results_df, output_prefix):
    """Create plots showing evolution trajectory."""
    start_time = time.time()
    log_progress("Creating 4-panel evolution plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Specificity over rounds
    axes[0, 0].plot(results_df['round'], results_df['specificity'], 'b-', alpha=0.7)
    axes[0, 0].set_xlabel('Evolution Round')
    axes[0, 0].set_ylabel('Specificity (Target - Background Mean)')
    axes[0, 0].set_title('Evolution of Specificity')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Target mean over rounds
    axes[0, 1].plot(results_df['round'], results_df['target_mean'], 'g-', alpha=0.7)
    axes[0, 1].set_xlabel('Evolution Round')
    axes[0, 1].set_ylabel('Target Cell Type Mean Activity')
    axes[0, 1].set_title('Target Activity Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Background mean over rounds
    axes[1, 0].plot(results_df['round'], results_df['background_mean'], 'r-', alpha=0.7)
    axes[1, 0].set_xlabel('Evolution Round')
    axes[1, 0].set_ylabel('Background Mean Activity')
    axes[1, 0].set_title('Background Suppression')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Both target and background
    axes[1, 1].plot(results_df['round'], results_df['target_mean'], 'g-', alpha=0.7, label='Target Mean')
    axes[1, 1].plot(results_df['round'], results_df['background_mean'], 'r-', alpha=0.7, label='Background Mean')
    axes[1, 1].set_xlabel('Evolution Round')
    axes[1, 1].set_ylabel('Activity')
    axes[1, 1].set_title('Target vs Background')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = f"{output_prefix}_evolution_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    log_progress(f"✓ Evolution plots saved to: {os.path.basename(plot_file)}", start_time)


def perform_ism_analysis(sequence, model, data_params, ad, target_celltype, timepoint, 
                        full_sequence, tss_offset, window_size):
    """
    Perform In Silico Mutagenesis (ISM) analysis.
    Returns 3D array of shape (4, sequence_length, n_celltypes)
    """
    EBFP_seq = ('ATGGCTAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACACTAGTGACCACCCTGTCCCACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTCGAGTACAACTTCAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGCCAACTTCAAGATCCGCCACAATATTGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGCATCACTCACGGCATGGACGAGCTGTACAAG')
    
    bases = ['A', 'T', 'G', 'C']
    n_bases = len(bases)
    seq_length = len(sequence)
    
    # Create task dataframe
    task_df = pd.DataFrame(data_params['tasks'])
    n_celltypes = len(task_df)
    
    # Initialize the ISM matrix
    ism_matrix = np.zeros((n_bases, seq_length, n_celltypes))
    
    # Get predictions for the original sequence
    full_element = sequence + EBFP_seq
    full_inserted_sequence = place_sequence(full_sequence, full_element, tss_offset)
    original_preds = make_pred(model, ad, full_inserted_sequence, full_element, window_size, tss_offset)
    
    log_progress(f"Starting ISM analysis: {seq_length} positions × 3 mutations = {seq_length * 3} total mutations")
    
    # Iterate over each position in the sequence
    for pos in range(seq_length):
        if pos % 20 == 0:
            progress = pos / seq_length * 100
            log_progress(f"ISM progress: position {pos}/{seq_length} ({progress:.1f}%)")
            
        original_base = sequence[pos]
        
        # Iterate over each possible base substitution
        for i, base in enumerate(bases):
            if base == original_base:
                continue  # Skip if it's the same as the original base
            
            # Create mutated sequence
            mutated_seq = sequence[:pos] + base + sequence[pos+1:]
            mutated_full_element = mutated_seq + EBFP_seq
            mutated_full_inserted_sequence = place_sequence(full_sequence, mutated_full_element, tss_offset)
            
            # Get predictions for the mutated sequence
            mutated_preds = make_pred(model, ad, mutated_full_inserted_sequence, mutated_full_element, window_size, tss_offset)
            
            # Calculate change in predictions for each cell type
            ism_matrix[i, pos, :] = mutated_preds - original_preds
    
    log_progress("✓ ISM analysis completed")
    return ism_matrix

def create_ism_plots(ism_results, sequence, target_celltype, timepoint, task_df, output_prefix):
    """Create ISM analysis plots matching the original notebook approach."""
    start_time = time.time()
    log_progress("Creating ISM plots...")
    
    # Filter to target cell types (matching original approach)
    target_celltypes, _ = filter_celltypes(task_df, target_celltype, timepoint)
    target_task_list = [target_celltypes.index.values]
    
    # Calculate mean ISM results across target cell types
    mean_ism_results = [ism_results[:, :, tasks].mean(axis=2) for tasks in target_task_list]
    mean_results_line = [mean_result.mean(axis=0) for mean_result in mean_ism_results]
    
    # Plot 1: Line plot showing mean change across positions
    plt.figure(figsize=(12, 4), dpi=200)
    for i, mean_line in enumerate(mean_results_line):
        plt.plot(range(len(sequence)), mean_line, label=f'{target_celltype}', alpha=0.6)
    
    plt.xlabel("Sequence Position")
    plt.ylabel("Mean Change in Prediction")
    plt.xlim(0, len(sequence))
    plt.axhline(y=0, color='darkred', linestyle='--', alpha=0.5)
    plt.title(f'Mean ISM across {target_celltype}')
    plt.legend()
    plt.tight_layout()
    
    line_plot_file = f"{output_prefix}_final_ism_line_plot.png"
    plt.savefig(line_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Heatmap showing all 4 bases (matching original plot_ism_heatmap function)
    bases = ['A', 'T', 'G', 'C']
    mean_ism_2d = mean_ism_results[0]  # Shape: (4, sequence_length)
    
    plt.figure(figsize=(20, 6))
    sns.heatmap(mean_ism_2d, cmap='RdBu_r', center=0, 
                xticklabels=list(sequence), yticklabels=bases)
    
    plt.title(f'ISM Heatmap for {target_celltype}')
    plt.xlabel("Sequence Position")
    plt.ylabel("Mutated Base")
    
    # Add colorbar label
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('Mean Change in Prediction', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    heatmap_file = f"{output_prefix}_final_ism_heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    log_progress(f"✓ ISM line plot saved to: {os.path.basename(line_plot_file)}")
    log_progress(f"✓ ISM heatmap saved to: {os.path.basename(heatmap_file)}", start_time)


def calculate_ism_weight(row: pd.Series, ism: np.ndarray) -> float:
    """Calculate ISM weight for a motif hit based on ISM results."""
    max_ism = np.abs(ism).max(axis=0)
    start, end = min(row["start"], row["end"]), max(row["start"], row["end"])
    return max_ism[start:end].mean()

def perform_tf_analysis(sequence, ism_results, target_celltype, timepoint, task_df, output_prefix):
    """Perform transcription factor motif analysis using the original approach with ISM weights."""
    start_time = time.time()
    log_progress("Starting TF motif scanning...")
    
    try:
        from grelu.interpret.motifs import scan_sequences
        
        # motif file path
        motif_file_path = "/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/9_design/JASPAR2020_CORE_vertebrates_non-redundant_pfms.meme"
        
        # Scan for motifs
        scan = scan_sequences(
            seqs=[sequence],
            motifs=motif_file_path,
            names=None,
            seq_ids=['final_evolved_sequence'],
            rc=True,
        )
        
        if scan is not None and len(scan) > 0:
            log_progress(f"✓ Found {len(scan)} motif hits")
            
            # Calculate motif center
            scan['motif_center'] = scan.apply(
                lambda row: min(row['end'], row['start']) + abs(row['end'] - row['start']) // 2, 
                axis=1
            )
            
            # Get target cell type ISM track
            target_celltypes, _ = filter_celltypes(task_df, target_celltype, timepoint)
            target_ism_track = ism_results[:, :, target_celltypes.index.values].mean(axis=2)
            
            # Calculate ISM weights for each motif hit
            log_progress("Calculating ISM weights for motif hits...")
            scan['ism_weight'] = scan.apply(calculate_ism_weight, ism=target_ism_track, axis=1)
            
            # Save motif results with ISM weights
            motif_file = f"{output_prefix}_final_motifs.csv"
            scan.to_csv(motif_file, index=False)
            log_progress(f"✓ Saved motif results with ISM weights to: {os.path.basename(motif_file)}")
            
            # Log ISM weight statistics
            log_progress(f"ISM weight statistics: mean={scan['ism_weight'].mean():.3f}, "
                        f"max={scan['ism_weight'].max():.3f}, "
                        f"min={scan['ism_weight'].min():.3f}")
            
            # Create motif visualization
            create_motif_plots(scan, sequence, output_prefix)
        else:
            log_progress("No motifs found")
            
        log_progress("TF motif analysis completed", start_time)
        
    except ImportError:
        log_progress("grelu motif scanning not available - skipping TF analysis")
    except Exception as e:
        log_progress(f"Error in TF analysis: {e}")
        import traceback
        traceback.print_exc()

def create_motif_plots(motif_results, sequence, output_prefix):
    """Create motif visualization plots with ISM weights."""
    start_time = time.time()
    log_progress("Creating motif plots...")
    
    try:
        # Plot top motifs by score
        top_motifs = motif_results.nlargest(10, 'score')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Top motifs by score
        for i, (_, motif) in enumerate(top_motifs.iterrows()):
            ax1.barh(i, motif['end'] - motif['start'], 
                    left=motif['start'], 
                    alpha=0.7,
                    label=f"{motif['motif'][:15]} (score: {motif['score']:.2f})")
        
        ax1.set_xlabel('Position in Sequence')
        ax1.set_ylabel('Motifs (by Score)')
        ax1.set_title('Top 10 TF Motifs by Score in Final Evolved Sequence')
        ax1.set_xlim(0, len(sequence))
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Top motifs by ISM weight
        if 'ism_weight' in motif_results.columns:
            top_ism_motifs = motif_results.nlargest(10, 'ism_weight')
            
            for i, (_, motif) in enumerate(top_ism_motifs.iterrows()):
                ax2.barh(i, motif['end'] - motif['start'], 
                        left=motif['start'], 
                        alpha=0.7,
                        label=f"{motif['motif'][:15]} (ISM: {motif['ism_weight']:.3f})")
            
            ax2.set_xlabel('Position in Sequence')
            ax2.set_ylabel('Motifs (by ISM Weight)')
            ax2.set_title('Top 10 TF Motifs by ISM Weight in Final Evolved Sequence')
            ax2.set_xlim(0, len(sequence))
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'ISM weights not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('ISM Weight Analysis (Not Available)')
        
        plt.tight_layout()
        motif_plot_file = f"{output_prefix}_final_motifs_plot.png"
        plt.savefig(motif_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create ISM weight distribution plot if available
        if 'ism_weight' in motif_results.columns:
            plt.figure(figsize=(10, 6))
            
            # Scatter plot: Score vs ISM weight
            plt.subplot(1, 2, 1)
            plt.scatter(motif_results['score'], motif_results['ism_weight'], alpha=0.7)
            plt.xlabel('Motif Score')
            plt.ylabel('ISM Weight')
            plt.title('Score vs ISM Weight')
            plt.grid(True, alpha=0.3)
            
            # Histogram of ISM weights
            plt.subplot(1, 2, 2)
            plt.hist(motif_results['ism_weight'], bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('ISM Weight')
            plt.ylabel('Frequency')
            plt.title('ISM Weight Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            ism_plot_file = f"{output_prefix}_final_motifs_ism_analysis.png"
            plt.savefig(ism_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            log_progress(f"✓ ISM analysis plot saved to: {os.path.basename(ism_plot_file)}")
        
        log_progress(f"✓ Motif plots saved to: {os.path.basename(motif_plot_file)}", start_time)
        
    except Exception as e:
        log_progress(f"Error creating motif plots: {e}")
        import traceback
        traceback.print_exc()

def perform_final_sequence_analysis(final_sequence_row, model, data_params, ad, target_celltype, timepoint,
                                  full_sequence, tss_offset, window_size, output_prefix):
    """Perform detailed analysis on the final evolved sequence only."""
    start_time = time.time()
    
    log_progress("Starting final sequence analysis...")
    log_progress(f"Final sequence (Round {final_sequence_row['round']}): {final_sequence_row['sequence']}")
    log_progress(f"Final specificity: {final_sequence_row['specificity']:.4f}")
    log_progress(f"Final target activity: {final_sequence_row['target_mean']:.4f}")
    log_progress(f"Final background activity: {final_sequence_row['background_mean']:.4f}")
    
    sequence = final_sequence_row['sequence']
    task_df = pd.DataFrame(data_params['tasks'])
    
    # 1. ISM Analysis
    log_progress("Performing ISM analysis on final sequence...")
    ism_results = perform_ism_analysis(sequence, model, data_params, ad, target_celltype, timepoint,
                                     full_sequence, tss_offset, window_size)
    
    # Save raw ISM results
    ism_file = f"{output_prefix}_final_ism_raw.npy"
    np.save(ism_file, ism_results)
    log_progress(f"✓ Saved raw ISM results to: {os.path.basename(ism_file)}")
    
    # Create ISM plots
    create_ism_plots(ism_results, sequence, target_celltype, timepoint, task_df, output_prefix)
    
    # 2. TF Motif Analysis with ISM weights
    log_progress("Performing TF motif analysis with ISM weight calculation...")
    try:
        perform_tf_analysis(sequence, ism_results, target_celltype, timepoint, task_df, output_prefix)
    except Exception as e:
        log_progress(f"Warning: TF analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    log_progress("Final sequence analysis completed", start_time)


def main():
    start_time = time.time()
    
    # Early logging to confirm Python execution starts
    log_progress("="*60)
    log_progress("PYTHON SCRIPT STARTED - EVOLVED SEQUENCE ANALYSIS")
    log_progress("="*60)
    
    try:
        args = parse_arguments()
        
        # Set up device
        log_progress("Setting up computational device...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_progress(f"✓ Using device: {device}")
        
        log_progress(f"Processing file: {args.file_path}")
        log_progress(f"Array task ID: {os.environ.get('SLURM_ARRAY_TASK_ID', 'Not set')}")
        log_progress(f"Model base directory: {args.model_base_dir}")
        log_progress(f"Data directory: {args.data_dir}")
        log_progress(f"Output directory: {args.output_dir}")
        
        # Check if file exists and has content
        log_progress("Validating input file...")
        if not os.path.exists(args.file_path):
            log_progress(f"ERROR: File does not exist: {args.file_path}")
            return
        
        file_size = os.path.getsize(args.file_path)
        log_progress(f"File size: {file_size} bytes")
        
        if file_size == 0:
            log_progress(f"ERROR: File is empty: {args.file_path}")
            return
        
        # Try to load the CSV with error handling
        log_progress("Loading and validating CSV file...")
        try:
            df = pd.read_csv(args.file_path)
            log_progress(f"✓ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            if len(df) == 0:
                log_progress(f"ERROR: CSV file has no data rows")
                return
                
            # Check if required columns exist
            required_columns = ['Current_Element', 'Round']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                log_progress(f"ERROR: Missing required columns: {missing_columns}")
                log_progress(f"Available columns: {list(df.columns)}")
                return
            
            log_progress(f"✓ CSV validation passed")
                
        except pd.errors.EmptyDataError:
            log_progress(f"ERROR: CSV file is empty or has no columns to parse")
            return
        except Exception as e:
            log_progress(f"ERROR: Could not parse CSV file: {e}")
            return
        
        # Extract information from filename
        filename = os.path.basename(args.file_path)
        try:
            rep_num, seed, celltype, timepoint = extract_info_from_filename(filename)
        except ValueError as e:
            log_progress(f"ERROR: {e}")
            return
        
        # # Determine model directory based on rep_num
        # if rep_num == 0:
        #     model_subdir = "0/task_0/lr_3e-05_bs_4_w_0.0001/version_0/"
        #     log_progress(f"Using model for rep0: {model_subdir}")
        # elif rep_num == 1:
        #     model_subdir = "1/task_1/lr_3e-05_bs_4_w_0.0001/version_0/"
        #     log_progress(f"Using model for rep1: {model_subdir}")
        # elif rep_num == 2:
        #     model_subdir = "2/task_2/lr_3e-05_bs_4_w_0.0001/version_0/"
        #     log_progress(f"Using model for rep2: {model_subdir}")
        # elif rep_num == 3:
        #     model_subdir = "3/task_3/lr_3e-05_bs_4_w_0.0001/version_0/"
        #     log_progress(f"Using model for rep3: {model_subdir}")
        # else:
        #     log_progress(f"ERROR: Unknown replicate number: {rep_num}")
        #     return
        
        # model_dir = os.path.join(args.model_base_dir, model_subdir)
        # log_progress(f"Full model path: {model_dir}")

        model_dir = args.model_base_dir
        log_progress(f"Using model directory directly: {model_dir}")
        
        # Find and load model
        try:
            checkpoint_path = find_checkpoint_file(model_dir)
            model, data_params = load_model(checkpoint_path, device)
        except Exception as e:
            log_progress(f"ERROR: Could not load model: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Load data
        try:
            ad = load_data(args.data_dir)
        except Exception as e:
            log_progress(f"ERROR: Could not load data: {e}")
            return
        
        # Get genomic sequence
        try:
            full_sequence = get_full_sequence(args.chrom, args.tss_start, args.window_size, args.tss_offset)
        except Exception as e:
            log_progress(f"ERROR: Could not get genomic sequence: {e}")
            return
        
        # Create output prefix
        output_prefix = os.path.join(args.output_dir, filename.replace('.csv', ''))
        log_progress(f"Output prefix: {output_prefix}")
        
        log_progress("="*60)
        log_progress("SETUP COMPLETED - STARTING ANALYSIS")
        log_progress("="*60)
        
        # Analyze evolution trajectory
        try:
            results_df, final_sequence_row = analyze_evolution_trajectory(
                df, model, data_params, ad, celltype, timepoint,
                full_sequence, args.tss_offset, args.window_size, output_prefix
            )
            
            if results_df is not None and final_sequence_row is not None:
                # Perform detailed analysis on final sequence only
                perform_final_sequence_analysis(
                    final_sequence_row, model, data_params, ad, celltype, timepoint,
                    full_sequence, args.tss_offset, args.window_size, output_prefix
                )
                
                log_progress("="*60)
                log_progress("ANALYSIS COMPLETED SUCCESSFULLY")
                log_progress(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
                log_progress("="*60)
            else:
                log_progress("Analysis failed due to missing target celltypes")
                
        except Exception as e:
            log_progress(f"ERROR: Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return
            
    except Exception as e:
        log_progress(f"CRITICAL ERROR in main(): {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()