#!/usr/bin/env python3
"""
Single sequence evolution per model for immediate analysis pipeline.
"""

import numpy as np
import pandas as pd
import anndata
import os, sys
import torch
from tqdm import tqdm
import argparse
import csv
import glob
import re
import time
from collections import deque

def parse_args():
    parser = argparse.ArgumentParser(description="Evolve single sequence per model")
    
    # Model and data paths
    parser.add_argument("--device", type=int, default=0, help="GPU device to use")
    parser.add_argument("--model_dir", required=True, help="Path to model checkpoint directory")
    parser.add_argument("--data_dir", required=True, help="Path to data directory containing h5ad files")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    
    # Evolution parameters
    parser.add_argument("--target_celltype", required=True, help="Target cell type to evolve for")
    parser.add_argument("--timepoint", default="16hpf", help="Timepoint to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rounds", type=int, default=50, help="Number of evolution rounds")
    parser.add_argument("--sequence_length", type=int, default=200, help="Length of promoter element")
    
    # Optimization parameters
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--min_improvement", type=float, default=0.001, help="Minimum improvement threshold")
    
    # Genomic coordinates
    parser.add_argument("--chrom", default="4", help="Chromosome for insertion")
    parser.add_argument("--tss_start", type=int, default=29480218, help="TSS start position")
    parser.add_argument("--window_size", type=int, default=524288, help="Sequence window size")
    parser.add_argument("--tss_offset", type=int, default=163840, help="TSS offset")
    
    return parser.parse_args()

def find_checkpoint_file(model_dir):
    """Find the epoch checkpoint file in the model directory"""
    checkpoint_files = glob.glob(os.path.join(model_dir, "**", "*epoch*.ckpt"), recursive=True)
    if not checkpoint_files:
        raise FileNotFoundError(f"No epoch checkpoint files found in {model_dir}")
    return sorted(checkpoint_files)[0]

def extract_model_info(model_dir):
    """Extract model information from directory path"""
    if 'decima_experiments_20250618_111138' in model_dir:
        if 'pretrained_decima-human' in model_dir:
            rep_match = re.search(r'rep(\d+)', model_dir)
            rep = rep_match.group(1) if rep_match else '0'
            return f"Human_Decima_{rep}"
    elif 'decima_experiments_20250617_225114' in model_dir:
        if 'pretrained_wandb-human' in model_dir:
            rep_match = re.search(r'rep(\d+)', model_dir)
            rep = rep_match.group(1) if rep_match else '0'
            return f"Human_Borzoi_{rep}"
        elif 'pretrained_wandb-mouse' in model_dir:
            rep_match = re.search(r'rep(\d+)', model_dir)
            rep = rep_match.group(1) if rep_match else '0'
            return f"Mouse_Borzoi_{rep}"
        elif 'random_lr3e-06' in model_dir:
            seed_match = re.search(r'seed(\d+)', model_dir)
            seed = seed_match.group(1) if seed_match else '42'
            seed_to_rep = {'42': '0', '43': '1', '44': '2', '45': '3'}
            rep = seed_to_rep.get(seed, '0')
            return f"Random_{rep}"
    
    return "Unknown_Model"

def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']
    model_params = ckpt['hyper_parameters']['model_params']
    train_params = ckpt['hyper_parameters']['train_params']
    data_params = ckpt['hyper_parameters'].get('data_params', {})


    # Add decima source to path
    src_dir = '/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/daniodecima-main/src/decima/'
    sys.path.insert(0, src_dir)
    from lightning import LightningModel
    
    model = LightningModel(model_params, train_params, data_params)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    print("Model loaded successfully")
    return model, data_params

def load_data(data_dir):
    """Load the anndata object"""
    # Look for data_out files first, then zebrahub_aggregated
    data_out_files = glob.glob(os.path.join(data_dir, "data_out_*.h5ad"))
    if data_out_files:
        adata_path = data_out_files[0]
    else:
        # Fallback to zebrahub data
        save_dir = "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/data/celltypes_chrom_split_v1/"
        adata_path = os.path.join(save_dir, "zebrahub_aggregated.h5ad")
    
    print(f"Loading data from: {adata_path}")
    ad = anndata.read_h5ad(adata_path)
    ad = ad[:, ad.var.dataset == "test"]
    
    return ad

def get_full_sequence(chrom, tss_start, window_size, tss_offset):
    """Generate the full genomic sequence for the specified region"""
    from grelu.sequence.format import intervals_to_strings
    
    sequence_start = tss_start - tss_offset
    sequence_end = sequence_start + window_size
    
    seq_df = pd.DataFrame([[chrom, sequence_start, sequence_end]], 
                         columns=['chrom', 'start', 'end'])
    
    full_sequence = intervals_to_strings(seq_df, genome="GRCz11")[0]
    return full_sequence

def place_sequence(full_seq, placed_seq, loc):
    """Place a sequence at a specific location within another sequence"""
    return full_seq[:loc] + placed_seq + full_seq[loc + len(placed_seq):]

def make_prediction(model, full_sequence, element_sequence, window_size, tss_offset):
    """Make prediction for a single sequence"""
    from grelu.sequence.format import strings_to_one_hot
    
    mask = np.zeros(window_size)
    mask[tss_offset:tss_offset + len(element_sequence)] = 1
    
    seq_one_hot = strings_to_one_hot(full_sequence, add_batch_axis=False)
    mask_tensor = torch.tensor(mask.reshape(1, -1))
    
    input_tensor = torch.cat((seq_one_hot, mask_tensor), dim=0).float()
    input_tensor = input_tensor.to(model.device)
    
    with torch.no_grad():
        predictions = model.forward(input_tensor).detach().cpu().numpy()
    
    return predictions.squeeze()

def filter_celltypes(task_df, target_celltype, timepoint):
    """Filter celltypes for target and background"""
    target_mask = (task_df['zebrafish_anatomy_ontology_class_fine'] == target_celltype) & \
                  (task_df['timepoint'] == timepoint)
    
    background_mask = (task_df['zebrafish_anatomy_ontology_class_fine'] != target_celltype) & \
                     (task_df['timepoint'] == timepoint)
    
    target_celltypes = task_df[target_mask]
    background_celltypes = task_df[background_mask]
    
    print(f"Target celltypes ({target_celltype}): {len(target_celltypes)}")
    print(f"Background celltypes: {len(background_celltypes)}")
    
    return target_celltypes, background_celltypes

def directed_evolution_single(model, full_sequence, promoter_element, tss_offset, rounds, 
                             target_celltypes, background_celltypes, output_csv, cargo_seq, 
                             early_stopping_patience=15, min_improvement=0.001,
                             checkpoint_interval=10):
    """Simple directed evolution without batching - processes one sequence at a time"""
    from grelu.sequence.mutate import mutate
    
    current_sequence = full_sequence
    element_length = len(promoter_element)
    window_size = len(full_sequence)
    
    # Pre-compute indices for efficiency
    target_indices = target_celltypes.index.values
    background_indices = background_celltypes.index.values
    
    # Track progress
    specificity_history = deque(maxlen=early_stopping_patience * 2)
    start_time = time.time()
    
    print(f"Starting simple directed evolution:")
    print(f"  Rounds: {rounds}")
    print(f"  Element length: {element_length}")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  Output file: {output_csv}")
    
    # Initialize CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Round', 'Position', 'Base', 'Specificity', 
                           'Target_Mean', 'Background_Mean', 'Current_Element', 
                           'Time_Elapsed', 'Convergence_Status'])
        
        for round_num in tqdm(range(rounds), desc="Evolution rounds"):
            round_start_time = time.time()
            
            best_mutation = {
                'position': -1,
                'base': '',
                'specificity': float('-inf'),
                'target_mean': 0,
                'background_mean': 0
            }
            
            # Test all mutations one by one (no batching)
            for position in range(tss_offset, tss_offset + element_length):
                current_base = current_sequence[position]
                
                for base in ['A', 'T', 'G', 'C']:
                    if base == current_base:
                        continue
                    
                    # Create mutated sequence
                    mutated_sequence = mutate(current_sequence, allele=base, pos=position)
                    current_element = mutated_sequence[tss_offset:tss_offset + element_length]
                    full_element = current_element + cargo_seq
                    
                    # Make prediction (single sequence)
                    predictions = make_prediction(model, mutated_sequence, full_element, 
                                                window_size, tss_offset)
                    
                    # Calculate specificity
                    target_mean = predictions[target_indices].mean()
                    background_mean = predictions[background_indices].mean()
                    specificity = target_mean - background_mean
                    
                    # Track best mutation
                    if specificity > best_mutation['specificity']:
                        best_mutation.update({
                            'position': position,
                            'base': base,
                            'specificity': specificity,
                            'target_mean': target_mean,
                            'background_mean': background_mean
                        })
            
            # Apply best mutation
            current_sequence = mutate(current_sequence, 
                                    allele=best_mutation['base'], 
                                    pos=best_mutation['position'])
            current_element = current_sequence[tss_offset:tss_offset + element_length]
            
            # Track specificity for early stopping
            specificity_history.append(best_mutation['specificity'])
            
            # Check for early stopping
            def should_early_stop(specificity_history, patience=15, min_improvement=0.001):
                if len(specificity_history) < patience:
                    return False
                recent_avg = np.mean(specificity_history[-patience//2:])
                older_avg = np.mean(specificity_history[-patience:-patience//2])
                improvement = recent_avg - older_avg
                return improvement < min_improvement
            
            converged = should_early_stop(list(specificity_history), 
                                        early_stopping_patience, min_improvement)
            
            # Calculate timing
            round_time = time.time() - round_start_time
            total_time = time.time() - start_time
            
            # Save results
            csvwriter.writerow([
                round_num + 1,
                best_mutation['position'] - tss_offset,  # Relative position
                best_mutation['base'],
                best_mutation['specificity'],
                best_mutation['target_mean'],
                best_mutation['background_mean'],
                current_element,
                f"{total_time:.1f}s",
                "converged" if converged else "continuing"
            ])
            
            # Flush every few rounds for real-time monitoring
            if round_num % 5 == 0:
                csvfile.flush()
            
            # Progress reporting
            if round_num % checkpoint_interval == 0 or round_num < 5:
                print(f"Round {round_num + 1}: "
                      f"Specificity = {best_mutation['specificity']:.4f}, "
                      f"Position = {best_mutation['position'] - tss_offset}, "
                      f"Base = {best_mutation['base']}, "
                      f"Time = {round_time:.1f}s")
            
            # Early stopping check
            if converged and round_num >= early_stopping_patience:
                print(f"Early stopping triggered at round {round_num + 1}")
                print(f"Converged after {total_time:.1f} seconds")
                break
            
            # Clear GPU cache every 20 rounds
            if round_num % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    final_time = time.time() - start_time
    print(f"Evolution completed in {final_time:.1f} seconds")
    print(f"Average time per round: {final_time/(round_num+1):.1f}s")
    
    return current_element, best_mutation['specificity'], round_num + 1

def main():
    args = parse_args()
    
    # Set up device
    torch.set_float32_matmul_precision("medium")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device(0)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    checkpoint_path = find_checkpoint_file(args.model_dir)
    model, data_params = load_model(checkpoint_path, device)
    
    # Extract model info
    model_info = extract_model_info(args.model_dir)
    
    # Load data
    ad = load_data(args.model_dir)  # Use model_dir to find model-specific data
    
    # Create task dataframe and filter celltypes
    task_df = pd.DataFrame(data_params['tasks'])
    target_celltypes, background_celltypes = filter_celltypes(
        task_df, args.target_celltype, args.timepoint)
    
    if len(target_celltypes) == 0:
        raise ValueError(f"No target celltypes found for '{args.target_celltype}' at {args.timepoint}")
    
    # Get genomic sequence
    full_sequence = get_full_sequence(args.chrom, args.tss_start, 
                                    args.window_size, args.tss_offset)
    
    # Generate starting sequence
    import grelu.sequence.utils
    promoter_element = grelu.sequence.utils.generate_random_sequences(
        args.sequence_length, seed=args.seed, output_format='strings')[0]
    
    # EBFP cargo sequence
    EBFP_seq = ('ATGGCTAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACACTAGTGACCACCCTGTCCCACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTCGAGTACAACTTCAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGCCAACTTCAAGATCCGCCACAATATTGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGCATCACTCACGGCATGGACGAGCTGTACAAG')
    
    # Place initial sequence
    full_inserted_sequence = place_sequence(full_sequence, promoter_element + EBFP_seq, args.tss_offset)
    
    # Create output filename
    celltype_clean = args.target_celltype.replace(' ', '_').replace('/', '_')
    output_filename = f"evolved_promoter_{model_info}_seed{args.seed}_{celltype_clean}_{args.timepoint}_single.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"Evolution setup:")
    print(f"  Model: {model_info}")
    print(f"  Target: {args.target_celltype}")
    print(f"  Output: {output_path}")
    
    # Run evolution
    final_element, final_specificity, total_rounds = directed_evolution_single(
        model=model,
        full_sequence=full_inserted_sequence,
        promoter_element=promoter_element,
        tss_offset=args.tss_offset,
        rounds=args.rounds,
        target_celltypes=target_celltypes,
        background_celltypes=background_celltypes,
        output_csv=output_path,
        cargo_seq=EBFP_seq,
        early_stopping_patience=args.early_stopping_patience,
        min_improvement=args.min_improvement
    )
    
    print(f"\nEvolution completed!")
    print(f"  Final specificity: {final_specificity:.4f}")
    print(f"  Total rounds: {total_rounds}")
    print(f"  Results: {output_path}")

if __name__ == "__main__":
    main()