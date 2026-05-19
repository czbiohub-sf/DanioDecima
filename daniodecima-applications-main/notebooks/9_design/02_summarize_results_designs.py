#!/usr/bin/env python3
"""
Create heatmaps and tables of motif occurrences by cell type for each model.
Enhanced with ISM weight percentile filtering and sequence specificity checks.
Updated for new evolution pipeline results.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def extract_metadata_from_filename(filename):
    """Extract experiment metadata from filename for new pipeline results."""
    # Remove file extension first
    name_without_ext = filename.replace('.csv', '')
    
    # Split by underscores and try to identify components
    parts = name_without_ext.split('_')
    
    try:
        # Expected format: evolved_promoter_ModelType_RepNum_seedSeed_CellType_Timepoint_single_FileType
        # Find the seed part
        seed_idx = None
        for i, part in enumerate(parts):
            if part.startswith('seed'):
                seed_idx = i
                seed = int(part[4:])  # Remove 'seed' prefix
                break
        
        if seed_idx is None:
            print(f"Warning: No seed found in filename: {filename}")
            return None
        
        # Model type and rep number should be before seed
        model_parts = []
        rep_num = None
        
        for i in range(2, seed_idx):  # Start from index 2 (skip 'evolved', 'promoter')
            part = parts[i]
            if part.isdigit():
                rep_num = int(part)
                break
            else:
                model_parts.append(part)
        
        if rep_num is None:
            print(f"Warning: No rep number found in filename: {filename}")
            return None
        
        model_type = '_'.join(model_parts)
        
        # Cell type and timepoint should be after seed
        cell_parts = []
        timepoint = None
        
        for i in range(seed_idx + 1, len(parts)):
            part = parts[i]
            if part.endswith('hpf'):
                timepoint = part
                break
            elif part not in ['single', 'trajectory', 'final', 'motifs']:
                cell_parts.append(part)
        
        if not cell_parts or timepoint is None:
            print(f"Warning: Could not extract cell type or timepoint from filename: {filename}")
            return None
        
        cell_type = ' '.join(cell_parts)
        
        # Standardize model names
        if 'Human' in model_type and 'Borzoi' in model_type:
            model = f"Human-Borzoi_rep{rep_num}"
        elif 'Human' in model_type and 'Decima' in model_type:
            model = f"Human-Decima_rep{rep_num}"
        elif 'Mouse' in model_type and 'Borzoi' in model_type:
            model = f"Mouse-Borzoi_rep{rep_num}"
        elif 'Random' in model_type:
            model = f"Random_rep{rep_num}"
        else:
            model = f"{model_type}_rep{rep_num}"
        
        return {
            'rep_num': rep_num,
            'seed': seed,
            'cell_type': cell_type,
            'timepoint': timepoint,
            'model': model,
            'model_type': model_type
        }
        
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None

def load_trajectory_data(results_dir):
    """Load trajectory data to get final sequence specificity."""
    results_dir = Path(results_dir)
    
    trajectory_data = {}
    trajectory_files = list(results_dir.glob("*_trajectory.csv"))
    
    print(f"Loading trajectory data from {len(trajectory_files)} files...")
    
    for traj_file in trajectory_files:
        try:
            metadata = extract_metadata_from_filename(traj_file.name)
            if metadata and metadata['seed'] == 42:  # Only seed=42
                df = pd.read_csv(traj_file)
                
                # Get final sequence specificity (last row)
                # FIXED: Use lowercase 'specificity' instead of 'Specificity'
                if len(df) > 0 and 'specificity' in df.columns:
                    final_specificity = df['specificity'].iloc[-1]
                    
                    # Create key to match with motif files
                    file_stem = traj_file.stem.replace('_trajectory', '_final_motifs')
                    trajectory_data[file_stem] = {
                        'final_specificity': final_specificity,
                        'metadata': metadata
                    }
                    
        except Exception as e:
            print(f"Error loading trajectory {traj_file}: {e}")
    
    print(f"✓ Loaded trajectory data for {len(trajectory_data)} experiments")
    return trajectory_data

def load_all_results(results_dir):
    """Load motif analysis results for seed=42 only with trajectory data."""
    results_dir = Path(results_dir)
    
    # First load trajectory data
    trajectory_data = load_trajectory_data(results_dir)
    
    all_motifs = []
    all_metadata = []
    
    print(f"Scanning results directory: {results_dir}")
    
    # Find all motif files
    motif_files = list(results_dir.glob("*_final_motifs.csv"))
    print(f"Found {len(motif_files)} motif files")
    
    seed42_count = 0
    excluded_specificity = 0
    
    # Load motif data - only seed=42
    for motif_file in motif_files:
        try:
            metadata = extract_metadata_from_filename(motif_file.name)
            if metadata and metadata['seed'] == 42:  # Only seed=42
                
                # Check trajectory data for specificity
                file_stem = motif_file.stem
                if file_stem in trajectory_data:
                    traj_info = trajectory_data[file_stem]
                    final_specificity = traj_info['final_specificity']
                    
                    # Add specificity info to metadata
                    metadata['final_specificity'] = final_specificity
                    
                    df = pd.read_csv(motif_file)
                    
                    # Add metadata to motifs FIRST
                    for key, value in metadata.items():
                        df[key] = value
                    df['file_stem'] = file_stem
                    
                    # THEN remove duplicates (after file_stem exists)
                    initial_count = len(df)
                    df = df.drop_duplicates(
                        subset=['motif', 'start', 'end', 'strand'])
                    final_count = len(df)
                    
                    if initial_count != final_count:
                        print(f"  Removed {initial_count - final_count} duplicates from {motif_file.name}")
                    
                    all_motifs.append(df)
                    all_metadata.append(metadata)
                    seed42_count += 1
                else:
                    print(f"  No trajectory data for {motif_file.name}, skipping...")
                    excluded_specificity += 1
                    
            elif metadata:
                print(f"  Skipping {motif_file.name} (seed={metadata['seed']}, not 42)")
                
        except Exception as e:
            print(f"Error loading motifs {motif_file}: {e}")
    
    print(f"✓ Found {seed42_count} experiments with seed=42 and trajectory data")
    if excluded_specificity > 0:
        print(f"✗ Excluded {excluded_specificity} experiments missing trajectory data")
    
    # Combine all data
    if all_motifs:
        combined_motifs = pd.concat(all_motifs, ignore_index=True)
        print(f"✓ Loaded {len(combined_motifs)} motif records from seed=42")
    else:
        combined_motifs = pd.DataFrame()
        
    if all_metadata:
        metadata_df = pd.DataFrame(all_metadata)
        print(f"✓ Loaded {len(metadata_df)} experiment metadata records")
        print(f"Models found: {sorted(metadata_df['model'].unique())}")
        print(f"Cell types found: {sorted(metadata_df['cell_type'].unique())}")
    else:
        metadata_df = pd.DataFrame()
    
    return combined_motifs, metadata_df

def calculate_replicate_success_rates(motifs_df):
    """Calculate how many replicates succeeded for each cell type."""
    print("Calculating replicate success rates per cell type...")
    
    # Expected number of replicates per model type (4 each: Human-Borzoi, Human-Decima, Mouse-Borzoi, Random)
    expected_replicates_per_model_type = 4
    expected_model_types = 4  # Human-Borzoi, Human-Decima, Mouse-Borzoi, Random
    total_expected_per_celltype = expected_replicates_per_model_type * expected_model_types
    
    # Count successful replicates per cell type
    replicate_counts = motifs_df.groupby(['cell_type', 'model']).size().reset_index()
    successful_per_celltype = replicate_counts.groupby('cell_type')['model'].nunique()
    
    replicate_stats = {}
    
    for cell_type in motifs_df['cell_type'].unique():
        successful_reps = successful_per_celltype.get(cell_type, 0)
        success_rate = successful_reps / total_expected_per_celltype
        
        replicate_stats[cell_type] = {
            'expected_replicates': total_expected_per_celltype,
            'successful_replicates': successful_reps,
            'failed_replicates': total_expected_per_celltype - successful_reps,
            'success_rate': success_rate
        }
        
        if successful_reps < total_expected_per_celltype:
            print(f"  {cell_type}: {successful_reps}/{total_expected_per_celltype} replicates successful ({success_rate:.1%})")
    
    return replicate_stats

def create_enhanced_plots(motifs_df, tables, output_dir):
    """Create enhanced comparison plots including ISM weight and specificity analysis."""
    output_dir = Path(output_dir)
    
    print("Creating enhanced model comparison plots...")
    
    # Enhanced comparison with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Subplot 1: Total motifs per model
    model_counts = motifs_df.groupby('model').size().sort_values(ascending=False)
    axes[0, 0].bar(model_counts.index, model_counts.values)
    axes[0, 0].set_title('Total Motifs per Model')
    axes[0, 0].set_ylabel('Number of Motifs')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Average ISM weight per model
    avg_ism = motifs_df.groupby('model')['ism_weight'].mean().sort_values(ascending=False)
    axes[0, 1].bar(avg_ism.index, avg_ism.values)
    axes[0, 1].set_title('Average ISM Weight per Model')
    axes[0, 1].set_ylabel('Average ISM Weight')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: Average specificity per model
    avg_spec = motifs_df.groupby('model')['final_specificity'].mean().sort_values(ascending=False)
    axes[0, 2].bar(avg_spec.index, avg_spec.values)
    axes[0, 2].set_title('Average Final Specificity per Model')
    axes[0, 2].set_ylabel('Average Specificity')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Subplot 4: ISM weight distribution
    axes[1, 0].hist(motifs_df['ism_weight'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(motifs_df['ism_weight'].mean(), color='red', linestyle='--', label=f'Mean: {motifs_df["ism_weight"].mean():.4f}')
    axes[1, 0].axvline(motifs_df['ism_weight'].median(), color='orange', linestyle='--', label=f'Median: {motifs_df["ism_weight"].median():.4f}')
    axes[1, 0].set_title('ISM Weight Distribution')
    axes[1, 0].set_xlabel('ISM Weight')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 5: Specificity distribution
    axes[1, 1].hist(motifs_df['final_specificity'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(motifs_df['final_specificity'].mean(), color='red', linestyle='--', label=f'Mean: {motifs_df["final_specificity"].mean():.2f}')
    axes[1, 1].axvline(motifs_df['final_specificity'].median(), color='orange', linestyle='--', label=f'Median: {motifs_df["final_specificity"].median():.2f}')
    axes[1, 1].set_title('Final Specificity Distribution')
    axes[1, 1].set_xlabel('Final Specificity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Subplot 6: ISM weight vs Specificity scatter
    axes[1, 2].scatter(motifs_df['ism_weight'], motifs_df['final_specificity'], alpha=0.6, s=10)
    axes[1, 2].set_xlabel('ISM Weight')
    axes[1, 2].set_ylabel('Final Specificity')
    axes[1, 2].set_title('ISM Weight vs Final Specificity')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = motifs_df['ism_weight'].corr(motifs_df['final_specificity'])
    axes[1, 2].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=axes[1, 2].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Enhanced Model Comparison with ISM Weight and Specificity Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'enhanced_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Enhanced comparison plots saved")

def create_heatmaps_by_model(tables, output_dir, top_n_motifs=30):
    """Create heatmaps for each model showing motif occurrences by cell type."""
    # (Keep the original function unchanged)
    output_dir = Path(output_dir)
    
    print(f"Creating heatmaps for each model (showing top {top_n_motifs} motifs)...")
    
    for model, table in tables.items():
        print(f"Creating heatmap for {model}...")
        
        # Remove the 'Total' row and column for visualization
        viz_table = table.iloc[:-1, :-1]  # Remove last row and column
        
        # Take top N motifs (already sorted by total)
        viz_table = viz_table.head(top_n_motifs)
        
        if viz_table.empty:
            print(f"  No data for {model}, skipping...")
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(8, len(viz_table) * 0.3)))
        
        # Create heatmap
        sns.heatmap(
            viz_table,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar_kws={'label': 'Number of Occurrences'},
            ax=ax
        )
        
        ax.set_title(f'{model}: Motif Occurrences by Cell Type\n(Top {len(viz_table)} motifs)', 
                     fontsize=14, pad=20)
        ax.set_xlabel('Cell Type', fontsize=12)
        ax.set_ylabel('Motif', fontsize=12)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save with model name in filename
        safe_model_name = model.replace('/', '_').replace(' ', '_')
        filename = f'motif_heatmap_{safe_model_name}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved heatmap: {filename}")

def filter_motifs_per_sequence_ism(motifs_df, ism_percentile=90.0, method='percentile'):
    """
    Filter motifs based on ISM weights calculated per-sequence using statistical principles.
    
    Parameters:
    - motifs_df: DataFrame with motif occurrences
    - ism_percentile: Percentile threshold (for percentile method)
    - method: 'percentile', 'zscore', or 'iqr'
    
    Returns:
    - Filtered DataFrame
    """
    if motifs_df.empty:
        return motifs_df
    
    print(f"Applying per-sequence ISM weight filtering using {method} method...")
    
    filtered_dfs = []
    total_sequences = 0
    sequences_with_motifs = 0
    
    # Group by sequence (file_stem) and apply filtering within each sequence
    for file_stem, seq_group in motifs_df.groupby('file_stem'):
        total_sequences += 1
        
        if len(seq_group) == 0:
            continue
            
        sequences_with_motifs += 1
        ism_weights = seq_group['ism_weight'].values
        
        if method == 'percentile':
            # Keep motifs above the specified percentile within this sequence
            if len(ism_weights) > 1:
                threshold = np.percentile(ism_weights, ism_percentile)
                mask = seq_group['ism_weight'] >= threshold
            else:
                mask = np.ones(len(seq_group), dtype=bool)  # Keep single motif
                
        elif method == 'zscore':
            # Keep motifs with z-score >= 1.0 (above mean + 1 std)
            if len(ism_weights) > 1:
                z_scores = (ism_weights - np.mean(ism_weights)) / np.std(ism_weights)
                threshold = 1.0  # Configurable threshold
                mask = z_scores >= threshold
            else:
                mask = np.ones(len(seq_group), dtype=bool)  # Keep single motif
                
        elif method == 'iqr':
            # Keep motifs above Q3 (75th percentile) - more selective than Q3 + 1.5*IQR
            if len(ism_weights) > 1:
                q3 = np.percentile(ism_weights, 75)
                mask = seq_group['ism_weight'] >= q3
            else:
                mask = np.ones(len(seq_group), dtype=bool)  # Keep single motif
                
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Always keep at least one motif per sequence (the highest ISM weight)
        if not mask.any():
            highest_idx = seq_group['ism_weight'].idxmax()
            mask = seq_group.index == highest_idx
            
        filtered_dfs.append(seq_group[mask])
    
    print(f"  Processed {total_sequences} total sequences, {sequences_with_motifs} had motifs")
    
    if filtered_dfs:
        result = pd.concat(filtered_dfs, ignore_index=True)
        print(f"  Kept {len(result)} motifs from per-sequence filtering")
        return result
    else:
        return pd.DataFrame()

def filter_motifs(motifs_df, max_pval=0.05, min_ism_weight=0.0, ism_percentile=50.0, min_specificity=1.0, ism_method='percentile'):
    """Filter motifs based on quality criteria including per-sequence ISM weight percentiles and specificity."""
    if motifs_df.empty:
        return motifs_df
    
    initial_count = len(motifs_df)
    print(f"Starting with {initial_count} total motifs")
    
    # Filter 1: Specificity check
    print(f"Filtering by final sequence specificity >= {min_specificity}")
    specificity_filtered = motifs_df[motifs_df['final_specificity'] >= min_specificity]
    print(f"After specificity filter: {len(specificity_filtered)} motifs ({len(motifs_df) - len(specificity_filtered)} excluded)")
    
    # Filter 2: P-value
    pval_filtered = specificity_filtered[specificity_filtered['pval'] <= max_pval]
    print(f"After p-value filter (<= {max_pval}): {len(pval_filtered)} motifs")
    
    # Filter 3: Basic ISM weight threshold
    basic_ism_filtered = pval_filtered[pval_filtered['ism_weight'] >= min_ism_weight]
    print(f"After basic ISM weight filter (>= {min_ism_weight}): {len(basic_ism_filtered)} motifs")
    
    # Filter 4: PER-SEQUENCE ISM weight filtering (UPDATED)
    if len(basic_ism_filtered) > 0:
        print(f"Applying per-sequence ISM weight filtering ({ism_method}, {ism_percentile}th percentile)...")
        percentile_filtered = filter_motifs_per_sequence_ism(
            basic_ism_filtered, 
            ism_percentile=ism_percentile, 
            method=ism_method
        )
    else:
        percentile_filtered = basic_ism_filtered
    
    # Summary
    print(f"Final filtered dataset: {len(percentile_filtered)} motifs")
    if len(percentile_filtered) > 0:
        print(f"ISM weight range: {percentile_filtered['ism_weight'].min():.4f} - {percentile_filtered['ism_weight'].max():.4f}")
        print(f"Specificity range: {percentile_filtered['final_specificity'].min():.2f} - {percentile_filtered['final_specificity'].max():.2f}")
        
        # Report per-sequence statistics
        per_seq_stats = percentile_filtered.groupby('file_stem').size()
        print(f"Motifs per sequence: mean={per_seq_stats.mean():.1f}, median={per_seq_stats.median():.1f}, range={per_seq_stats.min()}-{per_seq_stats.max()}")
    
    return percentile_filtered

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced motif occurrence analysis with per-sequence ISM weight percentiles and specificity filtering')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing individual analysis results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for summary')
    parser.add_argument('--max_motif_pval', type=float, default=0.05,
                       help='Maximum motif p-value threshold')
    parser.add_argument('--min_ism_weight', type=float, default=0.0,
                       help='Minimum ISM weight threshold (applied before percentile)')
    parser.add_argument('--ism_percentile', type=float, default=75.0,
                       help='Per-sequence ISM weight percentile cutoff (e.g., 75 for top 25%% within each sequence)')
    parser.add_argument('--ism_method', type=str, default='percentile',
                       choices=['percentile', 'zscore', 'iqr'],
                       help='Method for per-sequence ISM weight filtering')
    parser.add_argument('--min_specificity', type=float, default=1.0,
                       help='Minimum final sequence specificity threshold')
    
    return parser.parse_args()

def create_motif_occurrence_tables(motifs_df):
    """Create motif occurrence tables for each model with enhanced statistics and normalization."""
    
    print("Creating motif occurrence tables...")
    
    tables = {}
    summaries = {}
    normalized_tables = {}
    
    # Calculate replicate success rates per cell type
    replicate_stats = calculate_replicate_success_rates(motifs_df)
    
    for model in sorted(motifs_df['model'].unique()):
        print(f"Processing model: {model}")
        
        model_data = motifs_df[motifs_df['model'] == model]
        
        # Count motif occurrences by cell type
        occurrence_table = pd.crosstab(
            model_data['motif'], 
            model_data['cell_type'], 
            margins=True, 
            margins_name='Total'
        )
        
        # Sort by total occurrences (descending)
        occurrence_table = occurrence_table.sort_values('Total', ascending=False)
        
        tables[model] = occurrence_table
        
        # Create normalized table (excluding Total column for normalization)
        normalized_table = occurrence_table.copy()
        
        # Normalize each cell type column by number of successful replicates
        for cell_type in occurrence_table.columns:
            if cell_type != 'Total':  # Don't normalize the Total column
                if cell_type in replicate_stats:
                    successful_reps = replicate_stats[cell_type]['successful_replicates']
                    if successful_reps > 0:
                        normalized_table[cell_type] = normalized_table[cell_type] / successful_reps
                        
        # Recalculate Total for normalized table
        normalized_table['Total'] = normalized_table.drop('Total', axis=1).sum(axis=1)
        normalized_table = normalized_table.sort_values('Total', ascending=False)
        
        normalized_tables[model] = normalized_table
        
        # Create enhanced summary statistics
        summary = {
            'total_motifs': len(model_data),
            'unique_motifs': model_data['motif'].nunique(),
            'sequences_analyzed': model_data['file_stem'].nunique(),
            'cell_types': model_data['cell_type'].nunique(),
            'avg_motifs_per_sequence': len(model_data) / model_data['file_stem'].nunique() if model_data['file_stem'].nunique() > 0 else 0,
            'top_motif': occurrence_table.index[0] if len(occurrence_table) > 1 else 'None',
            'top_motif_count': occurrence_table.iloc[0]['Total'] if len(occurrence_table) > 1 else 0,
            'avg_ism_weight': model_data['ism_weight'].mean(),
            'median_ism_weight': model_data['ism_weight'].median(),
            'avg_specificity': model_data['final_specificity'].mean(),
            'median_specificity': model_data['final_specificity'].median()
        }
        summaries[model] = summary
        
        print(f"  - {summary['total_motifs']} total motifs, {summary['unique_motifs']} unique")
        print(f"  - {summary['sequences_analyzed']} sequences, {summary['cell_types']} cell types")
        print(f"  - Avg ISM weight: {summary['avg_ism_weight']:.4f}, Avg specificity: {summary['avg_specificity']:.2f}")
    
    return tables, summaries, normalized_tables, replicate_stats

def create_celltype_occurrence_tables(motifs_df):
    """Create motif occurrence tables organized by cell type (instead of model)."""
    print("Creating cell type-focused motif occurrence tables...")
    
    tables = {}
    summaries = {}
    normalized_tables = {}
    
    # Calculate replicate success rates per cell type
    replicate_stats = calculate_replicate_success_rates(motifs_df)
    
    # Create tables for each cell type
    for cell_type in motifs_df['cell_type'].unique():
        cell_data = motifs_df[motifs_df['cell_type'] == cell_type]
        
        # Create occurrence table: motifs × models
        occurrence_table = pd.crosstab(
            cell_data['motif'], 
            cell_data['model'],
            margins=True, 
            margins_name='Total'
        ).sort_values('Total', ascending=False)
        
        tables[cell_type] = occurrence_table
        
        # Create normalized table (accounting for failed replicates)
        normalized_table = occurrence_table.copy()
        
        # Get successful replicate count for this cell type
        if cell_type in replicate_stats:
            successful_reps = replicate_stats[cell_type]['successful_replicates']
            expected_reps = replicate_stats[cell_type]['expected_replicates']
            
            if successful_reps > 0:
                # Normalize by successful replicates (not total expected)
                for col in normalized_table.columns:
                    if col != 'Total':
                        # Count successful replicates for this model in this cell type
                        model_success_count = len(cell_data[cell_data['model'].str.contains(col.split('_')[0]) if '_' in col else cell_data['model'] == col]['model'].unique())
                        if model_success_count > 0:
                            normalized_table[col] = normalized_table[col] / model_success_count
        
        # Recalculate Total column for normalized table
        data_cols = [col for col in normalized_table.columns if col != 'Total']
        normalized_table['Total'] = normalized_table[data_cols].sum(axis=1)
        normalized_table = normalized_table.sort_values('Total', ascending=False)
        normalized_tables[cell_type] = normalized_table
        
        # Create summary statistics
        summaries[cell_type] = {
            'total_motifs': len(cell_data),
            'unique_motifs': cell_data['motif'].nunique(),
            'models_analyzed': cell_data['model'].nunique(),
            'sequences_analyzed': cell_data['file_stem'].nunique(),
            'avg_motifs_per_sequence': len(cell_data) / cell_data['file_stem'].nunique(),
            'top_motif': occurrence_table.drop('Total').iloc[0].name if len(occurrence_table) > 1 else 'N/A',
            'top_motif_count': occurrence_table.drop('Total').iloc[0]['Total'] if len(occurrence_table) > 1 else 0,
            'avg_ism_weight': cell_data['ism_weight'].mean(),
            'median_ism_weight': cell_data['ism_weight'].median(),
            'avg_specificity': cell_data['final_specificity'].mean(),
            'median_specificity': cell_data['final_specificity'].median(),
            'successful_replicates': replicate_stats[cell_type]['successful_replicates'] if cell_type in replicate_stats else 0,
            'expected_replicates': replicate_stats[cell_type]['expected_replicates'] if cell_type in replicate_stats else 0
        }
        
        print(f"✓ {cell_type}: {len(occurrence_table)-1} motifs × {len(occurrence_table.columns)-1} models")
    
    return tables, summaries, normalized_tables, replicate_stats

def save_celltype_focused_reports(tables, summaries, motifs_df, output_dir, args, normalized_tables=None, replicate_stats=None):
    """Save cell type-focused reports."""
    output_dir = Path(output_dir)
    
    print("Saving cell type-focused tables and reports...")
    
    # Save individual cell type tables (raw counts)
    for cell_type, table in tables.items():
        safe_celltype_name = cell_type.replace('/', '_').replace(' ', '_')
        filename = f'motif_occurrence_by_celltype_{safe_celltype_name}.csv'
        table.to_csv(output_dir / filename)
        print(f"✓ Saved cell type table: {filename}")
    
    # Save individual cell type normalized tables
    if normalized_tables:
        for cell_type, table in normalized_tables.items():
            safe_celltype_name = cell_type.replace('/', '_').replace(' ', '_')
            filename = f'motif_occurrence_by_celltype_normalized_{safe_celltype_name}.csv'
            table.to_csv(output_dir / filename)
            print(f"✓ Saved normalized cell type table: {filename}")
    
    # Create combined table: motifs × cell types (across all models)
    combined_by_celltype = pd.crosstab(
        motifs_df['motif'], 
        motifs_df['cell_type'],
        margins=True, 
        margins_name='Total'
    ).sort_values('Total', ascending=False)
    combined_by_celltype.to_csv(output_dir / 'motif_occurrence_by_celltype_combined.csv')
    print("✓ Saved combined cell type table")
    
    # IMPORTANT: Save the comprehensive CSV with all motif details
    motifs_df.to_csv(output_dir / 'all_motifs_comprehensive.csv', index=False)
    print("✓ Saved comprehensive motif CSV with all details")
    
    # Create cell type-focused summary report
    with open(output_dir / 'celltype_focused_analysis_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("CELL TYPE-FOCUSED MOTIF OCCURRENCE ANALYSIS (SEED=42)\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total motifs analyzed: {len(motifs_df)}\n")
        f.write(f"Unique motifs: {motifs_df['motif'].nunique()}\n")
        f.write(f"Cell types analyzed: {motifs_df['cell_type'].nunique()}\n")
        f.write(f"Models analyzed: {motifs_df['model'].nunique()}\n\n")
        
        # Cell type-specific summaries
        f.write("CELL TYPE-SPECIFIC SUMMARIES\n")
        f.write("-"*40 + "\n")
        for cell_type, summary in summaries.items():
            f.write(f"\n{cell_type}:\n")
            f.write(f"  Total motifs: {summary['total_motifs']}\n")
            f.write(f"  Unique motifs: {summary['unique_motifs']}\n")
            f.write(f"  Models with data: {summary['models_analyzed']}\n")
            f.write(f"  Successful replicates: {summary['successful_replicates']}/{summary['expected_replicates']}\n")
            f.write(f"  Most frequent motif: {summary['top_motif']} ({summary['top_motif_count']} occurrences)\n")
            f.write(f"  Average specificity: {summary['avg_specificity']:.2f}\n")
    
    print("✓ Cell type-focused reports saved")

def calculate_replicate_success_rates(motifs_df):
    """Calculate how many replicates succeeded for each cell type."""
    print("Calculating replicate success rates per cell type...")
    
    # Expected number of replicates (assuming rep0, rep1, rep2, rep3)
    expected_replicates = 4
    
    # Count successful replicates per cell type
    replicate_counts = motifs_df.groupby(['cell_type', 'model']).size().reset_index()
    successful_per_celltype = replicate_counts.groupby('cell_type')['model'].nunique()
    
    replicate_stats = {}
    
    for cell_type in motifs_df['cell_type'].unique():
        successful_reps = successful_per_celltype.get(cell_type, 0)
        success_rate = successful_reps / expected_replicates
        
        replicate_stats[cell_type] = {
            'expected_replicates': expected_replicates,
            'successful_replicates': successful_reps,
            'failed_replicates': expected_replicates - successful_reps,
            'success_rate': success_rate
        }
        
        if successful_reps < expected_replicates:
            print(f"  {cell_type}: {successful_reps}/{expected_replicates} replicates successful ({success_rate:.1%})")
    
    return replicate_stats

def create_replicate_success_report(replicate_stats, output_dir):
    """Create a detailed report of replicate success rates."""
    output_dir = Path(output_dir)
    
    # Create DataFrame from replicate stats
    stats_df = pd.DataFrame.from_dict(replicate_stats, orient='index')
    stats_df = stats_df.sort_values('success_rate', ascending=False)
    
    # Save CSV
    stats_df.to_csv(output_dir / 'replicate_success_rates.csv')
    
    # Create detailed report
    with open(output_dir / 'replicate_success_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("REPLICATE SUCCESS RATES BY CELL TYPE\n")
        f.write("="*80 + "\n\n")
        
        f.write("This report shows how many model replicates successfully completed\n")
        f.write("directed evolution with final_specificity >= 1.0 for each cell type.\n\n")
        
        f.write(f"{'Cell Type':<30} {'Success':<8} {'Failed':<8} {'Rate':<8} {'Normalization Factor':<20}\n")
        f.write("-" * 80 + "\n")
        
        for cell_type, stats in replicate_stats.items():
            norm_factor = f"÷{stats['successful_replicates']}" if stats['successful_replicates'] > 0 else "÷1"
            f.write(f"{cell_type[:29]:<30} {stats['successful_replicates']:<8} {stats['failed_replicates']:<8} {stats['success_rate']:<8.1%} {norm_factor:<20}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("NORMALIZATION EXPLANATION\n")
        f.write("="*80 + "\n")
        f.write("Raw motif counts are divided by the number of successful replicates\n")
        f.write("to enable fair comparison across cell types. For example:\n\n")
        f.write("- Cell type A: 20 motifs from 4 successful reps → 20÷4 = 5.0 normalized\n")
        f.write("- Cell type B: 15 motifs from 3 successful reps → 15÷3 = 5.0 normalized\n")
        f.write("→ Both cell types show equivalent motif enrichment per replicate\n\n")
    
    print("✓ Replicate success report saved")
    return stats_df

def create_averaged_normalized_table(normalized_tables, motifs_df):
    """Create a table with normalized motif occurrences averaged across model replicates."""
    print("Creating averaged normalized table across model replicates...")
    
    # Collect all normalized data
    all_normalized_data = []
    
    for model, norm_table in normalized_tables.items():
        # Convert to long format
        for cell_type in norm_table.columns:
            if cell_type != 'Total':  # Skip Total column
                for motif in norm_table.index:
                    if motif != 'Total':  # Skip Total row
                        normalized_count = norm_table.loc[motif, cell_type]
                        all_normalized_data.append({
                            'motif': motif,
                            'cell_type': cell_type,
                            'model': model,
                            'normalized_count': normalized_count
                        })
    
    # Convert to DataFrame
    norm_df = pd.DataFrame(all_normalized_data)
    
    # Average across models for each motif-celltype combination
    averaged_norm = norm_df.groupby(['motif', 'cell_type'])['normalized_count'].mean().reset_index()
    
    # Pivot to create motif × cell_type table
    averaged_table = averaged_norm.pivot(index='motif', columns='cell_type', values='normalized_count').fillna(0)
    
    # Add Total column (sum across cell types)
    averaged_table['Total'] = averaged_table.sum(axis=1)
    
    # Sort by Total (descending)
    averaged_table = averaged_table.sort_values('Total', ascending=False)
    
    print(f"✓ Created averaged normalized table: {len(averaged_table)} motifs × {len(averaged_table.columns)-1} cell types")
    
    # Report statistics
    non_zero_counts = (averaged_table.drop('Total', axis=1) > 0).sum().sum()
    total_cells = len(averaged_table) * (len(averaged_table.columns) - 1)
    sparsity = (total_cells - non_zero_counts) / total_cells
    
    print(f"  Average normalized count range: {averaged_table.drop('Total', axis=1).values.min():.3f} - {averaged_table.drop('Total', axis=1).values.max():.3f}")
    print(f"  Table sparsity: {sparsity:.1%} (fraction of zero entries)")
    
    return averaged_table

def save_enhanced_reports(tables, summaries, motifs_df, output_dir, args, normalized_tables=None, replicate_stats=None):
    """Save enhanced reports with ISM weight and specificity information, including normalized tables."""
    output_dir = Path(output_dir)
    
    print("Saving enhanced tables and creating reports...")
    
    # Save individual model tables (raw counts)
    for model, table in tables.items():
        safe_model_name = model.replace('/', '_').replace(' ', '_')
        filename = f'motif_occurrence_table_{safe_model_name}.csv'
        table.to_csv(output_dir / filename)
        print(f"✓ Saved raw table: {filename}")
    
    # Save individual model normalized tables
    if normalized_tables:
        for model, table in normalized_tables.items():
            safe_model_name = model.replace('/', '_').replace(' ', '_')
            filename = f'motif_occurrence_table_normalized_{safe_model_name}.csv'
            table.to_csv(output_dir / filename)
            print(f"✓ Saved normalized table: {filename}")
        
        # NEW: Create and save averaged normalized table for volcano plots
        averaged_normalized_table = create_averaged_normalized_table(normalized_tables, motifs_df)
        averaged_normalized_table.to_csv(output_dir / 'motif_occurrence_table_averaged_normalized.csv')
        print("✓ Saved averaged normalized table for volcano plot analysis")
        
        # Create volcano plot instructions
        create_volcano_plot_instructions(averaged_normalized_table, output_dir)
    
    # Create replicate success report
    if replicate_stats:
        create_replicate_success_report(replicate_stats, output_dir)
    
    # Create enhanced combined tables with additional statistics
    print("Creating enhanced combined motif occurrence tables...")
    
    # Detailed table with ISM weight and specificity stats
    motif_stats = motifs_df.groupby('motif').agg({
        'ism_weight': ['mean', 'median', 'std', 'count'],
        'final_specificity': ['mean', 'median', 'std'],
        'pval': 'mean'
    }).round(4)
    motif_stats.columns = ['_'.join(col).strip() for col in motif_stats.columns.values]
    
    # Combined occurrence table (raw counts)
    combined_detailed = pd.crosstab(
        motifs_df['motif'], 
        [motifs_df['model'], motifs_df['cell_type']], 
        margins=True, 
        margins_name='Total'
    )
    combined_detailed = combined_detailed.sort_values('Total', ascending=False)
    
    # FIXED: Combined normalized occurrence table
    if normalized_tables:
        # Combine all normalized tables
        all_normalized_data = []
        for model, norm_table in normalized_tables.items():
            # FIXED: Make sure to exclude 'Total' column completely
            data_columns = [col for col in norm_table.columns if col != 'Total']
            norm_long = norm_table[data_columns].stack().reset_index()
            norm_long.columns = ['motif', 'cell_type', 'normalized_count']
            norm_long['model'] = model
            all_normalized_data.append(norm_long)
        
        if all_normalized_data:
            combined_norm_long = pd.concat(all_normalized_data, ignore_index=True)
            
            # FIXED: Use different margins_name to avoid conflict
            combined_normalized = pd.crosstab(
                combined_norm_long['motif'],
                [combined_norm_long['model'], combined_norm_long['cell_type']],
                values=combined_norm_long['normalized_count'],
                aggfunc='sum',
                margins=True,
                margins_name='Grand_Total'  # CHANGED from 'Total' to 'Grand_Total'
            ).fillna(0)
            combined_normalized = combined_normalized.sort_values('Grand_Total', ascending=False)
            combined_normalized.to_csv(output_dir / 'motif_occurrence_table_combined_normalized.csv')
            print("✓ Saved combined normalized table")
    
    # Save other tables
    combined_detailed.to_csv(output_dir / 'motif_occurrence_table_combined_detailed.csv')
    motif_stats.to_csv(output_dir / 'motif_statistics_detailed.csv')
    
    # Create simple tables for easier analysis
    simple_combined = pd.crosstab(
        motifs_df['motif'], 
        motifs_df['model'],
        margins=True, 
        margins_name='Total'
    ).sort_values('Total', ascending=False)
    
    enhanced_simple = simple_combined.join(motif_stats, how='left')
    enhanced_simple.to_csv(output_dir / 'motif_occurrence_simple_with_stats.csv')
    
    print("✓ Saved enhanced combined tables")
    
    # Rest of the function remains the same...
    # [Include all the rest of the summary report generation code here]
    
    # Rest of the function for summary report...
    with open(output_dir / 'enhanced_motif_analysis_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("ENHANCED MOTIF OCCURRENCE ANALYSIS (SEED=42)\n")
        f.write("WITH PER-SEQUENCE ISM WEIGHT FILTERING AND REPLICATE NORMALIZATION\n")
        f.write("="*80 + "\n\n")
        
        # Filter parameters
        f.write("FILTER PARAMETERS\n")
        f.write("-"*40 + "\n")
        f.write(f"Max p-value: {args.max_motif_pval}\n")
        f.write(f"Min ISM weight: {args.min_ism_weight}\n")
        f.write(f"ISM weight method: {getattr(args, 'ism_method', 'percentile')}\n")
        f.write(f"ISM weight percentile cutoff: {args.ism_percentile}th percentile\n")
        f.write(f"Min final specificity: {args.min_specificity}\n\n")
        
        # Replicate success summary
        if replicate_stats:
            f.write("REPLICATE SUCCESS SUMMARY\n")
            f.write("-"*40 + "\n")
            total_expected = sum(stats['expected_replicates'] for stats in replicate_stats.values())
            total_successful = sum(stats['successful_replicates'] for stats in replicate_stats.values())
            overall_success_rate = total_successful / total_expected if total_expected > 0 else 0
            
            f.write(f"Total expected replicates: {total_expected}\n")
            f.write(f"Total successful replicates: {total_successful}\n")
            f.write(f"Overall success rate: {overall_success_rate:.1%}\n")
            f.write(f"Cell types analyzed: {len(replicate_stats)}\n\n")
            
            failed_celltypes = [ct for ct, stats in replicate_stats.items() if stats['failed_replicates'] > 0]
            if failed_celltypes:
                f.write(f"Cell types with failed replicates: {len(failed_celltypes)}\n")
                for ct in failed_celltypes[:5]:  # Show first 5
                    stats = replicate_stats[ct]
                    f.write(f"  - {ct}: {stats['successful_replicates']}/{stats['expected_replicates']} successful\n")
                if len(failed_celltypes) > 5:
                    f.write(f"  ... and {len(failed_celltypes) - 5} more (see replicate_success_report.txt)\n")
                f.write("\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total motifs analyzed: {len(motifs_df)}\n")
        f.write(f"Unique motifs: {motifs_df['motif'].nunique()}\n")
        f.write(f"Models analyzed: {motifs_df['model'].nunique()}\n")
        f.write(f"Cell types analyzed: {motifs_df['cell_type'].nunique()}\n")
        f.write(f"Total sequences: {motifs_df['file_stem'].nunique()}\n")
        f.write(f"ISM weight range: {motifs_df['ism_weight'].min():.4f} - {motifs_df['ism_weight'].max():.4f}\n")
        f.write(f"Average ISM weight: {motifs_df['ism_weight'].mean():.4f} ± {motifs_df['ism_weight'].std():.4f}\n")
        f.write(f"Specificity range: {motifs_df['final_specificity'].min():.2f} - {motifs_df['final_specificity'].max():.2f}\n")
        f.write(f"Average specificity: {motifs_df['final_specificity'].mean():.2f} ± {motifs_df['final_specificity'].std():.2f}\n\n")
        
        # Averaged normalized table info
        if normalized_tables:
            f.write("AVERAGED NORMALIZED TABLE\n")
            f.write("-"*40 + "\n")
            f.write("This table contains motif occurrences normalized by successful replicates\n")
            f.write("per cell type, then averaged across the 4 model replicates.\n")
            f.write("→ Use motif_occurrence_table_averaged_normalized.csv for volcano plots\n")
            f.write("→ See volcano_plot_instructions.txt for analysis guidance\n\n")
        
        # Enhanced model-specific summaries
        f.write("ENHANCED MODEL-SPECIFIC SUMMARIES\n")
        f.write("-"*40 + "\n")
        for model, summary in summaries.items():
            f.write(f"\n{model}:\n")
            f.write(f"  Total motifs: {summary['total_motifs']}\n")
            f.write(f"  Unique motifs: {summary['unique_motifs']}\n")
            f.write(f"  Sequences analyzed: {summary['sequences_analyzed']}\n")
            f.write(f"  Cell types: {summary['cell_types']}\n")
            f.write(f"  Avg motifs per sequence: {summary['avg_motifs_per_sequence']:.1f}\n")
            f.write(f"  Most frequent motif: {summary['top_motif']} ({summary['top_motif_count']} occurrences)\n")
            f.write(f"  Average ISM weight: {summary['avg_ism_weight']:.4f} (median: {summary['median_ism_weight']:.4f})\n")
            f.write(f"  Average specificity: {summary['avg_specificity']:.2f} (median: {summary['median_specificity']:.2f})\n")
        
        # Top motifs with enhanced stats
        f.write(f"\nTOP 20 MOTIFS WITH ENHANCED STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Motif':<25} {'Count':<6} {'Avg_ISM':<8} {'Med_ISM':<8} {'Avg_Spec':<8} {'Med_Spec':<8}\n")
        f.write("-" * 80 + "\n")
        
        top_motifs_stats = motifs_df.groupby('motif').agg({
            'motif': 'count',
            'ism_weight': ['mean', 'median'],
            'final_specificity': ['mean', 'median']
        }).round(4)
        
        top_motifs_stats.columns = ['count', 'ism_mean', 'ism_median', 'spec_mean', 'spec_median']
        top_motifs_stats = top_motifs_stats.sort_values('count', ascending=False).head(20)
        
        for motif_name, row in top_motifs_stats.iterrows():
            f.write(f"{motif_name[:24]:<25} {row['count']:<6.0f} {row['ism_mean']:<8.4f} {row['ism_median']:<8.4f} {row['spec_mean']:<8.2f} {row['spec_median']:<8.2f}\n")
    
    # Save enhanced summary table
    summary_df = pd.DataFrame.from_dict(summaries, orient='index')
    summary_df.to_csv(output_dir / 'model_summaries_enhanced.csv')
    
    # Save full filtered dataset with all columns
    motifs_df.to_csv(output_dir / 'filtered_motifs_all_models_enhanced.csv', index=False)
    
    print("✓ Enhanced reports and tables saved")


def main():
    """Enhanced main analysis function."""
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ENHANCED MOTIF OCCURRENCE ANALYSIS (SEED=42 ONLY)")
    print("WITH PER-SEQUENCE ISM WEIGHT FILTERING AND REPLICATE NORMALIZATION")
    print("UPDATED FOR NEW EVOLUTION PIPELINE RESULTS")
    print("="*80)
    print(f"Filters: seed=42, p-value <= {args.max_motif_pval}")
    print(f"         ISM weight >= {args.min_ism_weight}")
    print(f"         Per-sequence {args.ism_method}: {args.ism_percentile}th percentile")
    print(f"         Final specificity >= {args.min_specificity}")
    
    # Load results for seed=42 only (now includes trajectory data)
    motifs_df, metadata_df = load_all_results(args.results_dir)
    
    if motifs_df.empty:
        print("ERROR: No motif data found for seed=42!")
        return
    
    # Enhanced filtering with per-sequence ISM weight filtering
    filtered_motifs = filter_motifs(
        motifs_df, 
        args.max_motif_pval, 
        args.min_ism_weight, 
        args.ism_percentile,
        args.min_specificity,
        args.ism_method
    )
    
    if filtered_motifs.empty:
        print("ERROR: No motifs found after enhanced filtering!")
        return
    
    # Create occurrence tables for each model (now includes normalized tables)
    tables, summaries, normalized_tables, replicate_stats = create_motif_occurrence_tables(filtered_motifs)
    
    # Create original heatmaps
    create_heatmaps_by_model(tables, output_dir)
    
    # Create enhanced plots
    create_enhanced_plots(filtered_motifs, tables, output_dir)
    
    # Save enhanced reports (now includes averaged normalized table for volcano plots)
    save_enhanced_reports(tables, summaries, filtered_motifs, output_dir, args, normalized_tables, replicate_stats)

    # Create cell type-focused tables
    tables, summaries, normalized_tables, replicate_stats = create_celltype_occurrence_tables(filtered_motifs)
    
    # Save cell type-focused reports
    save_celltype_focused_reports(tables, summaries, filtered_motifs, output_dir, args, normalized_tables, replicate_stats)
    
    
    print("="*80)
    print("ENHANCED SEED=42 ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print(f"Created heatmaps for {len(tables)} models")
    print(f"Analyzed {len(filtered_motifs)} high-quality motifs")
    print(f"Per-sequence {args.ism_method} filtering at {args.ism_percentile}th percentile applied")
    print(f"Final specificity >= {args.min_specificity} required")
    print("✓ Normalized tables account for failed replicates per cell type")
    print("✓ Averaged normalized table ready for volcano plot analysis")
    print("="*80)

if __name__ == "__main__":
    main()