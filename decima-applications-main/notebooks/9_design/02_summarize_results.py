#!/usr/bin/env python3
"""
Create heatmaps and tables of motif occurrences by cell type for each model.
Shows motif frequency patterns across cell types with separate analysis per model.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import re
from collections import defaultdict, Counter
warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Motif occurrence analysis by cell type and model')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing individual analysis results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for summary')
    parser.add_argument('--max_motif_pval', type=float, default=0.05,
                       help='Maximum motif p-value threshold')
    parser.add_argument('--min_ism_weight', type=float, default=0.0,
                       help='Minimum ISM weight threshold')
    
    return parser.parse_args()

def extract_metadata_from_filename(filename):
    """Extract experiment metadata from filename."""
    # evolved_promoter_HumanBorzoi_rep0_seed123_neural_crest_16hpf_simple_trajectory.csv
    pattern = r'evolved_promoter_HumanBorzoi_rep(\d+)_seed(\d+)_(.+)_(\d+hpf)_simple'
    match = re.search(pattern, filename)
    
    if match:
        return {
            'rep_num': int(match.group(1)),
            'seed': int(match.group(2)),
            'cell_type': match.group(3).replace('_', ' '),
            'timepoint': match.group(4),
            'model': f"HumanBorzoi_rep{match.group(1)}"
        }
    return None

def load_all_results(results_dir):
    """Load motif analysis results for seed=42 only."""
    results_dir = Path(results_dir)
    
    all_motifs = []
    all_metadata = []
    
    print(f"Scanning results directory: {results_dir}")
    
    # Find all motif files
    motif_files = list(results_dir.glob("*_final_motifs.csv"))
    print(f"Found {len(motif_files)} motif files")
    
    seed42_count = 0
    
    # Load motif data - only seed=42
    for motif_file in motif_files:
        try:
            metadata = extract_metadata_from_filename(motif_file.name)
            if metadata and metadata['seed'] == 42:  # Only seed=42
                df = pd.read_csv(motif_file)
                
                # Add metadata to motifs FIRST
                for key, value in metadata.items():
                    df[key] = value
                df['file_stem'] = motif_file.stem.replace('_final_motifs', '')
                
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
                
            elif metadata:
                print(f"  Skipping {motif_file.name} (seed={metadata['seed']}, not 42)")
                
        except Exception as e:
            print(f"Error loading motifs {motif_file}: {e}")
    
    print(f"✓ Found {seed42_count} experiments with seed=42")
    
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

def filter_motifs(motifs_df, max_pval=0.05, min_ism_weight=0.0):
    """Filter motifs based on quality criteria."""
    if motifs_df.empty:
        return motifs_df
    
    initial_count = len(motifs_df)
    print(f"Starting with {initial_count} total motifs")
    
    # Apply filters
    filtered_df = motifs_df[
        (motifs_df['pval'] <= max_pval) & 
        (motifs_df['ism_weight'] >= min_ism_weight)
    ]
    
    print(f"After p-value filter (<= {max_pval}): {len(motifs_df[motifs_df['pval'] <= max_pval])} motifs")
    print(f"After ISM weight filter (>= {min_ism_weight}): {len(motifs_df[motifs_df['ism_weight'] >= min_ism_weight])} motifs")
    print(f"After both filters: {len(filtered_df)} motifs")
    
    return filtered_df

def create_motif_occurrence_tables(motifs_df):
    """Create motif occurrence tables for each model."""
    
    print("Creating motif occurrence tables...")
    
    tables = {}
    summaries = {}
    
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
        
        # Create summary statistics
        summary = {
            'total_motifs': len(model_data),
            'unique_motifs': model_data['motif'].nunique(),
            'sequences_analyzed': model_data['file_stem'].nunique(),
            'cell_types': model_data['cell_type'].nunique(),
            'avg_motifs_per_sequence': len(model_data) / model_data['file_stem'].nunique(),
            'top_motif': occurrence_table.index[0] if len(occurrence_table) > 1 else 'None',
            'top_motif_count': occurrence_table.iloc[0]['Total'] if len(occurrence_table) > 1 else 0
        }
        summaries[model] = summary
        
        print(f"  - {summary['total_motifs']} total motifs, {summary['unique_motifs']} unique")
        print(f"  - {summary['sequences_analyzed']} sequences, {summary['cell_types']} cell types")
    
    return tables, summaries

def create_heatmaps_by_model(tables, output_dir, top_n_motifs=30):
    """Create heatmaps for each model showing motif occurrences by cell type."""
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

def create_comparison_plots(motifs_df, tables, output_dir):
    """Create additional comparison plots across models."""
    output_dir = Path(output_dir)
    
    print("Creating model comparison plots...")
    
    # Plot 1: Total motifs per model
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Total motifs per model
    model_counts = motifs_df.groupby('model').size().sort_values(ascending=False)
    axes[0, 0].bar(model_counts.index, model_counts.values)
    axes[0, 0].set_title('Total Motifs per Model')
    axes[0, 0].set_ylabel('Number of Motifs')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Unique motifs per model
    unique_counts = motifs_df.groupby('model')['motif'].nunique().sort_values(ascending=False)
    axes[0, 1].bar(unique_counts.index, unique_counts.values)
    axes[0, 1].set_title('Unique Motifs per Model')
    axes[0, 1].set_ylabel('Number of Unique Motifs')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: Motifs per cell type (all models combined)
    celltype_counts = motifs_df.groupby('cell_type').size().sort_values(ascending=False)
    axes[1, 0].bar(celltype_counts.index, celltype_counts.values)
    axes[1, 0].set_title('Total Motifs per Cell Type (All Models)')
    axes[1, 0].set_ylabel('Number of Motifs')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 4: Average ISM weight per model
    avg_ism = motifs_df.groupby('model')['ism_weight'].mean().sort_values(ascending=False)
    axes[1, 1].bar(avg_ism.index, avg_ism.values)
    axes[1, 1].set_title('Average ISM Weight per Model')
    axes[1, 1].set_ylabel('Average ISM Weight')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Model Comparison Summary', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Model comparison plots saved")

def save_tables_and_reports(tables, summaries, motifs_df, output_dir):
    """Save all tables and create detailed reports."""
    output_dir = Path(output_dir)
    
    print("Saving tables and creating reports...")
    
    # Save individual model tables
    for model, table in tables.items():
        safe_model_name = model.replace('/', '_').replace(' ', '_')
        filename = f'motif_occurrence_table_{safe_model_name}.csv'
        table.to_csv(output_dir / filename)
        print(f"✓ Saved table: {filename}")
    
    # Create combined table across all models
    print("Creating combined motif occurrence table...")
    
    # Option 1: Combined table with model+celltype columns
    combined_detailed = pd.crosstab(
        motifs_df['motif'], 
        [motifs_df['model'], motifs_df['cell_type']], 
        margins=True, 
        margins_name='Total'
    )
    # Sort by total occurrences
    combined_detailed = combined_detailed.sort_values('Total', ascending=False)
    combined_detailed.to_csv(output_dir / 'motif_occurrence_table_combined_detailed.csv')
    print("✓ Saved combined detailed table: motif_occurrence_table_combined_detailed.csv")
    
    # Option 2: Combined table summed across models (just cell types)
    combined_summary = pd.crosstab(
        motifs_df['motif'], 
        motifs_df['cell_type'], 
        margins=True, 
        margins_name='Total'
    )
    # Sort by total occurrences
    combined_summary = combined_summary.sort_values('Total', ascending=False)
    combined_summary.to_csv(output_dir / 'motif_occurrence_table_combined_summary.csv')
    print("✓ Saved combined summary table: motif_occurrence_table_combined_summary.csv")
    
    # Create summary report
    with open(output_dir / 'motif_occurrence_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("MOTIF OCCURRENCE ANALYSIS BY MODEL AND CELL TYPE (SEED=42)\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total motifs analyzed: {len(motifs_df)}\n")
        f.write(f"Unique motifs: {motifs_df['motif'].nunique()}\n")
        f.write(f"Models analyzed: {motifs_df['model'].nunique()}\n")
        f.write(f"Cell types analyzed: {motifs_df['cell_type'].nunique()}\n")
        f.write(f"Total sequences: {motifs_df['file_stem'].nunique()}\n\n")
        
        # Model-specific summaries
        f.write("MODEL-SPECIFIC SUMMARIES\n")
        f.write("-"*40 + "\n")
        for model, summary in summaries.items():
            f.write(f"\n{model}:\n")
            f.write(f"  Total motifs: {summary['total_motifs']}\n")
            f.write(f"  Unique motifs: {summary['unique_motifs']}\n")
            f.write(f"  Sequences analyzed: {summary['sequences_analyzed']}\n")
            f.write(f"  Cell types: {summary['cell_types']}\n")
            f.write(f"  Avg motifs per sequence: {summary['avg_motifs_per_sequence']:.1f}\n")
            f.write(f"  Most frequent motif: {summary['top_motif']} ({summary['top_motif_count']} occurrences)\n")
        
        # Top motifs across all models
        f.write(f"\nTOP 20 MOTIFS ACROSS ALL MODELS\n")
        f.write("-"*40 + "\n")
        top_motifs = motifs_df['motif'].value_counts().head(20)
        for i, (motif, count) in enumerate(top_motifs.items(), 1):
            f.write(f"{i:2d}. {motif:<35} {count:3d} occurrences\n")
        
        # Cell type breakdown (combined across models)
        f.write(f"\nMOTIFS BY CELL TYPE (COMBINED ACROSS MODELS)\n")
        f.write("-"*40 + "\n")
        celltype_counts = motifs_df.groupby('cell_type').agg({
            'motif': 'count',
            'file_stem': 'nunique'
        }).round(1)
        celltype_counts.columns = ['total_motifs', 'sequences']
        
        for cell_type, row in celltype_counts.iterrows():
            f.write(f"{cell_type:<25} {row['total_motifs']:3.0f} motifs, {row['sequences']:2.0f} sequences\n")
        
        # Top motifs in combined summary table
        f.write(f"\nTOP 15 MOTIFS BY CELL TYPE (COMBINED)\n")
        f.write("-"*50 + "\n")
        top_combined = combined_summary.head(15).iloc[:-1, :-1]  # Remove Total row/col for display
        
        # Create a nice formatted table
        f.write(f"{'Motif':<35}")
        for cell_type in top_combined.columns:
            f.write(f"{cell_type[:12]:<13}")
        f.write(f"{'Total':<8}\n")
        f.write("-" * (35 + 13 * len(top_combined.columns) + 8) + "\n")
        
        for motif_name, row in top_combined.iterrows():
            f.write(f"{motif_name[:34]:<35}")
            for count in row:
                f.write(f"{count:<13}")
            f.write(f"{row.sum():<8}\n")
    
    # Save combined summary table
    summary_df = pd.DataFrame.from_dict(summaries, orient='index')
    summary_df.to_csv(output_dir / 'model_summaries.csv')
    
    # Save full filtered dataset
    motifs_df.to_csv(output_dir / 'filtered_motifs_all_models.csv', index=False)
    
    print("✓ Reports and tables saved")

def main():
    """Main analysis function."""
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("MOTIF OCCURRENCE ANALYSIS BY MODEL AND CELL TYPE (SEED=42 ONLY)")
    print("="*80)
    print(f"Filters: seed=42, p-value <= {args.max_motif_pval}, ISM weight >= {args.min_ism_weight}")
    
    # Load results for seed=42 only
    motifs_df, metadata_df = load_all_results(args.results_dir)
    
    if motifs_df.empty:
        print("ERROR: No motif data found for seed=42!")
        return
    
    # Filter motifs
    filtered_motifs = filter_motifs(motifs_df, args.max_motif_pval, args.min_ism_weight)
    
    if filtered_motifs.empty:
        print("ERROR: No motifs found after filtering!")
        return
    
    # Create occurrence tables for each model
    tables, summaries = create_motif_occurrence_tables(filtered_motifs)
    
    # Create heatmaps for each model
    create_heatmaps_by_model(tables, output_dir)
    
    # Create comparison plots
    create_comparison_plots(filtered_motifs, tables, output_dir)
    
    # Save tables and reports
    save_tables_and_reports(tables, summaries, filtered_motifs, output_dir)
    
    print("="*80)
    print("SEED=42 ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print(f"Created heatmaps for {len(tables)} models")
    print(f"Analyzed {len(filtered_motifs)} motifs across {filtered_motifs['cell_type'].nunique()} cell types")
    print("="*80)

if __name__ == "__main__":
    main()