# #!/usr/bin/env python3
# """
# Enhanced motif occurrence analysis with cell type focus and comprehensive CSV output.
# Updated for new evolution pipeline results with case-sensitive fix.
# """

# import os
# import sys
# import argparse
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import re
# from datetime import datetime

# def extract_metadata_from_filename(filename):
#     """Extract experiment metadata from filename for new pipeline results."""
#     patterns = [
#         r'evolved_promoter_([^_]+_[^_]+)_(\d+)_seed(\d+)_(.+)_(\d+hpf)_single_(?:trajectory|final_motifs)',
#         r'evolved_promoter_([^_]+)_(\d+)_seed(\d+)_(.+)_(\d+hpf)_single_(?:trajectory|final_motifs)'
#     ]
    
#     for pattern in patterns:
#         match = re.search(pattern, filename)
#         if match:
#             model_type = match.group(1)
#             rep_num = int(match.group(2))
#             seed = int(match.group(3))
#             cell_type = match.group(4).replace('_', ' ')
#             timepoint = match.group(5)
            
#             if 'Human_Borzoi' in model_type:
#                 model = f"Human-Borzoi_rep{rep_num}"
#             elif 'Human_Decima' in model_type:
#                 model = f"Human-Decima_rep{rep_num}"
#             elif 'Mouse_Borzoi' in model_type:
#                 model = f"Mouse-Borzoi_rep{rep_num}"
#             elif 'Random' in model_type:
#                 model = f"Random_rep{rep_num}"
#             else:
#                 model = f"{model_type}_rep{rep_num}"
            
#             return {
#                 'rep_num': rep_num,
#                 'seed': seed,
#                 'cell_type': cell_type,
#                 'timepoint': timepoint,
#                 'model': model,
#                 'model_type': model_type
#             }
    
#     print(f"Warning: Could not parse filename: {filename}")
#     return None

# def load_trajectory_data(results_dir):
#     """Load trajectory data to get final sequence specificity."""
#     results_dir = Path(results_dir)
    
#     trajectory_data = {}
#     trajectory_files = list(results_dir.glob("*_trajectory.csv"))
    
#     print(f"Loading trajectory data from {len(trajectory_files)} files...")
    
#     for traj_file in trajectory_files:
#         try:
#             metadata = extract_metadata_from_filename(traj_file.name)
#             if metadata and metadata['seed'] == 42:  # Only seed=42
#                 df = pd.read_csv(traj_file)
                
#                 # FIXED: Use lowercase 'specificity' instead of 'Specificity'
#                 if len(df) > 0 and 'specificity' in df.columns:
#                     final_specificity = df['specificity'].iloc[-1]
                    
#                     # Create key to match with motif files
#                     file_stem = traj_file.stem.replace('_trajectory', '_final_motifs')
#                     trajectory_data[file_stem] = {
#                         'final_specificity': final_specificity,
#                         'metadata': metadata
#                     }
                    
#         except Exception as e:
#             print(f"Error loading trajectory {traj_file}: {e}")
    
#     print(f"✓ Loaded trajectory data for {len(trajectory_data)} experiments")
#     return trajectory_data

# def load_all_results(results_dir):
#     """Load motif analysis results for seed=42 only with trajectory data."""
#     results_dir = Path(results_dir)
    
#     # First load trajectory data
#     trajectory_data = load_trajectory_data(results_dir)
    
#     all_motifs = []
#     all_metadata = []
    
#     print(f"Scanning results directory: {results_dir}")
    
#     # Find all motif files
#     motif_files = list(results_dir.glob("*_final_motifs.csv"))
#     print(f"Found {len(motif_files)} motif files")
    
#     seed42_count = 0
#     excluded_specificity = 0
    
#     # Load motif data - only seed=42
#     for motif_file in motif_files:
#         try:
#             metadata = extract_metadata_from_filename(motif_file.name)
#             if metadata and metadata['seed'] == 42:  # Only seed=42
                
#                 # Check trajectory data for specificity
#                 file_stem = motif_file.stem
#                 if file_stem in trajectory_data:
#                     traj_info = trajectory_data[file_stem]
#                     final_specificity = traj_info['final_specificity']
                    
#                     # Add specificity info to metadata
#                     metadata['final_specificity'] = final_specificity
                    
#                     df = pd.read_csv(motif_file)
                    
#                     # Add metadata to motifs
#                     for key, value in metadata.items():
#                         df[key] = value
#                     df['file_stem'] = file_stem
                    
#                     # Remove duplicates
#                     initial_count = len(df)
#                     df = df.drop_duplicates(subset=['motif', 'start', 'end', 'strand'])
#                     final_count = len(df)
                    
#                     if initial_count != final_count:
#                         print(f"  Removed {initial_count - final_count} duplicates from {motif_file.name}")
                    
#                     all_motifs.append(df)
#                     all_metadata.append(metadata)
#                     seed42_count += 1
#                 else:
#                     print(f"  No trajectory data for {motif_file.name}, skipping...")
#                     excluded_specificity += 1
                    
#         except Exception as e:
#             print(f"Error loading motifs {motif_file}: {e}")
    
#     print(f"✓ Found {seed42_count} experiments with seed=42 and trajectory data")
#     if excluded_specificity > 0:
#         print(f"✗ Excluded {excluded_specificity} experiments missing trajectory data")
    
#     # Combine all data
#     if all_motifs:
#         combined_motifs = pd.concat(all_motifs, ignore_index=True)
#         print(f"✓ Loaded {len(combined_motifs)} motif records from seed=42")
#     else:
#         combined_motifs = pd.DataFrame()
        
#     if all_metadata:
#         metadata_df = pd.DataFrame(all_metadata)
#         print(f"✓ Loaded {len(metadata_df)} experiment metadata records")
#     else:
#         metadata_df = pd.DataFrame()
    
#     return combined_motifs, metadata_df

# def filter_motifs(motifs_df, max_pval=0.05, min_ism_weight=0.0, ism_percentile=50.0, min_specificity=1.0, ism_method='percentile'):
#     """Filter motifs based on various criteria."""
#     print(f"Filtering motifs: p-value <= {max_pval}, ISM weight >= {min_ism_weight}, specificity >= {min_specificity}")
    
#     initial_count = len(motifs_df)
    
#     # Basic filters
#     filtered = motifs_df[
#         (motifs_df['pval'] <= max_pval) &
#         (motifs_df['ism_weight'] >= min_ism_weight) &
#         (motifs_df['final_specificity'] >= min_specificity)
#     ].copy()
    
#     print(f"✓ After basic filtering: {len(filtered)} motifs (from {initial_count})")
#     return filtered

# def calculate_replicate_success_rates(motifs_df):
#     """Calculate how many replicates succeeded for each cell type."""
#     print("Calculating replicate success rates per cell type...")
    
#     # Expected: 4 replicates per model type × 4 model types = 16 total per cell type
#     expected_replicates_per_celltype = 16
    
#     # Count successful replicates per cell type
#     replicate_counts = motifs_df.groupby(['cell_type', 'model']).size().reset_index()
#     successful_per_celltype = replicate_counts.groupby('cell_type')['model'].nunique()
    
#     replicate_stats = {}
    
#     for cell_type in motifs_df['cell_type'].unique():
#         successful_reps = successful_per_celltype.get(cell_type, 0)
#         success_rate = successful_reps / expected_replicates_per_celltype
        
#         replicate_stats[cell_type] = {
#             'expected_replicates': expected_replicates_per_celltype,
#             'successful_replicates': successful_reps,
#             'failed_replicates': expected_replicates_per_celltype - successful_reps,
#             'success_rate': success_rate
#         }
        
#         if successful_reps < expected_replicates_per_celltype:
#             print(f"  {cell_type}: {successful_reps}/{expected_replicates_per_celltype} replicates successful ({success_rate:.1%})")
    
#     return replicate_stats

# def create_celltype_occurrence_tables(motifs_df):
#     """Create motif occurrence tables organized by cell type."""
#     print("Creating cell type-focused motif occurrence tables...")
    
#     tables = {}
#     summaries = {}
#     normalized_tables = {}
    
#     # Calculate replicate success rates per cell type
#     replicate_stats = calculate_replicate_success_rates(motifs_df)
    
#     # Create tables for each cell type
#     for cell_type in motifs_df['cell_type'].unique():
#         cell_data = motifs_df[motifs_df['cell_type'] == cell_type]
        
#         # Create occurrence table: motifs × models
#         occurrence_table = pd.crosstab(
#             cell_data['motif'], 
#             cell_data['model'],
#             margins=True, 
#             margins_name='Total'
#         ).sort_values('Total', ascending=False)
        
#         tables[cell_type] = occurrence_table
        
#         # Create normalized table (accounting for failed replicates)
#         normalized_table = occurrence_table.copy()
#         normalized_tables[cell_type] = normalized_table
        
#         # Create summary statistics
#         summaries[cell_type] = {
#             'total_motifs': len(cell_data),
#             'unique_motifs': cell_data['motif'].nunique(),
#             'models_analyzed': cell_data['model'].nunique(),
#             'sequences_analyzed': cell_data['file_stem'].nunique(),
#             'avg_motifs_per_sequence': len(cell_data) / cell_data['file_stem'].nunique() if cell_data['file_stem'].nunique() > 0 else 0,
#             'top_motif': occurrence_table.drop('Total').iloc[0].name if len(occurrence_table) > 1 else 'N/A',
#             'top_motif_count': occurrence_table.drop('Total').iloc[0]['Total'] if len(occurrence_table) > 1 else 0,
#             'avg_ism_weight': cell_data['ism_weight'].mean(),
#             'median_ism_weight': cell_data['ism_weight'].median(),
#             'avg_specificity': cell_data['final_specificity'].mean(),
#             'median_specificity': cell_data['final_specificity'].median(),
#             'successful_replicates': replicate_stats[cell_type]['successful_replicates'] if cell_type in replicate_stats else 0,
#             'expected_replicates': replicate_stats[cell_type]['expected_replicates'] if cell_type in replicate_stats else 0
#         }
        
#         print(f"✓ {cell_type}: {len(occurrence_table)-1} motifs × {len(occurrence_table.columns)-1} models")
    
#     return tables, summaries, normalized_tables, replicate_stats

# def create_volcano_plot_instructions(averaged_table, output_dir):
#     """Create instructions for volcano plot analysis."""
#     instructions_file = output_dir / 'volcano_plot_instructions.txt'
    
#     with open(instructions_file, 'w') as f:
#         f.write("VOLCANO PLOT ANALYSIS INSTRUCTIONS\n")
#         f.write("="*50 + "\n\n")
#         f.write("Use the 'motif_occurrence_table_averaged_normalized.csv' file for volcano plot analysis.\n\n")
#         f.write("This table contains:\n")
#         f.write("- Motif occurrences normalized by successful replicates per cell type\n")
#         f.write("- Averaged across the 4 model replicates\n")
#         f.write("- Ready for statistical comparison between conditions\n\n")
#         f.write("Suggested analysis:\n")
#         f.write("1. Compare motif enrichment between cell types\n")
#         f.write("2. Identify cell type-specific motifs\n")
#         f.write("3. Create volcano plots for differential motif usage\n")

# def save_celltype_focused_reports(tables, summaries, motifs_df, output_dir, args, normalized_tables=None, replicate_stats=None):
#     """Save cell type-focused reports."""
#     output_dir = Path(output_dir)
    
#     print("Saving cell type-focused tables and reports...")
    
#     # Save individual cell type tables (raw counts)
#     for cell_type, table in tables.items():
#         safe_celltype_name = cell_type.replace('/', '_').replace(' ', '_')
#         filename = f'motif_occurrence_by_celltype_{safe_celltype_name}.csv'
#         table.to_csv(output_dir / filename)
#         print(f"✓ Saved cell type table: {filename}")
    
#     # Save individual cell type normalized tables
#     if normalized_tables:
#         for cell_type, table in normalized_tables.items():
#             safe_celltype_name = cell_type.replace('/', '_').replace(' ', '_')
#             filename = f'motif_occurrence_by_celltype_normalized_{safe_celltype_name}.csv'
#             table.to_csv(output_dir / filename)
#             print(f"✓ Saved normalized cell type table: {filename}")
    
#     # Create combined table: motifs × cell types (across all models)
#     combined_by_celltype = pd.crosstab(
#         motifs_df['motif'], 
#         motifs_df['cell_type'],
#         margins=True, 
#         margins_name='Total'
#     ).sort_values('Total', ascending=False)
#     combined_by_celltype.to_csv(output_dir / 'motif_occurrence_by_celltype_combined.csv')
#     print("✓ Saved combined cell type table")
    
#     # IMPORTANT: Save the comprehensive CSV with all motif details
#     motifs_df.to_csv(output_dir / 'all_motifs_comprehensive.csv', index=False)
#     print("✓ Saved comprehensive motif CSV with all details")
    
#     # Create additional useful tables
#     # Model comparison table: motifs × models (across all cell types)
#     combined_by_model = pd.crosstab(
#         motifs_df['motif'], 
#         motifs_df['model'],
#         margins=True, 
#         margins_name='Total'
#     ).sort_values('Total', ascending=False)
#     combined_by_model.to_csv(output_dir / 'motif_occurrence_by_model_combined.csv')
#     print("✓ Saved combined model table")
    
#     # Create volcano plot instructions
#     if normalized_tables:
#         create_volcano_plot_instructions(None, output_dir)
    
#     # Create cell type-focused summary report
#     with open(output_dir / 'celltype_focused_analysis_summary.txt', 'w') as f:
#         f.write("="*80 + "\n")
#         f.write("CELL TYPE-FOCUSED MOTIF OCCURRENCE ANALYSIS (SEED=42)\n")
#         f.write("="*80 + "\n\n")
        
#         # Overall statistics
#         f.write("OVERALL STATISTICS\n")
#         f.write("-"*40 + "\n")
#         f.write(f"Total motifs analyzed: {len(motifs_df)}\n")
#         f.write(f"Unique motifs: {motifs_df['motif'].nunique()}\n")
#         f.write(f"Cell types analyzed: {motifs_df['cell_type'].nunique()}\n")
#         f.write(f"Models analyzed: {motifs_df['model'].nunique()}\n")
#         f.write(f"ISM weight range: {motifs_df['ism_weight'].min():.4f} - {motifs_df['ism_weight'].max():.4f}\n")
#         f.write(f"Specificity range: {motifs_df['final_specificity'].min():.2f} - {motifs_df['final_specificity'].max():.2f}\n\n")
        
#         # Cell type-specific summaries
#         f.write("CELL TYPE-SPECIFIC SUMMARIES\n")
#         f.write("-"*40 + "\n")
#         for cell_type, summary in summaries.items():
#             f.write(f"\n{cell_type}:\n")
#             f.write(f"  Total motifs: {summary['total_motifs']}\n")
#             f.write(f"  Unique motifs: {summary['unique_motifs']}\n")
#             f.write(f"  Models with data: {summary['models_analyzed']}\n")
#             f.write(f"  Successful replicates: {summary['successful_replicates']}/{summary['expected_replicates']}\n")
#             f.write(f"  Most frequent motif: {summary['top_motif']} ({summary['top_motif_count']} occurrences)\n")
#             f.write(f"  Average specificity: {summary['avg_specificity']:.2f}\n")
        
#         # Top motifs overall
#         f.write(f"\nTOP 20 MOTIFS OVERALL\n")
#         f.write("-"*40 + "\n")
#         top_motifs = motifs_df['motif'].value_counts().head(20)
#         for motif, count in top_motifs.items():
#             f.write(f"  {motif}: {count} occurrences\n")
    
#     print("✓ Cell type-focused reports saved")

# def parse_arguments():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description="Enhanced motif occurrence analysis with cell type focus")
    
#     parser.add_argument('--results_dir', required=True, help='Directory containing motif analysis results')
#     parser.add_argument('--output_dir', required=True, help='Output directory for tables and plots')
#     parser.add_argument('--max_motif_pval', type=float, default=0.05, help='Maximum motif p-value')
#     parser.add_argument('--min_ism_weight', type=float, default=0.0, help='Minimum ISM weight')
#     parser.add_argument('--ism_percentile', type=float, default=75.0, help='ISM weight percentile cutoff')
#     parser.add_argument('--min_specificity', type=float, default=0.5, help='Minimum final specificity')
#     parser.add_argument('--ism_method', default='percentile', help='ISM filtering method')
    
#     return parser.parse_args()

# def main():
#     """Cell type-focused main analysis function."""
#     args = parse_arguments()
    
#     # Create output directory
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     print("Starting enhanced comprehensive evolution analysis...")
#     print(f"Results directory: {args.results_dir}")
#     print(f"Output directory: {args.output_dir}")
    
#     print("="*80)
#     print("ENHANCED MOTIF OCCURRENCE ANALYSIS (SEED=42 ONLY)")
#     print("WITH PER-SEQUENCE ISM WEIGHT FILTERING AND REPLICATE NORMALIZATION")
#     print("UPDATED FOR NEW EVOLUTION PIPELINE RESULTS")
#     print("="*80)
#     print(f"Filters: seed=42, p-value <= {args.max_motif_pval}")
#     print(f"         ISM weight >= {args.min_ism_weight}")
#     print(f"         Per-sequence percentile: {args.ism_percentile}th percentile")
#     print(f"         Final specificity >= {args.min_specificity}")
    
#     # Load and filter results
#     motifs_df, metadata_df = load_all_results(args.results_dir)
    
#     if motifs_df.empty:
#         print("ERROR: No motif data found for seed=42!")
#         return
    
#     # Apply filters
#     filtered_motifs = filter_motifs(
#         motifs_df, 
#         args.max_motif_pval, 
#         args.min_ism_weight, 
#         args.ism_percentile,
#         args.min_specificity,
#         args.ism_method
#     )
    
#     if filtered_motifs.empty:
#         print("ERROR: No motifs found after filtering!")
#         return
    
#     # Create cell type-focused tables
#     tables, summaries, normalized_tables, replicate_stats = create_celltype_occurrence_tables(filtered_motifs)
    
#     # Save cell type-focused reports
#     save_celltype_focused_reports(tables, summaries, filtered_motifs, output_dir, args, normalized_tables, replicate_stats)
    
#     print("="*80)
#     print("ENHANCED COMPREHENSIVE ANALYSIS COMPLETE!")
#     print(f"✓ Comprehensive motif CSV: all_motifs_comprehensive.csv")
#     print(f"✓ Cell type tables: motif_occurrence_by_celltype_*.csv")
#     print(f"✓ Combined tables: motif_occurrence_by_celltype_combined.csv")
#     print(f"✓ Model comparison: motif_occurrence_by_model_combined.csv")
#     print(f"Check outputs in: {output_dir}")
#     print("="*80)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Enhanced motif occurrence analysis with cell type focus and comprehensive CSV output.
Updated for new nested directory structure (analysis_25ct_20250619).
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import re

def find_analysis_directories(results_dir):
    """Find all analysis subdirectories in the new nested structure."""
    results_dir = Path(results_dir)
    
    analysis_dirs = []
    
    # Look for directories that contain CSV files
    for item in results_dir.iterdir():
        if item.is_dir():
            csv_files = list(item.glob("*.csv"))
            if csv_files:
                analysis_dirs.append(item)
    
    # If no direct subdirectories found, look one level deeper
    if not analysis_dirs:
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                for item in subdir.iterdir():
                    if item.is_dir():
                        csv_files = list(item.glob("*.csv"))
                        if csv_files:
                            analysis_dirs.append(item)
    
    print(f"Found {len(analysis_dirs)} analysis directories")
    for dir_path in analysis_dirs[:5]:  # Show first 5
        print(f"  {dir_path.name}")
    if len(analysis_dirs) > 5:
        print(f"  ... and {len(analysis_dirs) - 5} more")
    
    return analysis_dirs

def debug_filenames(results_dir):
    """Debug function to show all filenames in the analysis directories."""
    results_dir = Path(results_dir)
    analysis_dirs = find_analysis_directories(results_dir)
    
    print("=== FILENAME DEBUG ===")
    for analysis_dir in analysis_dirs[:3]:  # Show first 3 directories
        print(f"\nDirectory: {analysis_dir.name}")
        csv_files = list(analysis_dir.glob("*.csv"))
        for csv_file in csv_files:
            print(f"  File: {csv_file.name}")
            
            # Try to show what the regex would extract
            patterns = [
                r'evolved_promoter_([^_]+_[^_]+)_(\d+)_seed(\d+)_(.+?)_(\d+hpf)_single_',
                r'evolved_promoter_([^_]+)_(\d+)_seed(\d+)_(.+?)_(\d+hpf)_single_'
            ]
            
            for i, pattern in enumerate(patterns):
                match = re.search(pattern, csv_file.name)
                if match:
                    print(f"    Pattern {i+1} matched:")
                    print(f"      Model: {match.group(1)}")
                    print(f"      Rep: {match.group(2)}")
                    print(f"      Seed: {match.group(3)}")
                    print(f"      Cell type: {match.group(4)}")
                    print(f"      Timepoint: {match.group(5)}")
                    break
            else:
                print(f"    No patterns matched")

def extract_metadata_from_csv_content_improved(df, dirname, motif_file_path):
    """Extract metadata from CSV file content and infer from file patterns."""
    metadata = {
        'analysis_id': dirname,
        'rep_num': None,
        'seed': None,
        'cell_type': None,
        'timepoint': None,
        'model': None,
        'model_type': None
    }
    
    # Extract metadata from filename patterns
    filename = motif_file_path.name
    print(f"    Parsing filename: {filename}")
    
    # Pattern for the new filenames: evolved_promoter_Human_Decima_0_seed42_axaxial_cell_16hpf_single_*
    # Try multiple patterns to handle variations
    patterns = [
        r'evolved_promoter_([^_]+_[^_]+)_(\d+)_seed(\d+)_(.+?)_(\d+hpf)_single_',
        r'evolved_promoter_([^_]+)_(\d+)_seed(\d+)_(.+?)_(\d+hpf)_single_',
        r'.*_([^_]+_[^_]+)_(\d+)_seed(\d+)_(.+?)_(\d+hpf)_',
        r'.*_([^_]+)_(\d+)_seed(\d+)_(.+?)_(\d+hpf)_'
    ]
    
    parsed = False
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            model_type_raw = match.group(1)
            rep_num = int(match.group(2))
            seed = int(match.group(3))
            cell_type_raw = match.group(4)
            timepoint = match.group(5)
            
            # Clean up model type
            if 'Human_Decima' in model_type_raw or 'Human_Decima' in filename:
                model_type = 'Human_Decima'
            elif 'Human_Borzoi' in model_type_raw or 'Human_Borzoi' in filename:
                model_type = 'Human_Borzoi'
            elif 'Mouse_Borzoi' in model_type_raw or 'Mouse_Borzoi' in filename:
                model_type = 'Mouse_Borzoi'
            elif 'Random' in model_type_raw or 'Random' in filename:
                model_type = 'Random'
            else:
                model_type = model_type_raw
            
            # Clean up cell type - replace underscores with spaces and capitalize
            # Make sure we don't use generic terms
            if cell_type_raw and cell_type_raw not in ['evolved', 'sequence', 'final', 'single']:
                cell_type = cell_type_raw.replace('_', ' ').title()
            else:
                cell_type = None
            
            metadata.update({
                'model_type': model_type,
                'rep_num': rep_num,
                'seed': seed,
                'cell_type': cell_type,
                'timepoint': timepoint,
                'model': f"{model_type}_rep{rep_num}"
            })
            
            print(f"    ✓ Parsed: {cell_type} | {model_type} rep{rep_num} | seed{seed}")
            parsed = True
            break
    
    if not parsed:
        print(f"    Warning: Could not parse filename pattern for {filename}")
        print(f"    Trying alternative extraction methods...")
        
        # Try to extract seed from filename
        seed_match = re.search(r'seed(\d+)', filename)
        if seed_match:
            metadata['seed'] = int(seed_match.group(1))
            print(f"    Found seed: {metadata['seed']}")
        else:
            metadata['seed'] = 42  # Default assumption
            print(f"    Using default seed: 42")
            
        # Try to extract model type from filename
        if 'Human_Decima' in filename:
            metadata['model_type'] = 'Human_Decima'
        elif 'Human_Borzoi' in filename:
            metadata['model_type'] = 'Human_Borzoi'
        elif 'Mouse_Borzoi' in filename:
            metadata['model_type'] = 'Mouse_Borzoi'
        elif 'Random' in filename:
            metadata['model_type'] = 'Random'
        
        print(f"    Found model type: {metadata['model_type']}")
        
        # Extract replicate number
        rep_match = re.search(r'_(\d+)_seed', filename)
        if rep_match:
            metadata['rep_num'] = int(rep_match.group(1))
            print(f"    Found replicate: {metadata['rep_num']}")
        
        # Try to extract cell type from various parts of the filename
        # Look for patterns between seed and hpf
        celltype_match = re.search(r'seed\d+_(.+?)_\d+hpf', filename)
        if celltype_match:
            cell_type_raw = celltype_match.group(1)
            if cell_type_raw and cell_type_raw not in ['evolved', 'sequence', 'final', 'single']:
                metadata['cell_type'] = cell_type_raw.replace('_', ' ').title()
                print(f"    Found cell type: {metadata['cell_type']}")
        
        # Set default timepoint
        timepoint_match = re.search(r'(\d+)hpf', filename)
        if timepoint_match:
            metadata['timepoint'] = f"{timepoint_match.group(1)}hpf"
        else:
            metadata['timepoint'] = '16hpf'  # Default for this dataset
        
        print(f"    Found timepoint: {metadata['timepoint']}")
        
        # Create model name
        if metadata['model_type'] and metadata['rep_num'] is not None:
            metadata['model'] = f"{metadata['model_type']}_rep{metadata['rep_num']}"
        elif metadata['model_type']:
            metadata['model'] = metadata['model_type']
        else:
            metadata['model'] = dirname
        
        print(f"    Created model name: {metadata['model']}")
    
    # Final check - if cell_type is still None or generic, try to infer from directory structure
    if not metadata['cell_type'] or metadata['cell_type'] in ['Final Evolved Sequence', 'Evolved', 'Sequence']:
        print(f"    Cell type is generic or None, trying to infer from context...")
        
        # Look at all files in the directory to see if there's a pattern
        try:
            all_files = list(motif_file_path.parent.glob("*.csv"))
            for file in all_files:
                if file != motif_file_path:
                    # Try to extract cell type from other filenames in the same directory
                    other_celltype_match = re.search(r'seed\d+_(.+?)_\d+hpf', file.name)
                    if other_celltype_match:
                        cell_type_candidate = other_celltype_match.group(1)
                        if cell_type_candidate not in ['evolved', 'sequence', 'final', 'single']:
                            metadata['cell_type'] = cell_type_candidate.replace('_', ' ').title()
                            print(f"    Inferred cell type from other files: {metadata['cell_type']}")
                            break
        except (AttributeError, KeyError, IndexError, OSError):
            # filename-pattern fallback failed; metadata stays empty for this entry
            pass

    return metadata

def load_trajectory_data_nested(results_dir):
    """Load trajectory data from nested directory structure."""
    trajectory_data = {}
    
    analysis_dirs = find_analysis_directories(results_dir)
    
    print(f"Loading trajectory data from {len(analysis_dirs)} analysis directories...")
    
    for analysis_dir in analysis_dirs:
        try:
            # Look for trajectory files in this directory
            trajectory_files = list(analysis_dir.glob("*trajectory*.csv"))
            
            for traj_file in trajectory_files:
                df = pd.read_csv(traj_file)
                
                # Extract metadata from directory and file
                metadata = extract_metadata_from_csv_content_improved(df, analysis_dir.name, traj_file)
                
                # Only process seed=42 experiments
                if metadata.get('seed') == 42:
                    if len(df) > 0 and 'specificity' in df.columns:
                        final_specificity = df['specificity'].iloc[-1]
                        
                        # Use directory name as key
                        trajectory_data[analysis_dir.name] = {
                            'final_specificity': final_specificity,
                            'metadata': metadata,
                            'analysis_dir': analysis_dir
                        }
                        
        except Exception as e:
            print(f"Error loading trajectory from {analysis_dir}: {e}")
    
    print(f"✓ Loaded trajectory data for {len(trajectory_data)} experiments")
    return trajectory_data

def load_all_results_nested(results_dir):
    """Load motif analysis results from nested directory structure."""
    results_dir = Path(results_dir)
    
    # First load trajectory data
    trajectory_data = load_trajectory_data_nested(results_dir)
    
    all_motifs = []
    all_metadata = []
    
    print(f"Scanning nested results directory: {results_dir}")
    
    analysis_dirs = find_analysis_directories(results_dir)
    
    seed42_count = 0
    excluded_specificity = 0
    
    # Load motif data from each analysis directory
    for analysis_dir in analysis_dirs:
        print(f"\nProcessing directory: {analysis_dir.name}")
        try:
            # Look for different types of CSV files
            csv_files = list(analysis_dir.glob("*.csv"))
            
            # Separate different file types
            motif_files = []
            trajectory_files = []
            other_files = []
            
            for csv_file in csv_files:
                filename_lower = csv_file.name.lower()
                if 'trajectory' in filename_lower:
                    trajectory_files.append(csv_file)
                elif any(keyword in filename_lower for keyword in ['motif', 'final', 'evolved']):
                    motif_files.append(csv_file)
                else:
                    other_files.append(csv_file)
            
            print(f"  Found {len(motif_files)} motif files, {len(trajectory_files)} trajectory files, {len(other_files)} other files")
            
            for motif_file in motif_files:
                print(f"  Processing motif file: {motif_file.name}")
                
                try:
                    df = pd.read_csv(motif_file)
                    print(f"    Loaded {len(df)} rows, columns: {list(df.columns)}")
                    
                    # Extract metadata from directory and file content
                    metadata = extract_metadata_from_csv_content_improved(df, analysis_dir.name, motif_file)
                    
                    print(f"    Extracted metadata: cell_type='{metadata.get('cell_type')}', model='{metadata.get('model')}', seed={metadata.get('seed')}")
                    
                    # Only process seed=42 experiments
                    if metadata.get('seed') == 42:
                        
                        # Check if we have trajectory data for this experiment
                        if analysis_dir.name in trajectory_data:
                            traj_info = trajectory_data[analysis_dir.name]
                            final_specificity = traj_info['final_specificity']
                            print(f"    Found trajectory data, specificity: {final_specificity}")
                        else:
                            # If no trajectory data, still include but set default specificity
                            final_specificity = 1.0  # Default value
                            excluded_specificity += 1
                            print(f"    No trajectory data found, using default specificity: {final_specificity}")
                        
                        # Add specificity info to metadata
                        metadata['final_specificity'] = final_specificity
                        
                        # Add metadata to motifs
                        for key, value in metadata.items():
                            df[key] = value
                        df['file_stem'] = analysis_dir.name
                        df['analysis_dir'] = str(analysis_dir)
                        
                        # Remove duplicates if motif columns exist
                        if 'motif' in df.columns:
                            initial_count = len(df)
                            duplicate_cols = ['motif']
                            if 'start' in df.columns:
                                duplicate_cols.append('start')
                            if 'end' in df.columns:
                                duplicate_cols.append('end')
                            if 'strand' in df.columns:
                                duplicate_cols.append('strand')
                            
                            df = df.drop_duplicates(subset=duplicate_cols)
                            final_count = len(df)
                            
                            if initial_count != final_count:
                                print(f"    Removed {initial_count - final_count} duplicates")
                        
                        all_motifs.append(df)
                        all_metadata.append(metadata)
                        seed42_count += 1
                        print(f"    ✓ Added {len(df)} motif records")
                    else:
                        print(f"    Skipping: seed={metadata.get('seed')} (not 42)")
                        
                except Exception as e:
                    print(f"    Error loading {motif_file.name}: {e}")
                    
        except Exception as e:
            print(f"Error processing directory {analysis_dir}: {e}")
    
    print(f"\n✓ Found {seed42_count} experiments with seed=42")
    if excluded_specificity > 0:
        print(f"  Note: {excluded_specificity} experiments missing trajectory data (using default specificity=1.0)")
    
    # Combine all data
    if all_motifs:
        combined_motifs = pd.concat(all_motifs, ignore_index=True)
        print(f"✓ Loaded {len(combined_motifs)} motif records from seed=42")
        
        # Show summary of what was found - filter out None values before sorting
        if 'cell_type' in combined_motifs.columns:
            unique_cell_types = combined_motifs['cell_type'].dropna().unique()
            if len(unique_cell_types) > 0:
                print(f"✓ Cell types found: {sorted(unique_cell_types)}")
            else:
                print("⚠ No valid cell types found (all are None/NaN)")
                
        if 'model' in combined_motifs.columns:
            unique_models = combined_motifs['model'].dropna().unique()
            if len(unique_models) > 0:
                print(f"✓ Models found: {sorted(unique_models)}")
            else:
                print("⚠ No valid models found (all are None/NaN)")
            
    else:
        combined_motifs = pd.DataFrame()
        
    if all_metadata:
        metadata_df = pd.DataFrame(all_metadata)
        print(f"✓ Loaded {len(metadata_df)} experiment metadata records")
    else:
        metadata_df = pd.DataFrame()
    
    return combined_motifs, metadata_df

def filter_motifs(motifs_df, max_pval=0.05, min_ism_weight=0.0, ism_percentile=50.0, min_specificity=1.0, ism_method='percentile'):
    """Filter motifs based on quality criteria including per-sequence ISM weight percentiles."""
    if motifs_df.empty:
        return motifs_df
    
    initial_count = len(motifs_df)
    print(f"Starting with {initial_count} total motifs")
    
    # Filter 1: Specificity check
    print(f"Filtering by final sequence specificity >= {min_specificity}")
    specificity_filtered = motifs_df[motifs_df['final_specificity'] >= min_specificity]
    
    # Filter 2: P-value
    pval_filtered = specificity_filtered[specificity_filtered['pval'] <= max_pval]
    
    # Filter 3: Basic ISM weight threshold
    basic_ism_filtered = pval_filtered[pval_filtered['ism_weight'] >= min_ism_weight]
    
    # Filter 4: PER-SEQUENCE ISM weight filtering (THIS IS MISSING!)
    if len(basic_ism_filtered) > 0:
        print(f"Applying per-sequence ISM weight filtering ({ism_method}, {ism_percentile}th percentile)...")
        percentile_filtered = filter_motifs_per_sequence_ism(
            basic_ism_filtered, 
            ism_percentile=ism_percentile, 
            method=ism_method
        )
    else:
        percentile_filtered = basic_ism_filtered
    
    return percentile_filtered

def filter_motifs_per_sequence_ism(motifs_df, ism_percentile=90.0, method='percentile'):
    """
    Filter motifs based on ISM weights calculated per-sequence (per individual model).
    This ensures that the percentile is calculated within each individual sequence/model,
    not across all models globally.
    """
    if motifs_df.empty:
        return motifs_df
    
    print(f"Applying per-sequence ISM weight filtering using {method} method...")
    
    filtered_dfs = []
    
    # Group by sequence (file_stem) - this represents individual models/experiments
    for file_stem, seq_group in motifs_df.groupby('file_stem'):
        ism_weights = seq_group['ism_weight'].values
        
        if method == 'percentile':
            # Keep motifs above the specified percentile WITHIN THIS SEQUENCE/MODEL
            if len(ism_weights) > 1:
                threshold = np.percentile(ism_weights, ism_percentile)
                mask = seq_group['ism_weight'] >= threshold
            else:
                mask = np.ones(len(seq_group), dtype=bool)  # Keep single motif
        
        # Always keep at least one motif per sequence (the highest ISM weight)
        if not mask.any():
            highest_idx = seq_group['ism_weight'].idxmax()
            mask = seq_group.index == highest_idx
            
        filtered_dfs.append(seq_group[mask])
    
    if filtered_dfs:
        result = pd.concat(filtered_dfs, ignore_index=True)
        print(f"  Kept {len(result)} motifs from per-sequence filtering")
        return result
    else:
        return pd.DataFrame()

def calculate_replicate_success_rates(motifs_df):
    """Calculate how many replicates succeeded for each cell type."""
    print("Calculating replicate success rates per cell type...")
    
    # Expected: 4 replicates per model type × 4 model types = 16 total per cell type
    expected_replicates_per_celltype = 16
    
    # Count successful replicates per cell type
    if 'cell_type' in motifs_df.columns and 'model' in motifs_df.columns:
        replicate_counts = motifs_df.groupby(['cell_type', 'model']).size().reset_index()
        successful_per_celltype = replicate_counts.groupby('cell_type')['model'].nunique()
        
        replicate_stats = {}
        
        for cell_type in motifs_df['cell_type'].unique():
            if pd.notna(cell_type):
                successful_reps = successful_per_celltype.get(cell_type, 0)
                success_rate = successful_reps / expected_replicates_per_celltype
                
                replicate_stats[cell_type] = {
                    'expected_replicates': expected_replicates_per_celltype,
                    'successful_replicates': successful_reps,
                    'failed_replicates': expected_replicates_per_celltype - successful_reps,
                    'success_rate': success_rate
                }
                
                if successful_reps < expected_replicates_per_celltype:
                    print(f"  {cell_type}: {successful_reps}/{expected_replicates_per_celltype} replicates successful ({success_rate:.1%})")
    else:
        replicate_stats = {}
        print("  Warning: cell_type or model columns not found")
    
    return replicate_stats

def create_celltype_occurrence_tables(motifs_df):
    """Create cell type-focused occurrence tables."""
    print("Creating cell type-focused occurrence tables...")
    
    if 'cell_type' not in motifs_df.columns or 'motif' not in motifs_df.columns:
        print("ERROR: Required columns 'cell_type' or 'motif' not found!")
        return {}, {}, {}, {}
    
    tables = {}
    summaries = {}
    normalized_tables = {}
    
    # Calculate replicate success rates
    replicate_stats = calculate_replicate_success_rates(motifs_df)
    
    # Create tables for each cell type
    for cell_type in motifs_df['cell_type'].unique():
        if pd.notna(cell_type):
            cell_data = motifs_df[motifs_df['cell_type'] == cell_type]
            
            if len(cell_data) > 0:
                # Create occurrence table: motifs × models
                if 'model' in cell_data.columns:
                    occurrence_table = pd.crosstab(
                        cell_data['motif'], 
                        cell_data['model'],
                        margins=True, 
                        margins_name='Total'
                    ).sort_values('Total', ascending=False)
                    
                    tables[cell_type] = occurrence_table
                    
                    # Create normalized table (divide by successful replicates)
                    if cell_type in replicate_stats:
                        successful_reps = replicate_stats[cell_type]['successful_replicates']
                        if successful_reps > 0:
                            normalized_table = occurrence_table.copy()
                            # Normalize all columns except 'Total'
                            for col in normalized_table.columns:
                                if col != 'Total':
                                    normalized_table[col] = normalized_table[col] / successful_reps
                            normalized_tables[cell_type] = normalized_table
                    
                    # Create summary with standard deviation
                    summaries[cell_type] = {
                        'total_motifs': len(cell_data),
                        'unique_motifs': cell_data['motif'].nunique(),
                        'models_analyzed': cell_data['model'].nunique() if 'model' in cell_data.columns else 0,
                        'top_motif': occurrence_table.drop('Total').iloc[0].name if len(occurrence_table) > 1 else 'N/A',
                        'top_motif_count': occurrence_table.drop('Total').iloc[0]['Total'] if len(occurrence_table) > 1 else 0,
                        'avg_ism_weight': cell_data['ism_weight'].mean() if 'ism_weight' in cell_data.columns else 0,
                        'median_ism_weight': cell_data['ism_weight'].median() if 'ism_weight' in cell_data.columns else 0,
                        'avg_specificity': cell_data['final_specificity'].mean() if 'final_specificity' in cell_data.columns else 0,
                        'median_specificity': cell_data['final_specificity'].median() if 'final_specificity' in cell_data.columns else 0,
                        'std_specificity': cell_data['final_specificity'].std() if 'final_specificity' in cell_data.columns else 0,
                        'successful_replicates': replicate_stats[cell_type]['successful_replicates'] if cell_type in replicate_stats else 0,
                        'expected_replicates': replicate_stats[cell_type]['expected_replicates'] if cell_type in replicate_stats else 0
                    }
                    
                    print(f"✓ {cell_type}: {len(occurrence_table)-1} motifs × {len(occurrence_table.columns)-1} models")
    
    return tables, summaries, normalized_tables, replicate_stats

def create_volcano_plot_instructions(averaged_table, output_dir):
    """Create instructions for volcano plot analysis."""
    instructions_file = output_dir / 'volcano_plot_instructions.txt'
    
    with open(instructions_file, 'w') as f:
        f.write("VOLCANO PLOT ANALYSIS INSTRUCTIONS\n")
        f.write("="*50 + "\n\n")
        f.write("Use the 'motif_occurrence_table_averaged_normalized.csv' file for volcano plot analysis.\n\n")
        f.write("This table contains:\n")
        f.write("- Motif occurrences normalized by successful replicates per cell type\n")
        f.write("- Averaged across the 4 model replicates\n")
        f.write("- Ready for statistical comparison between conditions\n\n")
        f.write("Suggested analysis:\n")
        f.write("1. Compare motif enrichment between cell types\n")
        f.write("2. Identify cell type-specific motifs\n")
        f.write("3. Create volcano plots for differential motif usage\n")

def save_celltype_focused_reports(tables, summaries, motifs_df, output_dir, args, normalized_tables=None, replicate_stats=None):
    """Save cell type-focused reports."""
    output_dir = Path(output_dir)
    
    print("Saving cell type-focused tables and reports...")
    
    # Save individual cell type tables (raw counts)
    for cell_type, table in tables.items():
        safe_celltype_name = str(cell_type).replace('/', '_').replace(' ', '_').replace(':', '_')
        filename = f'motif_occurrence_by_celltype_{safe_celltype_name}.csv'
        table.to_csv(output_dir / filename)
        print(f"✓ Saved cell type table: {filename}")
    
    # Save individual cell type normalized tables
    if normalized_tables:
        for cell_type, table in normalized_tables.items():
            safe_celltype_name = str(cell_type).replace('/', '_').replace(' ', '_').replace(':', '_')
            filename = f'motif_occurrence_by_celltype_normalized_{safe_celltype_name}.csv'
            table.to_csv(output_dir / filename)
            print(f"✓ Saved normalized cell type table: {filename}")
    
    # Create combined table: motifs × cell types (across all models) - RAW COUNTS
    if 'cell_type' in motifs_df.columns and 'motif' in motifs_df.columns:
        combined_by_celltype = pd.crosstab(
            motifs_df['motif'], 
            motifs_df['cell_type'],
            margins=True, 
            margins_name='Total'
        ).sort_values('Total', ascending=False)
        combined_by_celltype.to_csv(output_dir / 'motif_occurrence_by_celltype_combined.csv')
        print("✓ Saved combined cell type table")
    
    # NEW: Create combined normalized table: motifs × cell types (normalized)
    if normalized_tables:
        print("Creating combined normalized table across cell types...")
        
        # Get all unique motifs across all cell types
        all_motifs = set()
        for table in normalized_tables.values():
            all_motifs.update(table.drop('Total', errors='ignore').index)
        all_motifs = sorted(all_motifs)
        
        # Create combined normalized table
        combined_normalized = pd.DataFrame(index=all_motifs)
        
        for cell_type, table in normalized_tables.items():
            # Get normalized counts (excluding 'Total' column)
            normalized_counts = table.drop('Total', errors='ignore').sum(axis=1)
            combined_normalized[cell_type] = normalized_counts
        
        # Fill NaN with 0 and add Total column
        combined_normalized = combined_normalized.fillna(0)
        combined_normalized['Total'] = combined_normalized.sum(axis=1)
        combined_normalized = combined_normalized.sort_values('Total', ascending=False)
        
        combined_normalized.to_csv(output_dir / 'motif_occurrence_by_celltype_combined_normalized.csv')
        print("✓ Saved combined normalized cell type table")
    
    # IMPORTANT: Save the comprehensive CSV with all motif details
    motifs_df.to_csv(output_dir / 'all_motifs_comprehensive.csv', index=False)
    print("✓ Saved comprehensive motif CSV with all details")
    
    # Create additional useful tables
    # Model comparison table: motifs × models (across all cell types)
    if 'motif' in motifs_df.columns and 'model' in motifs_df.columns:
        combined_by_model = pd.crosstab(
            motifs_df['motif'], 
            motifs_df['model'],
            margins=True, 
            margins_name='Total'
        ).sort_values('Total', ascending=False)
        combined_by_model.to_csv(output_dir / 'motif_occurrence_by_model_combined.csv')
        print("✓ Saved combined model table")
    
    # Create volcano plot instructions
    if normalized_tables:
        create_volcano_plot_instructions(None, output_dir)
    
    # Create cell type-focused summary report
    with open(output_dir / 'celltype_focused_analysis_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("CELL TYPE-FOCUSED MOTIF OCCURRENCE ANALYSIS (SEED=42)\n")
        f.write("UPDATED FOR NESTED DIRECTORY STRUCTURE\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total motifs analyzed: {len(motifs_df)}\n")
        if 'motif' in motifs_df.columns:
            f.write(f"Unique motifs: {motifs_df['motif'].nunique()}\n")
        if 'cell_type' in motifs_df.columns:
            f.write(f"Cell types analyzed: {motifs_df['cell_type'].nunique()}\n")
        if 'model' in motifs_df.columns:
            f.write(f"Models analyzed: {motifs_df['model'].nunique()}\n")
        if 'ism_weight' in motifs_df.columns:
            f.write(f"ISM weight range: {motifs_df['ism_weight'].min():.4f} - {motifs_df['ism_weight'].max():.4f}\n")
        if 'final_specificity' in motifs_df.columns:
            f.write(f"Specificity range: {motifs_df['final_specificity'].min():.2f} - {motifs_df['final_specificity'].max():.2f}\n\n")
        
        # Cell type-specific summaries
        f.write("CELL TYPE-SPECIFIC SUMMARIES\n")
        f.write("-"*40 + "\n")
        for cell_type, summary in summaries.items():
            f.write(f"\n{cell_type}:\n")
            f.write(f"  Total motifs: {summary['total_motifs']}\n")
            f.write(f"  Unique motifs: {summary['unique_motifs']}\n")
            f.write(f"  Models with data: {summary['models_analyzed']}\n")
            f.write(f"  Successful replicates: {summary.get('successful_replicates', 'N/A')}/{summary.get('expected_replicates', 'N/A')}\n")
            f.write(f"  Most frequent motif: {summary['top_motif']} ({summary['top_motif_count']} occurrences)\n")
            
            # Specificity statistics with standard deviation
            if 'avg_specificity' in summary:
                f.write(f"  Average specificity: {summary['avg_specificity']:.2f} ± {summary.get('std_specificity', 0):.2f}\n")
            
            # MODIFIED: Top 5 motifs for Human-Decima and Human-Borzoi models only, per cell type
            if 'model_type' in motifs_df.columns:
                cell_type_data = motifs_df[motifs_df['cell_type'] == cell_type]
                
                # Human-Decima top 5 motifs for this cell type
                human_decima_data = cell_type_data[cell_type_data['model_type'] == 'Human_Decima']
                if len(human_decima_data) > 0:
                    f.write(f"  Top 5 motifs (Human-Decima only):\n")
                    top_human_decima = human_decima_data['motif'].value_counts().head(5)
                    for i, (motif, count) in enumerate(top_human_decima.items(), 1):
                        f.write(f"    {i}. {motif}: {count} occurrences\n")
                else:
                    f.write(f"  Top 5 motifs (Human-Decima only): No data found\n")
                
                # Human-Borzoi top 5 motifs for this cell type
                human_borzoi_data = cell_type_data[cell_type_data['model_type'] == 'Human_Borzoi']
                if len(human_borzoi_data) > 0:
                    f.write(f"  Top 5 motifs (Human-Borzoi only):\n")
                    top_human_borzoi = human_borzoi_data['motif'].value_counts().head(5)
                    for i, (motif, count) in enumerate(top_human_borzoi.items(), 1):
                        f.write(f"    {i}. {motif}: {count} occurrences\n")
                else:
                    f.write(f"  Top 5 motifs (Human-Borzoi only): No data found\n")
            else:
                # Fallback to original approach if model_type column not available
                if cell_type in tables:
                    occurrence_table = tables[cell_type]
                    if len(occurrence_table) > 1:  # More than just the 'Total' row
                        top_5_motifs = occurrence_table.drop('Total').head(5)
                        f.write(f"  Top 5 motifs (all models):\n")
                        for i, (motif, row) in enumerate(top_5_motifs.iterrows(), 1):
                            total_count = row['Total']
                            f.write(f"    {i}. {motif}: {total_count} occurrences\n")
                    else:
                        f.write(f"  Top 5 motifs: No motifs found\n")
        
        # Remove the overall top motifs section entirely since we now show per cell type
    
    print("✓ Cell type-focused reports saved")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced motif occurrence analysis with cell type focus - Updated for nested directory structure")
    
    parser.add_argument('--results_dir', required=True, help='Directory containing nested motif analysis results')
    parser.add_argument('--output_dir', required=True, help='Output directory for tables and plots')
    parser.add_argument('--max_motif_pval', type=float, default=0.05, help='Maximum motif p-value')
    parser.add_argument('--min_ism_weight', type=float, default=0.0, help='Minimum ISM weight')
    parser.add_argument('--ism_percentile', type=float, default=75.0, help='ISM weight percentile cutoff')
    parser.add_argument('--min_specificity', type=float, default=0.5, help='Minimum final specificity')
    parser.add_argument('--ism_method', default='percentile', help='ISM filtering method')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode to show filenames')
    
    return parser.parse_args()

def main():
    """Cell type-focused main analysis function for nested directory structure."""
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting enhanced comprehensive evolution analysis...")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Debug mode - show filenames
    if args.debug:
        debug_filenames(args.results_dir)
        return
    
    print("="*80)
    print("ENHANCED MOTIF OCCURRENCE ANALYSIS (SEED=42 ONLY)")
    print("UPDATED FOR NESTED DIRECTORY STRUCTURE (analysis_25ct_20250619)")
    print("="*80)
    print(f"Filters: seed=42, p-value <= {args.max_motif_pval}")
    print(f"         ISM weight >= {args.min_ism_weight}")
    print(f"         Per-sequence percentile: {args.ism_percentile}th percentile")
    print(f"         Final specificity >= {args.min_specificity}")
    
    # Load and filter results using new nested structure
    motifs_df, metadata_df = load_all_results_nested(args.results_dir)
    
    if motifs_df.empty:
        print("ERROR: No motif data found for seed=42!")
        return
    
    # Apply filters
    filtered_motifs = filter_motifs(
        motifs_df, 
        args.max_motif_pval, 
        args.min_ism_weight, 
        args.ism_percentile,
        args.min_specificity,
        args.ism_method
    )
    
    if filtered_motifs.empty:
        print("ERROR: No motifs found after filtering!")
        return
    
    # Create cell type-focused tables
    tables, summaries, normalized_tables, replicate_stats = create_celltype_occurrence_tables(filtered_motifs)
    
    # Save cell type-focused reports
    save_celltype_focused_reports(tables, summaries, filtered_motifs, output_dir, args, normalized_tables, replicate_stats)
    
    print("="*80)
    print("ENHANCED COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"✓ Comprehensive motif CSV: all_motifs_comprehensive.csv")
    print(f"✓ Cell type tables: motif_occurrence_by_celltype_*.csv")
    print(f"✓ Combined tables: motif_occurrence_by_celltype_combined.csv")
    print(f"✓ Model comparison: motif_occurrence_by_model_combined.csv")
    print(f"Check outputs in: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()