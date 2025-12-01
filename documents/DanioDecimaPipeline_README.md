# DanioDecima Deep Learning Framework: Comprehensive Methods

This document provides comprehensive methods sections for all six analytical workflows implemented in the DanioDecima deep learning framework for predicting single-cell RNA-seq data from genomic DNA sequences.

## **1. Data Exploration and Preprocessing**

The first analytical workflow establishes the foundational dataset characteristics and preprocessing protocols for zebrafish single-cell expression data.

### **1.1 Single-Cell Data Processing**

The data exploration pipeline (`zebrafish_data_exploration.ipynb`) implements comprehensive quality control and feature selection:

- **Gene Filtering**: Highly variable genes were identified using scanpy with parameters: `min_mean=0.0125`, `max_mean=5`, `min_disp=0.5`, following established single-cell analysis protocols
- **Expression Variance Analysis**: Gene expression variance was computed across all cell types to assess the dynamic range available for model training
- **Model Performance Baseline**: Comparative analysis between human pretrained models and random initialization established performance baselines relative to gene expression variance distributions

### **1.2 Cross-Species Conservation Analysis**

The preprocessing workflow incorporates evolutionary conservation metrics:

- **Ortholog Mapping**: Human-zebrafish ortholog pairs were identified using Ensembl BioMart to enable cross-species model transfer
- **Conservation Scoring**: Gene-level conservation scores were computed based on sequence similarity and synteny relationships
- **Expression Conservation**: Cross-species expression pattern conservation was assessed to identify functionally preserved regulatory programs

---

## **2. Model Training and Experimental Design**

The second workflow implements the core model training pipeline with systematic experimental design across multiple initialization strategies.

### **2.1 Experimental Configuration**

The training framework (`decima_finetune.py`) implements a comprehensive 16-experiment design matrix:

- **Pretrained Initialization**: 12 experiments using three pretrained model sources (Human-Borzoi: 4 replicates, Human-Decima: 4 replicates, Mouse-Borzoi: 4 replicates)
- **Random Initialization**: 4 experiments with random weight initialization across different random seeds (42, 43, 44, 45)
- **Learning Rate Optimization**: Pretrained models used lr=3×10⁻⁵, random initialization used lr=3×10⁻⁶ based on preliminary hyperparameter tuning

### **2.2 Training Infrastructure**

The distributed training pipeline leverages high-performance computing resources:

- **SLURM Integration**: Array job submission (`submit_decima_finetune.sh`) with 16 parallel configurations across GPU clusters
- **Resource Allocation**: H100/H200 GPU constraints, 4-batch size with 5-step gradient accumulation for memory efficiency
- **Environment Management**: Isolated genomepy cache directories per job prevent genome access conflicts during concurrent training

### **2.3 Architecture and Loss Function**

The DanioDecima model extends the Borzoi architecture with zebrafish-specific modifications:

- **Input Representation**: 5-channel input (4 DNA bases + 1 gene mask) with 524,288bp sequence length cropped to 5,120bp effective resolution
- **Network Architecture**: 7 CNN blocks + 8 Transformer blocks generating 1,920 embedding channels
- **Task-Specific Output**: Gene-specific expression prediction via exponential activation
- **Loss Function**: TaskWisePoissonMultinomialLoss combining Poisson likelihood for count data with multinomial normalization

---

## **3. Prediction Generation and Model Inference**

The third workflow generates comprehensive predictions across all trained models for downstream evaluation and analysis.

### **3.1 Distributed Prediction Pipeline**

The prediction generation system (`decima_predictions.py`) implements scalable inference:

- **Checkpoint Management**: Automated discovery and loading of epoch checkpoints from model training directories
- **Sequence Augmentation**: Multiple sequence shifts (max_seq_shift=3) for robust prediction averaging across genomic contexts
- **Memory Optimization**: Batch size optimization (batch_size=6) with 16 worker processes for efficient data loading

### **3.2 Prediction Processing and Storage**

The inference pipeline generates structured prediction outputs:

- **Format Standardization**: All predictions stored in AnnData format with standardized gene and cell-type annotations
- **Quality Metrics**: Per-gene Pearson correlations computed between predicted and observed expression
- **Metadata Integration**: Experimental configuration, model architecture, and training metrics embedded in prediction objects

---

## **4. Model Evaluation and Performance Analysis**

The fourth workflow implements comprehensive model evaluation with emphasis on cell-type specific performance patterns and developmental stage analysis.

### **4.1 Evaluation Metrics and Statistical Framework**

The evaluation pipeline (`01_evaluate_celltypes.ipynb`) implements multi-faceted performance assessment:

- **Gene-Level Metrics**: Pearson correlation coefficients computed between predicted and observed expression for each gene
- **Cell-Type Metrics**: Performance evaluated separately for each of the 85 cell-type × timepoint combinations
- **Statistical Testing**: Mann-Whitney U tests with FDR correction (Benjamini-Hochberg) applied to assess significant performance differences between model configurations

### **4.2 Developmental Stage Analysis**

Performance evaluation incorporates temporal developmental patterns:

- **Timepoint Stratification**: Separate analysis for each developmental timepoint (6hpf through 5dpf) to capture stage-specific regulatory programs
- **Cell-Type Evolution**: Tracking of prediction accuracy across developmental trajectories for individual cell lineages
- **Comparative Analysis**: Cross-timepoint performance comparison to identify models with consistent temporal prediction capabilities

### **4.3 Model Architecture Comparison**

The evaluation framework enables systematic comparison across initialization strategies:

- **Pretrained vs Random**: Statistical comparison of transfer learning effectiveness versus random initialization
- **Cross-Species Transfer**: Assessment of human vs mouse pretrained model performance on zebrafish data
- **Replicate Consistency**: Analysis of performance variance across experimental replicates within each configuration

---

## **5. Attribution Analysis for Conservation Confounding**

The fifth analytical workflow examines model interpretability through attribution analysis to assess potential confounding between model predictions and evolutionary conservation patterns.

### **5.1 Attribution Score Calculation**

Attribution analysis was performed using the combined attribution calculation pipeline (`00_combined_attribution_analysis.py`) submitted via SLURM array jobs across all 16 experimental model configurations. The workflow implements the following approach:

- **Model Loading**: Each trained DanioDecima checkpoint was loaded and configured for inference mode with gradient computation disabled for computational efficiency
- **Attribution Method**: InputXGradient attribution from the Captum library was applied to calculate input sequence importance scores for each nucleotide position
- **Task Aggregation**: Cell-type specific attributions were computed using task-wise aggregation with mean pooling across target cell types (threshold > 0.5 expression)
- **Genomic Integration**: Attribution scores were calculated across the full 524,288bp genomic windows and stored in HDF5 format for efficient downstream analysis

The attribution pipeline processes each gene individually, computing attributions for cell types with detectable expression levels and aggregating results across the first four sequence channels (A, T, G, C) while excluding the gene mask channel.

### **5.2 Genomic Feature Annotation**

Attribution scores were systematically analyzed across distinct genomic regions to assess model focus patterns:

- **Gene Structure**: Promoters (±100bp from TSS), exons, introns, and exon-intron junctions (±10bp) were annotated using GTF coordinates
- **Regulatory Elements**: ATAC-seq peaks from zebrafish multi-omics data were overlapped with attributions to identify accessible chromatin regions
- **Distance Categories**: Attributions were stratified by distance from gene bodies: 1kb, 1-10kb, 10-100kb, and >100kb flanking regions
- **Region-Specific Analysis**: Mean attribution scores were calculated separately for CRE (ATAC peak) and non-CRE regions within each genomic category

### **5.3 Conservation Confounding Analysis**

The analysis examines whether high-performing models exhibit attribution patterns that correlate with evolutionary conservation rather than cell-type specific regulatory signals:

- **Gene Conservation Metrics**: Human-zebrafish ortholog conservation scores were integrated with model attribution patterns
- **Expression Controls**: Gene expression level distributions were used as covariates to control for expression-dependent attribution biases  
- **Statistical Framework**: Mann-Whitney U tests with FDR correction were applied to assess differential attribution patterns between conserved and non-conserved genomic regions
- **Propensity Score Matching**: Expression-level matched analyses were performed to isolate conservation-specific effects from general expression-level confounding

This analytical framework enables identification of whether attribution patterns reflect legitimate cell-type regulatory mechanisms or spurious correlations with evolutionary conservation signatures.

---

## **6. Regulatory Element Design with Cell-Type Specificity**

The sixth workflow implements directed evolution to design synthetic regulatory elements with enhanced cell-type specificity using trained DanioDecima models.

### **6.1 Evolutionary Design Framework**

The regulatory element design pipeline (`00_evolve_combined.py`) implements single-sequence directed evolution with the following optimization strategy:

- **Target Definition**: Cell-type specificity was defined as the difference between mean predicted expression in target cell types versus background cell types at 16hpf developmental timepoint
- **Sequence Initialization**: 200bp promoter elements were initialized with random sequences (seed=42) and placed upstream of EBFP reporter cargo sequences
- **Genomic Context**: Evolution occurred within realistic genomic contexts using chromosome 4 coordinates (TSS: 29480218) in 524,288bp windows with TSS offset at position 163840

### **6.2 Directed Evolution Algorithm**

The evolution algorithm implements exhaustive single-point mutagenesis across each round:

- **Mutation Space**: All possible single nucleotide substitutions (3 per position × sequence length) were evaluated at each evolutionary round
- **Fitness Function**: Specificity scores calculated as mean target cell-type expression minus mean background cell-type expression from model predictions
- **Selection Strategy**: The highest-scoring mutation was selected and applied to the sequence each round, with tracking of position-specific evolution patterns
- **Convergence Criteria**: Early stopping implemented with patience=50 rounds and minimum improvement threshold=0.001 to detect evolutionary plateau

### **6.3 High-Performance Computing Implementation**

The design pipeline was scaled across computational resources using SLURM array jobs:

- **Parallelization**: 400 total jobs (16 models × 25 cell types) distributed across GPU clusters with H100/H200 constraints
- **Resource Allocation**: 100GB memory, 4 CPU cores, single GPU per job with 8-hour time limits
- **Environment Management**: Genomepy cache directories isolated per job to prevent conflicts during concurrent genome access
- **Pipeline Integration**: Evolution directly feeds into analysis pipeline (`1_read_celltypes.py`) for immediate sequence characterization

### **6.4 Sequence Analysis and Interpretation**

Evolved sequences underwent comprehensive molecular analysis:

#### **6.4.1 In Silico Mutagenesis (ISM)**
- **Systematic Perturbation**: Every position in final evolved sequences was mutated to all alternative bases (3 mutations per position)
- **Effect Quantification**: Prediction changes calculated relative to original sequence across all cell types
- **Visualization**: ISM effects displayed as position-specific line plots and base-substitution heatmaps to identify critical regulatory positions

#### **6.4.2 Transcription Factor Motif Analysis**
- **Motif Scanning**: JASPAR2020 vertebrate transcription factor binding site database used for comprehensive motif detection
- **ISM Weight Integration**: Motif hits weighted by underlying ISM attribution scores to prioritize functionally important binding sites
- **Regulatory Prediction**: High ISM-weight motifs identified as candidate drivers of cell-type specificity

### **6.5 Comparative Analysis and Visualization**

#### **6.5.1 Cross-Model Performance Analysis**
Volcano plots were generated to compare motif enrichment patterns across cell types:
- **Statistical Framework**: Z-score based enrichment analysis with fold-change thresholds (FC≥1.2, Z-score≥1.0) 
- **Visualization**: Square format plots (8×8) with enhanced point sizes and axis labels for manuscript quality figures
- **Motif Classification**: Top motifs labeled by significance, fold-change, or both categories using color-coded annotation system

#### **6.5.2 Design Pattern Clustering**
Hierarchical clustering analysis revealed regulatory element design patterns:
- **Distance Metrics**: Correlation-based clustering of cell-type expression profiles using average-linkage
- **Heatmap Visualization**: Z-score normalized expression patterns displayed with dendrograms showing cell-type similarity relationships
- **Pattern Recognition**: Cluster analysis identified groups of cell types with shared regulatory sequence requirements

This comprehensive design framework enables systematic generation of cell-type specific regulatory elements while providing detailed mechanistic insights into sequence-function relationships in zebrafish development.

---

## **Key File Locations**

### Data Exploration and Preprocessing
- **Data Exploration**: `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/2_dataset/zebrafish_data_exploration.ipynb`

### Model Training Pipeline
- **Training Script**: `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-main/scripts/decima_finetune.py`
- **Submission Script**: `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-main/scripts/submit_decima_finetune.sh`
- **Core Lightning Module**: `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-main/src/decima/lightning.py`

### Prediction Generation
- **Prediction Script**: `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-main/scripts/decima_predictions.py`
- **Submission Script**: `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/4_evaluation/00_submit_predict_decima.sh`

### Model Evaluation
- **Evaluation Notebook**: `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/4_evaluation/01_evaluate_celltypes.ipynb`

### Attribution Analysis Pipeline
- **Main Script**: `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/5_specificity/00_combined_attribution_analysis.py`
- **Submission Script**: `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/5_specificity/00_submit_combined_attributions.sh`
- **Analysis Notebook**: `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/5_specificity/04_analyze_attributions_celltypes_decima.ipynb`

### Regulatory Element Design Pipeline
- **Evolution Script**: `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/9_design/00_evolve_combined.py`
- **Submission Script**: `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/9_design/00_submit_evolve_combined.sh`
- **Analysis Script**: `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/9_design/1_read_celltypes.py`
- **Volcano Plots**: `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/9_design/designs_volcano_plot.ipynb`
- **Clustering Analysis**: `/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/decima-applications-main/notebooks/9_design/designs_clustering_analysis.ipynb`

---

*Generated by Claude Code analysis of DanioDecima repository - September 2025*