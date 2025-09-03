# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import subprocess
import sys
import collections
import glob
import math

import numpy as np
import pandas as pd
import anndata
import os
import tqdm
import yaml
import wandb

from grelu.sequence.format import convert_input_type
from grelu.sequence.utils import reverse_complement
from grelu.transforms.prediction_transforms import get_compare_func, Aggregate, Specificity
from grelu.data.preprocess import filter_blacklist, filter_chromosomes
from grelu.variant import filter_variants

from sklearn.metrics import roc_auc_score, recall_score, accuracy_score
from sklearn import linear_model

import torch
# %matplotlib inline

pd.options.mode.chained_assignment = None 

# %% [markdown]
# ## Paths

# %%
# path to ensembl dict
ensembl_out_dir = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/results/ensemble"
with open(os.path.join(ensembl_out_dir,'ensembl_dict.yml'), 'r') as outfile:
    ensembl_dict = yaml.safe_load(outfile)
brozoi_out_dir = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/results/brozoi"
brozoi_tracks_path = '/gstore/data/resbioai/karollua/Decima/scborzoi/decima/data/borzoi_targets/targets_human.txt'

# anndata file
anndata_file = "/gstore/data/resbioai/grelu/decima/2024082/data.h5ad"

# path to gene h5
h5_file = "/gstore/data/resbioai/grelu/decima/20240823/data.h5"

# eqtl paths
susie_dir = '/gstore/data/resbioai/grelu/decima/onek1k/susie/QTS000038' # QTS000038 is OneK1K study ID
eqtl_sumstats_base_path = "/gstore/data/resbioai/grelu/decima/onek1k/sumstats/*.all.tsv.gz"

# where to save results
save_dir = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/"
brozima_ensembl_out_dir = os.path.join(save_dir, "results", 'brozoi',f'brozima_unsquash_ensembl')

# path for variants
susie_df_file = os.path.join(save_dir,'data/eQTL_processed/susie_df.csv')
variant_df_file = os.path.join(save_dir,'data/eQTL_processed/vars.csv')
neg_variant_df_file = os.path.join(save_dir,'data/eQTL_processed/neg_vars_all.csv')
matched_negative_file = os.path.join(save_dir,'data/eQTL_processed/matched_negative.csv')
matched_negative_dedup_file = os.path.join(save_dir,'data/eQTL_processed/matched_negative_dedup.csv')

# %%
ad = anndata.read_h5ad(anndata_file)
brozoi_tracks = pd.read_csv(brozoi_tracks_path,sep='\t')

# %% [markdown]
# ### Download eQTL metadata

# %%
eqtl_meta = pd.read_table('https://raw.githubusercontent.com/eQTL-Catalogue/eQTL-Catalogue-resources/master/data_tables/dataset_metadata.tsv')
eqtl_meta = eqtl_meta[eqtl_meta.quant_method == 'ge'] # gene exp. QTLs
eqtl_meta = eqtl_meta[eqtl_meta.study_label == 'OneK1K'] # Yazar et al.

# %%
ct_dict = eqtl_meta[['dataset_id', 'tissue_label']].set_index('dataset_id').to_dict()['tissue_label']

# %% [markdown]
# ## SuSiE (in)credible sets

# %%
# #!wget -r -e robots=off -P /home/karollua/projects/Decima/scborzoi/AKv1/data/eQTL https://ftp.ebi.ac.uk/pub/databases/spot/eQTL/susie/QTS000038 

# %%
ensembl_id_map = ad.var[['gene_id']].reset_index().set_index('gene_id')['index'].to_dict()

# %%
susie_df = []

for ct_id in tqdm.tqdm(list(ct_dict.keys())):
    df = pd.read_table(f'{susie_dir}/{ct_id}/{ct_id}.credible_sets.tsv.gz')
    df['chrom'] = [x.split('_')[0] for x in df.variant]
    df['pos'] = [int(x.split('_')[1]) for x in df.variant]
    df['ref'] = [x.split('_')[2] for x in df.variant]
    df['alt'] = [x.split('_')[3] for x in df.variant]
    df['gene_symbol'] = df.gene_id.map(ensembl_id_map)

    susie_df.append(df.assign(celltype=ct_dict[ct_id], celltype_id=ct_id))

susie_df = pd.concat(susie_df, axis=0).reset_index(drop=True)
susie_df.head()

# make complete list of "credible variants"
cs_vars = set(susie_df['variant'])

# %% [markdown]
# ### Filter variants

# %%
susie_df = susie_df[susie_df.gene_symbol.notna()]
susie_df = filter_variants(susie_df, max_del_len=0, max_insert_len=0, standard_bases=True) # remove indels
susie_df = filter_chromosomes(susie_df, include='autosomesXY') # keep standard chroms
susie_df = filter_blacklist(susie_df, genome="hg38", window=100) # remove variants in blacklisted regions

# add gene information and calculate relative variant positions (offset)
susie_df = susie_df.merge(ad.var[['gene_id', 'start', 'end', 'strand', 'gene_mask_start']]).rename(columns={'start': 'gene_window_start', 'end': 'gene_window_end', 'strand': 'gene_strand'}) # add window information
susie_df = susie_df[((susie_df.pos > susie_df.gene_window_start) & (susie_df.pos < susie_df.gene_window_end))] # keep variants within the sequence window
susie_df['pos_relative'] = susie_df.pos - susie_df.gene_window_start - 1

# use gene_end to calculate offset for - genes and rc() alleles
susie_df.loc[susie_df.gene_strand=='-', 'pos_relative'] = susie_df.gene_window_end[susie_df.gene_strand=='-'] - susie_df.pos[susie_df.gene_strand=='-']
susie_df.loc[susie_df.gene_strand=='-', 'ref'] = [reverse_complement(x) for x in susie_df.loc[susie_df.gene_strand=='-', 'ref']]
susie_df.loc[susie_df.gene_strand=='-', 'alt'] = [reverse_complement(x) for x in susie_df.loc[susie_df.gene_strand=='-', 'alt']]

# %%
susie_df['pos_rel_TSS'] = susie_df["pos_relative"] - susie_df["gene_mask_start"]
susie_df['abspos_rel_TSS'] = np.abs(susie_df['pos_rel_TSS'])

# %% [markdown]
# ## Pull in negatives from non-finemapped

# %% [markdown]
# ### Create data structures

# %%
# for each gene, we need boundaries, to exclude "unseeable" stuff
gene_to_boundary_dict = {v.gene_id:{'chr':v.chrom,'start':v.start,'end':v.end, 'strand':v.strand} for k,v in ad.var.iterrows()}
def check_scoreability(row):
    if row['gene_id'] not in gene_to_boundary_dict:
        return False
    boundary = gene_to_boundary_dict[row['gene_id']]
    pos = row['position']
    gene_window_start = boundary['start']
    gene_window_end = boundary['end']
    return (pos > gene_window_start) & (pos < gene_window_end)

# get all genes which have *some* positive
pos_genes = set(susie_df.query('pip > 0.5')['gene_id'])
high_pos_genes = set(susie_df.query('pip > 0.9')['gene_id'])

# %% [markdown]
# ### Load negatives

# %%
negvar_list = []

dtype_list = [np.dtype('O'),np.dtype('O'),np.dtype('int64'),np.dtype('O'),np.dtype('O'),np.dtype('O'),np.dtype('int64'),np.dtype('float64'),np.dtype('float64'),np.dtype('float64'),np.dtype('float64'),np.dtype('O'),np.dtype('int64'),np.dtype('int64'),np.dtype('float64'),np.dtype('O'),np.dtype('O'),np.dtype('float64'),np.dtype('O')]
names = ['molecular_trait_id', 'chromosome', 'position', 'ref', 'alt', 'variant','ma_samples', 'maf', 'pvalue', 'beta', 'se', 'type', 'ac', 'an', 'r2','molecular_trait_object_id', 'gene_id', 'median_tpm', 'rsid']
dtype_dict = {k:v for k,v in zip(names,dtype_list)}

negvar_paths = glob.glob(eqtl_sumstats_base_path)
for path in tqdm.tqdm(negvar_paths):
    negvar_df = pd.read_csv(path, sep="\t", dtype=dtype_dict)
    negvar_select = negvar_df.query('pvalue > 0.05 and type == "SNP" and maf > 0.05') # pre-select nonsignificant
    negvar_select = negvar_select.loc[negvar_select.gene_id.isin(pos_genes)] # consider only genes with some positive
    negvar_select = negvar_select.loc[~negvar_select.variant.isin(cs_vars)] # collect everything which never enters *any* credible set
    negvar_select = negvar_select.loc[negvar_select.apply(lambda row: check_scoreability(row), axis=1)] # check if they are in the right window
    negvar_select['ct_id'] = path.split('/')[-1].split('.')[0]
    negvar_list.append(negvar_select)
    del negvar_df

negvar_all = pd.concat(negvar_list)

# %% [markdown]
# ### Process

# %%
negvar_all['celltype'] = negvar_all['ct_id'].apply(lambda x: ct_dict[x])

# for each negative, compute the relpos and distance to the TSS 
negvar_dedup = negvar_all[['gene_id','variant','position','ref','alt']].drop_duplicates()

# add gene information and calculate relative variant positions (offset)
negvar_dedup = negvar_dedup.merge(ad.var[['gene_id', 'start', 'end', 'strand', 'gene_mask_start']]).rename(columns={'start': 'gene_window_start', 'end': 'gene_window_end', 'strand': 'gene_strand'}) # add window information
negvar_dedup['pos_relative'] = negvar_dedup.position - negvar_dedup.gene_window_start - 1

# use gene_end to calculate offset for - genes and rc() alleles
negvar_dedup.loc[negvar_dedup.gene_strand=='-', 'pos_relative'] = negvar_dedup.gene_window_end[negvar_dedup.gene_strand=='-'] - negvar_dedup.position[negvar_dedup.gene_strand=='-']
negvar_dedup.loc[negvar_dedup.gene_strand=='-', 'ref'] = [reverse_complement(x) for x in negvar_dedup.loc[negvar_dedup.gene_strand=='-', 'ref']]
negvar_dedup.loc[negvar_dedup.gene_strand=='-', 'alt'] = [reverse_complement(x) for x in negvar_dedup.loc[negvar_dedup.gene_strand=='-', 'alt']]

negvar_dedup['pos_rel_TSS'] = negvar_dedup["pos_relative"] - negvar_dedup['gene_mask_start']
negvar_dedup['abspos_rel_TSS'] = np.abs(negvar_dedup['pos_rel_TSS'])

negvar_all = negvar_all.drop(columns=['ref','alt']).merge(negvar_dedup[['gene_id','variant','gene_strand','ref','alt',"pos_relative","pos_rel_TSS","abspos_rel_TSS"]],on=['gene_id','variant'])
negvar_all.to_csv(neg_variant_df_file, index=None)

# %%
negvar_all.to_csv(neg_variant_df_file, index=None)

# %% [markdown]
# ### Match

# %%
negvar_matched = negvar_all.loc[negvar_all.gene_id.isin(high_pos_genes)]
negvar_matched['gene_symbol'] = negvar_matched.gene_id.map(ensembl_id_map)

# %%
# for each positive variant, collect target_negative_n negatives wich are as close to the TSS as possible, and not yet selected for this cell-type
target_negative_n = 20
selected_vars = {}
for celltype in tqdm.tqdm(set(susie_df['celltype'])):
    positive_df = susie_df.query('pip > 0.9 & celltype == @celltype')
    negative_df = negvar_matched.query('celltype == @celltype')#.loc[negvar_matched.gene_id.isin(set(positive_df['gene_id']))]
    selected_vars[celltype] = set()
    for _,positive in positive_df.iterrows():
        negative_sub = negative_df.loc[(negative_df.gene_id == positive['gene_id'])]
        negative_sub = negative_sub.sort_values('abspos_rel_TSS')
        i = 0
        for _,variant in negative_sub.iterrows():
            if variant['variant'] not in selected_vars[celltype]:
                selected_vars[celltype].add(variant['variant'])
                i += 1
                if i == target_negative_n:
                    break

# %%
rows = []
for celltype in selected_vars:
    rows.append({'celltype':celltype, 
                 'pos_genes':len(set(susie_df.query('pip > 0.9 & celltype == @celltype')['gene_symbol'])),
                 'pos_genes_with_neg_ct':len(set(negvar_matched.query('celltype == @celltype')['gene_symbol']) & set(susie_df.query('pip > 0.9 & celltype == @celltype')['gene_symbol'])),
                 'total_pos':len(set(susie_df.query('pip > 0.9 & celltype == @celltype')['variant'])),
                 'total_matched_neg':len(set(negvar_matched.query('celltype == @celltype').merge(susie_df.query('pip > 0.9 & celltype == @celltype')['gene_id'],on='gene_id')['variant'])),
                 'reduced_matched_neg':len(selected_vars[celltype]),
                 })

# %%
negvar_matched_reduced = pd.concat([negvar_matched.loc[negvar_matched.variant.isin(selected_vars[celltype]) & (negvar_matched.celltype == celltype)] for celltype in selected_vars])

# %%
negvar_matched_dedup = negvar_matched_reduced[['gene_symbol','gene_id','variant','position','ref','alt','gene_strand','pos_relative']].drop_duplicates()

# %%
negvar_matched_reduced.to_csv(matched_negative_file ,index=None)
negvar_matched_dedup.to_csv(matched_negative_dedup_file ,index=None)

# %% [markdown]
# ## Assemble Variant Dataset

# %%
# merge all finemapped and matched negatives for all celltypes
negvar_matched_reduced['pip'] = 0
negvar_matched_reduced['cs_id'] = 'negative'
negvar_matched_reduced['cs_size'] = 0
negvar_matched_reduced = negvar_matched_reduced.rename(columns={'ct_id':'celltype_id',
                                                                'chromosome':'chrom',
                                                                'position':'pos',
                                                                })
negvar_matched_reduced = negvar_matched_reduced.drop(columns=['ac','an','ma_samples','maf','median_tpm','molecular_trait_object_id','r2','type'])
susie_df = susie_df.drop(columns=['cs_min_r2','region','z','gene_window_start','gene_window_end'])
susie_df = pd.concat([susie_df, negvar_matched_reduced])
susie_df.to_csv(susie_df_file, index=None)

# %%
variant_df = susie_df[['gene_id', 'gene_symbol', 'variant', 'rsid', 'chrom', 'pos', 'ref', 'alt', 'gene_strand', 'pos_relative']].drop_duplicates(subset=['gene_id','variant'])
variant_df.to_csv(variant_df_file,index=None)

# %% [markdown]
# ## Predict variant effects with Decima

# %%
n_jobs = 8
df_len = len(variant_df)
job_size = math.ceil(df_len / n_jobs)

# %%
for ckpt_file in ckpt_files:
    for i in range(8):
       predict_script = "PredicteQTL.py"
       cmd = f"python {predict_script} -device {i} -task {i} -job_size {job_size} \
-ckpt_file {ckpt_file} -gene_h5_file {h5_file} -variant_df_file {variant_df_file} \
-out_dir {results_path}"
       print(cmd)

# %% [markdown]
# ### Ensemble the Decima scores

# %%
eqtl_ensembl_dict = collections.defaultdict(list)
starts = set()
ends = set()
for k in tqdm.tqdm(ensembl_dict):
    if k.startswith('test'):    
        run_id = ensembl_dict[k]['run_id']
        eqtl_results_path = os.path.join(save_dir, "results", run_id,'eqtl','eqtl_scores_*')
        pred_paths = sorted(glob.glob(os.path.join(eqtl_results_path)), key=lambda x: int(x.split('/')[-1].split("_")[2]))
        for i,pred in enumerate(pred_paths):
            preds = np.load(pred)
            starts.add(int(pred.split('/')[-1].split("_")[2]))
            ends.add(int(pred.split('/')[-1].split('.')[0].split("_")[3]))
            eqtl_ensembl_dict[i].append(preds)
assert len(eqtl_ensembl_dict) == n_jobs
ensembl_eqtl_out_dir = os.path.join(ensembl_out_dir,'eqtl'+suffix)
if not os.path.exists(ensembl_eqtl_out_dir):
    os.mkdir(ensembl_eqtl_out_dir)
starts = sorted(list(starts))
ends = sorted(list(ends))
for i in range(8):
    start = starts[i]
    end = ends[i]
    mean_pred = np.stack(eqtl_ensembl_dict[i]).mean(0)
    np.save(os.path.join(ensembl_eqtl_out_dir,f'eqtl_scores_{start}_{end}'), mean_pred)     

# %% [markdown]
# ## Predict variant effects using Borzoi

# %%
for fold in tqdm.tqdm([0,1,2,3]):
    for i in range(8):
       cmd = f"python Borzoi.py -device {i} -task {i} -job_size {job_size} -tracks {brozoi_tracks_path} -unsquash \
-fold {fold} -gene_h5_file {h5_file} -variant_df_file {variant_df_file} -out_dir {brozoi_out_dir}"
       print(cmd)

# %% [markdown]
# ### Ensemble Borzoi predictions

# %%
for score_type in ['gene', 'tss']:
    brozima_ensembl_dict = collections.defaultdict(list)
    starts = set()
    ends = set()
    for fold in tqdm.tqdm([0,1,2,3]):
        eqtl_results_path = os.path.join(save_dir, "results", 'brozoi',f'brozima_fold{fold}_unsquash',f'{score_type}_scores_*')
        pred_paths = sorted(glob.glob(os.path.join(eqtl_results_path)), key=lambda x: int(x.split('/')[-1].split("_")[2]))
        for i,pred in enumerate(pred_paths):
            preds = np.load(pred)
            starts.add(int(pred.split('/')[-1].split("_")[2]))
            ends.add(int(pred.split('/')[-1].split('.')[0].split("_")[3]))
            brozima_ensembl_dict[i].append(preds)
    assert len(brozima_ensembl_dict) == n_jobs
    brozima_ensembl_eqtl_out_dir = os.path.join(save_dir, "results", 'brozoi',f'brozima_eqtl_unsquash_ensembl')
    if not os.path.exists(brozima_ensembl_eqtl_out_dir):
        os.mkdir(brozima_ensembl_eqtl_out_dir)
    starts = sorted(list(starts))
    ends = sorted(list(ends))
    for i in range(8):
        start = starts[i]
        end = ends[i]
        mean_pred = np.stack(brozima_ensembl_dict[i]).mean(0)
        np.save(os.path.join(brozima_ensembl_eqtl_out_dir,f'{score_type}_scores_{start}_{end}'), mean_pred)
