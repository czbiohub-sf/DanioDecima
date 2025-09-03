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
import numpy as np
import pandas as pd
import anndata
import os
import glob
from tqdm import tqdm
from io import StringIO

import matplotlib.pyplot as plt
from plotnine import *
# %matplotlib inline

# %% [markdown]
# # Paths

# %%
matrix_file = "/gstore/data/resbioai/grelu/decima/20240823/data.h5ad"
ensembl_out_dir = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/results/ensemble"
borzoi_out_dir = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/results/brozoi"
borzoi_tracks_path = '/gstore/data/resbioai/karollua/Decima/scborzoi/decima/data/borzoi_targets/targets_human.txt'

save_dir = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/"
brozima_ensembl_out_dir = os.path.join(save_dir, "results", 'brozoi', 'brozima_unsquash_ensembl')
susie_df_file = os.path.join(save_dir,'data/eQTL_processed/susie_df.csv')
variant_df_file = os.path.join(save_dir,'data/eQTL_processed/vars.csv')

# %% [markdown]
# ## Load data

# %%
ad = anndata.read_h5ad(matrix_file)
susie_df = pd.read_csv(susie_df_file)
variant_df = pd.read_csv(variant_df_file)
borzoi_tracks = pd.read_csv(borzoi_tracks_path,sep='\t')

# %% [markdown]
# ## Load Decima scores

# %%
pred_paths = sorted(glob.glob(os.path.join(ensembl_out_dir,'eqtl',"eqtl_scores_*")), key=lambda x: int(x.split('/')[-1].split("_")[2]))
scores = np.concatenate([np.load(pred) for pred in tqdm(pred_paths)])

# %%
eqtl_ad = anndata.AnnData(scores, obs=variant_df.copy().reset_index(drop=True), var=ad.obs)
print(eqtl_ad.shape)
del scores

# %%
# %%time
gene_expr = {g: np.array(ad[:, ad.var.gene_id==g].X).squeeze() for g in variant_df.gene_id.unique()}
ref_expr = np.stack(variant_df.gene_id.map(gene_expr))
print(ref_expr.shape)
eqtl_ad.layers['ref_expr'] = ref_expr

# %%
eqtl_ad.write_h5ad('eqtl.h5ad')

# %%
# reduce to blood cells
eqtl_ad = eqtl_ad[:,(eqtl_ad.var.tissue == 'blood')].copy()
eqtl_ad.var = eqtl_ad.var.reset_index(drop=True)

# %% [markdown]
# ## Load borzoi predictions

# %%
scores_borzoi = []
scores_borzoi_tss = []
for result_list,score_type in zip([scores_borzoi, scores_borzoi_tss],['gene','tss']):
    pred_paths = sorted(glob.glob(os.path.join(borzoi_out_dir,f'brozima_eqtl_unsquash_ensembl',f"{score_type}_scores_*")), key=lambda x: int(x.split('/')[-1].split("_")[2]))
    for pred in tqdm(pred_paths):
        # load scores
        preds = np.load(pred)
        # convert to log1p
        preds = np.log(preds + 1)
        # get log fold change
        ref_preds = preds[:len(preds)//2]
        alt_preds = preds[len(preds)//2:]
        result_list.append(alt_preds - ref_preds)

scores_borzoi = np.concatenate(scores_borzoi)
scores_borzoi_tss = np.concatenate(scores_borzoi_tss)
print(scores_borzoi.shape, scores_borzoi_tss.shape)

# %%
# get relevant borzoi tracks
rna_tracks = borzoi_tracks.loc[borzoi_tracks.description.str.startswith('RNA:')]
rna_idx = np.array(rna_tracks.index)
rna_tracks = rna_tracks.reset_index(drop=True)
cage_tracks = borzoi_tracks.loc[borzoi_tracks.description.str.startswith('CAGE:')]
cage_idx = np.array(cage_tracks.index)
cage_tracks = cage_tracks.reset_index(drop=True)

# subset gene to RNA and TSS to CAGE
scores_borzoi = scores_borzoi[:,rna_idx]
scores_borzoi_tss = scores_borzoi_tss[:,cage_idx]

# Make anndata
eqtl_ad_borzoi = anndata.AnnData(np.concatenate([scores_borzoi, scores_borzoi_tss],axis=1), 
    obs=variant_df.copy().reset_index(drop=True), var=pd.concat([rna_tracks, cage_tracks]).reset_index(drop=True))

# %% [markdown]
# ## Matching the cell type - Decima

# %%
from eqtl_meta import cell_type_mapping
cell_type_mapping = pd.read_table(StringIO(cell_type_mapping))

# %%
# for each variant in susie_df, extract the predictions for matching cell type
# generate masking tensor with the celltype matching converted to a dict: eqtl_celltype --> mask

celltype_to_mask_dict = {}
for _,row in cell_type_mapping.groupby('eqtl_celltype')['model_celltype'].agg(list).reset_index().iterrows():
    eqtl_celltype = row['eqtl_celltype']
    model_celltypes = row['model_celltype']
    mask = np.zeros(len(eqtl_ad.var))
    for ix in eqtl_ad.var.query('cell_type in @model_celltypes').index:
        mask[int(ix)] = 1
    celltype_to_mask_dict[eqtl_celltype] = mask

# take global average for unknown cells
for ct in set(susie_df.celltype).difference(cell_type_mapping.eqtl_celltype): 
    celltype_to_mask_dict[ct] = np.ones(len(eqtl_ad.var))

# %%
# merge variants back to susie df to get mapping
susielen = len(susie_df)
susie_backmerged = susie_df.merge(eqtl_ad.obs.reset_index()[['index','gene_id','variant','rsid']],on=['gene_id','variant','rsid'])
susie_backmerged['index'] = susie_backmerged['index'].astype(int)

# extract masked predictions
scores = np.zeros(len(susie_backmerged))
ref_obs = np.zeros(len(susie_backmerged))
i = 0
for _, row in tqdm(susie_backmerged.iterrows()):
    mask = celltype_to_mask_dict[row.celltype]
    score = (eqtl_ad.X[row['index']]*mask).sum()/mask.sum()
    ref_expression = (eqtl_ad.layers['ref_expr'][row['index']]*mask).sum()/mask.sum()
    scores[i] = score
    ref_obs[i] = ref_expression
    i += 1

susie_backmerged['matched_score'] = scores
susie_backmerged['abs_matched_score'] = np.abs(scores)
susie_backmerged['ref_expr'] = ref_obs

# %% [markdown]
# ## Matching the cell type - Borzoi

# %%
borzoi_mapping = {
    "B cell":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('B cell') 
                        & ~(eqtl_ad_borzoi.var.description.str.contains('memory'))
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],
    "memory B cell":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('B cell') 
                        & (eqtl_ad_borzoi.var.description.str.contains('memory'))
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],
    "CD4+ T cell":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('CD4')
                        & ~eqtl_ad_borzoi.var.description.str.contains('CD25')
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],
    "CD4+ CTL cell":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('CD4')
                        & ~eqtl_ad_borzoi.var.description.str.contains('CD25')
                        & ~eqtl_ad_borzoi.var.description.str.contains('memory')
                        & eqtl_ad_borzoi.var.description.str.contains('activated|stimulated')
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],
    "CD4+ TCM cell":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('CD4')
                        & ~eqtl_ad_borzoi.var.description.str.contains('CD25')
                        & eqtl_ad_borzoi.var.description.str.contains('memory')
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],    
    "CD4+ TEM cell":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('CD4')
                        & ~eqtl_ad_borzoi.var.description.str.contains('CD25')
                        & eqtl_ad_borzoi.var.description.str.contains('memory')
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],
    "CD8+ T cell":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('CD8')
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],
    "CD8+ TCM cell":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('CD8')
                        & eqtl_ad_borzoi.var.description.str.contains('memory')
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],    
    "CD8+ TEM cell":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('CD8')
                        & eqtl_ad_borzoi.var.description.str.contains('memory')
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],
    "Treg memory":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('CD25')
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],
    "MAIT cell":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('CD4|CD8')
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))], # ???, T-cells as group seem closest
    "dendritic cell":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('CAGE') 
                        & eqtl_ad_borzoi.var.description.str.contains('dendri',case=False)
                        & ~eqtl_ad_borzoi.var.description.str.contains('plasma',case=False)
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],
    "plasmacytoid dendritic cell":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('CAGE') 
                        & eqtl_ad_borzoi.var.description.str.contains('dendri',case=False)
                        & eqtl_ad_borzoi.var.description.str.contains('plasma',case=False)
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],
    "hematopoietic precursor cell":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('hemato',case=False)
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],
    "monocyte":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('CD14',case=False)
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],
    "CD16+ monocyte":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('CAGE') 
                        & eqtl_ad_borzoi.var.description.str.contains('CD14-CD16',case=False)
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],
    "NK cell":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('killer',case=False)
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],
    "CD56+ NK cell":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('killer',case=False)
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))], 
    "plasmablast":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('B cell',case=False)
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))], # B cell seems closest
    "platelet":eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('CAGE') 
                        & eqtl_ad_borzoi.var.description.str.contains('megakaryo',case=False)
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))],  # Megakaryocyte seems closest??
    'dnT cell':eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('CD4|CD8')
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))], # ???, T-cells as group seem closest, 
    'gdT cell':eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description.str.contains('RNA') 
                        & eqtl_ad_borzoi.var.description.str.contains('CD4|CD8')
                        & ~(eqtl_ad_borzoi.var.identifier.str.endswith('-'))], # ???, T-cells as group seem closest,

}

# %%
blood_idx = np.array(eqtl_ad_borzoi.var.loc[eqtl_ad_borzoi.var.description == "RNA:blood"].index).astype('int')

# make cell mapping mask
celltype_to_mask_dict = {}
for eqtl_celltype, borzoi_celltype_df in borzoi_mapping.items():
    indices = list(borzoi_celltype_df.index)
    mask = np.zeros(len(eqtl_ad_borzoi.var))
    for ix in indices:
        mask[int(ix)] = 1
    celltype_to_mask_dict[eqtl_celltype] = mask

# extract masked predictions
scores = np.zeros(len(susie_backmerged))
scores_wholeblood = np.zeros(len(susie_backmerged))
i = 0
for _,row in tqdm(susie_backmerged.iterrows()):
    mask = celltype_to_mask_dict[row['celltype']]
    score = (eqtl_ad_borzoi.X[row['index']]*mask).sum()/mask.sum()
    scores[i] = score
    scores_wholeblood[i] = eqtl_ad_borzoi.X[row['index'],blood_idx].mean()
    i += 1
susie_backmerged['borzoi_matched_score'] = scores
susie_backmerged['abs_borzoi_matched_score'] = np.abs(scores)

# also extract RNA whole-blood results
susie_backmerged['borzoi_wholeblood_score'] = scores_wholeblood
susie_backmerged['abs_borzoi_wholeblood_score'] = np.abs(scores_wholeblood)

# %% [markdown]
# ## Remove duplicates

# %%
# for some celltypes, we have duplicate cell-type IDs. There are also some duplicate rsids
# We keep the best pip variant
susie_backmerged_dedup = susie_backmerged.sort_values(['variant','gene_id','celltype','pip']).drop_duplicates(
    subset=['gene_id','variant','celltype'], keep='last')
len(susie_backmerged_dedup)

# %% [markdown]
# ## Save model predictions

# %%
susie_backmerged_dedup.to_csv('susie_backmerged_dedup.csv', index=None)

# %%
