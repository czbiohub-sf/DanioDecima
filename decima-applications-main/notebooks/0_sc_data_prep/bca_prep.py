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

# %% [markdown]
# # Pseudobulk the Brain Cell Atlas (BCA) data

# %%
import scanpy as sc
import glob
import os
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

pd.options.display.max_columns = None

# %% [markdown]
# ## List files

# %%
files = os.listdir('.')
h5ads = []

for f in files:
    if not os.path.isdir(f) or not f.startswith('human'):
        continue

    h5ads.extend(glob.glob(f + '/processedData/*.h5ad'))

h5ads

# %% [markdown]
# ## Read files and filter cells

# %%
ads = {}

for h5 in tqdm(h5ads):
    ad = sc.read(h5)
    # transfer cell type labels from the adult ref
    if h5 == 'human_brain_wholebrain_Siletti_2022_10x/processedData/meta.h5ad':
        ad.obs['cell_type'] = ad.obs['Supercluster']

    if ad.obs.sample_type.unique()[0] in ['Organoid', 'Fetal']:
        print('skip')
        continue

    if ad.obs.seq_tech.unique()[0] != '10X':
        print('skip')        
        continue
    
    if not np.all(ad.X.data == ad.X.data.astype(int)):
        print('skip')
        continue

    ad.obs.donor_status = ad.obs.donor_status.astype(str)
    ad.obs.donor_ID = ad.obs.donor_ID.astype(str) + '-' + ad.obs.donor_age.astype(str) + '-' + ad.obs.donor_gender.astype(str) + '-' + ad.obs.donor_status.astype(str)
    ad.obs.subregion = ad.obs.subregion.astype(str)
    ad = ad[ad.obs.sample_ID.notna()]

    ads[h5] = ad

# %% [markdown]
# ## Pseudobulk

# %%
pb_cols = [
    'sample_ID',
    'cell_type',
    'donor_ID',
    'sample_status',
    'treatment',
    'region',
    'subregion',    
]

# %% [markdown]
# sc.get.aggregate(
#     ads['human_brain_CV_SunN_2022_10x/processedData/annot.h5ad'], 
#     [
#         #'project_code',
#         'sample_ID',
#         'cell_type',
#         'donor_ID',        
#         #'donor_age',
#         #'donor_gender',
#         #'donor_status',
#         'sample_status',
#         #'sample_type',
#         #'seq_method',
#         #'seq_tech',
#         'treatment',
#         #'reference',
#         'region',
#         'subregion',    
#     ], func='sum')

# %%
ads_pb = []

for name, ad in tqdm(list(ads.items())):
    print(name)

    project = ad.obs.project_code.values[0]
    ad.obs = ad.obs[pb_cols]
            
    ad_pb = sc.get.aggregate(ad, pb_cols, func='sum')
    ad_pb.X = ad_pb.layers['sum']
    del ad_pb.layers['sum']
    ad_pb.obs['project_code'] = project
    ad.obs['project_code'] = project    

    ad_pb.obs.index =  ad_pb.obs.project_code.astype(str) + '-' + ad_pb.obs.donor_ID.astype(str) + '-' + ad_pb.obs.sample_ID.astype(str) + '-' + ad_pb.obs.cell_type.astype(str) + '-' + ad_pb.obs.region.astype(str) + '-' + ad_pb.obs.subregion.astype(str)

    # add cell counts
    counts = ad.obs[pb_cols].value_counts().reset_index()
    counts['project_code'] = project
    counts.index = counts.project_code.astype(str) + '-' + counts.donor_ID.astype(str) + '-' + counts.sample_ID.astype(str) + '-' + counts.cell_type.astype(str) + '-' + counts.region.astype(str) + '-' + counts.subregion.astype(str)
    counts = counts.rename(columns={'count': 'n_cells'})
    
    ad_pb.obs['study_name'] = name
    ad_pb.obs['n_cells'] = counts.n_cells
    
    ad_pb = ad_pb[~ad_pb.obs.cell_type.isin(['unannotated', 'unannoted', 'Miscellaneous'])].copy()
    ad_pb = ad_pb[ad_pb.obs.n_cells > 10].copy()

    ad_pb.var = ad_pb.var.iloc[:, 0:0] #drop all var columns
    ad_pb.var.index.name = None

    display(ad_pb)
    
    ads_pb.append(ad_pb)

# %% [markdown]
# ## Save

# %%
final_ad = sc.concat(ads_pb, join='outer')
final_ad.write('bca_pseudobulk.h5ad')
