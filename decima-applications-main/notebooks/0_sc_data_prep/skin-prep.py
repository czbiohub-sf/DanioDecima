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
# # Pseudobulk the Skin Atopic Dermatitis atlas

# %%
import scanpy as sc
import pandas as pd
pd.options.display.max_columns = None

# %% [markdown]
# ## Load

# %%
ad = sc.read('scp-atlas-export.h5ad') # downloaded from https://singlecell.broadinstitute.org/single_cell/study/SCP2738/

# %%
ad.X = ad.layers['counts']
del ad.layers['counts']

# %%
ad.obs = ad.obs[['sample_ID', 'disease_status', 'study_ID', 'study_accession', 'celltype_granular', 'celltype_coarse']]

# %%
ad.obs.sample_ID = ad.obs.sample_ID.astype(str) + '-' + ad.obs.study_accession.astype(str)

# %%
ad.obs.disease_status = ad.obs.disease_status.astype(str)

# %%
ad.obs.loc[ad.obs.disease_status=='Lesional', 'disease_status'] = 'atopic dermatitis (lesional)'
ad.obs.loc[ad.obs.disease_status=='Non-lesional', 'disease_status'] = 'atopic dermatitis (non-lesional)'
ad.obs.loc[ad.obs.disease_status=='Lesional Dupilumab (16 wk)', 'disease_status'] = 'atopic dermatitis (lesional, Dupilumab, 16 wk)'
ad.obs.loc[ad.obs.disease_status=='Lesional Dupilumab (1 yr)', 'disease_status'] = 'atopic dermatitis (lesional, Dupilumab, 1 yr)'

# %%
ad.obs.study_ID = ad.obs.study_ID.astype(str)

# %%
ad.obs.loc[ad.obs.study_ID=='Current study', 'study_ID'] = 'Fiskin et al.'

# %% [markdown]
# ### Nicer cell types

# %%
celltype_map = {
 'KC 1': 'Keratinocyte 1',
 'KC cycling': 'Keratinocyte cycling',
 'KC 2': 'Keratinocyte 2',
 'KC 3': 'Keratinocyte 3',
 'KC 4': 'Keratinocyte 4',
 'KC 5': 'Keratinocyte 5',
 'Cornified KC 1': '',
 'Cornified KC 2': '',
 'HF': 'Hair follicle',
 'Sweat gland 1': '',
 'Sweat gland 2': '',
 'Sebaceous gland': '',
 'Sweat gland 3': '',
 'FB papillary': 'Fibroblast papillary',
 'FB CCL19+IL4I1+': 'Fibroblast CCL19+IL4I1+',
 'FB CCL19+APOE+': 'Fibroblast CCL19+APOE+',
 'FB APOC1+': 'Fibroblast APOC1+',
 'FB GDF10+': 'Fibroblast GDF10+',
 'FB CPE+': 'Fibroblast CPE+',
 'FB CDH19+': 'Fibroblast CDH19+',
 'FB NGFR+': 'Fibroblast NGFR+',
 'FB reticular': 'Fibroblast reticular',
 'FB DPEP1+': 'Fibroblast DPEP1+',
 'FB dermal papilla': 'Fibroblast dermal papilla',
 'PE TGFBI+': 'Pericyte TGFBI+',
 'PE RGS hi': 'Pericyte RGS hi',
 'PE/SMC GEM hi': 'Pericyte/SMC GEM hi',
 'SMC RERGL+': 'Smooth muscle cell RERGL+',
 'SMC DES+': 'Smooth muscle cell DES+',
 'ArtEC ICAM2 hi': 'Arterial endothelial ICAM2 hi',
 'ArtEC SOS1 hi': 'Arterial endothelial SOS1 hi',
 'CapEC FABP4+': 'Capillary endothelial FABP4+',
 'CapEC EDNRB hi': 'Capillary endothelial EDNRB hi',
 'CapEC INSR hi': 'Capillary endothelial INSR hi',
 'VenEC IL6+': 'Venous endothelial IL6+',
 'VenEC CCL15 hi': 'Venous endothelial CCL15 hi',
 'VenEC CCL14 hi': 'Venous endothelial CCL14 hi',
 'ArtEC RGS5+IGFBP3+': 'Arterial endothelial RGS5+IGFBP3+',
 'CapEC RGS5+FABP4+': 'Capillary endothelial RGS5+FABP4+',
 'CapEC RGS5+EDNRB hi': 'Capillary endothelial RGS5+EDNRB hi',
 'Cap EC RGS5+EDNRB hi MCAM hi': 'Capillary endothelial RGS5+EDNRB hi MCAM hi',
 'VenEC RGS5+ACKR1 hi': 'Venous endothelial RGS5+ACKR1 hi',
 'LEC': 'Lymphetic endothelial cell',
 'Melano S100A4-': '',
 'Melano S100A4+': '',
 'Melano IFI27 hi': '',
 'Schwann MBP+': '',
 'Schwann NRXN1 hi': '',
 'Schwann LAMP5+': '',
 'Schwann DCN+': '',
 'DC 1': '',
 'DC 1 cycling': '',
 'DC 2': '',
 'DC 2 CD83+': '',
 'DC 2 cycling': '',
 'LDC': 'DC (Langerhans)',
 'DC MMP12+': '',
 'mmDC': 'DC (mregDC/mmDC)',
 'DC IL1B+': '',
 'MΦ IL1B+': '',
 'MΦ FNIP2 hi': '',
 'MΦ EGR1+': '',
 'MΦ C1QA hi': '',
 'MΦ STAB1 hi': '',
 'MΦ SPP1+': '',
 'Neutrophil': '',
 'Mast CDC42EP3 hi': '',
 'Mast CD69 hi': '',
 'Mast CD63 hi': '',
 'Mast cycling': '',
 'B Naive/Mem': '',
 'Plasmablast': '',
 'Plasma IgA': '',
 'Plasma IgG': '',
 'NK': '',
 'ILC': '',
 'ILC cycling': '',
 'γδ T': 'gd T cell',
 'CD8+ CTL': '',
 'CD8+ CTL IFNG hi': '',
 'Treg': '',
 'Treg cycling': '',
 'T CREM hi': '',
 'T CREM lo FOS lo': '',
 'T FOS hi': '',
 'T FOS hi cycling': ''}

# %%
for k in celltype_map:
    if celltype_map[k] == '':
        celltype_map[k] = k

# %%
ad.obs.celltype_granular = ad.obs.celltype_granular.map(celltype_map)

# %% [markdown]
# ## Now the pseudobulking

# %%
ident_cols = ['sample_ID', 'disease_status', 'study_ID', 'study_accession', 'celltype_granular', 'celltype_coarse']

# %%
adp = sc.get.aggregate(ad, ident_cols, func='sum')

# %%
adp.X = adp.layers['sum'].astype(int)

# %%
del adp.layers['sum']

# %% [markdown]
# ### Add cell counts

# %%
counts = ad.obs[ident_cols].value_counts().reset_index()
counts = counts.rename(columns={'count': 'n_cells'})

# %%
adp.obs = adp.obs.merge(counts, how='left')

# %%
adp = adp[adp.obs.n_cells>=10].copy()

# %% [markdown]
# ## Save

# %%
adp.write('skin-pseudobulk.h5ad')
