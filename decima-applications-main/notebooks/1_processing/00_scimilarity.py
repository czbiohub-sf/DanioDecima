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
# # Process scimilarity dataset

# %%
import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import sys
sys.path.append('/code/decima/src/decima/')
sys.path.append('.')

import preprocess
import scimilarity

# %% [markdown]
# ## Paths

# %%
scimilarity_file = "/gstore/data/omni/scdb/models/human/model_2023_04_rep0/genesearch/pseudobulk.h5ad"
save_dir="/gstore/data/resbioai/grelu/decima/20240823/processed_pseudobulks"

# %% [markdown]
# ## Load count matrix

# %%
ad = anndata.read_h5ad(scimilarity_file)
ad.X = ad.layers['counts']
ad.obs.index = ad.obs.index.astype(str)
ad.obs_names_make_unique()

print(ad.shape)
display(ad.obs.head(1))
display(ad.var.head(1))

# %% [markdown]
# ## Process .obs

# %% [markdown]
# ### Column names

# %%
ad.obs = ad.obs.rename(columns={'prediction':'cell_type', 'cells': 'n_cells'})
ad.obs = ad.obs.drop(columns='data_type')
ad.obs = ad.obs.astype({'study':'str', 'sample':'str', 'cell_type':'str', 'tissue':'str', 'disease':'str',
       'in_vitro':'str', 'in_vivo':'str', 'ex_vivo':'str', 'organoid':'str', 'cell_line':'str', 'n_cells':'int'})

# %% [markdown]
# ### Drop cancers

# %%
print(ad.shape)
cancers = [x for x in ad.obs.disease.unique() if(('oma' in x) or ('tumor' in x) or ('cancer' in x) or ('leukemia' in x))]
ad = ad[~ad.obs.disease.isin(cancers), :].copy()
print(ad.shape)

# %% [markdown]
# ### Drop cell lines, organoids and unannotated

# %%
print(ad.shape)
ad = ad[ad.obs.cell_line!='True']
print(ad.shape)

ad = ad[ad.obs.organoid!="True"]
print(ad.shape)

ad = ad[ad.obs.tissue!="NA"]
print(ad.shape)

# %%
ad.obs = ad.obs.drop(columns=['cell_line', 'organoid'])

# %% [markdown]
# ### Drop fetal cells

# %%
fetal_terms = ['blastocyst', 'embryo', 'amniotic fluid', 'yolk sac', 'placenta', 'umbilical cord blood']

print(ad.shape)
ad = ad[~ad.obs.tissue.isin(fetal_terms)]
print(ad.shape)

# %% [markdown]
# ### Drop brain related terms

# %%
print(ad.shape)

ad = ad[~ad.obs.tissue.isin(scimilarity.scimilarity_brain_tissues)]
print(ad.shape)

ad = ad[~(ad.obs.cell_type.isin(scimilarity.scimilarity_brain_cts))]
print(ad.shape)

# %% [markdown]
# ### Drop skin related terms

# %%
print(ad.shape)
ad = ad[~ad.obs.tissue.isin(['skin epidermis', 'skin of body', 'skin of leg', 'skin of prepuce of penis', 'zone of skin', 'scrotum skin'])]

print(ad.shape)
ad = ad[~ad.obs.cell_type.isin(['keratinocyte', 'melanocyte'])]
print(ad.shape)

# %% [markdown]
# ### Drop retinal terms

# %%
print(ad.shape)
ad = ad[~ad.obs.tissue.isin([
    'eye', 'corneal epithelium', 'fovea centralis', 'sclera', 'lacrimal gland', 'pigment epithelium of eye', 'retina', 'macula lutea proper', 'peripheral region of retina'
])]
print(ad.shape)

# %% [markdown]
# ### Fix disease annotations

# %%
disease_dict = {
    'COVID-19;healthy':'COVID-19',
    'type 2 diabetes mellitus':'type II diabetes mellitus',
}
ad.obs = preprocess.change_values(ad.obs, col='disease', value_dict=disease_dict)

# %% [markdown]
# ### Fix tissue annotations

# %%
tissue_dict = {
    'adult mammalian kidney':'kidney', 
    'upper outer quadrant of breast':'breast',
    'venous blood':'blood',
    'bone tissue':'bone',
    'left colon':'descending colon',
    'right colon':'ascending colon',
}

ad.obs = preprocess.change_values(ad.obs, col="tissue", value_dict=tissue_dict)

# %% [markdown]
# ### Remove cells that don't make sense

# %%
print(ad.shape)
ad = ad[~ad.obs.cell_type.isin(['erythrocyte', 'neutrophil']), :]
print(ad.shape)

# %% [markdown]
# ### Drop mislabeled cells

# %%
drop = {

'alveolar macrophage': [
    'kidney', 'aorta', 'transverse colon', 'vasculature', 'trachea', 'islet of Langerhans', 'psoas muscle', 'synovial membrane of synovial joint', 'peritoneum', 
    'thoracic lymph node', 'mucosa of descending colon', 'blood', 'left cardiac atrium', 'adrenal gland', 'cardiac muscle of left ventricle', 'liver', 'descending colon', 
    'nasopharynx', 'heart left ventricle', 'ovary', 'thymus', 'bronchus', 'interventricular septum', 'tertiary ovarian follicle', 'ureter', 'prostate gland'
],

'ciliated cell': [
    'adrenal gland', 'muscle tissue', 'urothelium', 'lingula of left lung', 'vasculature',    
],
    
'club cell': [
    'transition zone of prostate;urethra', 'vasculature', 'fallopian tube', 'islet of Langerhans', 'colonic mucosa', 'epididymis epithelium', 'urinary bladder', 'mucosa of descending colon', 
    'muscle tissue', 'descending colon', 'ovary', 'colon', 'ureter', 'inferior nasal concha', 'prostate gland'
],
    
'common lymphoid progenitor': [
    'nasopharynx', 'adrenal gland', 'intestine', 'kidney', 'ileum', 'digestive tract'
],
    
'endothelial cell of hepatic sinusoid':	[
    'vasculature', 'trachea', 'renal medulla', 'renal papilla', 'spleen', 'mucosa of gallbladder', 'adrenal gland', 'heart left ventricle', 'ovary', 'pancreas'
],
    
'enterocyte': [
    'prostate gland', 'transition zone of prostate;urethra', 'vasculature', 'olfactory epithelium', 'urothelium', 'breast', 'islet of Langerhans', 'renal pelvis', 'renal medulla', 
    'renal papilla', 'respiratory airway', 'epididymis epithelium', 'aorta', 'bile duct', 'exocrine pancreas', 'urinary bladder', 'transition zone of prostate', 
    'adrenal gland', 'cortex of kidney', 'uterus', 'skin of prepuce of penis', 'kidney', 'nasopharynx', 'ovary', 'thymus', 'kidney blood vessel', 'tertiary ovarian follicle', 
    'ureter', 'pancreas', 'peripheral zone of prostate', 'inner medulla of kidney', 'testis', 'outer cortex of kidney', 'lung'
],
    
'enteroendocrine cell':	[
    'vasculature', 'epididymis epithelium', 'uterus', 'muscle tissue', 'thymus', 'lung'
],
    
'erythroid lineage cell': [
    'transverse colon', 'intestine', 'islet of Langerhans', 'renal medulla', 'renal papilla', 'spleen', 'thoracic lymph node', 'adrenal gland', 'heart', 'cardiac muscle of left ventricle', 
    'liver', 'cortex of kidney', 'kidney', 'ovary', 'thymus', 'testis', 'ileum', 'lung'
],
    
'goblet cell':	[
    'inferior nasal concha', 'islet of Langerhans', 'pancreas', 'exocrine pancreas', 'fallopian tube', 'uterus', 'vasculature', 'urothelium', 'breast', 'aorta', 'spleen', 
    'bile duct', 'liver', 'muscle tissue', 'caudate lobe of liver'
],
    
'hematopoietic stem cell':	[
    'thymus', 'respiratory airway', 'vasculature', 'subcutaneous adipose tissue', 'thoracic lymph node'
],
    
'hepatocyte': [
    'cardiac ventricle', 'vasculature', 'trachea', 'right cardiac atrium', 'urothelium', 'breast', 'islet of Langerhans', 'colonic mucosa', 'urine', 'respiratory airway', 
    'apex of heart', 'psoas muscle', 'esophagus muscularis mucosa', 'mucosa of gallbladder', 'upper lobe of right lung', 'bone', 'exocrine pancreas', 'peritoneum', 
    'mucosa of descending colon', 'blood', 'left cardiac atrium', 'lingula of left lung', 'adrenal gland', 'heart', 'cardiac muscle of left ventricle', 'cortex of kidney', 'gastrocnemius', 
    'kidney', 'heart left ventricle', 'descending colon', 'ovary', 'respiratory tract epithelium', 'colon', 'thymus', 'interventricular septum', 
    'tertiary ovarian follicle', 'pancreas', 'heart right ventricle', 'inner medulla of kidney', 'testis', 'prostate gland', 'outer cortex of kidney', 'lung'
],

'intestinal tuft cell':	['olfactory epithelium'],
'ionocyte':	['colon', 'vasculature', 'urothelium'],
'keratinocyte':	['kidney', 'vasculature'],
    
'kidney proximal convoluted tubule epithelial cell': [
    'mucosa of descending colon', 'colon', 'adrenal gland', 'vasculature', 'pancreas', 'mucosa of gallbladder', 'testis', 'bile duct', 'descending colon', 'islet of Langerhans', 'ovary'
],
    
'luminal cell of prostate epithelium': [
    'adrenal gland', 'kidney', 'right lobe of liver', 'fallopian tube', 'islet of Langerhans', 'colonic mucosa', 'mucosa', 'aorta', 'mucosa of gallbladder', 'bile duct', 'exocrine pancreas', 
    'mucosa of descending colon', 'liver', 'muscle tissue', 'descending colon', 'ovary', 'colon', 'caudate lobe of liver', 'pancreas'
],
    
'luminal epithelial cell of mammary gland':	[
    'aorta', 'bronchus', 'subcutaneous adipose tissue', 'urothelium', 'islet of Langerhans', 'peritoneum', 'mucosa', 'respiratory airway', 'bone', 
    'exocrine pancreas', 'urinary bladder', 'adrenal gland', 'uterus', 'respiratory tract epithelium', 'colon', 'pancreas', 'lung'
],
    
'lung secretory cell':	['nasal turbinal'],
'melanocyte':	[
    'thymus', 'heart', 'psoas muscle', 'portion of cartilage tissue in tibia', 'subcutaneous adipose tissue', 'vasculature', 'nasal cavity', 'gastrocnemius'
],
    
'paneth cell':	['vasculature', 'stomach', 'urinary bladder'],
    
'parietal epithelial cell':	['colon', 'adrenal gland', 'testis', 'bone', 'ovary'],

'pulmonary ionocyte':	[
    'vasculature', 'fallopian tube', 'colonic mucosa', 'renal medulla', 'renal papilla', 'epididymis epithelium', 'mucosa of descending colon', 'cortex of kidney', 'alveolar system', 
    'descending colon', 'kidney', 'thymus', 'ureter'
],
    
'respiratory basal cell':	[
    'periodontium', 'vasculature', 'fallopian tube', 'colonic mucosa', 'peritoneum', 'descending colon', 'kidney', 
    'colon', 'thymus', 'ureter', 'inner medulla of kidney', 'testis', 'prostate gland', 'outer cortex of kidney', 'urinary bladder'
],
    
'type I pneumocyte':	['left cardiac atrium', 'colon', 'vasculature', 'muscle tissue'],
'type II pneumocyte':	['vasculature', 'inferior nasal concha', 'muscle tissue'],

}

# %%
print(ad.shape)
for ct, tissues in drop.items():
    ad = ad[~((ad.obs.cell_type==ct) & (ad.obs.tissue.isin(tissues)))]
print(ad.shape)

# %%
ad.obs.groupby('cell_type').tissue.apply(lambda x: set(x)).to_csv('ct_tissue_map.txt', sep='\t', header=False)

# %% [markdown]
# ### Annotate organ

# %%
ad.obs['organ'] = ad.obs['tissue'].map(scimilarity.tissue_to_organ)

# %% [markdown]
# ## Save

# %%
ad.write_h5ad(os.path.join(save_dir, "scimilarity_processed.h5ad"))
