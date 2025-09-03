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
import h5py
from tqdm import tqdm
import bioframe as bf
import os, sys
from scipy.stats import mannwhitneyu, kruskal, wilcoxon
from plotnine import *
from grelu.sequence.utils import resize

sys.path.append('/code/decima/src/decima')
from resources import load_gtf

# %matplotlib inline

# %% [markdown]
# ## Paths

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823"
matrix_file = os.path.join(save_dir, "data.h5ad")
h5_file = os.path.join(save_dir, "attr.h5")

# %% [markdown]
# ## Load test genes

# %%
ad = anndata.read_h5ad(matrix_file)
ad = ad[:, ad.var.dataset == "test"].copy()

# %%
genes = ad.var.reset_index()
genes['gene'] = ad.var_names
genes['st'] = genes.gene_start - genes.start
genes['en'] = [min(524287, x) for x in genes.gene_end - genes.start]

# %% [markdown]
# ## Load CREs

# %%
# Encode CREs
# #!wget https://downloads.wenglab.org/V3/GRCh38-cCREs.bed

# %%
cre = pd.read_table('GRCh38-cCREs.bed', header=None, usecols=(0, 1, 2, 5))
cre.columns=['chrom', 'start', 'end', 'class']

# %% [markdown]
# ## Load exon annotations

# %%
gtf = load_gtf(
    '/gstore/data/resbioai/grelu/decima/refdata-gex-GRCh38-2020-A/genes/genes.gtf',
    feature='exon')
gtf = gtf[gtf.gene_name.isin(ad.var_names)]

# %% [markdown]
# ## Drop genes for which we have no exon annotations

# %%
genes = genes[genes.gene.isin(gtf.gene_name)]

# %% [markdown]
# ## Read attributions

# %%
attrs = {}
with h5py.File(h5_file, 'r') as f:
    for gene in ad.var_names:
        attr = np.array(f[gene])
        if ad.var.strand[gene]=='-':
            attr = attr[::-1]
        attrs[gene] = np.abs(attr)

# %% [markdown]
# ## Overlap annotations with gene intervals

# %%
cre_overlap = bf.overlap(genes, cre, how='inner')
cre_overlap['st'] = cre_overlap.start_ - cre_overlap.start
cre_overlap['en'] = cre_overlap.end_ - cre_overlap.start
cre_overlap['dist'] = np.abs(np.vstack([cre_overlap.start - cre_overlap.gene_start, cre_overlap.start - cre_overlap.gene_end])).min(0)

# %%
cre_overlap.loc[cre_overlap.dist < 100, 'dist_class'] = '0-100'
cre_overlap.loc[(cre_overlap.dist >= 100) & (cre_overlap.dist < 1000), 'dist_class'] = '100-1kb'
cre_overlap.loc[(cre_overlap.dist >= 1000) & (cre_overlap.dist < 10000), 'dist_class'] ='1-10kb'
cre_overlap.loc[(cre_overlap.dist >= 10000) & (cre_overlap.dist < 100000), 'dist_class'] ='10-100kb'
cre_overlap.loc[cre_overlap.dist >= 100000, 'dist_class'] ='>100kb'

# %%
annot = dict()
for gene in tqdm(ad.var_names):
    exons = gtf[(gtf.gene_name==gene) & (gtf.start >= ad.var.start[gene]) & (gtf.end <= ad.var.end[gene])].copy()
    exons['st'] = exons['start'] - ad.var.start[gene]
    exons['en'] = exons['end'] - ad.var.start[gene]
    annot[gene] = {'exons':exons}
    annot[gene]['cre'] = cre_overlap[cre_overlap.gene==gene]

# %% [markdown]
# ## Compare attributions in different classes

# %%
genes = genes.iloc[:, :23]
promoter_window = 100
junction_window = 10

for row in tqdm(genes.itertuples()):
    
    exons = annot[row.gene]['exons']
    cres = annot[row.gene]['cre']
    attr = attrs[row.gene]
    
    in_gene = np.array([False]*524288)
    in_promoter = np.array([False]*524288)
    in_exons = np.array([False]*524288)
    in_introns = np.array([False]*524288)
    in_junctions = np.array([False]*524288)
    in_cre = np.array([False]*524288)
    out_1k = np.array([False]*524288)
    out_1k_10k = np.array([False]*524288)
    out_10k_100k = np.array([False]*524288)
    out_100k = np.array([False]*524288)

    # select bases in gene/ promoter / exons / junctions / CREs
    in_gene[row.st: row.en] = True
    if row.strand=='+':
        in_promoter[row.st-promoter_window: row.st+promoter_window] = True
    else:
        in_promoter[row.en-promoter_window: row.en+promoter_window] = True
    
    for exon in exons.itertuples():
        in_exons[exon.st:exon.en] = True
        in_junctions[exon.st - junction_window: exon.st+junction_window] = True
        in_junctions[exon.en - junction_window: exon.en+junction_window] = True

    if len(cres) > 0:
        for cre in cres.itertuples():
            in_cre[cre.st:cre.en] = True

    # Select outer CREs by distance
    out_1k[row.st-1000:row.st] = True
    out_1k[row.en:row.en+1000] = True
    out_1k_10k[row.st-10000:row.st-1000] = True
    out_1k_10k[row.en+1000:row.en+10000] = True
    out_10k_100k[row.st-100000:row.st-10000] = True
    out_10k_100k[row.en+10000:row.en+100000] = True
    out_100k[:row.st-100000] = True
    out_100k[row.en+100000:] = True

    genes.loc[genes.gene==row.gene, 'Promoter'] = attr[in_promoter].mean()
    genes.loc[genes.gene==row.gene, 'Exons'] = attr[in_exons].mean()
    genes.loc[genes.gene==row.gene, 'Introns'] = attr[in_gene & (~in_exons)].mean()
    genes.loc[genes.gene==row.gene, 'Exon/Intron junctions'] = attr[in_junctions].mean()
    genes.loc[genes.gene==row.gene, 'Intronic CREs'] = attr[in_gene & (~in_exons) & in_cre].mean()
    genes.loc[genes.gene==row.gene, 'Outer CREs'] = attr[(~in_gene) & in_cre].mean()
    genes.loc[genes.gene==row.gene, 'Outer non-CREs'] = attr[(~in_gene) & (~in_cre)].mean()
    
    genes.loc[genes.gene==row.gene, '1k (CREs)'] = attr[out_1k & in_cre].mean()
    genes.loc[genes.gene==row.gene, '1k (Other)'] = attr[out_1k & (~in_cre)].mean()
    genes.loc[genes.gene==row.gene, '1k-10k (CREs)'] = attr[out_1k_10k & in_cre].mean()
    genes.loc[genes.gene==row.gene, '1k-10k (Other)'] = attr[out_1k_10k & (~in_cre)].mean()
    genes.loc[genes.gene==row.gene, '10k-100k (CREs)'] = attr[out_10k_100k & in_cre].mean()
    genes.loc[genes.gene==row.gene, '10k-100k (Other)'] = attr[out_10k_100k & (~in_cre)].mean()
    genes.loc[genes.gene==row.gene, '>=100k (CREs)'] = attr[out_100k & in_cre].mean()
    genes.loc[genes.gene==row.gene, '>=100k (Other)'] = attr[out_100k & (~in_cre)].mean()

# %%
df = genes[['Promoter', 'Exon/Intron junctions', 'Exons', 'Introns', 'Intronic CREs']].dropna().copy()
print(len(df))

df = df.melt()
df.variable = pd.Categorical(df.variable, categories=[
    'Promoter', 'Exon/Intron junctions', 'Exons', 'Introns', 'Intronic CREs'
])
(
    ggplot(df, aes(x='variable', y='value'))
    +geom_boxplot(outlier_size=.1) + theme_classic() + theme(figure_size=(4, 2.5))
    + scale_y_log10(limits = (5e-6, .5)) + ylab("    Mean Attribution\n(Promoter/gene body)") 
    + theme(axis_title_x=element_blank())
    +theme(axis_text_x=element_text(rotation=30, hjust=1))
)

# %%
df = genes[['1k (Other)', '1k (CREs)', '1k-10k (Other)', '1k-10k (CREs)',
    '10k-100k (Other)', '10k-100k (CREs)', '>=100k (Other)', '>=100k (CREs)']].dropna().copy()
df = df.rename(columns = {
    '1k (Other)':'<1kb (Other)', '1k (CREs)':'<1kb (CREs)',
    '1k-10k (Other)':'1kb-10kb (Other)', '1k-10k (CREs)': '1kb-10kb (CREs)',
    '10k-100k (Other)':'10kb-100kb (Other)', '10k-100k (CREs)': '10kb-100kb (CREs)',
    '>=100k (Other)':'>=100kb (Other)', '>=100k (CREs)':'>=100kb (CREs)'
})
print(len(df))
df = df.melt()
df['Distance'] = [x.split(' ')[0] for x in df.variable]
df.loc[df.variable.str.endswith('CREs)'), 'in CRE'] = True
df.loc[~df.variable.str.endswith('CREs)'), 'in CRE'] = False
df.Distance = pd.Categorical(df.Distance, categories=[
    '<1kb', '1kb-10kb', '10kb-100kb', '>=100kb'])

# %%
(
    ggplot(df, aes(x='Distance', fill='in CRE', y='value'))
    +geom_boxplot(outlier_size=.1) + theme_classic() + theme(figure_size=(4.7, 2.2))
    + scale_y_log10(limits = (5e-6, .5))
    + ylab("  Mean Attribution\n(outside gene body)") + xlab("Distance from gene")
    +theme(axis_text_x=element_text(rotation=30, hjust=.5))
)

# %%
