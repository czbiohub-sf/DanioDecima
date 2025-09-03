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
import tqdm

# %% [markdown]
# ## Paths

# %%
ckpts={
'kugrjb50': '/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/kugrjb50/checkpoints/epoch=3-step=2920.ckpt',
'i68hdsdk': '/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/i68hdsdk/checkpoints/epoch=2-step=2190.ckpt',
'0as9e8of': '/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/0as9e8of/checkpoints/epoch=7-step=5840.ckpt',
'i9zsp4nm': '/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/i9zsp4nm/checkpoints/epoch=8-step=6570.ckpt',
}

matrix_file = "/gstore/data/resbioai/grelu/decima/20240823/data.h5ad"
h5_file = "/gstore/data/resbioai/grelu/decima/20240823/data.h5"
meme_file_modisco = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/data/jaspar/H12CORE_meme_format.meme"

# where to save results
save_dir = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/"
ensembl_out_dir = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/results/ensemble"

# %% [markdown]
# ## Load data

# %%
ad = anndata.read_h5ad(matrix_file)
ad_full = ad.copy()
ad = ad[:,ad.var.dataset == "test"]

# %% [markdown]
# ## Compute attributions for each cell type and model replicate

# %%
all_cts = set(['respiratory basal cell','type II pneumocyte','type I pneumocyte','lung secretory cell','club cell','ciliated cell','goblet cell'])
tissues = set(ad.obs.query('cell_type in ["type II pneumocyte", "type I pneumocyte"]')['tissue']
marker_df = make_marker_df(ad_full[ad_full.obs.tissue.isin(set(ad.obs.query('cell_type in ["type II pneumocyte", "type I pneumocyte"]')['tissue']))])
top_n_cut=250

for ct in all_cts:
    ct_name = ct.replace(" ","_")
    background_cts = list(all_cts - set([ct]))
    devices = [3,4,6,7]
    i = 0
    for run_id, ckpt_file in ckpts.items():
        device = devices[i]        
        out_dir = os.path.join(save_dir, "results", run_id, f"{ct_name}__vs__lung")
        gene_df_file = os.path.join(out_dir,'gene_df.csv')
        targets_file = os.path.join(out_dir,'targets.csv')
        
        # make a data frame with top n genes according to cell-type z-score
        df = (marker_df.query('cell_type == @ct').sort_values('spring_score_pred', ascending=False).iloc[:top_n_cut])
    
        # make task df
        dfc = ad.obs[['cell_type','tissue','disease','study']].query('cell_type == @ct and tissue in @tissues').reset_index()
        task_df_on = pd.DataFrame([{"task_type":"on",'task':x} for x in list(set(dfc['index'].tolist()))])
        dfc =  ad.obs[['cell_type','tissue','disease','study']].query('cell_type in @background_cts').reset_index()
        dfc = dfc.loc[dfc.tissue.isin(tissues)]
        task_df_off = pd.DataFrame([{"task_type":"off",'task':x} for x in list(set(dfc['index'].tolist()))])
        task_df = pd.concat([task_df_on,task_df_off])
        
        # write gene and task df
        df.to_csv(gene_df_file, index=None)
        task_df.to_csv(targets_file, index=None)
        
        # make and execute command
        cmd = f"python Interpret.py -device {device} -ckpt_file {ckpt_file} -h5_file {h5_file} -gene_df_file {gene_df_file} \
            -targets_file {targets_file} -out_dir {out_dir}"
        cmd = " ".join(cmd.split())
        print(cmd)

        i += 1

# %% [markdown]
# ## Average attributions and run modisco for each cell type

# %%
for ct in tqdm.tqdm(all_cts):
    ct_name = ct.replace(" ","_")
    attrs = []
    results_path_ensemble = os.path.join(ensembl_out_dir, f"{ct_name}__vs__lung")
    
    for run_id, ckpt_file in ckpts.items():
        
        # collect attributons for all replicates
        results_path_model = os.path.join(save_dir, "results", run_id)
        attr = np.load(os.path.join(results_path_model, f"{ct_name}__vs__lung", "attributions.npy"))
        attrs.append(attr)
        
    # average the attributions
    attrs = np.stack(attrs).mean(0)
    attr_file = os.path.join(results_path_ensemble, "attributions.npy")
    np.save(attr_file, attrs)
    
    # save one sequence tensor csv
    seq = np.load(os.path.join(results_path_model, f"{ct_name}__vs__lung", "sequences.npy"))
    seq_file = os.path.join(results_path_ensemble, "sequences.npy")
    np.save(seq_file, seq)

    # Modisco
    modisco_dir = os.path.join(ensembl_out_dir, f"{ct_name}__vs__lung")
    cmd = f"python InterpretModisco.py -seq_file {seq_file} -attr_file {attr_file}  -meme_file {meme_file} -out_dir {modisco_dir}"
    print(cmd)
