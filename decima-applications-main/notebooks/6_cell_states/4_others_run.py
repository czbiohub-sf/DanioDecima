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

# %%
ckpts={
'kugrjb50': '/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/kugrjb50/checkpoints/epoch=3-step=2920.ckpt',
'i68hdsdk': '/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/i68hdsdk/checkpoints/epoch=2-step=2190.ckpt',
'0as9e8of': '/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/0as9e8of/checkpoints/epoch=7-step=5840.ckpt',
'i9zsp4nm': '/gstore/data/resbioai/grelu/decima/20240823/lightning_logs/i9zsp4nm/checkpoints/epoch=8-step=6570.ckpt',
}

matrix_file = "/gstore/data/resbioai/grelu/decima/20240823/data.h5ad"
h5_file = "/gstore/data/resbioai/grelu/decima/20240823/data.h5"
meme_file = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/data/jaspar/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
meme_file_modisco = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/data/jaspar/H12CORE_meme_format.meme"

# where to save results
save_dir = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/"
ensembl_out_dir = "/gstore/data/resbioai/karollua/Decima/scborzoi/decima/results/ensemble"

# %%
ad = anndata.read_h5ad(matrix_file)
ad_full = ad.copy()
ad = ad[:,ad.var.dataset == "test"]

# %%
devices = [3,4,6,7]

# %% [markdown]
# # Tregs

# %%
key = "Treg_cycling__vs__tregnoncycling"

# %%
on_tasks = ad[(ad.obs.cell_type=='Treg cycling') & (ad.obs.tissue=='skin')].obs.index.tolist()
off_tasks = ad[(ad.obs.cell_type=='Treg') & (ad.obs.tissue=='skin')].obs.index.tolist()

# %%
diff_df = pd.DataFrame({'gene':list(ad.var.index),
                        'pred_diff':ad[on_tasks].layers['preds'].mean(0) - ad[off_tasks].layers['preds'].mean(0),
                        })
diff_df = diff_df[diff_df.pred_diff > 0.75]

# %%
# make task df
task_df_on = pd.DataFrame([{"task_type":"on", 'task':x} for x in on_tasks])
task_df_off = pd.DataFrame([{"task_type":"off",'task':x} for x in off_tasks])
task_df = pd.concat([task_df_on,task_df_off])

# %%
i = 0
for run_id, ckpt_file in ckpts.items():
    device = devices[i]        
    out_dir = os.path.join(save_dir, "results", run_id, key)
    gene_df_file = os.path.join(out_dir,'gene_df.csv')
    targets_file = os.path.join(out_dir,'targets.csv')

    # write gene and task df
    diff_df.to_csv(gene_df_file, index=None)
    task_df.to_csv(targets_file, index=None)
    
    # make command
    cmd = f"python Interpret.py -device {device} -ckpt_file {ckpt_file} -h5_file {h5_file} -gene_df_file {gene_df_file} \
        -targets_file {targets_file} -out_dir {out_dir}"
    cmd = " ".join(cmd.split())
    print(cmd)

    i += 1

# %%
results_path_ensemble = os.path.join(ensembl_out_dir, key)
for run_id, ckpt_file in ckpts.items():
    
    # collect attributons for all replicates
    results_path_model = os.path.join(save_dir, "results", run_id)
    attr = np.load(os.path.join(results_path_model, key, "attributions.npy"))
    attrs.append(attr)
    
# average the attributions
attrs = np.stack(attrs).mean(0)
attr_file = os.path.join(results_path_ensemble, "attributions.npy")
np.save(attr_file, attrs)

# save one sequence tensor and gene df csv
seq = np.load(os.path.join(results_path_model, key, "sequences.npy"))
seq_file = os.path.join(results_path_ensemble, "sequences.npy")
np.save(seq_file, seq)

gene_df_file = os.path.join(results_path_ensemble, "gene_df.csv")
diff_df.to_csv(gene_df_file, index=None)

# Modisco
modisco_dir = os.path.join(ensembl_out_dir, key)
cmd = f"python InterpretModisco.py -seq_file {seq_file} -attr_file {attr_file} -meme_file {meme_file} -out_dir {modisco_dir}"
cmd = " ".join(cmd.split())
print(cmd)

# %% [markdown]
# ## Fibroblasts

# %%
key = "fibroblast__vs__noncardiac"

# %%
on_tasks = ad[(ad.obs.cell_type == "fibroblast") & (ad.obs.organ == 'heart')].obs.index.tolist()
off_tasks = ad[((ad.obs.cell_type == "fibroblast") | (ad.obs.celltype_coarse == "Fibroblasts")) & (ad.obs.organ != 'heart')].obs.index.tolist()

# %%
diff_df = pd.DataFrame({'gene':list(ad.var.index),
                        'pred_diff':ad[on_tasks].layers['preds'].mean(0) - ad[off_tasks].layers['preds'].mean(0),
                        })
diff_df = diff_df[diff_df.pred_diff > 1.25]

# %%
# make task df
task_df_on = pd.DataFrame([{"task_type":"on", 'task':x} for x in on_tasks])
task_df_off = pd.DataFrame([{"task_type":"off",'task':x} for x in off_tasks])
task_df = pd.concat([task_df_on,task_df_off])

# %%
i = 0
for run_id, ckpt_file in ckpts.items():
    device = devices[i]        
    out_dir = os.path.join(save_dir, "results", run_id, key)
    gene_df_file = os.path.join(out_dir,'gene_df.csv')
    targets_file = os.path.join(out_dir,'targets.csv')

    # write gene and task df
    diff_df.to_csv(gene_df_file, index=None)
    task_df.to_csv(targets_file, index=None)
    
    # make command
    cmd = f"python Interpret.py -device {device} -ckpt_file {ckpt_file} -h5_file {h5_file} -gene_df_file {gene_df_file} \
        -targets_file {targets_file} -out_dir {out_dir}"
    cmd = " ".join(cmd.split())
    print(cmd)

    i += 1

# %%
results_path_ensemble = os.path.join(ensembl_out_dir, key)
for run_id, ckpt_file in ckpts.items():
    
    # collect attributons for all replicates
    results_path_model = os.path.join(save_dir, "results", run_id)
    attr = np.load(os.path.join(results_path_model, key, "attributions.npy"))
    attrs.append(attr)
    
# average the attributions
attrs = np.stack(attrs).mean(0)
attr_file = os.path.join(results_path_ensemble, "attributions.npy")
np.save(attr_file, attrs)

# save one sequence tensor and gene df csv
seq = np.load(os.path.join(results_path_model, key, "sequences.npy"))
seq_file = os.path.join(results_path_ensemble, "sequences.npy")
np.save(seq_file, seq)

gene_df_file = os.path.join(results_path_ensemble, "gene_df.csv")
diff_df.to_csv(gene_df_file, index=None)

# Modisco
modisco_dir = os.path.join(ensembl_out_dir, key)
cmd = f"python InterpretModisco.py -seq_file {seq_file} -attr_file {attr_file} -meme_file {meme_file} -out_dir {modisco_dir}"
cmd = " ".join(cmd.split())
print(cmd)
