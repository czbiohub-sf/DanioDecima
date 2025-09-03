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
# # Plot Figure 7: cell-type and disease-specific design

# %% [markdown]
# ## Read evolution run

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import anndata
import torch
import logomaker
from grelu.visualize import plot_attributions
from grelu.interpret.motifs import scan_sequences
from grelu.sequence.format import *

import os, sys
sys.path.append('/home/gunsalul/tools/decima/src/decima/')

from lightning import LightningModel
import interpret

import matplotlib.pyplot as plt
# %matplotlib inline

# %%
output_file = 'fibroblast_sept29_200bp_part1.csv'
mutationF_1 = pd.read_csv(output_file)
output_file = 'fibroblast_sept30_200bp_part2.csv'
mutationF_2 = pd.read_csv(output_file)
mutationF_2['Round'] = mutationF_2['Round'] + 100

# %%
mutationF = pd.concat([mutationF_1, mutationF_2])

# %% [markdown]
# # Read model

# %%
EBFP_seq = 'ATGGCTAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACACTAGTGACCACCCTGTCCCACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTCGAGTACAACTTCAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGCCAACTTCAAGATCCGCCACAATATTGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGCATCACTCACGGCATGGACGAGCTGTACAAG'
hSyn = 'AGTGCAAGTGGGTTTTAGGACCAGGATGAGGCGGGGTGGGGGTGCCTACCTGACGACCGACCCCGACCCACTGGACAAGCACCCAACCCCCATTCCCCAAATTGCGCATCCCCTATCAGAGAGGGGGAGGGGAAACAGGATGCGGCGAGGCGCGTGCGCACTGCCAGCTTCAGCACCGCGGACAGTGCCTTCGCCCCCGCCTGGCGGCGCGCGCCACCGCCGCCTCAGCACTGAAGGCGCGCTGACGTCACTCGCCGGTCCCCCGCAAACTCCCCTTCCCGGCCACCTTGGTCGCGTCCGCGCCGCCGCCGGCCCAGCCGGACCGCACCACGCGAGGCGCGAGATAGGGGGGCACGGGCGCGACCATCTGCGCTGCGGCGCCGGCGACTCAGCGCTGCCTCAGTCTGCGGTGGGCAGCGGAGGAGTCGTGTCGTGCCTGAGAGCGCAG'

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823/"
matrix_file = os.path.join(save_dir, "aggregated.h5ad")
h5_file = os.path.join(save_dir, "data.h5")
ckpt_dir = os.path.join(save_dir, 'lightning_logs')
ckpts = [os.path.join(ckpt_dir, '0as9e8of/checkpoints/epoch=7-step=5840.ckpt'),  ]
model = LightningModel.load_from_checkpoint(ckpts[0]) 
device = 'cuda:1'
model = model.to(device)
model = model.eval()

# %%
window_size = 524288
TSS_offset = 5120*32
# chr22:29,480,218-29,491,390 NEFH
chrom = 'chr22'
TSS_start = 29480218
sequence_start_location = TSS_start - TSS_offset
sequence_end_location = sequence_start_location + window_size
seqF = pd.DataFrame([chrom, sequence_start_location,sequence_end_location]).T
seqF.columns = ['chrom', 'start', 'end']
full_sequence = intervals_to_strings(seqF,genome="hg38")[0]
TSS_drop_location = 67503862
chrom = 'chr1'
device = model.device

# %%
ad = anndata.read_h5ad(matrix_file)
ad = ad[:, ad.var.dataset=="test"]


# %%
def place_sequence(full_seq: str, placed_seq: str, loc: int) -> str:
    """
    Place a sequence at a specific location within another sequence.

    Args:
        full_seq (str): The full sequence to insert into.
        placed_seq (str): The sequence to be inserted.
        loc (int): The location to insert the sequence.

    Returns:
        str: The resulting sequence after insertion.
    """
    left_of_start = full_seq[0:loc]
    right_of_start = full_seq[loc:len(full_seq) - len(placed_seq)]
    new_seq = left_of_start + placed_seq + right_of_start
    return new_seq

def make_pred(full_inserted_sequence, inserted_sequence, window_size = 524288, TSS_offset=163840): # element + EBFP
    shape = (window_size)
    arr = np.zeros(shape=shape)
    for i, row in enumerate(ad.var.itertuples()):
        arr[TSS_offset:TSS_offset + len(inserted_sequence)] = 1
    full_seq_one_hot = strings_to_one_hot(full_inserted_sequence, add_batch_axis=False)
    arr_reshaped = torch.tensor(arr.reshape(1, -1))
    x = torch.cat((full_seq_one_hot, arr_reshaped), dim=0).float()
    x = x.to(model.device)
    with torch.no_grad():
        preds = model.forward(x).detach().cpu().numpy()
    preds = preds.squeeze()
    return preds 


# %%
taskF = pd.DataFrame(model.data_params['tasks'])


# %% [markdown]
# ## Define tasks

# %%
def get_cell_types(df, study, disease):
    return set(df[(df['study'] == study) & (df['disease'] == disease)]['cell_type'])

# Get the cell types for disease_tasks and healthy_tasks
disease_cell_types = get_cell_types(taskF, 'DS000010618', 'ulcerative colitis')
healthy_cell_types = get_cell_types(taskF, 'DS000010618', 'healthy')

# Combine the cell types
allowed_cell_types = disease_cell_types.union(healthy_cell_types)

def label_tasks(row):
    if row['study'] == 'DS000010618':
        if row['cell_type'] == 'fibroblast':
            if row['disease'] == 'ulcerative colitis':
                return 'fibroblast_disease'
            elif row['disease'] == 'healthy':
                return 'fibroblast_healthy'
        else:
            if row['disease'] == 'ulcerative colitis':
                return 'non_fibroblast_disease'
            elif row['disease'] == 'healthy':
                return 'non_fibroblast_healthy'
    return 'excluded'

def label_fibroblast(row):
    if row['study'] == 'DS000010618':
        if row['cell_type'] == 'fibroblast':
            return 'fibroblast'
        else:
            return 'non_fibroblast'
    return 'excluded'

# Apply the labeling functions to create new columns
taskF['label'] = taskF.apply(label_tasks, axis=1)
taskF['fibroblast_label'] = taskF.apply(label_fibroblast, axis=1)

# Filter the DataFrame based on these labels
fibroblast_disease = taskF[taskF['label'] == 'fibroblast_disease']
fibroblast_healthy = taskF[taskF['label'] == 'fibroblast_healthy']
non_fibroblast_disease = taskF[taskF['label'] == 'non_fibroblast_disease']
non_fibroblast_healthy = taskF[taskF['label'] == 'non_fibroblast_healthy']

# Filter based on the new fibroblast_label column
fibroblast = taskF[taskF['fibroblast_label'] == 'fibroblast']
non_fibroblast = taskF[taskF['fibroblast_label'] == 'non_fibroblast']

# %% [markdown]
# # Re-score mutations

# %%
tasks = []
for i,row in mutationF.iterrows():
    print(row['Round'])
    cur_taskF = taskF.copy()
    seq = row['Current_Sequence']
    full_inserted_sequence = place_sequence(full_sequence, seq + EBFP_seq , TSS_offset)
    preds = make_pred(full_inserted_sequence, seq + EBFP_seq)
    cur_taskF['preds'] = preds
    cur_taskF['round'] = row['Round']
    tasks.append(cur_taskF)
    
concat_tasks = [taskF] + tasks # ADD round 0

all_tasks = pd.concat(concat_tasks)

# %%
subF = all_tasks[all_tasks['label'] != 'excluded']

subF = subF[subF['label'].isin(['fibroblast_disease', 'fibroblast_healthy',
                               'non_fibroblast_disease', 'non_fibroblast_healthy'])]

# %%
# Assuming all_tasks and combined_diff dataframes are already created

# Create the combined plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), dpi=200)

label_mapping = {
    'fibroblast_disease': 'UC fibroblast',
    'fibroblast_healthy': 'Healthy fibroblast',
    'non_fibroblast_disease': 'UC non-fibroblast',
    'non_fibroblast_healthy': 'Healthy non-fibroblast',
    'fibroblast - non_fibroblast': 'Fibroblast vs Non-fibroblast',
    'fibroblast_disease - fibroblast_healthy': 'UC vs Healthy fibroblast'
}


# First subplot
subF = all_tasks[all_tasks['label'] != 'excluded']
subF = subF[subF['label'].isin(['fibroblast_disease', 'fibroblast_healthy',
                               'non_fibroblast_disease', 'non_fibroblast_healthy'])]
mean_data = subF.groupby(['round', 'label'])['preds'].mean().reset_index()
# Apply the mapping to the dataframes
mean_data['label'] = mean_data['label'].map(label_mapping)
sns.scatterplot(data=mean_data, x='round', y='preds', hue='label', s=50, alpha=.8, ax=ax1)
ax1.set_title('Predictions across directed evolution')
ax1.set_ylabel('Mean prediction')
ax1.set_xlabel('')  # Remove x-label from top subplot
ax1.set_xlim(0, 135)
ax1.axvline(x=100, linestyle='dashed', alpha=.8)
#ax1.tick_params(axis='x', rotation=45, ha='right')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


fibroblast = all_tasks[all_tasks['fibroblast_label'] == 'fibroblast']
non_fibroblast = all_tasks[all_tasks['fibroblast_label'] == 'non_fibroblast']

fibroblast_mean = fibroblast.groupby('round')['preds'].mean()
non_fibroblast_mean = non_fibroblast.groupby('round')['preds'].mean()

diff_fibro_non_fibro = (fibroblast_mean - non_fibroblast_mean).reset_index()
diff_fibro_non_fibro['group'] = 'fibroblast - non_fibroblast'

fibroblast_disease = all_tasks[all_tasks['label'] == 'fibroblast_disease']
fibroblast_healthy = all_tasks[all_tasks['label'] == 'fibroblast_healthy']

fibroblast_disease_mean = fibroblast_disease.groupby('round')['preds'].mean()
fibroblast_healthy_mean = fibroblast_healthy.groupby('round')['preds'].mean()

diff_disease_healthy = (fibroblast_disease_mean - fibroblast_healthy_mean).reset_index()
diff_disease_healthy['group'] = 'fibroblast_disease - fibroblast_healthy'

# Combine the differences
combined_diff = pd.concat([diff_fibro_non_fibro, diff_disease_healthy])
combined_diff['group'] = combined_diff['group'].map(label_mapping)

custom_palette = ["#b5b5b5", "#6e6e6e"]
sns.scatterplot(data=combined_diff, x='round', y='preds', hue='group', s=60, alpha=.8,
                palette=custom_palette, ax=ax2)
plt.axvline(x=100, linestyle='dashed', alpha=.8)
ax2.set_title('Specificity across directed evolution')
ax2.set_xlabel('Round')
ax2.set_ylabel('Mean prediction')
ax2.set_xlim(0, 135)
#ax2.tick_params(axis='x', rotation=45, ha='right')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Adjust layout
plt.tight_layout()
fig.subplots_adjust(right=0.8, hspace=0.3)

plt.show()

# %%
final_round = mean_data[mean_data['round'] == 150]


# %% [markdown]
# # ISM across groups

# %%
def ism(full_inserted_sequence, inserted_sequence, make_pred_func):
    """
    Perform In Silico Mutagenesis (ISM) on the given sequence.
    
    :param full_inserted_sequence: The full sequence including the inserted sequence
    :param inserted_sequence: The sequence to mutate (subset of full_inserted_sequence)
    :param make_pred_func: Function that takes (full_inserted_sequence, inserted_sequence) and returns predictions
    :return: 3D numpy array of shape (4, len(inserted_sequence), 8745)
    """
    bases = ['A', 'T', 'G', 'C']
    n_bases = len(bases)
    seq_length = len(inserted_sequence)
    n_celltypes = taskF.shape[0]
    
    # Initialize the ISM matrix
    ism_matrix = np.zeros((n_bases, seq_length, n_celltypes))
    
    # Get predictions for the original sequence
    original_preds = make_pred_func(full_inserted_sequence, inserted_sequence)
    
    # Find the start index of inserted_sequence within full_inserted_sequence
    start_index = full_inserted_sequence.index(inserted_sequence)
    
    # Iterate over each position in the inserted sequence
    for pos in tqdm(range(seq_length), desc="Positions"):
        original_base = inserted_sequence[pos]
        
        # Iterate over each possible base substitution
        for i, base in enumerate(bases):
            if base == original_base:
                continue  # Skip if it's the same as the original base
            
            # Create mutated full sequence
            mutated_full_seq = (full_inserted_sequence[:start_index + pos] + 
                                base + 
                                full_inserted_sequence[start_index + pos + 1:])
            
            # Create mutated inserted sequence
            mutated_inserted_seq = inserted_sequence[:pos] + base + inserted_sequence[pos+1:]
            
            # Get predictions for the mutated sequence
            mutated_preds = make_pred_func(mutated_full_seq, mutated_inserted_seq)
            
            # Calculate change in predictions for each cell type
            ism_matrix[i, pos, :] = mutated_preds - original_preds
    
    return ism_matrix



# %%
final_sequence = mutationF[mutationF['Round'] == 150].Current_Sequence.item()
final_sequence

# %%
placed_seq = place_sequence(full_sequence, final_sequence + EBFP_seq, TSS_offset)

# %%
fibroblast_ism_results = ism(placed_seq, final_sequence, make_pred)

# %%
#np.save('fibroblast_double_evolve_sept11.npy',fibroblast_ism_results)
np.save('fibroblast_double_evolve_sept30.npy',fibroblast_ism_results)


# %%
def plot_ism_heatmap(ism_results, sequence, title="ISM Heatmap for AD Brain Tasks"):
    """
    Plot a heatmap of the mean ISM results across AD brain tasks.
    
    :param ism_results: numpy array of shape (4, sequence_length) containing mean ISM results
    :param sequence: original DNA sequence
    :param title: title for the plot
    """
    bases = ['A', 'T', 'G', 'C']
    
    plt.figure(figsize=(20, 6))
    sns.heatmap(ism_results, cmap='RdBu_r', center=0, 
                xticklabels=list(sequence), yticklabels=bases)
    
    plt.title(title)
    plt.xlabel("Sequence Position")
    plt.ylabel("Mutated Base")
    
    # Add colorbar label
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('Mean Change in Prediction', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.show()


# %%
fibroblast_disease = taskF[taskF['label'] == 'fibroblast_disease']
fibroblast_healthy = taskF[taskF['label'] == 'fibroblast_healthy']
non_fibroblast_disease = taskF[taskF['label'] == 'non_fibroblast_disease']
non_fibroblast_healthy = taskF[taskF['label'] == 'non_fibroblast_healthy']

# %%
ism_results = fibroblast_ism_results

# %%
sequence = final_sequence

# %%
task_list = [ fibroblast_disease.index.values, 
             fibroblast_healthy.index.values, 
             non_fibroblast_disease.index.values, 
             non_fibroblast_healthy.index.values] 

mean_ism_results = [ism_results[:, :, tasks].mean(axis=2) for tasks in task_list]
mean_results_line = [mean_result.mean(axis=0) for mean_result in mean_ism_results]
labels= ['UC fibroblast', 'Healthy fibroblast', 'UC non-fibroblast', 'Healthy non-fibroblast', ]

plt.figure(figsize=(10, 3), dpi=200)
for i in range(len(task_list)):
    plt.plot(range(len(sequence)), mean_results_line[i], label = labels[i], alpha=.6)
    plt.xlabel("Sequence Position")
    plt.ylabel("Mean Change in Prediction")
    plt.xlim(0,len(sequence))
    plt.axhline(y=0, color='darkred', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.title(f'Mean ISM across')

plt.show()

# %% [markdown]
# # Scan for motifs in final evolved sequence

# %%
scan_halfway = scan_sequences(
        seqs=[halfway_sequence],
        motifs = "/gstore/data/resbioai/gunsalul/annotations/H12CORE_jaspar_format.meme", # HOCOMOCOv12
        names=None,
        seq_ids=['fibroblast_evolved'],
        rc=True, 
    )

# %%
scan_halfway['motif_center'] = scan.apply(lambda row: min(row['end'], row['start']) + abs(row['end'] - row['start']) // 2, axis=1)
scanF_halfway_disease = scan_halfway.copy()
scanF_halfway_healthy = scan_halfway.copy()

disease_ism_track = halfway_ism_results[:,:,fibroblast_disease.index.values].mean(axis=2)
healthy_ism_track = halfway_ism_results[:,:,non_fibroblast_healthy.index.values].mean(axis=2)

scanF_halfway_disease['ism_weight'] = scan.apply(calculate_ism_weight, ism=disease_ism_track, axis=1)
scanF_halfway_healthy['ism_weight'] = scan.apply(calculate_ism_weight, ism=healthy_ism_track, axis=1)

# %%
scan = scan_sequences(
        seqs=[sequence],
        motifs = "/gstore/data/resbioai/gunsalul/annotations/H12CORE_jaspar_format.meme", # HOCOMOCOv12
        names=None,
        seq_ids=['fibroblast_evolved'],
        rc=True, 
    )


# %%
def calculate_ism_weight(row: pd.Series, ism: np.ndarray) -> float:
    max_ism = np.abs(ism).max(axis=0)
    start, end = min(row["start"], row["end"]), max(row["start"], row["end"])
    return max_ism[start:end].mean()


# %%
scan['motif_center'] = scan.apply(lambda row: min(row['end'], row['start']) + abs(row['end'] - row['start']) // 2, axis=1)
scanF_disease = scan.copy()
scanF_healthy = scan.copy()

disease_ism_track = ism_results[:,:,fibroblast_disease.index.values].mean(axis=2)
healthy_ism_track = ism_results[:,:,non_fibroblast_healthy.index.values].mean(axis=2)

scanF_disease['ism_weight'] = scan.apply(calculate_ism_weight, ism=disease_ism_track, axis=1)
scanF_healthy['ism_weight'] = scan.apply(calculate_ism_weight, ism=healthy_ism_track, axis=1)

# %% [markdown]
# # Logo plots

# %%
ismF = pd.DataFrame(disease_ism_track)
ismF.columns = list(sequence)

# %%
plot_ISM(ismF, method="logo", figsize=(20, 1.5),ymax=1.2, ymin=-.2)  

# %%
plot_ISM(ismF_partial, method="logo", figsize=(20,1.5), ymax=1.2, ymin=-.2)  


# %%
def plot_ISM(
    ism_preds: pd.DataFrame,
    start_pos: Optional[int] = None,
    end_pos: Optional[int] = None,
    figsize: Tuple[float, float] = (8, 1.5),
    method: str = "heatmap",
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,/
    **kwargs,
):
    """
    Return in silico mutagenesis plot

    Args:
        ism_preds: ISM dataframe produced by `grelu.model.interpret.ISM_predict`
        start_pos: Start position of region to plot
        end_pos: End position of region to plot
        figsize: Tuple containing (width, height)
        method: 'heatmap' or 'logo'
        ymin: Minimum value for the y-axis (color scale for heatmap, height for logo)
        ymax: Maximum value for the y-axis (color scale for heatmap, height for logo)
        **kwargs: Additional arguments to be passed to sns.heatmap (in case type='heatmap')
        or plot_attributions (in case type = 'logo')

    Returns:
        Heatmap or sequence logo for the specified region.
    """
    # Positions to plot
    if start_pos is None:
        start_pos = 0
    if end_pos is None:
        end_pos = ism_preds.shape[1]

    # Subset dataframe
    ism_preds = ism_preds.iloc[:, start_pos:end_pos].copy()

    # Plot heatmap
    if method == "heatmap":
        fig, ax = plt.subplots(figsize=figsize)
        heatmap_kwargs = kwargs.copy()
        if ymin is not None or ymax is not None:
            heatmap_kwargs['vmin'] = ymin if ymin is not None else ism_preds.min().min()
            heatmap_kwargs['vmax'] = ymax if ymax is not None else ism_preds.max().max()
        g = sns.heatmap(
            ism_preds,
            xticklabels=1,
            yticklabels=1,
            cmap="vlag",
            center=0,
            **heatmap_kwargs,
        )
        g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=8)
        g.set_xticklabels(g.get_xticklabels(), rotation=0, fontsize=8)

    # Plot logo
    elif method == "logo":
        from grelu.sequence.format import BASE_TO_INDEX_HASH

        # Calculate mean mutation effect
        means = -ism_preds.mean(0)

        # Make attribution array - everything is set to 0
        attrs = np.zeros((4, end_pos - start_pos)).astype(np.float32)

        # Add score for the reference base
        for i in range(end_pos - start_pos):
            attrs[BASE_TO_INDEX_HASH[means.index[i]], i] = np.float32(means.iloc[i])

        # Make logo
        logo_kwargs = kwargs.copy()
        if ymin is not None or ymax is not None:
            logo_kwargs['ylim'] = (ymin if ymin is not None else attrs.min(), 
                                   ymax if ymax is not None else attrs.max())
        g = plot_attributions(attrs, figsize=figsize, **logo_kwargs)

    return g


# %%
plot_ISM(ismF, method="logo", figsize=(8,4), 
                         start_pos = 9, end_pos=19, ymin=0,ymax=1.2)   # JUN/FOS

# %%
plot_ISM(ismF, method="logo", figsize=(8,4), 
                         start_pos = 69, end_pos=78, ymax=1.2, ymin=0)   # SMAD

# %%
plot_ISM(ismF, method="logo", figsize=(8,4), 
                         start_pos = 109, end_pos=119, ymax=1.2, ymin=0)   # JUN/FOS

# %%
plot_ISM(ismF, method="logo", figsize=(8,4), 
                         start_pos = 127, end_pos=138, ymin=0, ymax=1.2)   # TATA binding domain

# %% [markdown]
# # Plot motifs before disease design

# %%
halfway_sequence = mutationF[mutationF['Round'] == 99].Current_Sequence.item()
placed_halfway_seq = place_sequence(full_sequence, halfway_sequence + EBFP_seq, TSS_offset)
halfway_ism_results = ism(placed_halfway_seq, halfway_sequence, make_pred)

# %%
halfway_disease_ism_track = halfway_ism_results[:,:,fibroblast_disease.index.values].mean(axis=2)
ismF_partial = pd.DataFrame(halfway_disease_ism_track)
ismF_partial.columns = list(halfway_sequence)

# %%
plot_ISM(ismF_partial,method="logo", figsize=(8,4), start_pos = 9, end_pos=19, ymin=0, ymax=1.2)   # JUN/FOS

# %%
plot_ISM(ismF_partial, method="logo", figsize=(8,4), start_pos = 69, end_pos=78,ymin=0, ymax=1.2)   # SMAD

# %%
plot_ISM(ismF_partial, method="logo", figsize=(8,4), start_pos = 109, end_pos=119, ymin=0, ymax=1.2)# JUN/FOS

# %%
plot_ISM(ismF_partial, method="logo", figsize=(8,4), start_pos = 127, end_pos=138, ymin=0, ymax=1.2)   # TATA binding domain

# %%
plot_ISM(ismF, method="logo", figsize=(10,4), start_pos = 4, end_pos=19, ymin=0, ymax=1.2)   

# %%
plot_ISM(ismF_partial, method="logo", figsize=(10,4), start_pos = 4, end_pos=19, ymin=0, ymax=1.2)   

# %% [markdown]
# # Plot HOMOCOCO motifs

# %%
### MOTIF INFO FROM hocomoco

fosb_dict = {'A': [427.0, 8.0, 6.0, 921.0, 187.0, 25.0, 45.0, 980.0, 47.0],
 'C': [229.0, 4.0, 20.0, 13.0, 0.0, 17.0, 944.0, 5.0, 373.0],
 'G': [276.0, 9.0, 875.0, 3.0, 816.0, 17.0, 13.0, 4.0, 188.0],
 'T': [71.0, 982.0, 102.0, 66.0, 0.0, 944.0, 1.0, 14.0, 395.0]}

fosB_pcm_RC = {
    'T': [47.0, 980.0, 45.0, 25.0, 187.0, 921.0, 6.0, 8.0, 427.0],
    'G': [373.0, 5.0, 944.0, 17.0, 0.0, 13.0, 20.0, 4.0, 229.0],
    'C': [188.0, 4.0, 13.0, 17.0, 816.0, 3.0, 875.0, 9.0, 276.0],
    'A': [395.0, 14.0, 1.0, 944.0, 0.0, 66.0, 102.0, 982.0, 71.0]}

tbd_dict = {
    "A": [132, 67, 841, 25, 922, 685, 959, 830, 279, 135],
    "C": [568, 34, 7, 14, 1, 1, 6, 5, 19, 299],
    "G": [219, 24, 10, 18, 2, 9, 28, 63, 652, 515],
    "T": [78, 872, 139, 940, 72, 302, 4, 99, 47, 48]
}

smad4_dict = {
    "A": [156, 4, 248, 5, 29, 3, 569, 990],
    "C": [6, 23, 29, 24, 948, 945, 19, 10],
    "G": [20, 56, 50, 290, 234, 532, 698, 124],
    "T": [2, 223, 969, 303, 20, 347, 11, 129]
}

smad4_pcm = {
    "A": [156.0, 6.0, 20.0, 2.0, 4.0, 23.0, 56.0, 223.0],
    "C": [248.0, 29.0, 50.0, 969.0, 5.0, 24.0, 290.0, 303.0],
    "G": [29.0, 948.0, 234.0, 20.0, 3.0, 945.0, 532.0, 347.0],
    "T": [569.0, 19.0, 698.0, 11.0, 990.0, 10.0, 124.0, 129.0]
}

smad4_pcm_RC = { 
    "A": [129.0, 124.0, 10.0, 990.0, 11.0, 698.0, 19.0, 569.0],
    "C": [347.0, 532.0, 945.0, 3.0, 20.0, 234.0, 948.0, 29.0],
    "G": [303.0, 290.0, 24.0, 5.0, 969.0, 50.0, 29.0, 248.0],
    "T": [223.0, 56.0, 23.0, 4.0, 2.0, 20.0, 6.0, 156.0]
}

klf_pcm = {
    "A": [321.0, 436.0, 14.0, 21.0, 5.0, 76.0, 10.0, 22.0, 115.0, 71.0, 70.0, 104.0, 134.0],
    "C": [309.0, 41.0, 2.0, 7.0, 7.0, 591.0, 9.0, 35.0, 12.0, 24.0, 659.0, 500.0, 326.0],
    "G": [232.0, 274.0, 972.0, 965.0, 980.0, 0.0, 945.0, 554.0, 765.0, 814.0, 90.0, 183.0, 374.0],
    "T": [134.0, 245.0, 8.0, 3.0, 4.0, 329.0, 32.0, 385.0, 104.0, 87.0, 177.0, 209.0, 162.0]
}


jaspar_data = smad4_pcm_RC

# %%
jaspar_data = fosB_pcm_RC
pcm_df = pd.DataFrame(jaspar_data) 

total_counts = pcm_df.sum(axis=1)
pfm_df = pcm_df.div(total_counts, axis=0)
information_content = 2 + np.sum(pfm_df * np.log2(pfm_df.clip(lower=1e-6)), axis=1)
icm_df = pfm_df.multiply(information_content, axis=0)
fig, ax = plt.subplots(figsize=(8,4))
logo = logomaker.Logo(icm_df, ax=ax)
logo.style_spines(visible=False)  # Remove axis spines
logo.style_xticks(anchor=0, spacing=1)  # Position x-ticks appropriately
plt.show()

# %%
jaspar_data = fosb_dict
pcm_df = pd.DataFrame(jaspar_data) # fosb_dict
total_counts = pcm_df.sum(axis=1)
pfm_df = pcm_df.div(total_counts, axis=0)
information_content = 2 + np.sum(pfm_df * np.log2(pfm_df.clip(lower=1e-6)), axis=1)
icm_df = pfm_df.multiply(information_content, axis=0)
fig, ax = plt.subplots(figsize=(8,4))
logo = logomaker.Logo(icm_df, ax=ax)
logo.style_spines(visible=False)  # Remove axis spines
logo.style_xticks(anchor=0, spacing=1)  # Position x-ticks appropriately
plt.show()

# %%
jaspar_data = tbd_dict
pcm_df = pd.DataFrame(jaspar_data) # fosb_dict
total_counts = pcm_df.sum(axis=1)
pfm_df = pcm_df.div(total_counts, axis=0)
information_content = 2 + np.sum(pfm_df * np.log2(pfm_df.clip(lower=1e-6)), axis=1)
icm_df = pfm_df.multiply(information_content, axis=0)
fig, ax = plt.subplots(figsize=(8,4))
logo = logomaker.Logo(icm_df, ax=ax)
logo.style_spines(visible=False)  # Remove axis spines
logo.style_xticks(anchor=0, spacing=1)  # Position x-ticks appropriately
plt.show()

# %%
jaspar_data = smad4_pcm_RC
pcm_df = pd.DataFrame(jaspar_data) # fosb_dict

# Calculate the total counts at each position
total_counts = pcm_df.sum(axis=1)
pfm_df = pcm_df.div(total_counts, axis=0)
information_content = 2 + np.sum(pfm_df * np.log2(pfm_df.clip(lower=1e-6)), axis=1)
icm_df = pfm_df.multiply(information_content, axis=0)
fig, ax = plt.subplots(figsize=(8,4))
logo = logomaker.Logo(icm_df, ax=ax)
logo.style_spines(visible=False)  # Remove axis spines
logo.style_xticks(anchor=0, spacing=1)  # Position x-ticks appropriately
plt.show()

# %%
jaspar_data = klf_pcm
pcm_df = pd.DataFrame(jaspar_data) # fosb_dict

# Calculate the total counts at each position
total_counts = pcm_df.sum(axis=1)
pfm_df = pcm_df.div(total_counts, axis=0)
information_content = 2 + np.sum(pfm_df * np.log2(pfm_df.clip(lower=1e-6)), axis=1)
icm_df = pfm_df.multiply(information_content, axis=0)
fig, ax = plt.subplots(figsize=(8,4))
logo = logomaker.Logo(icm_df, ax=ax)
logo.style_spines(visible=False)  # Remove axis spines
logo.style_xticks(anchor=0, spacing=1)  # Position x-ticks appropriately
plt.show()
