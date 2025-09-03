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

EBFP_seq = 'ATGGCTAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACACTAGTGACCACCCTGTCCCACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTCGAGTACAACTTCAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGCCAACTTCAAGATCCGCCACAATATTGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGCATCACTCACGGCATGGACGAGCTGTACAAG'
hSyn = 'AGTGCAAGTGGGTTTTAGGACCAGGATGAGGCGGGGTGGGGGTGCCTACCTGACGACCGACCCCGACCCACTGGACAAGCACCCAACCCCCATTCCCCAAATTGCGCATCCCCTATCAGAGAGGGGGAGGGGAAACAGGATGCGGCGAGGCGCGTGCGCACTGCCAGCTTCAGCACCGCGGACAGTGCCTTCGCCCCCGCCTGGCGGCGCGCGCCACCGCCGCCTCAGCACTGAAGGCGCGCTGACGTCACTCGCCGGTCCCCCGCAAACTCCCCTTCCCGGCCACCTTGGTCGCGTCCGCGCCGCCGCCGGCCCAGCCGGACCGCACCACGCGAGGCGCGAGATAGGGGGGCACGGGCGCGACCATCTGCGCTGCGGCGCCGGCGACTCAGCGCTGCCTCAGTCTGCGGTGGGCAGCGGAGGAGTCGTGTCGTGCCTGAGAGCGCAG'


# %%
import sys
sys.path.append('/home/gunsalul/tools/decima/src/decima/')

# %%
import numpy as np
import pandas as pd
import anndata
from tqdm import tqdm
import torch
import os
from grelu.sequence.format import *
from grelu.sequence.mutate import mutate
import grelu.sequence.utils
import pandas as pd
import seaborn as sns
import csv
import matplotlib.pyplot as plt

from lightning import LightningModel
import interpret
# %matplotlib inline

# %% [markdown]
# # Load Decima and data

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240808"
matrix_file = os.path.join(save_dir, "final_filtered.h5ad")
h5_file = os.path.join(save_dir, "data.h5")
ad = anndata.read_h5ad(matrix_file)
ad = ad[:, ad.var.dataset=="test"]
device = 'cuda:1'

# %%
save_dir="/gstore/data/resbioai/grelu/decima/20240823/"
matrix_file = os.path.join(save_dir, "aggregated.h5ad")
h5_file = os.path.join(save_dir, "data.h5")
ckpt_dir = os.path.join(save_dir, 'lightning_logs')

# %%
ckpt_dir = os.path.join(save_dir, 'lightning_logs')
ckpts = [os.path.join(ckpt_dir, '0as9e8of/checkpoints/epoch=7-step=5840.ckpt'),  ]
model = LightningModel.load_from_checkpoint(ckpts[0]) 
model = model.to(device)
model = model.eval()

# %%
taskF = pd.DataFrame(model.data_params['tasks'])

# %% [markdown]
# # Define start location

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
device


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
# # Generate starting sequence

# %%
sequence_length = 200
random_starting_sequence = grelu.sequence.utils.generate_random_sequences(sequence_length, seed=42,
                                                                  output_format = 'strings')[0]

# %%
full_inserted_sequence = place_sequence(full_sequence, random_starting_sequence + EBFP_seq, TSS_offset)
preds = make_pred(full_inserted_sequence, random_starting_sequence + EBFP_seq)
taskF['starting_preds'] = preds

# %%
# Assuming taskF is your DataFrame
subF = taskF[taskF['label'] != 'excluded']

# Calculate mean starting_preds for each cell_type and label combination
mean_preds = subF.groupby(['cell_type', 'label'])['starting_preds'].mean().reset_index()

# Create the plot
plt.figure(figsize=(14, 6), dpi=200)
sns.barplot(data=mean_preds, x='cell_type', y='starting_preds', hue='label')

# Customize the plot
plt.title('Mean Starting Predictions by Cell Type and Label')
plt.xlabel('Cell Type')
plt.ylabel('Mean Starting Predictions')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(right=0.85)

# Show the plot
plt.show()

# %% [markdown]
# # Evolve

# %%
rounds = 100


# %%
def directed_evolution(full_inserted_sequence, inserted_seq, TSS_offset, rounds, diseaseF, healthyF, output_csv, cargo=EBFP_seq):
    current_sequence = full_inserted_sequence
    
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Round', 'Position', 'Base', 'Specificity', 'Current_Sequence'])
        
        for round_num in tqdm(range(rounds), desc="Rounds"):
            best_mutation = {
                'position': -1,
                'base': '',
                'specificity': float('-inf')
            }
            
            for position in range(TSS_offset, TSS_offset + len(inserted_seq)):
                for base in ['A', 'T', 'G', 'C']:
                    if base == current_sequence[position]:
                        continue
                    
                    new_seq = mutate(current_sequence, allele=base, pos=position)
                    current_element = new_seq[TSS_offset:TSS_offset+len(inserted_seq)]
                    
                    preds = make_pred(new_seq, current_element+cargo)
                    healthy_preds = preds[healthyF.index].mean()
                    disease_preds = preds[diseaseF.index].mean()
                    specificity = disease_preds - healthy_preds
                    
                    if specificity > best_mutation['specificity']:
                        best_mutation = {
                            'position': position,
                            'base': base,
                            'specificity': specificity
                        }
            
            current_sequence = mutate(current_sequence, allele=best_mutation['base'], pos=best_mutation['position'])
            current_element = current_sequence[TSS_offset:TSS_offset+len(inserted_seq)]
            
            csvwriter.writerow([
                round_num + 1,
                best_mutation['position'],
                best_mutation['base'],
                best_mutation['specificity'],
                current_element
            ])
            csvfile.flush()  # Ensure data is written to the file immediately
            
            print(f"Round {round_num+1}: Best specificity = {best_mutation['specificity']:.4f}, "
                  f"Position = {best_mutation['position']}, Base = {best_mutation['base']}")
    
    return current_element, best_mutation['specificity']



# %%
inserted_seq = random_starting_sequence
filename = 'fibroblast_sept29_200bp_part1.csv'
diseaseF = fibroblast
healthyF = non_fibroblast
result, specificity = directed_evolution(full_inserted_sequence, inserted_seq, TSS_offset, rounds, diseaseF, healthyF, filename)

# %%
mutationF_1 = pd.read_csv(filename)

# %%
evolved_seq = mutationF_1.iloc[99].Current_Sequence

# %% [markdown]
# # Evolve w/ disease - P2

# %%
full_inserted_sequence_evolved = place_sequence(full_sequence, evolved_seq + EBFP_seq, TSS_offset)

# %%
filename = 'fibroblast_sept30_200bp_part2.csv'
diseaseF = fibroblast_disease
healthyF = fibroblast_healthy
result, specificity = directed_evolution(full_inserted_sequence_evolved, evolved_seq, 
                                         TSS_offset, rounds, diseaseF, healthyF, filename)
