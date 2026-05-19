# Modified from the original Genentech/decima source by the DanioDecima authors
# (Chan Zuckerberg Biohub) on 2026-05-10. See daniodecima-main/FORK_NOTES.md for the scope of
# modifications. Original copyright Genentech, Inc., 2024 (Genentech Non-Commercial
# Software License v1.0).

import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from grelu.sequence.format import indices_to_one_hot, BASE_TO_INDEX_HASH
from grelu.data.augment import Augmenter, _split_overall_idx


def count_genes(h5_file, key=None):
    with h5py.File(h5_file, "r") as f:
        genes = np.array(f["genes"]).astype(str)
    if key is None:
        return genes.shape[0]
    else:
        return np.sum(genes[:, 1] == key)


def index_genes(h5_file, key=None):
    with h5py.File(h5_file, "r") as f:
        genes = np.array(f["genes"]).astype(str)
    if key is None:
        return np.array(range(len(genes)))
    else:
        return np.where(genes[:, 1] == key)[0]


def list_genes(h5_file, key=None):
    with h5py.File(h5_file, "r") as f:
        genes = np.array(f["genes"]).astype(str)
    if key is None:
        return genes[:, 0]
    else:
        return genes[genes[:, 1] == key, 0]


def get_gene_idx(h5_file, gene, key=None):
    gene_ord = list_genes(h5_file, key=None)
    return np.where(gene_ord == gene)[0][0]


def _extract_center(x, seq_len, shift=0):
    start = (x.shape[-1] - seq_len)//2
    start -= shift
    return x[..., start:start+seq_len]


def extract_gene_data(h5_file, gene, seq_len=524288, merge=True):
    gene_idx = get_gene_idx(h5_file, key=None, gene=gene)
    
    with h5py.File(h5_file, "r") as f:
        seq = np.array(f["sequences"][gene_idx])
        seq = indices_to_one_hot(seq)
        mask = torch.Tensor(np.array(f["masks"][[gene_idx]]))

    seq = _extract_center(seq, seq_len=seq_len)
    mask = _extract_center(mask, seq_len=seq_len)
    
    if merge:
        return torch.vstack([seq, mask])
    else:
        return seq, mask


def mutate(seq, allele, pos):
    idx = BASE_TO_INDEX_HASH[allele]
    seq[:4, pos] = 0
    seq[idx, pos] = 1
    return seq


class HDF5Dataset(Dataset):
    def __init__(self, key, h5_file, ad=None, seq_len=524288, max_seq_shift=0, seed=0, augment_mode = "random"):
        super().__init__()

        # Save data params
        self.h5_file = h5_file
        self.seq_len = seq_len
        self.key = key

        # Save augmentation params
        self.max_seq_shift = max_seq_shift
        self.augmenter = Augmenter(
                rc=False,
                max_seq_shift=self.max_seq_shift,
                max_pair_shift=0,
                seq_len=self.seq_len,
                label_len=None,
                seed=seed,
                mode=augment_mode,
            )
        self.n_augmented = len(self.augmenter)
        self.padded_seq_len = self.seq_len + (2 * self.max_seq_shift)
        
        # Index genes
        self.gene_index = index_genes(self.h5_file, key = self.key)
        self.n_seqs = len(self.gene_index)
        
        # Setup
        self.dataset = h5py.File(self.h5_file, "r")
        self.extract_tasks(ad)
        self.predict = False
        self.n_alleles = 1

    def __len__(self):
        return self.n_seqs * self.n_augmented 

    def close(self):
        self.dataset.close()

    def extract_tasks(self, ad=None):
        tasks = np.array(self.dataset["tasks"]).astype(str)
        if ad is not None:
            assert np.all(tasks==ad.obs_names)
            self.tasks = ad.obs
        else:
            self.tasks = pd.DataFrame(index=tasks)

    def extract_seq(self, idx):
        seq = self.dataset['sequences'][idx]
        seq = indices_to_one_hot(seq) # 4, L
        mask = self.dataset['masks'][[idx]] # 1, L
        seq = np.concatenate([seq, mask]) # 5, L
        seq = _extract_center(seq, seq_len=self.padded_seq_len)
        return torch.Tensor(seq)

    def extract_label(self, idx):
        return torch.Tensor(self.dataset['labels'][idx])

    def __getitem__(self, idx):

        # Augment
        seq_idx, augment_idx = _split_overall_idx(idx, (self.n_seqs, self.n_augmented))

        # Extract the sequence
        gene_idx = self.gene_index[seq_idx]
        seq = self.extract_seq(gene_idx)

        # Augment the sequence
        seq = self.augmenter(seq=seq, idx=augment_idx)
       
        if self.predict:
            return seq

        else:
            label = self.extract_label(gene_idx)
            return seq, label



class VariantDataset(Dataset):
    def __init__(self, variants, h5_file, seq_len=524288, max_seq_shift=0, test_ref=False):
        super().__init__()

        # Save data params
        self.h5_file = h5_file
        self.seq_len = seq_len

        # Save variant params
        self.variants = variants[['gene', 'rel_pos', 'ref_tx', 'alt_tx']].copy()
        self.n_seqs = len(self.variants)
        self.n_alleles = 2
        self.test_ref = test_ref

        # Map each variant to the corresponding gene in the h5 file
        gene_map = {gene : get_gene_idx(self.h5_file, gene) for gene in self.variants.gene.unique()}
        self.variants['gene_idx'] = self.variants.gene.map(gene_map)

        # Save augmentation params
        self.max_seq_shift = max_seq_shift
        self.augmenter = Augmenter(
                rc=False,
                max_seq_shift=self.max_seq_shift,
                max_pair_shift=0,
                seq_len=self.seq_len,
                label_len=None,
                mode='serial',
            )
        self.n_augmented = len(self.augmenter)
        self.padded_seq_len = self.seq_len + (2 * self.max_seq_shift)
        
        # Setup
        self.dataset = h5py.File(self.h5_file, "r")
        self.pad = self.dataset["pad"]

    def __len__(self):
        return self.n_seqs * self.n_augmented * self.n_alleles

    def close(self):
        self.dataset.close()

    def extract_seq(self, idx):
        seq = self.dataset['sequences'][idx]
        seq = indices_to_one_hot(seq) # 4, L
        mask = self.dataset['masks'][[idx]] # 1, L
        seq = np.concatenate([seq, mask]) # 5, L
        return torch.Tensor(seq)

    def __getitem__(self, idx):

        # Get indices
        seq_idx, augment_idx, allele_idx = _split_overall_idx(
            idx, (self.n_seqs, self.n_augmented, self.n_alleles)
        )

        # Extract the sequence
        variant = self.variants.iloc[seq_idx]
        seq = self.extract_seq(variant.gene_idx)

        if self.test_ref: # check that ref is actually present 
            assert ["A","C","G","T"][seq[:4,variant.rel_pos+self.pad].argmax()] == variant.ref_tx, variant.ref_tx + "_vs_" + seq[:4,variant.rel_pos+self.pad] + "__" + str(seq_idx)

        # Insert the allele
        if allele_idx:
            seq = mutate(seq, variant.alt_tx, variant.rel_pos + self.pad)
        else:
            seq = mutate(seq, variant.ref_tx, variant.rel_pos + self.pad)

        # Augment the sequence
        seq = _extract_center(seq, seq_len=self.padded_seq_len)
        seq = self.augmenter(seq=seq, idx=augment_idx)
        return seq

class GeneForecastDataset(Dataset):
    def __init__(self, key, h5_file, ad=None, history_length=7, forecast_horizon=3, 
                 gene_indices=None, seq_len=524288, max_seq_shift=0, seed=0, augment_mode="random"):
        """
        Args:
            key: Key for filtering genes
            h5_file: Path to HDF5 file containing sequences and masks.
            ad: An AnnData object containing gene expression time series.
                ad.X is assumed to be of shape [n_timepoints, n_genes] (dense or convertible to dense)
            history_length: Number of timepoints used as input for forecasting.
            forecast_horizon: Number of future timepoints to predict.
            gene_indices: Optional list/array of gene indices to use (for train/test splits).
            seq_len: Sequence length to crop the genomic sequence.
            max_seq_shift: Maximum sequence shift for augmentation.
            seed: Random seed.
            augment_mode: Mode for sequence augmentation.
        """
        super().__init__()
        self.h5_file = h5_file
        self.seq_len = seq_len
        self.key = key
        self.history_length = history_length
        self.forecast_horizon = forecast_horizon
        self.max_seq_shift = max_seq_shift
        self.ad = ad  # AnnData object
        
        # Set up augmentation
        self.augmenter = Augmenter(
            rc=False,
            max_seq_shift=self.max_seq_shift,
            max_pair_shift=0,
            seq_len=self.seq_len,
            label_len=None,
            seed=seed,
            mode=augment_mode,
        )
        self.n_augmented = len(self.augmenter)
        self.padded_seq_len = self.seq_len + (2 * self.max_seq_shift)
        
        # Index genes from HDF5 file; if gene_indices are provided (for train/test split), use them.
        if gene_indices is not None:
            self.gene_index = gene_indices
        else:
            self.gene_index = index_genes(self.h5_file, key=self.key)
        self.n_seqs = len(self.gene_index)
        
        # Open the HDF5 file (keep open during dataset lifetime)
        self.dataset = h5py.File(self.h5_file, "r")
        self.extract_tasks(ad)
        self.predict = False
        self.n_alleles = 1
        #self.cell_types = sorted(self.ad.obs["cell_type"].unique())
        self.cell_types = ['central_nervous_system', 'endoderm', 'lateral_mesoderm']
        self.n_cell_types = len(self.cell_types)

    def __len__(self):
        return self.n_seqs * self.n_augmented

    def close(self):
        self.dataset.close()

    def extract_tasks(self, ad=None):
        tasks = np.array(self.dataset["tasks"]).astype(str)
        if ad is not None:
            assert np.all(tasks==ad.obs_names)
            self.tasks = ad.obs
        else:
            self.tasks = pd.DataFrame(index=tasks)

    def extract_seq(self, idx):
        """
        Extract the genomic sequence (and mask) for the gene at index idx from the HDF5 file.
        Returns a tensor of shape [5, padded_seq_len].
        """
        seq = self.dataset['sequences'][idx]           # raw sequence indices
        seq = indices_to_one_hot(seq)                   # one-hot: shape [4, L]
        mask = self.dataset['masks'][[idx]]             # shape [1, L]
        seq = np.concatenate([seq, mask])               # shape [5, L]
        seq = _extract_center(seq, seq_len=self.padded_seq_len)
        return torch.Tensor(seq)
    
    def extract_label(self, gene_idx):
        """
        Extract the time series for the given gene (by index) across all tissues.
        ad.X is assumed to be of shape [n_obs, n_genes] and ad.obs contains "tissue" and "timepoint".
        For each tissue, we sort observations by timepoint and then extract the expression for this gene.
        Returns:
            expr_dict: A dict mapping tissue name to a 1D numpy array of expression values.
        """
        # Get gene expression for the gene (column gene_idx)
        expr_mat = self.ad.X
        # ad.obs should be a DataFrame
        obs_df = self.ad.obs.copy()
        expr_dict = {}
        for cell_type in self.cell_types:
            cell_type_mask = obs_df["cell_type"] == cell_type
            cell_type_obs = obs_df[cell_type_mask]
            # Sort by timepoint
            cell_type_obs = cell_type_obs.sort_values("timepoint")
            # Get the indices corresponding to these observations
            indices = cell_type_obs.index.astype(int).to_numpy()
            # Extract expression values for this gene at these observations
            expr_values = expr_mat[indices, gene_idx]
            expr_dict[cell_type] = expr_values  # numpy array of shape [n_timepoints_for_tissue]
        return expr_dict

    def __getitem__(self, idx):
        """
        For each sample, returns a dictionary with:
          - "static_sequence": the one-hot encoded sequence (plus mask) tensor of shape [5, padded_seq_len]
          - "expr_series": the historical gene expression time series (input) of shape [history_length, 1]
          - "target": the target future expression values of shape [forecast_horizon]
        """
        # Split global index into gene index and augmentation index
        seq_idx, augment_idx = _split_overall_idx(idx, (self.n_seqs, self.n_augmented))
        gene_idx = self.gene_index[seq_idx]
        
        # Extract and augment the static sequence
        seq = self.extract_seq(gene_idx)
        seq = self.augmenter(seq=seq, idx=augment_idx)
        
        # Extract full gene expression time series from AnnData:
        expr_dict = self.extract_label(gene_idx)  # shape: [n_timepoints]
        
        input_series = []
        target_series = []
        for cell_type in self.cell_types:
            expr = expr_dict[cell_type]
            if expr.shape[0] < self.history_length + self.forecast_horizon:
                raise IndexError(
                    f"Cell type {cell_type} has too few timepoints for gene index {gene_idx}."
                )
            # For simplicity, take the first history_length as input and the next forecast_horizon as target.
            input_series.append(expr[:self.history_length])
            target_series.append(expr[self.history_length:self.history_length + self.forecast_horizon])
        
        # Stack across cell types: resulting shapes [history_length, n_cell_types] and [forecast_horizon, n_cell_types]
        expr_series = np.stack(input_series, axis=-1)  # each column is a cell type
        target = np.stack(target_series, axis=-1)
        
        # Convert to tensors
        expr_series = torch.tensor(expr_series, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        
        return {
            "static_sequence": seq,       # Tensor: [5, padded_seq_len]
            "expr_series": expr_series,   # Tensor: [history_length, n_tissues]
            "target": target              # Tensor: [forecast_horizon, n_tissues]
        }

