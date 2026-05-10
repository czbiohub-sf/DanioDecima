# Modified from the original Genentech/decima source by the DanioDecima authors
# (Chan Zuckerberg Biohub) on 2026-05-10. See daniodecima-main/FORK_NOTES.md for the scope of
# modifications. Original copyright Genentech, Inc., 2024 (Genentech Non-Commercial
# Software License v1.0).

import anndata
import os, sys
import argparse
import wandb
from pytorch_lightning.loggers import WandbLogger

src_dir = f'{os.path.dirname(__file__)}/../src/decima/'
sys.path.insert(0, src_dir)  # before installed lightning pkg
from read_hdf5 import HDF5Dataset
from lightning import LightningModel

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="decima_test")
parser.add_argument("--dir", type=str, default="/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/data/")
parser.add_argument("--lr", type=float)
parser.add_argument("--weight", type=float)
parser.add_argument("--grad", type=int)
parser.add_argument("--replicate", type=int, default=0)
parser.add_argument("--bs", type=int, default=4)
parser.add_argument("--init_mode", type=str, default="random", 
                   choices=["pretrained", "random", "xavier", "kaiming", "zeros"],
                   help="Weight initialization mode")
parser.add_argument("--pretrained_source", type=str, default="wandb-human",
                   choices=["wandb-human", "wandb-mouse", "local"],
                   help="Source of pretrained weights")
parser.add_argument("--wandb_project", type=str, default="grelu/borzoi",
                   help="WandB project path for pretrained weights")
parser.add_argument("--checkpoint_path", type=str, default=None,
                   help="Path to local checkpoint file")

args = parser.parse_args()

os.environ["WANDB_START_METHOD"] = "thread"

def main():
    wandb.login()
    run = wandb.init(
        project="decima-zebrafish", 
        entity="czbsf-comp-bio",
        dir=args.dir,      # Logs saved under a specific directory
        name=args.name      # Name of the run
    )

    # Get paths
    data_dir = args.dir
    matrix_file = os.path.join(data_dir, "zebrahub_aggregated.h5ad")
    h5_file = os.path.join(data_dir, "data.h5")
    print(f"Data paths: {matrix_file}, {h5_file}")

    # Load data
    print("Reading anndata")
    ad = anndata.read_h5ad(matrix_file)

    # Make datasets
    print("Making dataset objects")
    train_dataset = HDF5Dataset(h5_file=h5_file, ad=ad, key="train", max_seq_shift=5000, augment_mode="random", seed=0)
    example = train_dataset[2]
    print(f"Example: {example}")
    print(f"Example shape: {example[0].shape}")
    print(f"Example shape: {example[1].shape}")
    
    val_dataset = HDF5Dataset(h5_file=h5_file, ad=ad, key="val", max_seq_shift=0)

    # Make paramdicts
    train_params = {
        "optimizer": "adam",
        "batch_size": args.bs,
        "num_workers": 16,
        "devices": 0,
        "logger": "wandb",
        "save_dir": data_dir,
        "max_epochs": 100,
        "lr":args.lr,
        "total_weight": args.weight,
        "accumulate_grad_batches": args.grad,
        "loss": 'poisson_multinomial'
        #"log_every_n_steps": 50,
        #"pairs": ad.uns["disease_pairs"].values
    }
    
    model_params = {
    "n_tasks": ad.shape[0],
    "replicate": args.replicate,
    "init_mode": args.init_mode,
    "pretrained_source": args.pretrained_source,
    "wandb_project": args.wandb_project,
    "checkpoint_path": args.checkpoint_path
    }

    print(f"train_params: {train_params}")
    print(f"model_params: {model_params}")

    # Make model
    print("Initializing model")
    model = LightningModel(model_params=model_params, train_params=train_params)
    
    # Fine-tune model
    print("Training")
    model.train_on_dataset(train_dataset, val_dataset)
    
    train_dataset.close()
    val_dataset.close()
    run.finish()

if __name__ == "__main__":
    main()