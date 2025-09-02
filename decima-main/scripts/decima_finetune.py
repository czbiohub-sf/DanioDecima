#!/usr/bin/env python3
import os
import sys
import argparse
import random
import torch
import numpy as np
import anndata
import tempfile
import json
from pytorch_lightning.loggers import TensorBoardLogger

def set_all_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_single_experiment(config):
    """Train a single experiment with given configuration"""
    
    # Set seeds first thing
    set_all_seeds(config["seed"])
    
    print(f"=== EXPERIMENT CONFIG ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print(f"========================")
    
    # Set up paths
    src_dir = f'{os.path.dirname(__file__)}/../src/decima/'
    sys.path.append(src_dir)
    
    # Import modules
    from read_hdf5 import HDF5Dataset
    from lightning import LightningModel
    
    # Use scratch space for temp directories instead of /tmp
    base_scratch = "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima"
    unique_id = f"{os.getpid()}_{config['seed']}"
    temp_dir = os.path.join(base_scratch, "temp_dirs", f"genomepy_tmp_{unique_id}")
    
    # Create directories
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["GENOMEPY_CONFIG"] = os.path.join(temp_dir, "config")
    os.environ["GENOMEPY_CACHE_DIR"] = os.path.join(temp_dir, "cache")
    os.makedirs(os.environ["GENOMEPY_CONFIG"], exist_ok=True)
    os.makedirs(os.environ["GENOMEPY_CACHE_DIR"], exist_ok=True)
    
    print(f"Temp directories:")
    print(f"   Config: {os.environ['GENOMEPY_CONFIG']}")
    print(f"   Cache: {os.environ['GENOMEPY_CACHE_DIR']}")
    
    # # Set deterministic port based on seed
    # port = 20000 + (config["seed"] % 1000)
    # os.environ["MASTER_PORT"] = str(port)
    
    # Load data
    matrix_file = os.path.join(config["dir"], "zebrahub_aggregated.h5ad")
    h5_file = os.path.join(config["dir"], "data.h5")
    
    ad = anndata.read_h5ad(matrix_file)
    
    # Use seed for dataset as well
    train_dataset = HDF5Dataset(
        h5_file=h5_file, ad=ad, key="train", 
        max_seq_shift=5000, augment_mode="random", seed=config["seed"]
    )
    val_dataset = HDF5Dataset(h5_file=h5_file, ad=ad, key="val", max_seq_shift=0)
    
    # Set up logging
    exp_name = f"{config['init_mode']}"
    if config['init_mode'] == 'pretrained':
        exp_name += f"_{config['pretrained_source']}_rep{config['replicate']}"
    exp_name += f"_lr{config['lr']:.0e}_seed{config['seed']}"
    
    log_dir = os.path.join(config["log_dir"], exp_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=log_dir, name="")
    
    # Training parameters (full config from the original)
    train_params = {
        "optimizer": "adam",
        "batch_size": config["bs"],
        "num_workers": 4,
        "devices": [0] if torch.cuda.is_available() else None,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "logger": logger,
        "precision": "16-mixed" if torch.cuda.is_available() else "32",
        "save_dir": log_dir,
        "max_epochs": config["max_epochs"],
        "lr": config["lr"],
        "total_weight": config["weight"],
        "accumulate_grad_batches": config["grad"],
        "loss": 'poisson_multinomial',
        "weight_decay": config["weight_decay"],
        "gradient_clip_val": config["gradient_clip_val"],
        "gradient_clip_algorithm": config["gradient_clip_algorithm"],
        "early_stopping_patience": config["early_stopping_patience"],
        "early_stopping_min_delta": config["early_stopping_min_delta"],
        "strategy": "auto",
        "seed": config["seed"],
    }
    
    # Model parameters (full config)
    model_params = {
        "n_tasks": ad.shape[0],
        "init_mode": config["init_mode"],
        "seed": config["seed"],  # Pass seed to model
    }
    
    # Add pretrained-specific parameters
    if config["init_mode"] == "pretrained":
        model_params["replicate"] = config["replicate"]
        model_params["pretrained_source"] = config["pretrained_source"]
        model_params["wandb_project"] = config["wandb_project"]
    
    # Create and train model
    model = LightningModel(model_params=model_params, train_params=train_params)
    trainer = model.train_on_dataset(train_dataset, val_dataset)
    
    # Save configuration and final metrics
    final_metrics = {}
    for key, value in trainer.callback_metrics.items():
        if isinstance(value, torch.Tensor):
            final_metrics[key] = value.item()
        else:
            final_metrics[key] = value
    
    # Save experiment info
    non_serializable_keys = {"logger", "devices"}  # Add other problematic keys as needed
    
    experiment_info = {
        "config": config,
        "final_metrics": final_metrics,
        "model_params": model_params,
        "train_params": {k: v for k, v in train_params.items() 
                        if k not in non_serializable_keys and not callable(v)}
    }
    
    with open(os.path.join(log_dir, "experiment_info.json"), "w") as f:
        json.dump(experiment_info, f, indent=2)
    
    # Cleanup
    train_dataset.close()
    val_dataset.close()
    
    print(f"Completed experiment: {exp_name}")
    print(f"Final validation loss: {final_metrics.get('val_loss', 'N/A')}")

    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp directory: {temp_dir}")
    except Exception as e:
        print(f"Could not clean up temp directory {temp_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train single Decima experiment")
    parser.add_argument("--experiment_id", type=int, required=True, help="Experiment ID (0-15)")
    parser.add_argument("--data_dir", type=str, 
                       default="/hpc/projects/data.science/yangjoon.kim/zebrafish-seq2func-data/celltypes_chrom_split_v1/",
                       help="Data directory")
    parser.add_argument("--log_dir", type=str, required=True, help="Log directory")
    
    args = parser.parse_args()
    
    # Define all 16 experiments
    experiments = []
    
    # Base config template (full config from your original)
    base_config = {
        "dir": args.data_dir,
        "log_dir": args.log_dir,
        "bs": 4,
        "weight": 1e-4,
        "weight_decay": 0,
        "grad": 5,
        "wandb_project": "grelu/borzoi",
        "gradient_clip_val": 1,
        "gradient_clip_algorithm": "norm",
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.0001,
        "max_epochs": 40,
    }
    
    # 1. Human-Borzoi pretrained (replicates 0-3, lr=3e-5, seed=42)
    for rep in range(4):
        config = base_config.copy()
        config.update({
            "init_mode": "pretrained",
            "pretrained_source": "wandb-human",
            "replicate": rep,
            "lr": 3e-5,
            "seed": 42,
        })
        experiments.append(config)
    
    # 2. Human-Decima pretrained (replicates 0-3, lr=3e-5, seed=42)
    for rep in range(4):
        config = base_config.copy()
        config.update({
            "init_mode": "pretrained",
            "pretrained_source": "decima-human",
            "replicate": rep,
            "lr": 3e-5,
            "seed": 42,
        })
        experiments.append(config)
    
    # 3. Mouse-Borzoi pretrained (replicates 0-3, lr=3e-5, seed=42)
    for rep in range(4):
        config = base_config.copy()
        config.update({
            "init_mode": "pretrained",
            "pretrained_source": "wandb-mouse",
            "replicate": rep,
            "lr": 3e-5,
            "seed": 42,
        })
        experiments.append(config)
    
    # 4. Random initialization (lr=3e-6, seeds=42,43,44,45)
    for i in range(4):
        config = base_config.copy()
        config.update({
            "init_mode": "random",
            "lr": 3e-6,
            "seed": 42 + i,
        })
        experiments.append(config)
    
    # Validate experiment ID
    if args.experiment_id < 0 or args.experiment_id >= len(experiments):
        raise ValueError(f"experiment_id must be between 0 and {len(experiments)-1}")
    
    # Get the specific experiment config
    config = experiments[args.experiment_id]
    
    # Train
    train_single_experiment(config)

if __name__ == "__main__":
    main()