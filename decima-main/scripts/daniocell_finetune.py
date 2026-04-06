#!/usr/bin/env python3
"""
Fine-tuning script for Decima on the DanioCell atlas.

Experiment matrix (12 experiments):
  [0-3]  Human Decima pretrained backbone → DanioCell  (replicates 0-3, lr=3e-5)
  [4-7]  Random initialization baseline                (seeds 42-45, lr=3e-6)
  [8-11] 3-stage transfer: Zebrahub fine-tuned → DanioCell (replicates 0-3, lr=3e-5)

Usage:
  python daniocell_finetune.py --experiment_id 0 --data_dir <path> --log_dir <path>
"""
import os
import sys
import argparse
import random
import torch
import numpy as np
import anndata
import json
from pytorch_lightning.loggers import CSVLogger


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

    set_all_seeds(config["seed"])

    print(f"=== EXPERIMENT CONFIG ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print(f"========================")

    src_dir = f'{os.path.dirname(__file__)}/../src/decima/'
    sys.path.insert(0, src_dir)  # insert before installed 'lightning' package

    from read_hdf5 import HDF5Dataset
    from lightning import LightningModel

    base_scratch = config.get("scratch_dir", "/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell")
    unique_id = f"{os.getpid()}_{config['seed']}"
    temp_dir = os.path.join(base_scratch, "temp_dirs", f"genomepy_tmp_{unique_id}")

    os.makedirs(temp_dir, exist_ok=True)
    os.environ["GENOMEPY_CONFIG"] = os.path.join(temp_dir, "config")
    os.environ["GENOMEPY_CACHE_DIR"] = os.path.join(temp_dir, "cache")
    os.makedirs(os.environ["GENOMEPY_CONFIG"], exist_ok=True)
    os.makedirs(os.environ["GENOMEPY_CACHE_DIR"], exist_ok=True)

    print(f"Temp directories:")
    print(f"   Config: {os.environ['GENOMEPY_CONFIG']}")
    print(f"   Cache:  {os.environ['GENOMEPY_CACHE_DIR']}")

    matrix_file = os.path.join(config["dir"], config.get("matrix_name", "daniocell_aggregated.h5ad"))
    h5_file = os.path.join(config["dir"], "data.h5")

    ad = anndata.read_h5ad(matrix_file)

    train_dataset = HDF5Dataset(
        h5_file=h5_file, ad=ad, key="train",
        max_seq_shift=5000, augment_mode="random", seed=config["seed"]
    )
    val_dataset = HDF5Dataset(h5_file=h5_file, ad=ad, key="val", max_seq_shift=0)

    exp_name = f"{config['init_mode']}"
    if config['init_mode'] == 'pretrained':
        exp_name += f"_{config['pretrained_source']}_rep{config['replicate']}"
    exp_name += f"_lr{config['lr']:.0e}_seed{config['seed']}"

    log_dir = os.path.join(config["log_dir"], exp_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = CSVLogger(save_dir=log_dir, name="")

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

    model_params = {
        "n_tasks": ad.shape[0],
        "init_mode": config["init_mode"],
        "seed": config["seed"],
    }

    if config["init_mode"] == "pretrained":
        model_params["replicate"] = config["replicate"]
        model_params["pretrained_source"] = config["pretrained_source"]
        model_params["wandb_project"] = config["wandb_project"]
        if config["pretrained_source"] == "local":
            model_params["checkpoint_path"] = config["checkpoint_path"]

    model = LightningModel(model_params=model_params, train_params=train_params)
    trainer = model.train_on_dataset(train_dataset, val_dataset)

    final_metrics = {}
    for key, value in trainer.callback_metrics.items():
        if isinstance(value, torch.Tensor):
            final_metrics[key] = value.item()
        else:
            final_metrics[key] = value

    non_serializable_keys = {"logger", "devices"}
    experiment_info = {
        "config": config,
        "final_metrics": final_metrics,
        "model_params": model_params,
        "train_params": {k: v for k, v in train_params.items()
                        if k not in non_serializable_keys and not callable(v)}
    }

    with open(os.path.join(log_dir, "experiment_info.json"), "w") as f:
        json.dump(experiment_info, f, indent=2)

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
    parser = argparse.ArgumentParser(description="Fine-tune Decima on DanioCell")
    parser.add_argument("--experiment_id", type=int, required=True,
                       help="Experiment ID (0-11 for core; 0-7 for initial runs)")
    parser.add_argument("--data_dir", type=str,
                       default="/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/celltypes_chrom_split_v1/",
                       help="Directory containing daniocell_aggregated.h5ad and data.h5")
    parser.add_argument("--log_dir", type=str, required=True,
                       help="Directory for experiment logs and checkpoints")
    parser.add_argument("--matrix_name", type=str,
                       default="daniocell_aggregated.h5ad",
                       help="Filename of the aggregated h5ad matrix within data_dir")
    parser.add_argument("--scratch_dir", type=str,
                       default="/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell",
                       help="Scratch directory for genomepy temp files")

    args = parser.parse_args()

    base_config = {
        "dir":           args.data_dir,
        "log_dir":       args.log_dir,
        "matrix_name":   args.matrix_name,
        "scratch_dir":   args.scratch_dir,
        "bs":            4,
        "weight":        1e-4,
        "weight_decay":  0,
        "grad":          5,
        "wandb_project": "grelu/borzoi",
        "gradient_clip_val":          1,
        "gradient_clip_algorithm":    "norm",
        "early_stopping_patience":    10,
        "early_stopping_min_delta":   0.0001,
        "max_epochs":    40,
    }

    experiments = []

    # [0-3] Human Decima pretrained backbone → DanioCell
    # Loads from Mathias's checkpoint dir; head is stripped, backbone reused.
    for rep in range(4):
        config = base_config.copy()
        config.update({
            "init_mode":        "pretrained",
            "pretrained_source": "decima-human",
            "replicate":         rep,
            "lr":                3e-5,
            "seed":              42,
        })
        experiments.append(config)

    # [4-7] Random initialization baseline
    for i in range(4):
        config = base_config.copy()
        config.update({
            "init_mode": "random",
            "lr":        3e-6,
            "seed":      42 + i,
        })
        experiments.append(config)

    # [8-11] Optional: 3-stage transfer (Zebrahub fine-tuned → DanioCell)
    # Populate checkpoint_paths once Zebrahub runs are complete.
    # Placeholder paths below — update before running array=8-11.
    zebrahub_ckpt_dir = "/hpc/scratch/group.data.science/yang-joon.kim/zebrahub-decima/experiments"
    for rep in range(4):
        config = base_config.copy()
        config.update({
            "init_mode":         "pretrained",
            "pretrained_source": "local",
            "replicate":         rep,
            "lr":                3e-5,
            "seed":              42,
            "checkpoint_path":   os.path.join(
                zebrahub_ckpt_dir,
                f"pretrained_decima-human_rep{rep}_lr3e-05_seed42",
                "best_model.ckpt"
            ),
        })
        experiments.append(config)

    if args.experiment_id < 0 or args.experiment_id >= len(experiments):
        raise ValueError(f"experiment_id must be between 0 and {len(experiments)-1}")

    config = experiments[args.experiment_id]
    train_single_experiment(config)


if __name__ == "__main__":
    main()
