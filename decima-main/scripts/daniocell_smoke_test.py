#!/usr/bin/env python3
"""
Smoke test harness for DanioCell Decima fine-tuning.

Runs 6 numbered checks before committing GPU hours to full training.
Each check prints PASS / FAIL and exits non-zero on the first failure.

Checks:
  1. HDF5 file sanity     — shapes, expected datasets, NaN in labels
  2. DataLoader batches   — seq shape [bs,5,524288], label shape [bs,1047,1]
  3. Model init (pretrained) — load decima-human weights, n_tasks=1047
  4. Forward pass + loss  — output shape correct, values finite, loss finite
  5. Model init (random)  — random baseline path (no WandB needed)
  6. 1-epoch mini-training — 5 train + 5 val batches, full Lightning loop,
                             optimizer step, checkpoint written

Usage:
  python daniocell_smoke_test.py [--skip_pretrained] [--n_batches N]

  --skip_pretrained  Skip checks 3-4 (useful if WandB is unavailable)
  --n_batches N      Batches per epoch in check 6 (default: 5)
"""
import os
import sys
import argparse
import traceback

import torch
import numpy as np
import h5py
import anndata
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = "/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/celltypes_chrom_split_v1"
H5_FILE     = os.path.join(DATA_DIR, "data.h5")
MATRIX_FILE = os.path.join(DATA_DIR, "daniocell_aggregated.h5ad")
SCRATCH_DIR = "/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell"
LOG_DIR     = os.path.join(SCRATCH_DIR, "smoke_test_logs")

EXPECTED_GENES    = 28201
EXPECTED_TASKS    = 1047
EXPECTED_SEQ_LEN  = 524288
EXPECTED_PAD_LEN  = 534288  # seq_len + 2*pad = 524288 + 10000

# ── helpers ──────────────────────────────────────────────────────────────────
_src = os.path.join(os.path.dirname(__file__), "..", "src", "decima")
sys.path.insert(0, os.path.abspath(_src))  # insert before installed 'lightning' package


def _header(n, title):
    print(f"\n{'='*60}")
    print(f"CHECK {n}: {title}")
    print('='*60)


def _pass(msg=""):
    print(f"  [PASS]{' ' + msg if msg else ''}")


def _fail(msg):
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def _check_finite(arr, name):
    if isinstance(arr, torch.Tensor):
        has_nan = torch.isnan(arr).any().item()
        has_inf = torch.isinf(arr).any().item()
    else:
        has_nan = bool(np.isnan(arr).any())
        has_inf = bool(np.isinf(arr).any())
    if has_nan:
        _fail(f"{name} contains NaN")
    if has_inf:
        _fail(f"{name} contains Inf")


# ── Check 1: HDF5 file sanity ─────────────────────────────────────────────────
def check_hdf5():
    _header(1, "HDF5 file sanity")

    if not os.path.exists(H5_FILE):
        _fail(f"data.h5 not found at {H5_FILE}")

    with h5py.File(H5_FILE, "r") as f:
        keys = list(f.keys())
        for required in ["sequences", "masks", "labels", "tasks", "genes"]:
            if required not in keys:
                _fail(f"Missing dataset '{required}' in data.h5 (found: {keys})")
        _pass(f"All required datasets present: {keys}")

        # shapes
        seq_shape   = f["sequences"].shape   # (genes, padded_seq_len)
        mask_shape  = f["masks"].shape       # (genes, padded_seq_len)
        label_shape = f["labels"].shape      # (genes, tasks, 1)
        tasks_shape = f["tasks"].shape       # (tasks,)
        genes_shape = f["genes"].shape       # (genes, 2)

        if seq_shape[0] != EXPECTED_GENES:
            _fail(f"sequences: expected {EXPECTED_GENES} genes, got {seq_shape[0]}")
        if seq_shape[1] != EXPECTED_PAD_LEN:
            _fail(f"sequences: expected padded_len={EXPECTED_PAD_LEN}, got {seq_shape[1]}")
        if label_shape != (EXPECTED_GENES, EXPECTED_TASKS, 1):
            _fail(f"labels shape {label_shape} != ({EXPECTED_GENES},{EXPECTED_TASKS},1)")
        if tasks_shape[0] != EXPECTED_TASKS:
            _fail(f"tasks: expected {EXPECTED_TASKS}, got {tasks_shape[0]}")
        _pass(f"sequences: {seq_shape}")
        _pass(f"labels:    {label_shape}")
        _pass(f"tasks:     {tasks_shape}")

        # NaN / Inf in labels (sample 100 genes)
        sample_labels = f["labels"][:100]
        if np.isnan(sample_labels).any():
            _fail("labels[0:100] contain NaN")
        if np.isinf(sample_labels).any():
            _fail("labels[0:100] contain Inf")
        n_zero = (sample_labels == 0).sum()
        pct_zero = 100 * n_zero / sample_labels.size
        _pass(f"labels finite (sample 100 genes; {pct_zero:.1f}% zeros — expected for pseudobulk)")

        # sequence dtype
        if f["sequences"].dtype != np.int8:
            _fail(f"sequences dtype should be int8, got {f['sequences'].dtype}")
        _pass(f"sequences dtype: int8")

        # pad scalar
        if "pad" in f:
            pad_val = int(f["pad"][()])
            if pad_val != 5000:
                _fail(f"pad={pad_val}, expected 5000")
            _pass(f"pad: {pad_val}")


# ── Check 2: DataLoader batches ───────────────────────────────────────────────
def check_dataloader():
    _header(2, "DataLoader batches (train + val, 1 batch each)")

    from read_hdf5 import HDF5Dataset

    ad = anndata.read_h5ad(MATRIX_FILE)
    _pass(f"AnnData loaded: {ad.shape} (tracks × genes)")

    # tasks in h5 must match ad.obs_names
    with h5py.File(H5_FILE, "r") as f:
        h5_tasks = np.array(f["tasks"]).astype(str)
    if not np.all(h5_tasks == ad.obs_names.values):
        _fail("tasks in data.h5 don't match ad.obs_names — was h5 built from the same h5ad?")
    _pass("tasks in data.h5 match ad.obs_names")

    for split in ["train", "val"]:
        ds = HDF5Dataset(h5_file=H5_FILE, ad=ad, key=split, max_seq_shift=0)
        n = len(ds)
        _pass(f"{split} dataset: {n} items ({ds.n_seqs} genes × {ds.n_augmented} augmentations)")
        if n == 0:
            _fail(f"{split} dataset is empty")

        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        seq, label = next(iter(loader))

        # seq: [bs, 5, padded_seq_len] — but after augmenter crop: [bs, 5, seq_len]
        if seq.shape[0] != 2:
            _fail(f"{split} seq batch_size={seq.shape[0]}, expected 2")
        if seq.shape[1] != 5:
            _fail(f"{split} seq channels={seq.shape[1]}, expected 5 (4 DNA + 1 mask)")
        if seq.shape[2] != EXPECTED_SEQ_LEN:
            _fail(f"{split} seq length={seq.shape[2]}, expected {EXPECTED_SEQ_LEN}")
        if label.shape != (2, EXPECTED_TASKS, 1):
            _fail(f"{split} label shape={tuple(label.shape)}, expected (2,{EXPECTED_TASKS},1)")

        _check_finite(seq, f"{split} seq")
        _check_finite(label, f"{split} label")
        _pass(f"{split} seq:   {tuple(seq.shape)}  dtype={seq.dtype}")
        _pass(f"{split} label: {tuple(label.shape)} dtype={label.dtype}")
        ds.close()


# ── Check 3: Model init — pretrained (decima-human) ───────────────────────────
def check_model_pretrained():
    _header(3, "Model instantiation — pretrained (decima-human, replicate=0)")

    from lightning import LightningModel

    try:
        model = LightningModel(
            model_params={
                "n_tasks":          EXPECTED_TASKS,
                "init_mode":        "pretrained",
                "pretrained_source": "decima-human",
                "replicate":        0,
                "wandb_project":    "grelu/borzoi",
                "seed":             42,
            },
            train_params={
                "lr": 3e-5,
                "batch_size": 2,
                "num_workers": 0,
                "max_epochs": 1,
                "total_weight": 1e-4,
                "weight_decay": 0,
                "accumulate_grad_batches": 1,
                "loss": "poisson_multinomial",
                "logger": "csv",
                "save_dir": LOG_DIR,
            },
        )
    except Exception as e:
        _fail(f"LightningModel init failed:\n{traceback.format_exc()}")

    head_tasks = model.model.head.n_tasks
    if head_tasks != EXPECTED_TASKS:
        _fail(f"head.n_tasks={head_tasks}, expected {EXPECTED_TASKS}")
    _pass(f"head.n_tasks: {head_tasks}")

    n_params = model.count_params()
    _pass(f"trainable parameters: {n_params:,}")

    return model


# ── Check 4: Forward pass + loss ──────────────────────────────────────────────
def check_forward_pass(model):
    _header(4, "Forward pass + loss (1 batch, pretrained model)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    _pass(f"device: {device}")

    # Synthetic batch — same shapes as real data
    seq   = torch.zeros(2, 5, EXPECTED_SEQ_LEN, device=device)
    label = torch.ones(2, EXPECTED_TASKS, 1, device=device)

    with torch.no_grad():
        logits = model.forward(seq, logits=True)

    if logits.shape != (2, EXPECTED_TASKS, 1):
        _fail(f"logits shape={tuple(logits.shape)}, expected (2,{EXPECTED_TASKS},1)")
    _check_finite(logits, "logits")
    _pass(f"logits shape: {tuple(logits.shape)}")

    preds = model.forward(seq, logits=False)
    if (preds < 0).any():
        _fail("predictions contain negative values (exp activation failed?)")
    _pass(f"predictions non-negative (range [{preds.min():.4f}, {preds.max():.4f}])")

    # loss
    loss_val, poisson_term, multinomial_term = model.decima_loss(logits, label)
    if not torch.isfinite(loss_val):
        _fail(f"loss is not finite: {loss_val.item()}")
    _pass(f"loss: {loss_val.item():.4f}  (poisson={poisson_term.item():.4f}, "
          f"multinomial={multinomial_term.item():.4f})")

    # gradient check
    model.train()
    logits_grad = model.forward(seq, logits=True)
    loss_grad, _, _ = model.decima_loss(logits_grad, label)
    loss_grad.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    if len(grad_norms) == 0:
        _fail("No gradients computed — backward pass may have broken")
    max_grad = max(grad_norms)
    if not np.isfinite(max_grad):
        _fail(f"Gradient norm is not finite: {max_grad}")
    _pass(f"gradients flow  (max norm: {max_grad:.4f} across {len(grad_norms)} param groups)")


# ── Check 5: Model init — random ──────────────────────────────────────────────
def check_model_random():
    _header(5, "Model instantiation — random initialization (no WandB)")

    from lightning import LightningModel

    try:
        model = LightningModel(
            model_params={
                "n_tasks":   EXPECTED_TASKS,
                "init_mode": "random",
                "seed":      42,
            },
            train_params={
                "lr": 3e-6,
                "batch_size": 2,
                "num_workers": 0,
                "max_epochs": 1,
                "total_weight": 1e-4,
                "weight_decay": 0,
                "accumulate_grad_batches": 1,
                "loss": "poisson_multinomial",
                "logger": "csv",
                "save_dir": LOG_DIR,
            },
        )
    except Exception as e:
        _fail(f"Random-init LightningModel failed:\n{traceback.format_exc()}")

    head_tasks = model.model.head.n_tasks
    if head_tasks != EXPECTED_TASKS:
        _fail(f"head.n_tasks={head_tasks}, expected {EXPECTED_TASKS}")
    _pass(f"random-init model ok — head.n_tasks={head_tasks}, "
          f"params={model.count_params():,}")

    return model


# ── Check 6: 1-epoch mini-training ────────────────────────────────────────────
def check_mini_training(n_batches, skip_pretrained):
    _header(6, f"1-epoch mini-training ({n_batches} train + {n_batches} val batches)")

    from read_hdf5 import HDF5Dataset
    from lightning import LightningModel

    os.makedirs(LOG_DIR, exist_ok=True)

    # Use random init to avoid WandB dependency during training check
    # (pretrained weights already verified in checks 3-4)
    init_mode = "random" if skip_pretrained else "pretrained"
    pretrained_kwargs = {} if skip_pretrained else {
        "pretrained_source": "decima-human",
        "replicate": 0,
        "wandb_project": "grelu/borzoi",
    }
    _pass(f"init_mode: {init_mode}")

    ad = anndata.read_h5ad(MATRIX_FILE)

    train_ds = HDF5Dataset(h5_file=H5_FILE, ad=ad, key="train",
                           max_seq_shift=5000, augment_mode="random", seed=42)
    val_ds   = HDF5Dataset(h5_file=H5_FILE, ad=ad, key="val",   max_seq_shift=0)

    train_dl = DataLoader(train_ds, batch_size=2, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=2)

    model = LightningModel(
        model_params={
            "n_tasks":   EXPECTED_TASKS,
            "init_mode": init_mode,
            "seed":      42,
            **pretrained_kwargs,
        },
        train_params={
            "lr":                        3e-6 if init_mode == "random" else 3e-5,
            "batch_size":                2,
            "num_workers":               2,
            "max_epochs":                1,
            "total_weight":              1e-4,
            "weight_decay":              0,
            "accumulate_grad_batches":   1,
            "gradient_clip_val":         1,
            "gradient_clip_algorithm":   "norm",
            "loss":                      "poisson_multinomial",
            "logger":                    "csv",
            "save_dir":                  LOG_DIR,
        },
    )

    ckpt_dir = os.path.join(LOG_DIR, "mini_training_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=n_batches,
        limit_val_batches=n_batches,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0] if torch.cuda.is_available() else None,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        logger=CSVLogger(save_dir=ckpt_dir, name=""),
        callbacks=[ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss",
                                   mode="min", save_last=True)],
        accumulate_grad_batches=1,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        enable_model_summary=False,
        deterministic=True,
    )

    # Required for deterministic=True on CUDA >= 10.2 with CuBLAS ops
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    pl.seed_everything(42, workers=True)

    try:
        trainer.fit(model, train_dl, val_dl)
    except Exception as e:
        _fail(f"trainer.fit raised an exception:\n{traceback.format_exc()}")

    # Verify checkpoint was written
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpt_files:
        _fail(f"No checkpoint found in {ckpt_dir} after training")
    _pass(f"Checkpoint saved: {ckpt_files}")

    # Check val_loss was logged
    metrics = trainer.callback_metrics
    if "val_loss" not in metrics:
        _fail(f"val_loss not in trainer.callback_metrics: {list(metrics.keys())}")
    val_loss = metrics["val_loss"].item()
    if not np.isfinite(val_loss):
        _fail(f"val_loss is not finite: {val_loss}")
    _pass(f"val_loss: {val_loss:.4f}")

    train_ds.close()
    val_ds.close()

    _pass(f"Mini-training complete — {n_batches} train + {n_batches} val batches, "
          f"1 epoch, no errors")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DanioCell Decima smoke test")
    parser.add_argument("--skip_pretrained", action="store_true",
                        help="Skip checks 3-4 (WandB pretrained weight loading)")
    parser.add_argument("--n_batches", type=int, default=5,
                        help="Batches per epoch in check 6 mini-training (default: 5)")
    args = parser.parse_args()

    print("DanioCell Decima — Smoke Test Harness")
    print(f"  H5 file:     {H5_FILE}")
    print(f"  Matrix file: {MATRIX_FILE}")
    print(f"  Log dir:     {LOG_DIR}")
    print(f"  GPU:         {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")

    os.makedirs(LOG_DIR, exist_ok=True)

    # Set up genomepy dirs in scratch
    temp_dir = os.path.join(SCRATCH_DIR, "temp_dirs", "smoke_test")
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["GENOMEPY_CONFIG"]    = os.path.join(temp_dir, "config")
    os.environ["GENOMEPY_CACHE_DIR"] = os.path.join(temp_dir, "cache")
    os.makedirs(os.environ["GENOMEPY_CONFIG"],    exist_ok=True)
    os.makedirs(os.environ["GENOMEPY_CACHE_DIR"], exist_ok=True)

    check_hdf5()
    check_dataloader()

    if not args.skip_pretrained:
        model_pretrained = check_model_pretrained()
        check_forward_pass(model_pretrained)
        del model_pretrained
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("\n  [SKIPPED] Checks 3-4 (--skip_pretrained)")

    check_model_random()
    check_mini_training(n_batches=args.n_batches, skip_pretrained=args.skip_pretrained)

    print(f"\n{'='*60}")
    print("ALL CHECKS PASSED — safe to submit full training jobs")
    print('='*60)


if __name__ == "__main__":
    main()
