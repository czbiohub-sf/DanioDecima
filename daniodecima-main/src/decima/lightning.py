# Modified from the original Genentech/decima source by the DanioDecima authors
# (Chan Zuckerberg Biohub) on 2026-05-10. See daniodecima-main/FORK_NOTES.md for the scope of
# modifications. Original copyright Genentech, Inc., 2024 (Genentech Non-Commercial
# Software License v1.0).

"""
The LightningModel class.
"""

from datetime import datetime
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from grelu.lightning.metrics import MSE, PearsonCorrCoef
from grelu.utils import make_list

import os, sys
sys.path.append(os.path.dirname(__file__))
sys.path.insert(0, '/code/decima/src/decima')

from decima_model import DecimaModel
from loss import TaskWisePoissonMultinomialLoss


default_train_params = {
    "lr": 4e-5,
    "batch_size": 4,
    "num_workers": 1,
    "devices": 0,
    "logger": "csv",
    "save_dir": ".",
    "max_epochs": 1,
    "accumulate_grad_batches":1,
    "total_weight": 1e-4,
    "disease_weight":1e-2,
    "weight_decay": 1e-4
}


class LightningModel(pl.LightningModule):
    """
    Wrapper for predictive sequence models

    Args:
        model_params: Dictionary of parameters specifying model architecture
        train_params: Dictionary specifying training parameters
        data_params: Dictionary specifying parameters of the training data.
            This is empty by default and will be filled at the time of
            training.
    """

    def __init__(
        self,
        model_params: dict,
        train_params: dict | None = None,
        data_params: dict | None = None,
    ) -> None:
        super().__init__()
        train_params = {} if train_params is None else train_params
        data_params = {} if data_params is None else data_params

        self.save_hyperparameters(ignore=["model"])

        # Add default training parameters
        for key in default_train_params.keys():
            if key not in train_params:
                train_params[key] = default_train_params[key]

        # Save params
        self.model_params = model_params
        self.train_params = train_params
        self.data_params = data_params

        # Build model
        self.model = DecimaModel(
            n_tasks=self.model_params["n_tasks"],
            replicate=self.model_params.get("replicate", 0),
            init_mode=self.model_params.get("init_mode", "pretrained"),
            pretrained_source=self.model_params.get("pretrained_source", "wandb-human"),
            wandb_project=self.model_params.get("wandb_project", "grelu/borzoi"),
            checkpoint_path=self.model_params.get("checkpoint_path", None)
        )
        # Set up loss function
        self.decima_loss = TaskWisePoissonMultinomialLoss(
                total_weight=self.train_params["total_weight"],
                debug=True
                )
        self.val_losses = []
        self.test_losses = []

        # Set up activation function
        self.activation = torch.exp

        # Inititalize metrics
        metrics = MetricCollection(
            {
                "mse": MSE(num_outputs=self.model.head.n_tasks, average=False),
                "pearson": PearsonCorrCoef(
                    num_outputs=self.model.head.n_tasks, average=False
                ),
            }
        )
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Initialize prediction transform
        self.reset_transform()
        

    def format_input(self, x: Union[Tuple[Tensor, Tensor], Tensor]) -> Tensor:
        """
        Extract the one-hot encoded sequence from the input
        """
        # if x is a tuple of sequence, label, return the sequence
        if isinstance(x, Tensor):
            if x.ndim == 3:
                return x
            else:
                return x.unsqueeze(0)
        elif isinstance(x, Tuple):
            return x[0]
        else:
            raise Exception("Cannot perform forward pass on the given input format.")

    def forward(
        self,
        x: Union[Tuple[Tensor, Tensor], Tensor, str, List[str]],
        logits: bool = False,
    ) -> Tensor:
        """
        Forward pass
        """
        # Format the input as a one-hot encoded tensor
        x = self.format_input(x)

        # Run the model
        x = self.model(x)

        # forward() produces prediction (e.g. post-activation)
        # unless logits=True, which is used in loss functions
        if not logits:
            x = self.activation(x)

        # Apply transform
        x = self.transform(x)
        return x

    def training_step(self, batch: Tensor, _batch_idx: int) -> Tensor:
        x, y = batch
        logits = self.forward(x, logits=True)
        decima_loss, poisson_term, multinomial_term = self.decima_loss(logits, y)
        self.log("train_loss", decima_loss, logger=True, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx="mean")
        self.log("train_poisson_loss", poisson_term, logger=True, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx="mean")
        self.log("train_multinomial_loss", multinomial_term, logger=True, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx="mean")
        loss = decima_loss
        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int) -> Tensor:
        x, y = batch
        logits = self.forward(x, logits=True)
        decima_loss, poisson_term, multinomial_term = self.decima_loss(logits, y)
        y_hat = self.activation(logits)
        print(f"Pred variance: {y_hat.var():.6f}, Target variance: {y.var():.6f}")
        self.log("val_poisson_loss", poisson_term, logger=True, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("val_multinomial_loss", multinomial_term, logger=True, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.val_metrics.update(y_hat, y)
        loss = decima_loss
        self.val_losses.append(loss)
        return loss

    def on_validation_epoch_end(self):
        """
        Calculate metrics for entire validation set
        """
        # Compute metrics
        val_metrics = self.val_metrics.compute()
        mean_val_metrics = {k: v.mean() for k, v in val_metrics.items()}
        # Compute loss
        losses = torch.stack(self.val_losses)
        mean_losses = torch.mean(losses)
        # Log
        self.log_dict(mean_val_metrics, sync_dist=True)
        self.log("val_loss", mean_losses, sync_dist=True)
        self.val_metrics.reset()
        self.val_losses = []

    def test_step(self, batch: Tensor, _batch_idx: int) -> Tensor:
        """
        Calculate metrics after a single test step
        """
        x, y = batch
        logits = self.forward(x, logits=True)
        decima_loss, poisson_term, multinomial_term = self.decima_loss(logits, y)
        y_hat = self.activation(logits)
        self.log("test_poisson_loss", poisson_term, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_multinomial_loss", multinomial_term, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics.update(y_hat, y)
        loss = decima_loss
        self.test_losses.append(loss)
        return loss

    def on_test_epoch_end(self) -> None:
        """
        Calculate metrics for entire test set
        """
        self.computed_test_metrics = self.test_metrics.compute()
        self.log_dict({k: v.mean() for k, v in self.computed_test_metrics.items()}, sync_dist=True)
        losses = torch.stack(self.test_losses)
        self.log("test_loss", torch.mean(losses), sync_dist=True)
        self.test_metrics.reset()
        self.test_losses = []

    def configure_optimizers(self) -> None:
        """
        Configure optimizer for training
        """
        return optim.Adam(self.parameters(), 
                        lr=self.train_params["lr"], 
                        weight_decay=self.train_params.get("weight_decay", 1e-4))

    def count_params(self) -> int:
        """
        Number of gradient enabled parameters in the model
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def parse_logger(self) -> str:
        if "name" not in self.train_params:
            self.train_params["name"] = datetime.now().strftime("%Y_%d_%m_%H_%M")
        
        if self.train_params["logger"] == "wandb":
            logger = WandbLogger(
                name=self.train_params["name"],
                log_model=True,
                save_dir=self.train_params["save_dir"],
            )
        elif self.train_params["logger"] == "csv":
            logger = CSVLogger(
                name=self.train_params["name"], 
                save_dir=self.train_params["save_dir"]
            )
        elif self.train_params["logger"] == "tensorboard":
            # Add support for TensorBoard logger
            from pytorch_lightning.loggers import TensorBoardLogger
            logger = TensorBoardLogger(
                save_dir=self.train_params["save_dir"],
                name=self.train_params["name"]
            )
        else:
            # Support passing a logger object directly
            if isinstance(self.train_params["logger"], pl.loggers.Logger):
                logger = self.train_params["logger"]
            else:
                raise NotImplementedError
        return logger

    def add_transform(self, prediction_transform: Callable) -> None:
        """
        Add a prediction transform
        """
        if prediction_transform is not None:
            self.transform = prediction_transform

    def reset_transform(self) -> None:
        """
        Remove a prediction transform
        """
        self.transform = nn.Identity()

    def make_train_loader(
        self,
        dataset: Callable,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> Callable:
        """
        Make dataloader for training
        """
        return DataLoader(
            dataset,
            batch_size=batch_size or self.train_params["batch_size"],
            shuffle=True,
            num_workers=num_workers or self.train_params["num_workers"],
        )

    def make_test_loader(
        self,
        dataset: Callable,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> Callable:
        """
        Make dataloader for validation and testing
        """
        return DataLoader(
            dataset,
            batch_size=batch_size or self.train_params["batch_size"],
            shuffle=False,
            num_workers=num_workers or self.train_params["num_workers"],
        )

    def make_predict_loader(
        self,
        dataset: Callable,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> Callable:
        """
        Make dataloader for prediction
        """
        dataset.predict = True
        return DataLoader(
            dataset,
            batch_size=batch_size or self.train_params["batch_size"],
            shuffle=False,
            num_workers=num_workers or self.train_params["num_workers"],
        )

    def train_on_dataset(
        self,
        train_dataset: Callable,
        val_dataset: Callable,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Train model and optionally log metrics to wandb.

        Args:
            train_dataset (Dataset): Dataset object that yields training examples
            val_dataset (Dataset) : Dataset object that yields training examples
            checkpoint_path (str): Path to model checkpoint from which to resume training.
                The optimizer will be set to its checkpointed state.

        Returns:
            PyTorch Lightning Trainer
        """
        torch.set_float32_matmul_precision("medium")
        # Required for deterministic=True on CUDA >= 10.2 with CuBLAS ops
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        # Set up logging
        logger = self.parse_logger()

        # Set up callbacks
        callbacks = [
            ModelCheckpoint(monitor="val_loss", mode="min", save_last=True)
        ]
        # Add early stopping callback if specified
        if "early_stopping_patience" in self.train_params:
            from pytorch_lightning.callbacks import EarlyStopping
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=self.train_params["early_stopping_patience"],
                min_delta=self.train_params.get("early_stopping_min_delta", 0.0),
                verbose=True,
                mode="min"
            )
            callbacks.append(early_stopping)
        

        # Get trainer settings from train_params or use defaults
        accelerator = self.train_params.get("accelerator", "gpu" if torch.cuda.is_available() else "cpu")
        devices = self.train_params.get("devices", None)
        precision = self.train_params.get("precision", "16-mixed" if accelerator == "gpu" else "32")
        strategy = self.train_params.get("strategy", None)
        
        print(f"Training with: accelerator={accelerator}, devices={devices}, precision={precision}")
        
        # Set up trainer
        trainer = pl.Trainer(
            max_epochs=self.train_params["max_epochs"],
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            callbacks=callbacks,
            default_root_dir=self.train_params["save_dir"],
            accumulate_grad_batches=self.train_params["accumulate_grad_batches"],
            precision=precision,
            strategy=strategy,
            gradient_clip_val=self.train_params.get("gradient_clip_val", None),
            gradient_clip_algorithm=self.train_params.get("gradient_clip_algorithm", None),
            deterministic=True,
            enable_model_summary=True,
        )

        # Set seed right before training if provided
        if "seed" in self.train_params:
            pl.seed_everything(self.train_params["seed"], workers=True)
            print(f"Set PyTorch Lightning seed to: {self.train_params['seed']}")
            
        # Make dataloaders
        train_dataloader = self.make_train_loader(train_dataset)
        val_dataloader = self.make_test_loader(val_dataset)


        # Add data parameters
        self.data_params["tasks"] = train_dataset.tasks.reset_index(
            names="name"
        ).to_dict(orient="list")

        for attr, value in self._get_dataset_attrs(train_dataset):
            self.data_params["train_" + attr] = value

        for attr, value in self._get_dataset_attrs(val_dataset):
            self.data_params["val_" + attr] = value

        # Training
        trainer.fit(
            model=self,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=checkpoint_path,
        )
        return trainer

    def _get_dataset_attrs(self, dataset: Callable) -> None:
        """
        Read data parameters from a dataset object
        """
        for attr in dir(dataset):
            if not attr.startswith("_") and not attr.isupper():
                value = getattr(dataset, attr)
                if (
                    (isinstance(value, str))
                    or (isinstance(value, int))
                    or (isinstance(value, float))
                    or (value is None)
                ):
                    yield attr, value

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["hyper_parameters"]["data_params"] = self.data_params

    def predict_on_dataset(
        self,
        dataset: Callable,
        devices: int = 0,
        num_workers: int = 1,
        batch_size: int = 6,
        _augment_aggfunc: Union[str, Callable] = "mean",
        _compare_func: Optional[Union[str, Callable]] = None,
    ):
        """
        Predict for a dataset of sequences or variants

        Args:
            dataset: Dataset object that yields one-hot encoded sequences
            devices: Device IDs to use
            num_workers: Number of workers for data loader
            batch_size: Batch size for data loader

        Returns:
            Model predictions as a numpy array or dataframe
        """
        torch.set_float32_matmul_precision("medium")
        dataloader = self.make_predict_loader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        trainer = pl.Trainer(accelerator="gpu", devices=make_list(devices), logger=None)

        # Predict
        preds = torch.concat(trainer.predict(self, dataloader)).squeeze(-1)

        # Reshape predictions
        preds = rearrange(
            preds,
            "(b n a) t -> b n a t",
            n=dataset.n_augmented,
            a=dataset.n_alleles,
        )

        # Convert predictions to numpy array
        preds = preds.detach().cpu().numpy()

        if dataset.n_alleles==2:
            preds = preds[:, :, 1, :] - preds[:, :, 0, :] # BNT
        else:
            preds = preds.squeeze(2)  # B N T

        preds = np.mean(preds, axis=1)  # B T
        return preds

    def get_task_idxs(
        self,
        tasks: Union[int, str, List[int], List[str]],
        key: str = "name",
    ) -> Union[int, List[int]]:
        """
        Given a task name or metadata entry, get the task index.
        If integers are provided, return them unchanged.

        Args:
            tasks: A string corresponding to a task name or metadata entry,
                or an integer indicating the index of a task, or a list of strings/integers
            key: key to model.data_params["tasks"] in which the relevant task data is
                stored. "name" will be used by default.

        Returns:
            The index or indices of the corresponding task(s) in the model's
            output.
        """
        # If a string is provided, extract the index
        if isinstance(tasks, str):
            return self.data_params["tasks"][key].index(tasks)
        # If an integer is provided, return it as the index
        if isinstance(tasks, int):
            return tasks
        # If a list is provided, return the index for each element
        if isinstance(tasks, list):
            return [self.get_task_idxs(task) for task in tasks]
        raise TypeError("Input must be a list, string or integer")
