import warnings
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from einops import rearrange
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from grelu.lightning.metrics import MSE, PearsonCorrCoef
from grelu.sequence.format import strings_to_one_hot
from grelu.utils import get_aggfunc, get_compare_func, make_list

import os, sys
sys.path.append(os.path.dirname(__file__))
sys.path.insert(0, '/code/decima/src/decima')

from decima_model import DecimaModel 
from loss import TaskWisePoissonMultinomialLoss 

#TODO: replace tissue with celltype
class GeneTissueSpecificLSTM(nn.Module):
    def __init__(self, expr_input_dim, hidden_dim=64, n_tissues=6, forecast_horizon=3, 
                 num_layers=2, dropout=0.2, gene_embedding_dim=1920):
        """
        LSTM-based model that captures gene-celltype specific dynamics for forecasting.
        
        Args:
            expr_input_dim: Dimension of input features per celltype (not used for celltype LSTMs)
            hidden_dim: Hidden dimension of the LSTM
            n_tissues: Number of celltypes to forecast
            forecast_horizon: Number of future time points to predict
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            gene_embedding_dim: Dimension of the gene embedding from Decima
        """
        super().__init__()
        self.n_tissues = n_tissues
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        
        # Gene embedding projection - creates gene-specific context
        self.gene_proj = nn.Linear(gene_embedding_dim, hidden_dim)
        
        # Tissue-specific LSTMs - each takes a single feature as input
        self.tissue_lstms = nn.ModuleList([
            nn.LSTM(
                input_size=1,  # Each tissue has a single feature
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            ) for _ in range(n_tissues)
        ])
        
        # Gene-tissue interaction module
        self.gene_tissue_interaction = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(n_tissues)
        ])
        
        # Cross-tissue attention to capture tissue relationships
        self.cross_tissue_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Prediction heads (one per tissue)
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, forecast_horizon)
            ) for _ in range(n_tissues)
        ])
        
        # Global context integration
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim * n_tissues, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Learnable tissue importance
        self.tissue_importance = nn.Parameter(torch.ones(n_tissues))
        
    def forward(self, expr_series, static_emb):
        """
        Forward pass through the gene-tissue specific LSTM model.
        
        Args:
            expr_series: Historical expression values [batch, history_length, n_tissues]
            static_emb: Static gene embeddings [batch, embedding_dim]
            
        Returns:
            forecast: Predicted future expression [batch, forecast_horizon, n_tissues]
        """
        batch_size, history_length, n_features = expr_series.shape
        
        # Ensure we have the expected number of tissues
        assert n_features == self.n_tissues, f"Expected {self.n_tissues} tissues, got {n_features}"
        
        # Process gene embedding to create gene-specific context
        if static_emb.dim() > 2:
            static_emb = static_emb.squeeze(-1)
        gene_context = self.gene_proj(static_emb)  # [batch, hidden_dim]
        
        # Process each tissue separately
        tissue_features = []
        for tissue_idx in range(self.n_tissues):
            # Extract expression for this tissue
            tissue_expr = expr_series[:, :, tissue_idx:tissue_idx+1]  # [batch, history, 1]
            
            # Process with tissue-specific LSTM
            lstm_out, (h_n, _) = self.tissue_lstms[tissue_idx](tissue_expr)
            tissue_hidden = h_n[-1]  # Last layer's hidden state [batch, hidden_dim]
            
            # Combine with gene context to create gene-tissue specific representation
            gene_tissue_combined = torch.cat([tissue_hidden, gene_context], dim=1)
            tissue_features.append(self.gene_tissue_interaction[tissue_idx](gene_tissue_combined))
        
        # Stack tissue features
        tissue_features_stacked = torch.stack(tissue_features, dim=1)  # [batch, n_tissues, hidden_dim]
        
        # Apply cross-tissue attention to capture relationships between tissues
        tissue_weights = F.softmax(self.tissue_importance, dim=0).view(1, -1, 1)  # [1, n_tissues, 1]
        weighted_features = tissue_features_stacked * tissue_weights  # [batch, n_tissues, hidden_dim]
        
        # Self-attention across tissues
        attn_output, _ = self.cross_tissue_attention(
            weighted_features, weighted_features, weighted_features
        )  # [batch, n_tissues, hidden_dim]
        
        # Compute global context
        global_features = self.global_context(
            weighted_features.reshape(batch_size, -1)  # Flatten tissue dimension
        )  # [batch, hidden_dim]
        
        # Generate predictions for each tissue
        forecasts = []
        for tissue_idx in range(self.n_tissues):
            # Combine tissue-specific features with global context
            tissue_with_global = torch.cat([
                attn_output[:, tissue_idx], 
                global_features
            ], dim=1)  # [batch, hidden_dim*2]
            
            # Generate forecast for this tissue
            tissue_forecast = self.prediction_heads[tissue_idx](tissue_with_global)  # [batch, forecast_horizon]
            forecasts.append(tissue_forecast)
        
        # Stack tissue forecasts
        forecast = torch.stack(forecasts, dim=2)  # [batch, forecast_horizon, n_tissues]
        
        return forecast
    
    def get_tissue_importance(self):
        """Returns the learned importance of each tissue"""
        return F.softmax(self.tissue_importance, dim=0).detach().cpu().numpy()
    
    def get_gene_tissue_interactions(self, gene_embeddings):
        """
        Analyzes how different genes interact with different tissues.
        
        Args:
            gene_embeddings: Tensor of gene embeddings [n_genes, embedding_dim]
            
        Returns:
            interaction_scores: Matrix of gene-tissue interaction scores [n_genes, n_tissues]
        """
        n_genes = gene_embeddings.size(0)
        interaction_scores = torch.zeros(n_genes, self.n_tissues)
        
        with torch.no_grad():
            # Process each gene
            for gene_idx in range(n_genes):
                gene_emb = gene_embeddings[gene_idx:gene_idx+1]  # [1, embedding_dim]
                gene_context = self.gene_proj(gene_emb)  # [1, hidden_dim]
                
                # Compute interaction with each tissue
                for tissue_idx in range(self.n_tissues):
                    # Create a dummy tissue hidden state (zeros)
                    tissue_hidden = torch.zeros(1, self.hidden_dim, device=gene_emb.device)
                    
                    # Combine with gene context
                    gene_tissue_combined = torch.cat([tissue_hidden, gene_context], dim=1)
                    interaction = self.gene_tissue_interaction[tissue_idx](gene_tissue_combined)
                    
                    # Use the norm as an interaction score
                    interaction_scores[gene_idx, tissue_idx] = torch.norm(interaction)
        
        return interaction_scores.detach().cpu().numpy()

# --- Expression Autoencoder Module ---
class ExpressionAutoencoder(nn.Module):
    def __init__(self, history_length, n_tissues, latent_dim=16):
        """
        Autoencoder to learn a latent representation of the historical expression.
        
        Args:
            history_length: Number of timepoints in the historical expression.
            n_tissues: Number of tissues (each gene's time series is [history_length, n_tissues]).
            latent_dim: Dimensionality of the latent representation.
        """
        super().__init__()
        self.history_length = history_length
        self.n_tissues = n_tissues
        self.input_dim = history_length * n_tissues
        
        # Encoder: flatten the input and reduce dimension.
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        # Decoder: reconstruct the flattened input.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.input_dim)
        )
    
    def forward(self, x):
        # x shape: [batch, history_length, n_tissues]
        batch_size = x.size(0)
        flat = x.view(batch_size, -1)
        latent = self.encoder(flat)
        reconstruction = self.decoder(latent)
        reconstruction = reconstruction.view(batch_size, self.history_length, self.n_tissues)
        return latent, reconstruction
    

class JointModel(nn.Module):
    def __init__(self, decima_model: DecimaModel, forecast_model: GeneTissueSpecificLSTM, expr_autoencoder: nn.Module, ablate_static_emb: bool = True):
        super().__init__()
        self.decima = decima_model
        self.forecast_model = forecast_model
        self.expr_autoencoder = expr_autoencoder
        self.activation = torch.exp
        self.ablate_static_emb = ablate_static_emb
        self.combined_proj = nn.Linear(1920 + 16, 1920)  # Project back to DECIMA embedding size


    def forward(self, static_seq, expr_series, logits: bool = False):
        """
        Args:
            static_seq: Tensor for static gene sequence input, shape [batch, 5, padded_seq_len]
            expr_series: Tensor of historical expression values, shape [batch, history_length, n_tissues]
        Returns:
            decima_pred: Decima model’s prediction (e.g. reconstruction of historical expression)
            forecast_pred: Forecasted future expression values, shape [batch, forecast_horizon, n_tissues]
            embedding: Static gene embedding from DecimaModel
        """
        # Get decima prediction and static embedding.
        decima_pred = self.decima(static_seq)   # Original Decima prediction
        if not logits:
            decima_pred = self.activation(decima_pred)

        embedding = self.decima.embedding(static_seq)  # Extract decima embedding.
        embedding = self.decima.head.pool(embedding)
        
        if embedding.dim() > 2:
            embedding = embedding.squeeze(-1)
        
        if self.ablate_static_emb:
            embedding = torch.zeros_like(embedding)

        # Process autoencoder branch if provided.
        if self.expr_autoencoder is not None:
            ae_latent, ae_reconstruction = self.expr_autoencoder(expr_series)
        else:
            ae_latent, ae_reconstruction = None, None
        
        # Fuse static embedding with autoencoder latent (if available)
        if ae_latent is not None:
            # Check if shapes are compatible for concatenation
            if embedding.shape[0] == ae_latent.shape[0]:
                try:
                    combined_emb = torch.cat([embedding, ae_latent], dim=1)
                    combined_emb = self.combined_proj(combined_emb)
                except RuntimeError as e:
                    print(f"Error concatenating tensors: {e}")
                    print(f"Embedding shape: {embedding.shape}, AE latent shape: {ae_latent.shape}")
                    # Fall back to using just the embedding
                    combined_emb = embedding
            else:
                print(f"Batch size mismatch: Embedding {embedding.shape[0]}, AE latent {ae_latent.shape[0]}")
                combined_emb = embedding
        else:
            combined_emb = embedding
        
        # Forecasting branch: use the historical expression along with static sequence conditioning.
        forecast_pred = self.forecast_model(expr_series, combined_emb)
        return decima_pred, forecast_pred, embedding, ae_reconstruction

# Create a Lightning module to train jointly.
class JointLightningModel(pl.LightningModule):
    def __init__(self, model_params: dict, train_params: dict = {}, data_params: dict = {}):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.activation = torch.exp
        # Set default train params if missing.
        default_train_params = {
            "lr": 4e-5,
            "batch_size": 4,
            "num_workers": 1,
            "devices": 0,
            "max_epochs": 1,
            "accumulate_grad_batches": 1,
            "decima_loss_weight": 1.0,
            "forecast_loss_weight": 1.0,
            "ae_loss_weight": 1.0,
        }
        for key, value in default_train_params.items():
            if key not in train_params:
                train_params[key] = value
        self.train_params = train_params
        self.model_params = model_params
        self.data_params = data_params

        # Build DecimaModel.
        self.decima = DecimaModel(
            n_tasks=self.model_params["n_tasks"],
            replicate=self.model_params.get("replicate", 0),
            init_mode=self.model_params.get("init_mode", "pretrained"),
            pretrained_source=self.model_params.get("pretrained_source", "wandb"),
            wandb_project=self.model_params.get("wandb_project", "grelu/borzoi"),
            checkpoint_path=self.model_params.get("checkpoint_path", None)
        )
        
        #print(f"self.model_params: {self.model_params}")

        self.forecast_model = GeneTissueSpecificLSTM(
            expr_input_dim=self.model_params.get("expr_input_dim", self.model_params["cell_types"]),  # input dimension is n_cell_types
            hidden_dim=self.model_params.get("expr_hidden_dim", 256),
            n_tissues=self.model_params["cell_types"],
            forecast_horizon=self.model_params.get("forecast_horizon", 3),
            num_layers=2,#model_params.get("lstm_layers", 2),
            dropout=self.model_params.get("dropout", 0.2),
            gene_embedding_dim=1920
        )
        
        # Optionally build Expression Autoencoder if desired.
        self.expr_autoencoder = ExpressionAutoencoder(
            history_length=self.model_params.get("history_length", 5),
            n_tissues=self.model_params["cell_types"],
            latent_dim=self.model_params.get("ae_latent_dim", 16)
        )
        
        # Joint model combining both branches.
        self.model = JointModel(self.decima, self.forecast_model, expr_autoencoder=self.expr_autoencoder)
        
        # Loss functions:
        # For decima, use TaskWisePoissonMultinomialLoss on historical expression.
        self.decima_loss_fn = TaskWisePoissonMultinomialLoss(
                total_weight=self.train_params["total_weight"],
                debug=True
                )
        # For forecasting, use MSE loss (for now)
        self.forecast_loss_fn = nn.MSELoss()
        self.ae_loss_fn = nn.MSELoss()
        self.val_losses = []
        self.test_losses = []
        
        # Initialize metrics (example with MSE and Pearson)
        metrics = MetricCollection({
            #"mse_cell_type": MSE(num_outputs=self.model_params["cell_types"], average=False),
            #"pearson_cell_type": PearsonCorrCoef(num_outputs=self.model_params["cell_types"], average=False),
            "mse_forecast_horizon": MSE(num_outputs=self.model_params["forecast_horizon"], average=False),
            "pearson_forecast_horizon": PearsonCorrCoef(num_outputs=self.model_params["forecast_horizon"], average=False),
        })
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        
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
    
    def forward(self, batch, logits: bool = False):
        """
        Expects a batch dictionary with keys:
            "static_sequence": [batch, 5, padded_seq_len]
            "expr_series": [batch, history_length, n_tissues]
        Returns:
            decima_pred, forecast_pred, embedding
        """ 
        static_seq = batch["static_sequence"]
        static_seq = self.format_input(static_seq)
        expr_series = batch["expr_series"]
        decima_pred, forecast_pred, embedding, ae_reconstruction = self.model(static_seq, expr_series, logits)
        # Apply transform
        decima_pred = self.transform(decima_pred)
        return decima_pred, forecast_pred, embedding, ae_reconstruction

    def training_step(self, batch, batch_idx):
        # Assume the batch dict includes: "static_sequence", "expr_series", "target"
        decima_pred, forecast_pred, embedding, ae_reconstruction = self.forward(batch, logits=True)
        # For decima loss, assume we want to reconstruct the historical expression.
        # Use the historical input as ground truth.
        historical_target = batch["expr_series"]  # shape: [batch, history_length, n_tissues]
        historical_target = historical_target.reshape(historical_target.size(0), -1)

        loss_decima = self.decima_loss_fn(decima_pred, historical_target)
        
        # Forecasting loss:
        forecast_target = batch["target"]  # shape: [batch, forecast_horizon, n_tissues]
        loss_forecast = self.forecast_loss_fn(forecast_pred, forecast_target)

        loss_ae = self.ae_loss_fn(ae_reconstruction, batch["expr_series"])
        
        # Joint loss (weighted sum)
        loss = (self.train_params["decima_loss_weight"] * loss_decima +
                self.train_params["forecast_loss_weight"] * loss_forecast +
                self.train_params["ae_loss_weight"] * loss_ae)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        #self.log("train_decima_loss", loss_decima, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        #self.log("train_forecast_loss", loss_forecast, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Check if embeddings are being updated
        if batch_idx == 0 and self.current_epoch % 5 == 0:  # Check every 5 epochs
            # Get a sample parameter from the embedding
            sample_param = next(self.decima.embedding.parameters())
            param_norm = sample_param.norm().item()
            
            # If we haven't stored the previous value, store it
            if not hasattr(self, 'prev_param_norm'):
                self.prev_param_norm = param_norm
                self.log("embedding_param_norm", param_norm)
                return
            
            # Calculate change in parameter norm
            param_change = abs(param_norm - self.prev_param_norm)
            self.prev_param_norm = param_norm
            
            # Log the change
            self.log("embedding_param_change", param_change)
            self.log("embedding_param_norm", param_norm)
    
    def validation_step(self, batch, batch_idx):
        decima_pred, forecast_pred, _, _ = self.forward(batch, logits=True)
        historical_target = batch["expr_series"]#.mean(dim=1)
        historical_target = historical_target.reshape(historical_target.size(0), -1)
        loss_decima = self.decima_loss_fn(decima_pred, historical_target)
        y_hat = self.activation(decima_pred)

        forecast_target = batch["target"]
        loss_forecast = self.forecast_loss_fn(forecast_pred, forecast_target)
        loss = self.train_params["decima_loss_weight"] * loss_decima + self.train_params["forecast_loss_weight"] * loss_forecast
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_decima_loss", loss_decima, sync_dist=True)
        self.log("val_forecast_loss", loss_forecast, sync_dist=True)
        # Update metrics for forecast branch
        self.val_metrics.update(forecast_pred.squeeze(-1), forecast_target.squeeze(-1))
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
        self.log_dict(mean_val_metrics)
        self.log("val_loss", mean_losses, sync_dist=True)

        self.val_metrics.reset()
        self.val_losses = []

    def test_step(self, batch, batch_idx):
        decima_pred, forecast_pred, _, _ = self.forward(batch, logits=True)
        historical_target = batch["expr_series"]#.mean(dim=1)
        historical_target = historical_target.reshape(historical_target.size(0), -1)
        loss_decima = self.decima_loss_fn(decima_pred, historical_target)
        forecast_target = batch["target"]
        loss_forecast = self.forecast_loss_fn(forecast_pred, forecast_target)
        loss = self.train_params["decima_loss_weight"] * loss_decima + self.train_params["forecast_loss_weight"] * loss_forecast
        self.log("test_loss", loss, sync_dist=True)
        self.test_metrics.update(forecast_pred.squeeze(-1), forecast_target.squeeze(-1))
        self.test_losses.append(loss)
        return loss

    def on_test_epoch_end(self):
        """
        Calculate metrics for entire test set
        """
        self.computed_test_metrics = self.test_metrics.compute()
        self.log_dict({k: v.mean() for k, v in self.computed_test_metrics.items()})
        losses = torch.stack(self.test_losses)
        self.log("test_loss", torch.mean(losses), sync_dist=True)
        self.test_metrics.reset()
        self.test_losses = []

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.train_params["lr"])
    
    def count_params(self) -> int:
        """
        Number of gradient enabled parameters in the model
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def parse_logger(self) -> str:
        """
        Parses the name of the logger supplied in train_params.
        """
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
                name=self.train_params["name"], save_dir=self.train_params["save_dir"]
            )
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

    def make_train_loader(self, dataset, batch_size=None, num_workers=None):
        return DataLoader(dataset, batch_size=batch_size or self.train_params["batch_size"], shuffle=True, num_workers=num_workers or self.train_params["num_workers"])

    def make_test_loader(self, dataset, batch_size=None, num_workers=None):
        return DataLoader(dataset, batch_size=batch_size or self.train_params["batch_size"], shuffle=False, num_workers=num_workers or self.train_params["num_workers"])

    def make_predict_loader(
        self,
        dataset: Callable,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        limit=None
    ) -> Callable:
        """
        Make dataloader for prediction
        """
        dataset.predict = True

        if limit is not None:
            from torch.utils.data import Subset
            indices = list(range(min(limit, len(dataset))))
            dataset = Subset(dataset, indices)
            print(f"Created subset with {len(dataset)} examples (limited from {len(dataset)})")
    
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
        
        # Set up logging
        logger = self.parse_logger()

        # Set up trainer
        trainer = pl.Trainer(
            max_epochs=self.train_params["max_epochs"],
            accelerator='gpu',
            devices=[0,1,2,3],#make_list(self.train_params["devices"]),
            logger=logger,
            callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", save_last=True)],
            default_root_dir=self.train_params["save_dir"],
            accumulate_grad_batches=self.train_params["accumulate_grad_batches"],
            precision="16-mixed",
            strategy="ddp",
        )

        # Make dataloaders
        train_dataloader = self.make_train_loader(train_dataset)
        val_dataloader = self.make_test_loader(val_dataset)

        if checkpoint_path is None:
            # First validation pass
            trainer.validate(model=self, dataloaders=val_dataloader)
            self.val_metrics.reset()

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
        augment_aggfunc: Union[str, Callable] = "mean",
        compare_func: Optional[Union[str, Callable]] = None,
    ):
        """
        Predict for a dataset of sequences or variants using the joint model.
    
        Args:
            dataset: Dataset object that yields a dict with keys "static_sequence", "expr_series", and "target".
            devices: Device IDs to use.
            num_workers: Number of workers for the data loader.
            batch_size: Batch size for the data loader.
            augment_aggfunc: Function or string (default "mean") to aggregate predictions over augmentations.
            compare_func: (Optional) Function to compare predictions.
            
        Returns:
            A numpy array with forecast predictions of shape [num_genes, forecast_horizon, n_tissues],
            aggregated over augmentations.
        """
        torch.set_float32_matmul_precision("medium")
        
        dataloader = self.make_predict_loader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        #first_item = dataloader.dataset[1]
        #print(f"First item: {first_item}")
        trainer = pl.Trainer(accelerator="gpu", devices=make_list(devices), logger=None)

        all_forecasts = []
        for batch_out in trainer.predict(self, dataloader):
            # batch_out is a tuple of tensors
            # We extract the second element: forecast_pred
            _, forecast_pred, _ = batch_out
            all_forecasts.append(forecast_pred)
        
        preds = torch.cat(all_forecasts, dim=0)  # Expected shape: [total_samples, forecast_horizon, n_tissues]
        
        # If the dataset is augmented, total_samples = (num_genes * n_augmented)
        if hasattr(dataset, "n_augmented") and dataset.n_augmented > 1:
            num_genes = preds.shape[0] // dataset.n_augmented
            # Reshape to group augmentations per gene: [num_genes, n_augmented, forecast_horizon, n_tissues]
            preds = preds.view(num_genes, dataset.n_augmented, preds.shape[1], preds.shape[2])
            # Aggregate over the augmentation dimension.
            if isinstance(augment_aggfunc, str) and augment_aggfunc == "mean":
                preds = preds.mean(dim=1)
            elif callable(augment_aggfunc):
                preds = augment_aggfunc(preds, dim=1)
        # Otherwise, preds remains as [total_samples, forecast_horizon, n_tissues]
        
        preds = preds.detach().cpu().numpy()
        return preds

    def get_task_idxs(
        self,
        tasks: Union[int, str, List[int], List[str]],
        key: str = "name",
        invert: bool = False,
    ) -> Union[int, List[int]]:
        """
        Given a task name or metadata entry, get the task index
        If integers are provided, return them unchanged

        Args:
            tasks: A string corresponding to a task name or metadata entry,
                or an integer indicating the index of a task, or a list of strings/integers
            key: key to model.data_params["tasks"] in which the relevant task data is
                stored. "name" will be used by default.
            invert: Get indices for all tasks except those listed in tasks

        Returns:
            The index or indices of the corresponding task(s) in the model's
            output.
        """
        # If a string is provided, extract the index
        if isinstance(tasks, str):
            return self.data_params["tasks"][key].index(tasks)
        # If an integer is provided, return it as the index
        elif isinstance(tasks, int):
            return tasks
        # If a list is provided, return teh index for each element
        elif isinstance(tasks, list):
            return [self.get_task_idxs(task) for task in tasks]
        else:
            raise TypeError("Input must be a list, string or integer")
        if invert:
            return [
                i
                for i in range(self.model_params["n_tasks"])
                if i not in make_list(tasks)
            ]
