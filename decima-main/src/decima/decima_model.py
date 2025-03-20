import torch
import grelu
from torch import nn
from grelu.model.models import BorzoiModel, BaseModel
from grelu.model.heads import ConvHead
from grelu.resources import get_artifact
from tempfile import TemporaryDirectory
from pathlib import Path
import wandb


class DecimaModel(BaseModel):

    def __init__(self, n_tasks: int, replicate: int = 0, mask=True, init_mode="pretrained", 
                 pretrained_source="wandb", wandb_project="grelu/borzoi", 
                 checkpoint_path=None):
        """
        Initialize the Decima model with different weight initialization strategies.
        
        Args:
            n_tasks: Number of prediction tasks (genes)
            replicate: Replicate number for pretrained weights
            mask: Whether to add a channel for gene mask
            init_mode: Initialization mode - "pretrained", "random", "xavier", "kaiming", or "zeros"
            pretrained_source: Source of pretrained weights - "wandb" or "local"
            wandb_project: WandB project path (if using wandb)
            checkpoint_path: Path to local checkpoint file (if using local)
        """
        self.mask = mask

        # Create the base model architecture
        model = BorzoiModel(
            crop_len=5120,
            n_tasks=7611,
            stem_channels=512,
            stem_kernel_size=15,
            init_channels=608,
            n_conv=7,
            kernel_size=5,
            n_transformers=8,
            key_len=64,
            value_len=192,
            pos_dropout=0.0,
            attn_dropout=0.0,
            n_heads=8,
            n_pos_features=32,
            final_act_func=None,
            final_pool_func=None,
        )
        
        if init_mode == "pretrained":
            print(f"Initializing with pretrained weights from {pretrained_source}")
            if pretrained_source == "wandb":
                try:
                    api = wandb.Api()
                    art = api.artifact(f'{wandb_project}/human_state_dict_fold{replicate}:latest')
                    with TemporaryDirectory() as d:
                        art.download(d)
                        state_dict = torch.load(Path(d) / f"fold{replicate}.h5")
                    model.load_state_dict(state_dict)
                    print("Successfully loaded pretrained weights from WandB")
                except Exception as e:
                    print(f"Error loading pretrained weights from WandB: {e}")
                    print("Falling back to random initialization")
                    init_mode = "random"
            elif pretrained_source == "local" and checkpoint_path:
                try:
                    state_dict = torch.load(checkpoint_path)
                    model.load_state_dict(state_dict)
                    print(f"Successfully loaded pretrained weights from {checkpoint_path}")
                except Exception as e:
                    print(f"Error loading pretrained weights from local path: {e}")
                    print("Falling back to random initialization")
                    init_mode = "random"
            else:
                print("Invalid pretrained source or missing checkpoint path")
                print("Falling back to random initialization")
                init_mode = "random"
        
        if init_mode != "pretrained":
            print(f"Initializing with {init_mode} initialization")
            for name, param in model.named_parameters():
                if 'weight' in name:
                    if init_mode == "xavier":
                        nn.init.xavier_uniform_(param)
                    elif init_mode == "kaiming":
                        nn.init.kaiming_normal_(param)
                    elif init_mode == "zeros":
                        nn.init.zeros_(param)
                    # Random is PyTorch default, so no action needed
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        # Change head
        head = ConvHead(n_tasks=n_tasks, in_channels=1920, pool_func="avg")

        super().__init__(embedding=model.embedding, head=head)

        # Add a channel for the gene mask
        if self.mask:
            weight = self.embedding.conv_tower.blocks[0].conv.weight
            new_layer = nn.Conv1d(
                5, 512, kernel_size=(15,), stride=(1,), padding="same"
            )
            new_weight = nn.Parameter(
                torch.cat([weight, new_layer.weight[:, [-1], :]], axis=1)
            )
            self.embedding.conv_tower.blocks[0].conv.weight = new_weight
