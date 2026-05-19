# Modified from the original Genentech/decima source by the DanioDecima authors
# (Chan Zuckerberg Biohub) on 2026-05-10. See daniodecima-main/FORK_NOTES.md for the scope of
# modifications. Original copyright Genentech, Inc., 2024 (Genentech Non-Commercial
# Software License v1.0).

import torch
from torch import nn
from grelu.model.models import BorzoiModel, BaseModel
from grelu.model.heads import ConvHead
from huggingface_hub import hf_hub_download
import random
import numpy as np


class DecimaModel(BaseModel):

    def __init__(self, n_tasks: int, replicate: int = 0, mask=True, init_mode="pretrained",
                 pretrained_source="wandb-human", wandb_project="grelu/borzoi",
                 checkpoint_path=None, seed=42, hf_cache_dir=None):
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
            seed: Random seed for reproducibility (default: 42)
            hf_cache_dir: Directory to cache HuggingFace downloads (e.g. scratch path).
                Defaults to None (uses HF_HOME or ~/.cache/huggingface/hub).
        """
        # Set random seeds for reproducibility
        self._set_seed(seed)

        self.mask = mask

        
        if init_mode == "pretrained":
            print(f"Initializing with pretrained weights from {pretrained_source}")
            if pretrained_source == "wandb-human":
                # WandB is deprecated (403 Forbidden). Weights now on HuggingFace:
                # https://huggingface.co/Genentech/borzoi-model
                # Files: human_state_dict_rep{N}.h5  (renamed from fold{N}.h5)
                try:
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
                    weights_path = hf_hub_download(
                        repo_id="Genentech/borzoi-model",
                        filename=f"human_state_dict_rep{replicate}.h5",
                        cache_dir=hf_cache_dir,
                    )
                    state_dict = torch.load(weights_path, weights_only=True)
                    model.load_state_dict(state_dict)
                    print(f"Loaded Borzoi human rep{replicate} weights from HuggingFace "
                          f"(Genentech/borzoi-model)")

                    needs_mask_channel = True

                except Exception as e:
                    print(f"Error loading Borzoi human weights from HuggingFace: {e}")
                    raise RuntimeError(f"Pretrained weight loading failed: {e}") from e
            elif pretrained_source == "decima-human":
                try:
                    # Create the base model architecture for human
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
                    
                    # Modify the first conv layer to have 5 input channels to match checkpoint
                    first_conv = model.embedding.conv_tower.blocks[0].conv
                    new_first_conv = nn.Conv1d(
                        5,  # 5 input channels (4 DNA + 1 mask)
                        first_conv.out_channels,
                        kernel_size=first_conv.kernel_size,
                        stride=first_conv.stride,
                        padding=first_conv.padding,
                        bias=first_conv.bias is not None
                    )
                    model.embedding.conv_tower.blocks[0].conv = new_first_conv

                    checkpoint = torch.load(f"/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/decima_checkpoints_lal/rep{replicate}.ckpt", weights_only=False)
                    state_dict = checkpoint['state_dict']
                    print(f"Loaded checkpoint from /hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/decima_checkpoints_lal/rep{replicate}.ckpt")

                    filtered_state_dict = {}
                    for key, value in state_dict.items():
                        # Keep all parameters that don't belong to the head
                        if not key.startswith('model.head.'):
                            # Remove the 'model.' prefix to match the new model structure
                            new_key = key.replace('model.', '', 1)  # Remove only the first occurrence
                            filtered_state_dict[new_key] = value
                    
                    # Load with more verbose output to confirm loading
                    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
                    
                    # Print diagnostics to confirm weights are actually loaded
                    print(f"Successfully processed {len(filtered_state_dict)} parameters from checkpoint")
                    print(f"Model has {len(model.state_dict())} parameters")
                    print(f"Missing keys: {len(missing_keys)} (expected for head layers)")
                    print(f"Unexpected keys: {len(unexpected_keys)}")
                    
                    if len(missing_keys) > 0:
                        print(f"Sample missing keys: {missing_keys[:3]}")
                    if len(unexpected_keys) > 0:
                        print(f"Sample unexpected keys: {unexpected_keys[:3]}")

                    needs_mask_channel = False

                except Exception as e:
                    print(f"Error loading pretrained weights from WandB: {e}")
                    raise RuntimeError(f"Pretrained weight loading failed: {e}") from e
            elif pretrained_source == "wandb-mouse":
                # WandB is deprecated (403 Forbidden). Weights now on HuggingFace:
                # https://huggingface.co/Genentech/borzoi-model
                # Files: mouse_state_dict_rep{N}.h5  (renamed from fold{N}.h5)
                try:
                    model = BorzoiModel(
                        crop_len=5120,
                        n_tasks=2608,
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
                    weights_path = hf_hub_download(
                        repo_id="Genentech/borzoi-model",
                        filename=f"mouse_state_dict_rep{replicate}.h5",
                        cache_dir=hf_cache_dir,
                    )
                    state_dict = torch.load(weights_path, weights_only=True)
                    model.load_state_dict(state_dict)
                    print(f"Loaded Borzoi mouse rep{replicate} weights from HuggingFace "
                          f"(Genentech/borzoi-model)")

                    needs_mask_channel = True

                except Exception as e:
                    print(f"Error loading Borzoi mouse weights from HuggingFace: {e}")
                    raise RuntimeError(f"Pretrained weight loading failed: {e}") from e
            elif pretrained_source == "local" and checkpoint_path:
                try:
                    state_dict = torch.load(checkpoint_path, weights_only=True)
                    model.load_state_dict(state_dict)
                    print(f"Successfully loaded pretrained weights from {checkpoint_path}")

                    needs_mask_channel = True

                except Exception as e:
                    print(f"Error loading pretrained weights from local path: {e}")
                    print("Falling back to random initialization")
                    raise RuntimeError(f"Pretrained weight loading failed: {e}") from e
            else:
                print("Invalid pretrained source or missing checkpoint path")
                raise RuntimeError("Invalid pretrained source or missing checkpoint path")
        
        if init_mode != "pretrained":
            print(f"Initializing with {init_mode} initialization")
            # Create the base model architecture for human when doing random initialization
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
            for name, param in model.named_parameters():
                if 'weight' in name:
                    if init_mode == "xavier":
                        nn.init.xavier_uniform_(param)
                    elif init_mode == "kaiming":
                        if param.dim() >= 2:
                            nn.init.kaiming_normal_(param)
                        else:
                            # For 1D tensors (like biases), use a normal distribution initialization
                            nn.init.normal_(param, mean=0, std=0.01)
                    elif init_mode == "zeros":
                        nn.init.zeros_(param)
                    # Random is PyTorch default, so no action needed
                elif 'bias' in name:
                    nn.init.zeros_(param)

            needs_mask_channel = True
        
        # Change head
        head = ConvHead(n_tasks=n_tasks, in_channels=1920, pool_func="avg")

        super().__init__(embedding=model.embedding, head=head)

        # Add a channel for the gene mask ONLY if needed
        if self.mask and needs_mask_channel:
            weight = self.embedding.conv_tower.blocks[0].conv.weight
            new_layer = nn.Conv1d(
                5, 512, kernel_size=(15,), stride=(1,), padding="same"
            )
            new_weight = nn.Parameter(
                torch.cat([weight, new_layer.weight[:, [-1], :]], axis=1)
            )
            self.embedding.conv_tower.blocks[0].conv.weight = new_weight
            print("Added mask channel to model")
        elif self.mask and not needs_mask_channel:
            print("Mask channel already present in pretrained weights")
    
    def _set_seed(self, seed):
        """
        Set random seeds for reproducibility.
        
        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False