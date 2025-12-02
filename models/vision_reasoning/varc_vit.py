from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import torch
from torch import nn
from pydantic import BaseModel
import sys
import os

# Add VARC to sys.path to allow imports from it
varc_path = os.path.join(os.path.dirname(__file__), "VARC")
if varc_path not in sys.path:
    sys.path.append(varc_path)

try:
    from src.ARC_ViT import ARCViT
except ImportError:
    # Try importing with modified path if src is not found directly
    # This handles the case where VARC structure might be slightly different or path issues
    sys.path.append(os.path.join(varc_path, "src"))
    from ARC_ViT import ARCViT

@dataclass
class VARCViTInnerCarry:
    # ViT is stateless in this context, but we need a placeholder
    dummy: torch.Tensor

@dataclass
class VARCViTCarry:
    inner_carry: VARCViTInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]

class VARCViTConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0 # Not used by ARCViT directly but required by pretrain.py logic
    num_puzzle_identifiers: int
    vocab_size: int

    # ViT specific config
    image_size: int = 30
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.1
    num_task_tokens: int = 1
    patch_size: int = 2
    
    # Halting config (for compatibility)
    halt_max_steps: int = 1
    halt_exploration_prob: float = 0.0

    forward_dtype: str = "bfloat16"

class VARCViTInner(nn.Module):
    def __init__(self, config: VARCViTConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        
        # Auto-detect image size from seq_len
        # Assuming square grid
        detected_image_size = int(self.config.seq_len ** 0.5)
        if detected_image_size * detected_image_size != self.config.seq_len:
            # If not square, we might have an issue, but let's warn and proceed if it matches config
            if self.config.image_size * self.config.image_size != self.config.seq_len:
                print(f"Warning: seq_len {self.config.seq_len} is not a perfect square and doesn't match image_size {self.config.image_size} squared.")
        else:
            if self.config.image_size != detected_image_size:
                print(f"Overriding image_size {self.config.image_size} with detected size {detected_image_size} from seq_len {self.config.seq_len}")
                self.config.image_size = detected_image_size

        # Adjust patch_size if necessary
        if self.config.image_size % self.config.patch_size != 0:
            print(f"Warning: image_size {self.config.image_size} is not divisible by patch_size {self.config.patch_size}. Falling back to patch_size=1.")
            self.config.patch_size = 1

        self.vit = ARCViT(
            num_tasks=config.num_puzzle_identifiers,
            image_size=config.image_size,
            num_colors=config.vocab_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            num_task_tokens=config.num_task_tokens,
            patch_size=config.patch_size
        )
        
        # Dummy parameter for carry
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def empty_carry(self, batch_size: int):
        return VARCViTInnerCarry(
            dummy=torch.zeros(batch_size, 1, device=self.dummy_param.device)
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: VARCViTInnerCarry):
        # Stateless, nothing to reset
        return carry

    def forward(self, carry: VARCViTInnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[VARCViTInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        inputs = batch["inputs"] # (B, 900)
        task_ids = batch["puzzle_identifiers"] # (B,)
        
        B, L = inputs.shape
        H = W = self.config.image_size
        
        # Reshape inputs to (B, H, W)
        # Assuming inputs are flattened row-major
        pixel_values = inputs.view(B, H, W)
        
        # Forward pass
        # ARCViT returns (B, num_colors, H, W)
        logits = self.vit(pixel_values, task_ids)
        
        # Reshape logits to (B, L, num_colors)
        # (B, C, H, W) -> (B, H, W, C) -> (B, H*W, C)
        logits = logits.permute(0, 2, 3, 1).reshape(B, L, -1)
        
        # Dummy Q-values (always halt)
        q_halt_logits = torch.full((B,), 10.0, device=inputs.device, dtype=torch.float32)
        q_continue_logits = torch.full((B,), -10.0, device=inputs.device, dtype=torch.float32)
        
        return carry, logits, (q_halt_logits, q_continue_logits)

    @property
    def puzzle_emb(self):
        # Expose task_token_embed as puzzle_emb for optimizer compatibility if needed
        # The pretrain.py checks for puzzle_emb_ndim > 0 to use special optimizer
        # ARCViT has task_token_embed
        return self.vit.task_token_embed

class VARCViT(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = VARCViTConfig(**config_dict)
        self.inner = VARCViTInner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return VARCViTCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=batch["inputs"].device),
            halted=torch.zeros((batch_size,), dtype=torch.bool, device=batch["inputs"].device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: VARCViTCarry, batch: Dict[str, torch.Tensor], compute_target_q: bool = False) -> Tuple[VARCViTCarry, Dict[str, torch.Tensor]]:
        # Update data (removing halted sequences - though for 1 step it doesn't matter much)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }
        
        # Forward inner
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(carry.inner_carry, new_current_data)
        
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        # Step logic
        new_steps = carry.steps + 1
        halted = new_steps >= self.config.halt_max_steps
        
        return VARCViTCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
