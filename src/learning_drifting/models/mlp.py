import torch
import torch.nn as nn
from torch.nn import Module
from jaxtyping import Float
from torch import Tensor, nn


class Mlp(Module):
    """
    MLP: noise -> output. 3 hidden layers with SiLU.
    """
    def __init__(self, dim: int =2, hidden_dim: int =512) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(
        self,
        z: Float[Tensor, "batch_dim"]
    ) -> Float[Tensor, "batch dim"]:
        
        z = self.backbone(z)
        return z