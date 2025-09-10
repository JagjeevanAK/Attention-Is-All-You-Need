import torch
import torch.nn as nn
from .LayerNorm import LayerNormalization


class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        # Post-LN as per original paper: LayerNorm(x + sublayer(x))
        return self.norm(x + self.dropout(sublayer(x)))
