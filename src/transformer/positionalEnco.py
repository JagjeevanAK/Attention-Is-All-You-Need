import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.pe: torch.Tensor

        # Matrix of Shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Vector of Shape (seq_len, 1)
        pos = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp( torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        temp = pos * div_term
        
        # Even position encoding
        pe[: , 0::2 ] = torch.sin(temp)
        # Odd position encoding
        pe[:, 1::2] = torch.cos(temp)
        
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, d_model)
        return self.dropout(x)