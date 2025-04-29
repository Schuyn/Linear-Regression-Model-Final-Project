'''
author:        Schuyn <98257102+Schuyn@users.noreply.github.com>
date:          2025-04-28 20:50:43
'''
import torch
import torch.nn as nn

class SimpleEmbedding(nn.Module):
    """
    Simple embedding for numeric and time features:
    - Projects raw numeric features (c_in) to d_model
    - Projects 3-dimensional time features (month, day, weekday) to d_model
    - Combines them and applies dropout
    """
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        # Numeric feature projection
        self.value_proj = nn.Linear(c_in, d_model)
        # Time feature projection (month, day, weekday)
        self.time_proj = nn.Linear(3, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_val: torch.Tensor, x_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_val: Tensor of shape (B, L, c_in) containing raw numeric features
            x_time: Tensor of shape (B, L, 3) containing time features
                    in order [month, day, weekday]
        Returns:
            Tensor of shape (B, L, d_model)
        """
        # Project numeric features
        v = self.value_proj(x_val)       # (B, L, d_model)
        # Project time features
        t = self.time_proj(x_time)       # (B, L, d_model)
        # Combine and dropout
        return self.dropout(v + t)

from embed import PositionalEmbedding
class RichEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_proj = nn.Linear(c_in, d_model)
        self.time_proj  = nn.Linear(3, d_model)
        self.pos_emb    = PositionalEmbedding(d_model)
        self.dropout    = nn.Dropout(dropout)
    def forward(self, x_val, x_time):
        v = self.value_proj(x_val)
        t = self.time_proj(x_time)
        p = self.pos_emb(x_val)           # (1, L, d_model) 切片到 (B, L, d_model)
        return self.dropout(v + t + p)
