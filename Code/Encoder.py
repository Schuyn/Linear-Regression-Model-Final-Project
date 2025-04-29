'''
author:        Schuyn <98257102+Schuyn@users.noreply.github.com>
date:          2025-04-28 19:31:59
'''
import torch
import torch.nn as nn
from ProbAttention import ProbAttention
import numpy as np
from math import sqrt

class MultiHeadProbAttention(nn.Module):
    """
    Multi-head wrapper around ProbSparse Attention.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        factor: int = 5,
        dropout: float = 0.1,
        output_attention: bool = False
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.attn = ProbAttention(
            mask_flag=True,
            factor=factor,
            scale=None,
            attention_dropout=dropout,
            output_attention=output_attention
        )
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        B, L, D = x.shape
        # Linear projections and split heads
        Q = self.proj_q(x).view(B, L, self.n_heads, self.d_head)
        K = self.proj_k(x).view(B, L, self.n_heads, self.d_head)
        V = self.proj_v(x).view(B, L, self.n_heads, self.d_head)

        # Apply ProbSparse Attention
        context, attn_weights = self.attn(Q, K, V, attn_mask)
        # Merge heads: (B, L, H, D_head) -> (B, L, D)
        context = context.contiguous().view(B, L, -1)
        return self.dropout(self.fc_out(context)), attn_weights

class PositionwiseFeedForward(nn.Module):
    """
    Two-layer feed-forward network.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class ConvDistill(nn.Module):
    """
    1D convolution + pooling for distillation (downsampling by 2).
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode='circular')
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)      # (B, D, L)
        x = self.act(self.norm(self.conv(x)))
        x = self.pool(x)           # (B, D, L//2)
        return x.transpose(1, 2)   # (B, L//2, D)

class EncoderLayer(nn.Module):
    """
    Single encoder layer: self-attention + feed-forward.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        factor: int = 5
    ):
        super().__init__()
        self.self_attn = MultiHeadProbAttention(d_model, n_heads, factor=factor, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        attn_out, attn_weights = self.self_attn(x, attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights

class EncoderStack(nn.Module):
    """
    Multi-branch encoder with multiple distillation levels.
    Each branch applies `num_layers` of EncoderLayer, inserting ConvDistill
    for the first `distill` layers accordingly.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        distill_levels: list,
        dropout: float = 0.1,
        factor: int = 5
    ):
        super().__init__()
        # For each distillation level, store a list of modules (EncoderLayer or ConvDistill)
        self.branches = nn.ModuleList()
        for distill in distill_levels:
            modules = nn.ModuleList()
            for i in range(num_layers):
                modules.append(
                    EncoderLayer(
                        d_model=d_model,
                        n_heads=n_heads,
                        d_ff=d_ff,
                        dropout=dropout,
                        factor=factor
                    )
                )
                if i < distill:
                    modules.append(ConvDistill(d_model))
            self.branches.append(modules)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, L, d_model)
            attn_mask: Optional mask tensor
        Returns:
            Tensor of shape (B, sum_{lvl}(L // 2^lvl), d_model)
        """
        outputs = []
        # Apply each branch sequentially, handling tuple outputs correctly
        for modules in self.branches:
            x_branch = x
            for module in modules:
                if isinstance(module, EncoderLayer):
                    # EncoderLayer returns (x, attn)
                    x_branch, _ = module(x_branch, attn_mask)
                else:
                    # ConvDistill returns only x
                    x_branch = module(x_branch)
            outputs.append(x_branch)
        # Concatenate along sequence length
        return torch.cat(outputs, dim=1)
