'''
author:        Schuyn <98257102+Schuyn@users.noreply.github.com>
date:          2025-04-28 19:31:59
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ProbAttention import ProbAttention

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
        # expects shapes (B, L, H, D)
        context, attn_weights = self.attn(Q, K, V, attn_mask)
        # Merge heads: (B, L, H, D) -> (B, L, H*D)
        context = context.contiguous().view(B, L, -1)
        out = self.dropout(self.fc_out(context))
        return out, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    Feed-forward network with two linear layers and activation.
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
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class ConvDistill(nn.Module):
    """
    1D convolution + pooling for sequence length distillation.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode='circular')
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        # x: (B, L, D)
        x = x.transpose(1, 2)          # (B, D, L)
        x = self.act(self.norm(self.conv(x)))
        x = self.pool(x)               # (B, D, L//2)
        return x.transpose(1, 2)       # (B, L//2, D)


class EncoderLayer(nn.Module):
    """
    Single Encoder layer: self-attention + feed-forward.
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
        # Self-attention
        attn_out, attn_weights = self.self_attn(x, attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights


class Encoder(nn.Module):
    """
    Stack of EncoderLayers with optional convolutional distillation.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        factor: int = 5,
        distill: bool = True
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Add Encoder layer
            self.layers.append(
                EncoderLayer(d_model, n_heads, d_ff, dropout, factor)
            )
            # After each except last, optionally add distillation conv
            if distill and i < num_layers - 1:
                self.layers.append(ConvDistill(d_model))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        """
        Args:
            x: (B, L, d_model)
        Returns:
            x: (B, L', d_model)
            attns: list of attention weights from each EncoderLayer
        """
        attns = []
        for layer in self.layers:
            if isinstance(layer, EncoderLayer):
                x, attn_w = layer(x, attn_mask)
                attns.append(attn_w)
            else:
                # ConvDistill layer
                x = layer(x)
        return x, attns
