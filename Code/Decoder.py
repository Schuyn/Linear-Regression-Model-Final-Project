'''
author:        Schuyn <98257102+Schuyn@users.noreply.github.com>
date:          2025-04-28 22:09:50
'''
import torch
import torch.nn as nn
from ProbAttention import ProbAttention

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention using ProbSparse Attention.
    """
    def __init__(self, d_model: int, n_heads: int, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # ProbSparse attention core
        self.attn = ProbAttention(
            mask_flag=True,
            factor=factor,
            scale=None,
            attention_dropout=dropout,
            output_attention=False
        )
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, L, D = x.shape
        # Linear projections and split into heads
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head)
        # ProbSparse attention
        context, _ = self.attn(Q, K, V, mask)
        # Merge heads and output projection
        out = context.contiguous().view(B, L, D)
        return self.dropout(self.fc_out(out))

class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention: queries from decoder, keys/values from encoder.
    """
    def __init__(self, d_model: int, n_heads: int, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # mask_flag=False for cross-attention (full context)
        self.attn = ProbAttention(
            mask_flag=False,
            factor=factor,
            scale=None,
            attention_dropout=dropout,
            output_attention=False
        )
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,       # decoder queries (B, L_dec, D)
        context: torch.Tensor, # encoder keys/values (B, L_enc, D)
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        B, L_dec, D = x.shape
        L_enc = context.size(1)
        # Project and split
        Q = self.q_proj(x).view(B, L_dec, self.n_heads, self.d_head)
        K = self.k_proj(context).view(B, L_enc, self.n_heads, self.d_head)
        V = self.v_proj(context).view(B, L_enc, self.n_heads, self.d_head)
        # ProbSparse cross-attention
        context_out, _ = self.attn(Q, K, V, mask)
        out = context_out.contiguous().view(B, L_dec, D)
        return self.dropout(self.fc_out(out))

class PositionwiseFeedForward(nn.Module):
    """
    Two-layer feed-forward network.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class DecoderLayer(nn.Module):
    """
    Single decoder layer: self-attention, cross-attention, and feed-forward.
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
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, n_heads, factor, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        self_mask: torch.Tensor = None,
        cross_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # 1) Self-attention with residual
        x2 = self.self_attn(x, self_mask)
        x = self.norm1(x + x2)
        # 2) Cross-attention with residual
        x2 = self.cross_attn(x, enc_out, cross_mask)
        x = self.norm2(x + x2)
        # 3) Feed-forward with residual
        x2 = self.ff(x)
        x = self.norm3(x + x2)
        return x

class Decoder(nn.Module):
    """
    Stack of DecoderLayer(s).
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        factor: int = 5
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, factor)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        self_mask: torch.Tensor = None,
        cross_mask: torch.Tensor = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_out, self_mask, cross_mask)
        return x
