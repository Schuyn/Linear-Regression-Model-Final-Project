'''
author:        Schuyn <98257102+Schuyn@users.noreply.github.com>
date:          2025-04-28 22:09:50
'''
import torch
import torch.nn as nn
from ProbAttention import ProbAttention

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
        context: torch.Tensor  # encoder keys/values (B, L_enc, D)
    ) -> torch.Tensor:
        B, L_dec, D = x.shape
        L_enc = context.size(1)
        # Project and split
        Q = self.q_proj(x).view(B, L_dec, self.n_heads, self.d_head)
        K = self.k_proj(context).view(B, L_enc, self.n_heads, self.d_head)
        V = self.v_proj(context).view(B, L_enc, self.n_heads, self.d_head)
        # Cross-attention
        context_out, _ = self.attn(Q, K, V, None)
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

class SimpleDecoder(nn.Module):
    """
    Non-autoregressive decoder with a self-attention among queries,
    followed by multiple cross-attention + feed-forward layers.
    Uses time feature embedding for future steps.
    """
    def __init__(
        self,
        pred_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int = 2,
        factor: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.pred_len = pred_len
        # Learned query embeddings for future positions
        self.query_embed = nn.Parameter(torch.randn(pred_len, d_model))
        # Time features projection (month, day, weekday)
        self.time_proj = nn.Linear(3, d_model)
        # Self-attention among queries (full, no mask)
        self.self_attn = MultiHeadCrossAttention(d_model, n_heads, factor, dropout)
        self.norm0 = nn.LayerNorm(d_model)
        # Cross-attention + feed-forward layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'cross_attn': MultiHeadCrossAttention(d_model, n_heads, factor, dropout),
                'norm1': nn.LayerNorm(d_model),
                'ff': PositionwiseFeedForward(d_model, d_ff, dropout),
                'norm2': nn.LayerNorm(d_model)
            }))
        self.dropout = nn.Dropout(dropout)
        # Final projection to scalar
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()          # ➜ 输出∈[0,1]
        )
        nn.init.constant_(self.proj[2].bias, 0.8)   # 最后一层 Linear 的 bias

    def forward(
        self,
        enc_out: torch.Tensor,
        x_time_dec: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            enc_out: Encoder output, shape (B, L_enc, d_model)
            x_time_dec: Time features for future steps, shape (B, pred_len, 3)
        Returns:
            Tensor of shape (B, pred_len)
        """
        B = enc_out.size(0)
        # Expand query embeddings to batch
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        # Project time features and add
        t = self.time_proj(x_time_dec)
        x = queries + t
        # Self-attention among queries
        x2 = self.self_attn(x, x)
        x = self.norm0(x + self.dropout(x2))
        # Cross-attention + feed-forward layers
        for layer in self.layers:
            x2 = layer['cross_attn'](x, enc_out)
            x = layer['norm1'](x + self.dropout(x2))
            x2 = layer['ff'](x)
            x = layer['norm2'](x + self.dropout(x2))
        # Project to scalar predictions
        out_raw = self.proj(x).squeeze(-1)          # (B, pred_len) ∈ [0,1]
        # 取 Encoder 最后一个时间步 target 作为基准（已是 scaled 值）
        last_scaled = enc_out[:, -1, -1]            # shape (B,)

        # 预测“增量”再加回基准
        pred_scaled = out_raw + last_scaled.unsqueeze(-1)

        return pred_scaled