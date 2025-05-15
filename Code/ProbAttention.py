import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

class ProbAttention(nn.Module):
    """
    ProbSparse Attention for Informer:
    - First, perform sampling on the Keys to quickly estimate the importance of each Query;
    - Then, compute exact attention for the top-u Queries, and approximate the rest using mean or cumulative sum.
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Q: (B, H, L_Q, D)
        K: (B, H, L_K, D)
       sample_k: number of sampled Keys ≈ factor × ln(L_K)
       n_top: number of top Queries selected ≈ factor × ln(L_Q)
        return:
          scores_top: (B, H, n_top, L_K)
       index:  (n_top,) indices of the selected top Queries in the original sequence for each head
        """
        B, H, L_K, D = K.shape
        _, _, L_Q, _ = Q.shape

        # Expand K to align with each Q
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        # Randomly sample sample_k positions from the Keys
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # (B, H, L_Q, sample_k, D)

        # First compute the Q-K similarity using the sampled Keys
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)  # (B, H, L_Q, sample_k)

        # Sparsity metric: max minus mean
        M = Q_K_sample.max(-1)[0] - Q_K_sample.mean(-1)
        # Select the top-u most important Queries
        n_top = min(n_top, L_Q)
        M_top = M.topk(n_top, sorted=False)[1]  # (B, H, u)

        # Compute full QK attention for these selected Queries
        Q_reduce = torch.gather(Q, 2, M_top.unsqueeze(-1).expand(-1, -1, -1, D))
        # 全量计算相似度得分
        scores_top = torch.matmul(Q_reduce, K.transpose(-2, -1))  # (B, H, u, L_K)

        return scores_top, M_top

    def _get_initial_context(self, V, L_Q):
        """
        For Queries not in the top set, approximate the context using mean or cumulative sum
        V: (B, H, L_V, D)
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # Mean concatenation
            V_mean = V.mean(dim=2)  # (B, H, D)
            context = V_mean.unsqueeze(2).expand(B, H, L_Q, D).clone()
        else:
            # Cumulative sum: only applicable to self-attention and requires L_Q == L_V
            context = V.cumsum(dim=2)
        return context  # (B, H, L_Q, D)

    def _update_context(self, context, V, scores, index, L_Q):
        """
Use the exact attention scores to update the context at the positions of the top Queries

context: (B, H, L_Q, D) — initial context tensor

scores: (B, H, u, L_K) — exact attention scores for the top-u Queries

index: (B, H, u) — indices of the top-u Queries in the original sequence

        """
        B, H, L_V, D = V.shape
        attn = torch.softmax(scores, dim=-1)  # (B, H, u, L_K)
        context_v = torch.matmul(attn, V)     # (B, H, u, D)
        context_new = context.clone()
        index_expanded = index.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, H, u, D)
        context_new.scatter_(2, index_expanded, context_v)
        
        return context_new, attn

    def forward(self, queries, keys, values, attn_mask=None):
        """
        queries: (B, L_Q, H, D)
        keys:    (B, L_K, H, D)
        values:  (B, L_K, H, D)
        """
        # Reshape to (B, H, L, D)
        Q = queries.transpose(1, 2)
        K = keys.transpose(1, 2)
        V = values.transpose(1, 2)
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # Compute the number of samples and the number of top Queries
        sample_k = min(self.factor * int(np.ceil(np.log(L_K))), L_K)
        n_top    = min(self.factor * int(np.ceil(np.log(L_Q))), L_Q)

        #Step 1: Sampling-based estimation & top Query selection
        scores_top, index = self._prob_QK(Q, K, sample_k, n_top)

        #  Scaling
        scale = self.scale or 1.0 / sqrt(D)
        scores_top = scores_top * scale

        # Construct the initial context
        context = self._get_initial_context(V, L_Q)

        # Update the context for the top Queries
        context, attn = self._update_context(context, V, scores_top, index, L_Q)

        # Reshape back to (B, L_Q, H, D)
        context = context.transpose(1, 2).contiguous()
        if self.output_attention:
            return context, attn
        else:
            return context, None
