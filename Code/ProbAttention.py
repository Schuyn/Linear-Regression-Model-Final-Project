import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

class ProbAttention(nn.Module):
    """
    ProbSparse Attention for Informer:
    - 先对 Key 做抽样，快速估算每个 Query 的重要性；
    - 只对 top u 个 Query 计算精确注意力，其余用均值/累积和近似。
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
        sample_k: 抽样 Key 数量 ≈ factor * ln(L_K)
        n_top: 选取最重要的 Query 数量 ≈ factor * ln(L_Q)
        返回:
          scores_top: (B, H, n_top, L_K)
          index:      (n_top,) 每头选出的 Query 在原序列中的下标
        """
        B, H, L_K, D = K.shape
        _, _, L_Q, _ = Q.shape

        # 扩展 K 以便对应每个 Q
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        # 在 Key 上随机抽样 sample_k 个位置
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # (B, H, L_Q, sample_k, D)

        # 先用抽样 Key 计算 Q-K 相似度
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)  # (B, H, L_Q, sample_k)

        # 稀疏度度量：max - 平均
        M = Q_K_sample.max(-1)[0] - Q_K_sample.mean(-1)
        # 选出 top u 个最重要的 Query
        n_top = min(n_top, L_Q)
        M_top = M.topk(n_top, sorted=False)[1]  # (B, H, u)

        # 对这部分 Query 计算全量 QK
        Q_reduce = torch.gather(Q, 2, M_top.unsqueeze(-1).expand(-1, -1, -1, D))
        # 全量计算相似度得分
        scores_top = torch.matmul(Q_reduce, K.transpose(-2, -1))  # (B, H, u, L_K)

        return scores_top, M_top

    def _get_initial_context(self, V, L_Q):
        """
        对不在 top Query 的位置，用均值或累积和近似上下文
        V: (B, H, L_V, D)
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # 均值拼接
            V_mean = V.mean(dim=2)  # (B, H, D)
            context = V_mean.unsqueeze(2).expand(B, H, L_Q, D).clone()
        else:
            # 累积和：只能用于自注意力，要求 L_Q == L_V
            context = V.cumsum(dim=2)
        return context  # (B, H, L_Q, D)

    def _update_context(self, context, V, scores, index, L_Q):
        """
        用精确计算的 attention 更新 top Query 的上下文位置
        context: (B, H, L_Q, D) 初始上下文
        scores:  (B, H, u, L_K) 精确得分
        index:   (B, H, u)         top Query 下标
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
        # 调整到 (B, H, L, D)
        Q = queries.transpose(1, 2)
        K = keys.transpose(1, 2)
        V = values.transpose(1, 2)
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # 计算样本数和 top 数
        sample_k = min(self.factor * int(np.ceil(np.log(L_K))), L_K)
        n_top    = min(self.factor * int(np.ceil(np.log(L_Q))), L_Q)

        # 第一步：抽样估算 & 筛出 top Query
        scores_top, index = self._prob_QK(Q, K, sample_k, n_top)

        # 缩放
        scale = self.scale or 1.0 / sqrt(D)
        scores_top = scores_top * scale

        # 构造初始上下文
        context = self._get_initial_context(V, L_Q)

        # 更新 top Query 的上下文
        context, attn = self._update_context(context, V, scores_top, index, L_Q)

        # 转回 (B, L_Q, H, D)
        context = context.transpose(1, 2).contiguous()
        if self.output_attention:
            return context, attn
        else:
            return context, None